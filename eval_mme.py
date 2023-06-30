# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from lavin.eval_model import ModelArgs, Transformer
from lavin.tokenizer import Tokenizer
from lavin.generator import LaVIN_Generator
from lavin.mm_adapter import set_MMAdapter,set_Clip_Adapter
from util.base_prompt import build_prompt
from dataclasses import dataclass
import re
import random

import warnings
import pandas as pd
from PIL import Image

from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist
from util.apply_delta import apply_model_delta_online

warnings.filterwarnings('ignore')

@dataclass
class PromptArgs:
    prompt_format='QCM-ALE'
    use_caption=True
    options=["A", "B", "C", "D", "E"]

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]))
        checkpoint = torch.load(checkpoints[mp_rank], map_location='cpu')
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
            loaded = []
            for x in checkpoints:
                print('loading from', x)
                loaded.append(torch.load(x, map_location='cpu'))

            full_state_dict = {}
            split_dims = {}

            def add_weight_with_split_dim(name, dim):
                if dim < 0:  # bcast without split
                    full_state_dict[name] = loaded[0][name].clone()
                else:
                    full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
                for x in loaded:
                    del x[name]
                split_dims[name] = dim

            add_weight_with_split_dim('tok_embeddings.weight', 1)
            add_weight_with_split_dim('norm.weight', -1)
            add_weight_with_split_dim('output.weight', 0)
            for i in range(params['n_layers']):
                print('gathering layer %d of %d' % (i, params['n_layers']))
                layer_prefix = f'layers.{i}.'
                bcast_names = [
                    'attention_norm.weight',
                    'ffn_norm.weight',
                ]
                column_parallel_names = [
                    'attention.wq.weight',
                    'attention.wk.weight',
                    'attention.wv.weight',
                    'feed_forward.w1.weight',
                    'feed_forward.w3.weight',
                ]
                row_parallel_names = [
                    'attention.wo.weight',
                    'feed_forward.w2.weight',
                ]
                for key in bcast_names:
                    add_weight_with_split_dim(layer_prefix + key, -1)
                for key in column_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 0)
                for key in row_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 1)

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else:  # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st: shard_ed]
                elif dim == 1:
                    value = value[:, shard_st: shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params



def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_file, data_file):
    # read result file
    results = json.load(open(result_file))
    num = len(results)
    assert num == 4241

    sqa_data = json.load(open(data_file))

    # construct pandas data
    sqa_pd = pd.DataFrame(sqa_data).T
    res_pd = sqa_pd[sqa_pd['split'] == 'test']  # test set

    # update data
    for index, row in res_pd.iterrows():

        res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
        res_pd.loc[index, 'has_text'] = True if row['hint'] else False
        res_pd.loc[index, 'has_image'] = True if row['image'] else False
        res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

        label = row['answer']
        pred = int(results[index])
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100

    scores = {
        'acc_natural':
        get_acc_with_contion(res_pd, 'subject', 'natural science'),
        'acc_social':
        get_acc_with_contion(res_pd, 'subject', 'social science'),
        'acc_language':
        get_acc_with_contion(res_pd, 'subject', 'language science'),
        'acc_has_text':
        get_acc_with_contion(res_pd, 'has_text', True),
        'acc_has_image':
        get_acc_with_contion(res_pd, 'has_image', True),
        'acc_no_context':
        get_acc_with_contion(res_pd, 'no_context', True),
        'acc_grade_1_6':
        get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12':
        get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average':
        "{:.2f}".format(acc_average),
    }

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)

def load(
    ckpt_dir: str,
llm_model:str,
    tokenizer_path: str,
    adapter_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    adapter_type: str,
    adapter_dim:int,
    adapter_scale:float,
    hidden_proj:int,
    visual_adapter_type: str,
    temperature: float,
use_vicuna: bool
) -> LaVIN_Generator:
    start_time = time.time()
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(ckpt_dir, llm_model)

    print("Loading")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")


    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,hidden_proj=hidden_proj, **params
    )
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    set_MMAdapter(model, adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)
    set_Clip_Adapter(model.backbone.visual, visual_adapter_type, dim=adapter_dim, s=adapter_scale,t=temperature)

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    if use_vicuna:
        apply_model_delta_online(model,'../data/weights/vicuna_'+llm_model)

    state_dict={}
    for key in adapter_checkpoint['model']:
        state_dict[key.replace('module.','')]=adapter_checkpoint['model'][key]

    model.load_state_dict(state_dict, strict=False)

    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))

def prepare_data(dir):
    if os.path.exists(os.path.join(dir,'images')):
        image_dir=os.path.join(dir,'images')
    else:
        image_dir=dir
    if os.path.exists(os.path.join(dir,'questions_answers_YN')):
        ann_dir=os.path.join(dir,'questions_answers_YN')
    else:
        ann_dir=dir
    image_list=[]
    image_path=[]
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            # 检查文件后缀名是否为.jpg或.png
            if file.endswith(".jpg") or file.endswith(".png"):
                # 拼接文件的完整路径
                image_list.append(file)
                image_path.append(os.path.join(image_dir,file))
    ann_list=[]
    for img_id in image_list:
        ann_file=img_id.replace('.jpg','.txt').replace('.png','.txt')
        ann={}
        with open(os.path.join(ann_dir,ann_file)) as f:
            pos=f.readline().split('\t')[0]
            neg=f.readline().split('\t')[0]
            ann['pos']=pos
            ann['neg']=neg
        ann_list.append(ann)

    return image_path,ann_list


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    adapter_path: str,
    data_root:str,
    caption_file:str,
    max_seq_len: int,
    max_batch_size: int,
    llm_model:str='7B',
    generation_temperature: float = 0.1,
    top_p: float = 0.75,
    split='val',
    prompt_format='QCM-ALE',
    use_caption=False,
    options=["A", "B", "C", "D", "E"],
    adapter_type='repattn',
    adapter_dim=8,
    adapter_scale=1,
    n_prompt=10,
    hidden_proj=128,
    visual_adapter_type='normal',
    temperature=10.,
    use_vicuna=False,
root_dir_='../data/mme'
):
    print(max_batch_size,max_seq_len)
    print('use caption: ',use_caption)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir,llm_model, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size,
        adapter_type,adapter_dim,adapter_scale,hidden_proj,visual_adapter_type,
    temperature,use_vicuna)

    subsets=os.listdir(root_dir_)
    total_score=0
    cognition_score=0
    perception_score=0
    for subset in subsets:
        root_dir=os.path.join(root_dir_,subset)
        print('split: ', subset)
        img_list,ann_list=prepare_data(root_dir)
        qids=range(len(img_list))
        total_items=len(img_list)
        print('total_items: ',total_items)


        image_transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

        prompt_args=PromptArgs()
        prompt_args.prompt_format = prompt_format
        prompt_args.use_caption = use_caption
        prompt_args.options = options

        pattern = re.compile(r'The answer is ([A-Z]).')

        answers = []
        preds=[]
        max_batch_size=8
        for i in range(total_items//max_batch_size+1):
            batch_qids=qids[i*max_batch_size:(i+1)*max_batch_size]
            if len(batch_qids)==0:
                break
            indicators = []
            prompts=[]
            images = []
            for qid in batch_qids:
                #pos
                prompt= 'Instruction: '+ ann_list[qid]['pos']+'\n'+\
                               'Response: '
                prompt = prompt.replace("  ", " ").strip()
                answer='yes'
                image = Image.open(img_list[qid]).convert('RGB')
                image = image_transforms(image)
                indicator = 1
                prompts.append(prompt)
                answers.append(answer)
                images.append(image.unsqueeze(0))
                indicators.append(indicator)

                #neg
                prompt= 'Instruction: '+ ann_list[qid]['neg']+'\n'+\
                               'Response: '
                prompt = prompt.replace("  ", " ").strip()
                answer='no'
                indicator = 1
                prompts.append(prompt)
                answers.append(answer)
                images.append(image.unsqueeze(0))
                indicators.append(indicator)
            images=torch.cat(images,0)


            results = generator.generate(
                prompts,images=images,indicators=indicators, max_gen_len=20, temperature=generation_temperature, top_p=top_p,n_feats=n_prompt
            )

            for result in results:
                result=result.lower().strip().split('response:')[1]
                if 'yes' in result[:4]:
                    pred='yes'
                elif 'no' in result[:4]:
                    pred='no'
                else:
                    pred='other'
                preds.append(pred)

        #evaluations
        correct=0
        corrects=[]
        assert len(preds)==len(answers)
        for i, prediction in enumerate(preds):
            if prediction == answers[i]:
                correct += 1
                corrects.append(1)
            else:
                corrects.append(0)
        import numpy as np
        corrects=np.array(corrects)
        acc = correct / len(preds) * 100
        acc_plus= (corrects.reshape(-1,2).sum(1)==2).sum()/ (len(preds)//2)* 100
        total_score+=acc
        total_score+=acc_plus
        if subset in ['commonsense_reasoning','numerical_calculation','text_translation','code_reasoning']:
            cognition_score+=acc_plus
            cognition_score+=acc
        else:
            perception_score+=acc_plus
            perception_score+=acc
        print('subset: ', subset)
        print('overall accuracy: ', acc)
        print('overall accuracy+: ', acc_plus)
        with open('mme_eval.txt','a') as f:
            f.write('subset: '+ subset+'\n')
            f.write('accuracy: '+ str(acc)+'\n')
            f.write('accuracy+: '+ str(acc_plus)+'\n')

    print('total_score: ',total_score)
    print('perception_score: ',perception_score)
    print('cognition_score: ',cognition_score)
    with open('mme_eval.txt', 'a') as f:
        f.write('total_score: ' + str(total_score) + '\n')
        f.write('perception_score: ' + str(perception_score) + '\n')
        f.write('cognition_score: ' + str(cognition_score) + '\n')

if __name__ == "__main__":
    fire.Fire(main)

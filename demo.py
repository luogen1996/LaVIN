import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image
# from minigpt4.common.config import Config
from util.misc import get_rank
# from minigpt4.common.registry import registry
from conversation.conversation import Chat, CONV_VISION
from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from eval import load
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from typing import Tuple
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="server name")
    parser.add_argument("--ckpt_dir", type=str, default="../data/weights/", help="dir of pre-trained weights.")
    parser.add_argument("--llm_model", type=str, default="13B", help="the type of llm.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="decoder length")
    parser.add_argument('--adapter_type', type=str, default='attn', metavar='LENGTH',choices=['block','attn'],
                        help='the insert position  of adapter layer')
    parser.add_argument('--adapter_path', type=str, default='./15-eph-pretrain.pth',  help='path of pre-trained adapter')
    parser.add_argument('--temperature', type=float, default=5., metavar='LENGTH',
                        help='the temperature of router')
    parser.add_argument('--use_vicuna',  action='store_true',   help='use vicuna weights')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()


local_rank, world_size = setup_model_parallel()
lavin=load(
    ckpt_dir=args.ckpt_dir,
    llm_model=args.llm_model,
    adapter_path=args.adapter_path,
    max_seq_len=512,
    max_batch_size=4,
    adapter_type='attn',
    adapter_dim=8,
    adapter_scale=1,
    hidden_proj=128,
    visual_adapter_type='router',
    temperature=args.temperature,
    tokenizer_path='',
    local_rank=local_rank,
    world_size=world_size,
    use_vicuna=args.use_vicuna
)

vis_processor = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
chat = Chat(lavin, vis_processor, device=torch.device('cuda'))
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Type and press Enter',
                                                                    interactive=True), gr.update(
        value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(
        value="Start Chatting", interactive=False), chat_state, img_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    if chat_state is None:
        chat_state=CONV_VISION.copy()
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


title = """<h1 align="center">Demo of LaVIN</h1>"""
description = """<h3>This is the demo of LaVIN. Upload your images and start chatting!</h3>"""



with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='LaVIN-13B')
            text_input = gr.Textbox(label='User', placeholder='Type and press Enter', interactive=True)

    upload_button.click(upload_img, [image, text_input, chat_state],
                        [image, text_input, upload_button, chat_state, img_list])

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list],
                queue=False)

demo.launch(share=True, enable_queue=True,server_name=args.server_name)
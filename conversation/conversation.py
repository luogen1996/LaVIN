import argparse
import time
from PIL import Image

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import re

ERROR_CODE= [260, 1794, 11440]
ERROR_MESSAGE=[1, 7423, 29892, 474, 508, 29915, 29873, 1234, 445, 1139, 29889, 2]


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = ""
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="",
    roles=("Instruction", "Responese"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
)



class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.lavin = model
        self.vis_processor = vis_processor

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000,n_feats=6):
        conv.append_message(conv.roles[1], None)
        prompt, indicator,  img = self.get_context_emb(conv, img_list)

        current_max_len = len(prompt) + max_new_tokens+n_feats
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        prompt = prompt[begin_idx:]
        CODE=self.lavin.tokenizer.encode(prompt, bos=False, eos=False)
        if ERROR_CODE in [CODE[i:i+len(ERROR_CODE)] for i in range(len(CODE)-len(ERROR_CODE)+1)]:
            output_text=self.lavin.tokenizer.decode(ERROR_MESSAGE).split('Responese:')[-1].strip()
        else:
            outputs = self.lavin.generate(
                prompts= [prompt],
                images= img,
                indicators=[indicator],
                max_gen_len=max_length,
                n_feats=n_feats,
                temperature = 0.1,
                top_p = 0.75,
            )

            output_text = outputs[0].split('Responese:')[-1].strip()

        conv.messages[-1][1] = output_text
        return output_text

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        # image_emb, _ = self.lavin.backbone.encode_img(image)
        img_list.append(image)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()

        if '<Img><ImageHere></Img>' in prompt:
            indicator=1
            prompt=prompt.replace('<Img><ImageHere></Img>','')
        else:
            indicator=0
        assert img_list is None or len(img_list) <=  1

        return prompt, indicator,  img_list[0] if indicator==1 else torch.Tensor(torch.zeros(1,3, 224, 224).float())
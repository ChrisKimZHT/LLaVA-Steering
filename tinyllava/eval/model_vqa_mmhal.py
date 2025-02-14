import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from PIL import Image
import math, random

import requests
from io import BytesIO


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):

    def load_image(image_file):
        file_name = image_file.split('/')[-1]
        local_image_file = os.path.join(args.image_folder, file_name)
        if os.path.exists(local_image_file):
            image = Image.open(local_image_file).convert('RGB')
        else:
            if image_file.startswith('http') or image_file.startswith('https'):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                raise ValueError("image_path doesnot exist or isnot a valid http link.")
        return image

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    
    # text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    model.load_mores_adaptor_from_ckpt(model_path) if os.path.isdir(model_path) and 'mores' in os.listdir(model_path) else None
    text_processor = TextPreprocessMoReS(tokenizer, args.conv_mode, model.mores_pos_configs) \
        if hasattr(model, 'mores_pos_configs') else TextPreprocess(tokenizer, args.conv_mode)
    
    data = json.load(open(args.response_template, 'r'))

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    model.to(device='cuda')
    for i, line in enumerate(tqdm(data)):
        image_src = line['image_src']
        image = load_image(image_src)
        question = line['question']

        image_sizes = [image.size]
        image = image_processor(image)
        images = image.unsqueeze(0).half().cuda()

        question = "<image>\n" + question

        msg = Message()
        msg.add_message(question)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        input_ids = input_ids.unsqueeze(0).cuda()
        kwargs = dict(
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id
        )
        kwargs['intervention_locations'] = torch.tensor([result['intervention_locations']]) if 'intervention_locations' in result and result['intervention_locations'] is not None else None
        kwargs['len_two_pos_configs'] = [list(result['len_two_pos_configs'])] if 'len_two_pos_configs' in result and result['len_two_pos_configs'] != None else None
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                **kwargs
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        line['model_answer'] = outputs

    with open(answers_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/two_stage_pretrain/base-lora-zero2-r128")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/eval_tinyllava/mmhal-bench/images")
    parser.add_argument("--response_template", type=str, default="/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/eval_tinyllava/mmhal-bench/response_template.json")
    parser.add_argument("--answers-file", type=str, default="/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/eval_tinyllava/mmhal-bench/answers/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    args = parser.parse_args()

    eval_model(args)



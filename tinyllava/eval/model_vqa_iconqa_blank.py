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
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def generate_questions_imgs(test_folder):
    data_list = []
    test_folder = os.path.expanduser(test_folder)
    ids = os.listdir(test_folder)
    for id in ids:
        img_abs_path = os.path.join(os.path.join(test_folder, id), 'image.png')
        json_path = os.path.join(os.path.join(test_folder, id), 'data.json')
        data = json.load(open(json_path, "r"))
        data_list.append({
            'id': id,
            'question': data['question'],
            'answer': data['answer'],
            'image': img_abs_path
        })
    return data_list

def eval_model(args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    model.load_mores_adaptor_from_ckpt(model_path) if os.path.isdir(model_path) and 'mores' in os.listdir(model_path) else None
    text_processor = TextPreprocessMoReS(tokenizer, args.conv_mode, model.mores_pos_configs) \
        if hasattr(model, 'mores_pos_configs') else TextPreprocess(tokenizer, args.conv_mode)
    
    data_list = generate_questions_imgs(args.test_folder)
    # questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    data_list = get_chunk(data_list, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    model.to(device='cuda')
    for i, line in enumerate(tqdm(data_list)):
        idx = line["id"]
        question = line['question']
        question = question.replace('<image>', '').strip()
        if 'image' in line:
            image_file = line["image"]
            image = Image.open(image_file)
            image_sizes = [image.size]
            image = image_processor(image)
            images = image.unsqueeze(0).half().cuda()
            question = '<image>' + '\n' + question
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            question = question + ' ' + "Fill in the blanks in (_) or answer this question"
        msg = Message()
        msg.add_message(question)

        result = text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']
        prompt = result['prompt']
        input_ids = input_ids.unsqueeze(0).cuda()
        kwargs = dict(
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
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
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_path.split('/')[-1],
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/two_stage_pretrain/base-lora-zero2-r128")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--test-folder", type=str, default='iconqa/test/fill_in_blank')
    parser.add_argument("--answers-file", type=str, default="answer_sqa.jsonl")
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    args = parser.parse_args()

    eval_model(args)



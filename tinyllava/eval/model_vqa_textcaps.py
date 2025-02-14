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


def eval_model(args):

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
    
    data = json.load(open(os.path.expanduser(args.question_file), "r"))['data']
    data = get_chunk(data, args.num_chunks, args.chunk_idx)
    ocr_tokens = json.load(open(os.path.expanduser(args.token_file), "r"))['data']
    ocr_tokens_dict = {elem['image_id']: elem['ocr_tokens'] for elem in ocr_tokens}
    ocr_tokens_keys = list(ocr_tokens_dict.keys())
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    model.to(device='cuda')
    for i, line in enumerate(tqdm(data)):
        idx = line["image_id"]
        
        image_file = line["image_name"]+'.jpg'
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_sizes = [image.size]
        if image.layers == 1:
            image = Image.merge("RGB", (image, image, image))
        elif image.mode == "CMYK":
            image = image.convert("RGB")
        image = image_processor(image)
        images = image.unsqueeze(0).half().cuda()


        question = "<image>\nProvide a one-sentence caption for the provided image.\nReference OCR token: "
        if idx in ocr_tokens_keys:
            for ocr_token in ocr_tokens_dict[idx]:
                question += ocr_token + ' '
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
        ans_file.write(json.dumps({
                                "image_id": idx,
                                "caption": outputs
                                }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/atuin/b211dd/b211dd19/data/checkpoints/llava_factory/two_stage_pretrain/base-lora-zero2-r128")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/eval_tinyllava/textcaps/test_images")
    parser.add_argument("--question-file", type=str, default="/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/eval_tinyllava/textcaps/TextCaps_0.1_test.json")
    parser.add_argument("--answers-file", type=str, default="answer_sqa.jsonl")
    parser.add_argument("--token-file", type=str, default="/home/atuin/b211dd/b211dd19/data/dataset/tinyllava/eval/eval_tinyllava/textcaps/TextVQA_Rosetta_OCR_v0.2_test.json")
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    args = parser.parse_args()

    eval_model(args)



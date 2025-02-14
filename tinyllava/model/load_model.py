import os
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from huggingface_hub import HfApi

from .modeling_tinyllava import TinyLlavaForConditionalGeneration
from .configuration_tinyllava import TinyLlavaConfig
 
def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('.base_layer', '')
        new_ckp[new_k] = v
    return new_ckp

def load_base_ckp_for_adapter(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        if 'adapter_' not in k:
            new_k = k.replace('base_model.model.', '')
            new_ckp[new_k] = v
    return new_ckp

def load_base_ckp_for_ia3(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        if 'ia3_' not in k:
            new_k = k.replace('base_model.model.', '')
            new_k = new_k.replace('base_layer.', '')
            new_ckp[new_k] = v
    return new_ckp

def generate_model_args(config_in_model, **kwargs):
    # TODO: do not consider model_name_or_path2 (for vision_tower 2)
    pretrained_model_path = kwargs['pretrained_model_path'] if 'pretrained_model_path' in kwargs else None

    model_args = {
        "llm": {
            "model_name_or_path": config_in_model.llm_model_name_or_path,
            "cache_dir": config_in_model.cache_dir,
            "torch_dtype": getattr(kwargs, "torch_dtype", torch.float16),
            "pretrained_llm_path": os.path.join(pretrained_model_path, "language_model")
        },
        "vision_tower": {
            "model_name_or_path": config_in_model.vision_model_name_or_path,
            "pretrained_vision_tower_path": os.path.join(pretrained_model_path, "vision_tower")
        },
        "connector": {
            "connector_type": config_in_model.connector_type,
            "pretrained_connector_path": os.path.join(pretrained_model_path, "connector")
        }
    }
    if not ("attn_implementation" in kwargs and kwargs["attn_implementation"] == ""):
        model_args['llm']['attn_implementation'] = getattr(kwargs, "attn_implementation", "flash_attention_2")
    return model_args

def load_(pretrained_model_path, model, model_args={}):
    # same as training_recipe.load()
    # if not ('lora' in pretrained_model_path and os.path.exists(os.path.join(pretrained_model_path, 'adapter_config.json'))): # loading model for non-lora/non-qlora pretraining
    #     model.load_llm(**model_args['llm'])
    #     model.load_vision_tower(**model_args['vision_tower'])
    #     model.load_connector(**model_args['connector'])
    # else:
    #     model.language_model = model.language_model.from_pretrained(model_args['llm']['model_name_or_path'],attn_implementation='flash_attention_2',torch_dtype=model_args['llm']['torch_dtype'])
    #     model.load_vision_tower(**model_args['vision_tower'])
    #     model.load_connector(**model_args['connector'])
    #     model.to(model_args['llm']['torch_dtype'])
    #     from peft import PeftModel
    #     print('Loading LoRA weights...')
    #     model = PeftModel.from_pretrained(model, pretrained_model_path)
    #     print('Merging LoRA weights...')
    #     model = model.merge_and_unload()
    #     print('Model is loaded...')

    model.load_llm(**model_args['llm'])
    model.load_vision_tower(**model_args['vision_tower'])
    model.load_connector(**model_args['connector'])
    return model
    
def is_valid_hf_repo(repo_id):
    """
    Check if the given Hugging Face repository ID exists.

    Parameters:
    - repo_id (str): The full ID of the repository, for example, "username/repo-name"

    Returns:
    - bool: Returns True if the repository exists, otherwise False
    """
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def load_pretrained_model(model_name_or_path, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if os.path.isdir(model_name_or_path):
        if 'lora' in model_name_or_path:
            if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
                model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
                model = TinyLlavaForConditionalGeneration(model_config)
                language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
                language_model_ckp = load_base_ckp_for_lora(language_model_ckp_path)
                model.language_model.load_state_dict(language_model_ckp)
                vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
                vision_tower_ckp = load_base_ckp_for_lora(vision_tower_ckp_path)
                model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
                connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
                connector_ckp = load_base_ckp_for_lora(connector_ckp_path)
                model.connector.load_state_dict(connector_ckp)
                model.to(torch.float16)
                from peft import PeftModel
                print('Loading LoRA weights...')
                model = PeftModel.from_pretrained(model, model_name_or_path)
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
                print('Model is loaded...')
            else:
                raise ValueError("No adapter_config.json when loading peft model!")
        elif 'adapter' in model_name_or_path:
            if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
                model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
                model = TinyLlavaForConditionalGeneration(model_config)
                language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
                language_model_ckp = load_base_ckp_for_adapter(language_model_ckp_path)
                model.language_model.load_state_dict(language_model_ckp)
                vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
                vision_tower_ckp = load_base_ckp_for_adapter(vision_tower_ckp_path)
                model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
                connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
                connector_ckp = load_base_ckp_for_adapter(connector_ckp_path)
                model.connector.load_state_dict(connector_ckp)

                from ..utils.peft.src.peft import PeftModel
                print('Loading adapter-Peftmodel...')
                model.language_model = PeftModel.from_pretrained(model.language_model, model_name_or_path)
                model.to(torch.float16)
            else:
                raise ValueError("No adapter_config.json when loading peft model!")
        elif 'ia3' in model_name_or_path:
            if os.path.exists(os.path.join(model_name_or_path, 'adapter_config.json')):
                model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
                model = TinyLlavaForConditionalGeneration(model_config)
                language_model_ckp_path = os.path.join(model_name_or_path, 'language_model/pytorch_model.bin')
                language_model_ckp = load_base_ckp_for_ia3(language_model_ckp_path)
                model.language_model.load_state_dict(language_model_ckp)
                vision_tower_ckp_path = os.path.join(model_name_or_path, 'vision_tower/pytorch_model.bin')
                vision_tower_ckp = load_base_ckp_for_ia3(vision_tower_ckp_path)
                model.vision_tower._vision_tower.load_state_dict(vision_tower_ckp)
                connector_ckp_path = os.path.join(model_name_or_path, 'connector/pytorch_model.bin')
                connector_ckp = load_base_ckp_for_ia3(connector_ckp_path)
                model.connector.load_state_dict(connector_ckp)

                from peft import PeftModel
                print('Loading IA3-Peftmodel...')
                model.language_model = PeftModel.from_pretrained(model.language_model, model_name_or_path)
                model.to(torch.float16)
            else:
                raise ValueError("No adapter_config.json when loading peft model!")
        else:
            # model = TinyLlavaForConditionalGeneration.from_pretrained(model_name_or_path,low_cpu_mem_usage=True,torch_dtype=torch.float16)
            model_config = TinyLlavaConfig.from_pretrained(model_name_or_path)
            model = TinyLlavaForConditionalGeneration(model_config)

            model_args = generate_model_args(config_in_model=model_config,
                                            pretrained_model_path=model_name_or_path,
                                            attn_implementation="")
        
            model = load_(model_name_or_path, model, model_args)
            model.to(torch.float16)

    elif is_valid_hf_repo(model_name_or_path):
        model = TinyLlavaForConditionalGeneration.from_pretrained(model_name_or_path, trust_remote_code=True)
    else:
        raise ValueError("model_name_or_path is neither a local ckpt path nor a huggingface repo!")
        
    image_processor = model.vision_tower._image_processor
    context_len = getattr(model.config, 'max_sequence_length', 2048)
    # tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_name_or_path, use_fast=False, padding_side="right")
    tokenizer = model.tokenizer
    #tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, image_processor, context_len

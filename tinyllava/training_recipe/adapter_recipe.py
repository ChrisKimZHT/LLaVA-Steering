import os

from collections import OrderedDict

import torch
from transformers import BitsAndBytesConfig

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils.train_utils import *
from ..utils import log
from ..utils.peft.src.peft import (  # noqa: E402
    BottleneckConfig,
    get_peft_model
)


ADAPTER_TARGET_MODULES_MAPPING = {
    'llama': ["gate_proj", "up_proj", "down_proj"],
    'phi': ['fc1', 'fc2']
}

@register_training_recipe('adapter')
class AdapterTrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.training_arguments = training_arguments
        self.adapter_skip_module = ['connector', 'vision_tower', 'language_model']
        
    def training_model_converse(self, model):
        if self.training_arguments.tune_type_connector == 'adapter':
            self.adapter_skip_module.remove('connector')
            raise NotImplementedError('adapter does NOT support connector now!')
        if self.training_arguments.tune_type_llm == 'adapter':
            self.adapter_skip_module.remove('language_model')
        if self.training_arguments.tune_type_vision_tower == 'adapter':
            self.adapter_skip_module.remove('vision_tower')
            raise NotImplementedError('adapter does NOT support vision tower now!')
        assert len(self.adapter_skip_module) != 3, 'There should be at least one of [llm, vt, connector] using adapter when AdapterTrainingRecipe is used.'
        if self.training_arguments.bits == 16:
            if self.training_arguments.bf16:
                model.to(torch.bfloat16)
            if self.training_arguments.fp16:
                model.to(torch.float16)
        if 'language_model' not in self.adapter_skip_module:
            bottleneck_config = BottleneckConfig(
                    bottleneck_size=self.training_arguments.adapter_bottleneck_size,
                    adapter_dropout=self.training_arguments.adapter_dropout,
                    use_parallel_adapter=self.training_arguments.adapter_use_parallel_adapter,
                    use_adapterp=self.training_arguments.adapter_use_adapterp,
                    target_modules=ADAPTER_TARGET_MODULES_MAPPING[model.language_model.config.model_type],
                    bias=self.training_arguments.adapter_bias,
                    task_type="CAUSAL_LM",
                )
            log("Adding Adapters ...")
            model.language_model = get_peft_model(model.language_model, bottleneck_config)  

        return model
        

    # Borrowed from peft.utils.get_peft_model_state_dict
    @staticmethod
    def get_peft_state_of_adapter_maybe_zero_3(named_params, bias):
        if bias == "none":
            to_return = {k: t for k, t in named_params if "adapter_" in k}
        elif bias == "all":
            to_return = {k: t for k, t in named_params if "adapter_" in k or "bias" in k}
        elif bias == "adapter_only":
            to_return = {}
            maybe_adapter_bias = {}
            adapter_bias_names = set()
            for k, t in named_params:
                if "adapter_" in k:
                    to_return[k] = t
                    bias_name = k.split("adapter_")[0] + "bias"
                    adapter_bias_names.add(bias_name)
                elif "bias" in k:
                    maybe_adapter_bias[k] = t
            for k, t in maybe_adapter_bias:
                if bias_name in adapter_bias_names:
                    to_return[bias_name] = t
        else:
            raise NotImplementedError
        to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
        return to_return
    
    def save(self, model, trainer):
        model.config.use_cache = True
        #save tokenizer       
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        #save entire model config
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        #save trainer
        trainer.save_state() 

        #save language model base params
        language_model_state_dict = get_peft_state_non_lora_maybe_zero_3(model.language_model.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)
        #save vision tower base params
        vision_tower_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(self.training_arguments.output_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            model.config.vision_config.save_pretrained(vision_tower_output_dir, from_pt=True)
        #save connector base params
        connector_state_dict = get_peft_state_non_lora_maybe_zero_3(model.connector.named_parameters(),  False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)
        
        # save adapter params
        adapter_state_dict = self.get_peft_state_of_adapter_maybe_zero_3(
            model.language_model.named_parameters(), self.training_arguments.adapter_bias
        )
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            # currently only supports for llm
            model.language_model.save_pretrained(self.training_arguments.output_dir, state_dict=adapter_state_dict)
        


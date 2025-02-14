import os
from typing import Any

from collections import OrderedDict

import torch
from transformers import BitsAndBytesConfig

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils.train_utils import *
from ..utils import log
# from ..utils.peft.src.peft import (  # noqa: E402
#     BottleneckConfig,
#     get_peft_model
# )
from peft import OFTModel, OFTConfig, get_peft_model


OFT_TARGET_MODULES_MAPPING = {
    'llama': ["k_proj", "q_proj", "v_proj"],
    # 'phi': ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    'phi': ["k_proj", "q_proj", "v_proj"]
}


# temporarily modify the function in peft library to solve a minor problem
from peft.tuners.oft.layer import Linear
def _get_delta_activations_(
    self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    delta_weight = self.get_delta_weight(adapter_name)

    base_layer = self.get_base_layer()
    base_weight = base_layer.weight.data
    delta_weight = delta_weight[: base_weight.shape[0], : base_weight.shape[0]]

    delta_weight = delta_weight.to(input.dtype)
    return torch.matmul(input, delta_weight)

Linear._get_delta_activations = _get_delta_activations_


@register_training_recipe('oft')
class OFTTrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.training_arguments = training_arguments
        self.oft_skip_module = ['connector', 'vision_tower', 'language_model']
        
    def training_model_converse(self, model):
        if self.training_arguments.tune_type_connector == 'oft':
            self.oft_skip_module.remove('connector')
            raise NotImplementedError('oft does NOT support connector now!')
        if self.training_arguments.tune_type_llm == 'oft':
            self.oft_skip_module.remove('language_model')
        if self.training_arguments.tune_type_vision_tower == 'oft':
            self.oft_skip_module.remove('vision_tower')
            raise NotImplementedError('oft does NOT support vision tower now!')
        assert len(self.oft_skip_module) != 3, 'There should be at least one of [llm, vt, connector] using oft when OFTTrainingRecipe is used.'
        # if self.training_arguments.bits == 16:
        #     if self.training_arguments.bf16:
        #         model.to(torch.bfloat16)
        #     if self.training_arguments.fp16:
        #         model.to(torch.float16)
        if 'language_model' not in self.oft_skip_module:
            oft_config = OFTConfig(
                r=self.training_arguments.oft_rank,
                target_modules=OFT_TARGET_MODULES_MAPPING[model.language_model.config.model_type],
                module_dropout=0.0,
                init_weights=True,
            )
            log("Adding OFT adapter ...")
            model.language_model = get_peft_model(model.language_model, oft_config)  
        if self.training_arguments.bits == 16:
            if self.training_arguments.bf16:
                model.to(torch.bfloat16)
            if self.training_arguments.fp16:
                model.to(torch.float16)
        return model
        
    @staticmethod
    def get_peft_state_of_oft_maybe_zero_3(named_params):
        to_return = {k: t for k, t in named_params if "oft_" in k}
        to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
        return to_return
    
    @staticmethod
    def get_peft_state_non_oft_maybe_zero_3(named_params, require_grad_only=True):
        to_return = {k: t for k, t in named_params if "oft" not in k}
        if require_grad_only:
            to_return = {k: t for k, t in to_return.items() if t.requires_grad}
        to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
        return to_return
    
    @staticmethod
    def modify_oft_peft_model_params(state_dict):
        return {k.replace('base_model.model.', '').replace('base_layer.', ''): v for k, v in state_dict.items()}

    def save(self, model, trainer):
        model.config.use_cache = True
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            #save tokenizer       
            model.tokenizer.save_pretrained(self.training_arguments.output_dir)
            #save entire model config
            model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
            #save trainer
            trainer.save_state() 

        #save language model base params
        language_model_state_dict = self.get_peft_state_non_oft_maybe_zero_3(model.language_model.named_parameters(), False)
        language_model_state_dict = self.modify_oft_peft_model_params(language_model_state_dict)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)
        #save vision tower base params
        vision_tower_state_dict = self.get_peft_state_non_oft_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(self.training_arguments.output_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            model.config.vision_config.save_pretrained(vision_tower_output_dir, from_pt=True)
        #save connector base params
        connector_state_dict = self.get_peft_state_non_oft_maybe_zero_3(model.connector.named_parameters(),  False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)
        
        # save adapter params
        ia3_state_dict = self.get_peft_state_of_oft_maybe_zero_3(model.language_model.named_parameters())
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            # currently only supports for llm
            model.language_model.save_pretrained(self.training_arguments.output_dir, state_dict=ia3_state_dict)
        


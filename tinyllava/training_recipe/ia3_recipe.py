import os

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
from peft import IA3Config, get_peft_model


IA3_TARGET_MODULES_MAPPING = {
    'llama': ["k_proj", "v_proj", "down_proj"],
    'phi': ["k_proj", "v_proj", "fc2"]
}

IA3_FEEDFORWARD_MODULES_MAPPING = {
    'llama': ["down_proj"],
    'phi': ['fc2']
}


@register_training_recipe('ia3')
class IA3TrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.training_arguments = training_arguments
        self.ia3_skip_module = ['connector', 'vision_tower', 'language_model']
        
    def training_model_converse(self, model):
        if self.training_arguments.tune_type_connector == 'ia3':
            self.ia3_skip_module.remove('connector')
            raise NotImplementedError('ia3 does NOT support connector now!')
        if self.training_arguments.tune_type_llm == 'ia3':
            self.ia3_skip_module.remove('language_model')
        if self.training_arguments.tune_type_vision_tower == 'ia3':
            self.ia3_skip_module.remove('vision_tower')
            raise NotImplementedError('ia3 does NOT support vision tower now!')
        assert len(self.ia3_skip_module) != 3, 'There should be at least one of [llm, vt, connector] using ia3 when IA3TrainingRecipe is used.'
        if self.training_arguments.bits == 16:
            if self.training_arguments.bf16:
                model.to(torch.bfloat16)
            if self.training_arguments.fp16:
                model.to(torch.float16)
        if 'language_model' not in self.ia3_skip_module:
            ia3_config = IA3Config(
                    target_modules=IA3_TARGET_MODULES_MAPPING[model.language_model.config.model_type],
                    feedforward_modules=IA3_FEEDFORWARD_MODULES_MAPPING[model.language_model.config.model_type],
                    task_type="CAUSAL_LM",
                )
            log("Adding IA3 adapter ...")
            model.language_model = get_peft_model(model.language_model, ia3_config)  

        return model
        
    @staticmethod
    def get_peft_state_of_ia3_maybe_zero_3(named_params):
        to_return = {k: t for k, t in named_params if "ia3_" in k}
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
        ia3_state_dict = self.get_peft_state_of_ia3_maybe_zero_3(model.language_model.named_parameters())
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            # currently only supports for llm
            model.language_model.save_pretrained(self.training_arguments.output_dir, state_dict=ia3_state_dict)
        


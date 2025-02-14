# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass, field

import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from ..utils import PeftType, PromptLearningConfig
import deepspeed
# torch.set_printoptions(profile="full")

@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PrefixEncoder`].

    Args:
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        prefix_projection (`bool`): Whether to project the prefix embeddings.
    """

    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the encoder"},
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to project the prefix tokens"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.PREFIX_TUNING
def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
class PrefixEncoder(torch.nn.Module):
    r"""
    The torch.nn model to encode the prefix

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example::

        >>> from peft import PrefixEncoder, PrefixTuningConfig >>> config = PrefixTuningConfig(
                peft_type="PREFIX_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20, token_dim=768,
                num_transformer_submodules=1, num_attention_heads=12, num_layers=12, encoder_hidden_size=768
            )
        >>> prefix_encoder = PrefixEncoder(config)


    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) --
            The embedding layer of the prefix encoder.
        - **transform** (`torch.nn.Sequential`) -- The
        two-layer MLP to transform the prefix embeddings if `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (batch_size, num_virtual_tokens)

    Output shape: (batch_size, num_virtual_tokens, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        self.m = []

        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)
        #self.previous_past_key_values = None
    def forward(self, prefix: torch.Tensor):
    #   for n,v in self.embedding.named_parameters():
    #        if hasattr(v, 'ds_id'):
    #            with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
    #                                                                       ]),
    #                                                  enabled=True):
    #               print(v)
    #               v_p = v.data.cpu()
    #               if(self.m!=[]):
    #                   print('打印是否有变化',self.m[-1].equal(v_p))
                        # diff_indices = torch.nonzero(self.m[-1] != v_p)
                        # print(diff_indices)
                        # for index in diff_indices:
                        #     xi = index[0]
                        #     xj = index[1]
                        #     print(f"Difference at position {(xi,xj)}: tensor1[{(xi,xj)}] = {self.m[-1][xi,xj]}, tensor2[{(xi,xj)}] = {v_p[xi,xj]}")

    #               else:
    #                   self.m.append(v_p)
    #       grad_data = deepspeed.utils.safe_get_full_grad(v)
    #       print(v.requires_grad)

                    # print('打印embedding',v_p)
        # for n, pa in self.transform.named_parameters():
        #     print('dayintidu', pa.grad)
    #   print('打印weight', self.embedding.weight.data)
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        #print(prefix)
        #print(past_key_values[0])
        return past_key_values

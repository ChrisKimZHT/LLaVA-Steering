import torch
from torch.nn.parameter import Parameter
from collections import OrderedDict
from typing import Iterator, Tuple

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN
from pyreft.interventions import LowRankRotateLayer


class LowRankRotateLayer_32(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m, dtype=torch.float32), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        self.weight = self.weight.to(torch.float32)
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class MoReSIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    MoReSIntervention(h) = h + R^T(Wh + b − Rh)

    NOTE: The parametrization only supports torch.float32!!!
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer_32(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer, orthogonal_map='householder') # cayley matrix_exp householder
        self.rotate_layer.parametrizations.weight.original = self.rotate_layer.parametrizations.weight.original.to(torch.float32)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.float16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        orig_dtype = base.dtype
        base = base.to(torch.float32)
        self.rotate_layer.to(torch.float32)
        rotated_base = self.rotate_layer(base)
        rotated_base = rotated_base.to(orig_dtype)
        base = base.to(orig_dtype)

        inter_value = self.act_fn(self.learned_source(base)) - rotated_base   # 16
        inter_value = inter_value.to(torch.float32)   # 32
        output = base + torch.matmul(inter_value, self.rotate_layer.weight.T)
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        # change self.rotate_layer to float32, otherwise raise error  -->  "orgqr_cuda" not implemented for 'Half'
        self.rotate_layer.to(torch.float32)
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        return


class NoRotMoReSIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NoRotMoReS(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))
    
class MultiMoReSIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.mores_vis = MoReSIntervention(**kwargs)
        self.mores_text = MoReSIntervention(**kwargs)

    def forward(
        self, base, source=None, subspaces=None
    ):
        assert subspaces is not None, "subspaces should not be None, it\'s used to split base into two modalities."
        vis_len, text_len = subspaces[0]
        intervened_vis = self.mores_vis(base[:, :vis_len], source, subspaces=None)
        intervened_text = self.mores_text(base[:, vis_len:vis_len+text_len], source, subspaces=None)
        return torch.cat([intervened_vis, intervened_text], dim=1)
    
    def state_dict(self, *args, **kwargs):
        state_dict_vis = self.mores_vis.state_dict()
        state_dict_vis = {'1_'+k: v for k, v in state_dict_vis.items()}
        state_dict_text = self.mores_text.state_dict()
        state_dict_text = {'2_'+k: v for k, v in state_dict_text.items()}
        
        return {**state_dict_vis, **state_dict_text}

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict_vis = OrderedDict()
        state_dict_text = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('1_'):
                state_dict_vis[k[2:]] = v
            elif k.startswith('2_'):
                state_dict_text[k[2:]] = v
        self.mores_vis.load_state_dict(state_dict_vis)
        self.mores_text.load_state_dict(state_dict_text)
        return
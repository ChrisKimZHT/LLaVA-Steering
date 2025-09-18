from .intervention_models import (
    MoReSIntervention,
    NoRotMoReSIntervention,
    MultiMoReSIntervention
)


INTERVENTION_TYPE_MAPPING = {
    'mores': MoReSIntervention,
    'norot_mores': NoRotMoReSIntervention,
    'm_mores': MultiMoReSIntervention
}


EXTENSION_type_to_module_mapping = {
    'microsoft/phi-2': {
        'block_input': ('model.layers[%s]', 'register_forward_pre_hook'),
        'block_output': ('model.layers[%s]', 'register_forward_hook'),
        'mlp_activation': ('model.layers[%s].mlp.activation_fn', 'register_forward_hook'),
        'mlp_output': ('model.layers[%s].mlp', 'register_forward_hook'),
        'mlp_input': ('model.layers[%s].mlp', 'register_forward_pre_hook'),
        'attention_output': ('model.layers[%s].self_attn', 'register_forward_hook'),
        'attention_input': ('model.layers[%s].self_attn', 'register_forward_pre_hook'),
        'query_output': ('model.layers[%s].self_attn.q_proj', 'register_forward_hook'),
        'key_output': ('model.layers[%s].self_attn.k_proj', 'register_forward_hook'),
        'value_output': ('model.layers[%s].self_attn.v_proj', 'register_forward_hook'),
        'head_query_output': ('model.layers[%s].self_attn.q_proj', 'register_forward_hook'),
        'head_key_output': ('model.layers[%s].self_attn.k_proj', 'register_forward_hook'),
        'head_value_output': ('model.layers[%s].self_attn.v_proj', 'register_forward_hook')
    }
}

model_name_to_module_path_mapping = {
    'microsoft/phi-2': {
        "embed_tokens": "model.embed_tokens",
        "layers": "model.layers",
        "lm_head": "lm_head"
    },
    'lmsys/vicuna-7b-v1.5': {
        "embed_tokens": "model.embed_tokens",
        "layers": "model.layers",
        "lm_head": "lm_head"
    },
    'lmsys/vicuna-13b-v1.5': {
        "embed_tokens": "model.embed_tokens",
        "layers": "model.layers",
        "lm_head": "lm_head"
    },
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0': {
        "embed_tokens": "model.embed_tokens",
        "layers": "model.layers",
        "lm_head": "lm_head"
    },
    '/home/public_space/zhangxiaohong/yintaoo/vicuna-7b': {
        "embed_tokens": "model.embed_tokens",
        "layers": "model.layers",
        "lm_head": "lm_head"
    }
}
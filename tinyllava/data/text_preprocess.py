from typing import Any

from .template import TemplateFactory


class TextPreprocess:
    def __init__(self, tokenizer, version):
        self.tokenizer = tokenizer
        self.template = TemplateFactory(version)()
    
    def __call__(self, messages, mode='train', return_sys_len=False):
        return self.template.encode(messages, self.tokenizer, mode, return_sys_len)
    
class TextPreprocessMoReS(TextPreprocess):
    def __init__(self, tokenizer, version, mores_pos_configs):
        super().__init__(tokenizer, version)
        self.template = TemplateFactory(version)(mores_pos_configs=mores_pos_configs)
import torch

from fairseq.tokenizer import tokenize_line
from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.data import data_utils


class LangWithEntityDictionary(object):
    def __init__(self, lang_dict: Dictionary, ne_dict: Dictionary):
        self.lang_dict = lang_dict
        self.ne_dict = ne_dict

    def __eq__(self, other):
        return self.lang_dict == other.lang_dict and self.ne_dict == other.ne_dict

    def __getitem__(self, idx):
        if idx < len(self.lang_dict):
            return self.lang_dict[idx]
        return self.ne_dict[idx - len(self.lang_dict)]

    def __len__(self):
        return len(self.lang_dict) + len(self.ne_dict)

    def __contains__(self, sym):
        return sym in self.lang_dict or sym in self.ne_dict

    def index(self, sym):
        if sym in self.lang_dict:
            return self.lang_dict.index(sym)
        return self.ne_dict.index(sym) + len(self.lang_dict)

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, 'bos_index'):
            sent = ' '.join(token_string(i) for i in tensor if (i != self.eos()) and (i != self.bos()))
        else:
            sent = ' '.join(token_string(i) for i in tensor if i != self.eos())
        return data_utils.process_bpe_symbol(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.lang_dict.unk_word)
        else:
            return self.lang_dict.unk_word

    def bos(self):
        return self.lang_dict.bos_index

    def pad(self):
        return self.lang_dict.pad_index

    def eos(self):
        return self.lang_dict.eos_index

    def unk(self):
        return self.lang_dict.unk_index

    def encode_line(self, line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                    consumer=None, append_eos=True, reverse_order=False):
        return self.lang_dict.encode_line(line, line_tokenizer=tokenize_line, add_if_not_exist=True,
                                          consumer=None, append_eos=True, reverse_order=False)

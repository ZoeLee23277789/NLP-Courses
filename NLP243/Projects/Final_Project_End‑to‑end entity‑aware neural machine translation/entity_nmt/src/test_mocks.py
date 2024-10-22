import string

import numpy as np
import torch.nn.functional as F


from fairseq.data import FairseqDataset, LanguagePairDataset
from fairseq.data.dictionary import Dictionary
from fairseq.tasks import FairseqTask

from .entity_translation_dataset import EntityTranslationDataset
from .entity_dictionary import LangWithEntityDictionary

MAX_NE_COUNT = 100


class MockDataset(FairseqDataset):
    def __init__(self, dictionary, data, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.size = len(data)

        for line in data:
            tokens = dictionary.encode_line(
                line, add_if_not_exist=False,
                append_eos=self.append_eos, reverse_order=self.reverse_order,
            ).long()
            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if isinstance(i, slice):
            if i.start is not None:
                self.check_index(i.start)
            if i.stop is not None:
                self.check_index(i.stop)
        else:
            if i < 0 or i >= self.size:
                raise IndexError('index out of range')

    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]


class MockTask(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        lang_dict = LangWithEntityDictionary(make_lang_dict(), make_ne_dict())

        self.src_dict = lang_dict
        self.source_dictionary = lang_dict

        self.tgt_dict = lang_dict
        self.target_dictionary = lang_dict

        self.ne_dict = lang_dict.ne_dict


class Args(object):
    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, v)
        

    def __str__(self):
        result = ''
        for k, v in self.__dict__.items():
            result += f'{k}: {v}\n'
        return result


class MockModel(object):
    def __init__(self, output):
        self.output = output

    def __call__(self, *args, **kwargs):
        return self.output

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def get_targets(self, sample, net_output):
        return sample['target']


def make_ne_dict(count=MAX_NE_COUNT):
    ne_dict = Dictionary()
    all_types = ['ORG', 'EVENT', 'PRODUCT', 'FAC', 'PERCENT', 'WORK_OF_ART', 'ORDINAL', 'LOC',
                 'LANGUAGE', 'LAW', 'PERSON', 'TIME', 'CARDINAL', 'GPE', 'QUANTITY', 'DATE', 'NORP', 'MONEY']

    ne_dict.add_symbol('O', 46)
    for entity in all_types:
        ne_dict.add_symbol(f'B-{entity}', 46)
        ne_dict.add_symbol(f'I-{entity}', 46)
        for i in range(count):
            ne_dict.add_symbol(f'{entity}-{i}', 46)

    return ne_dict


def make_lang_dict():
    lang_dict = Dictionary()
    for v in string.ascii_letters:
        lang_dict.add_symbol(v, 1)

    return lang_dict


def make_lang_pair(src_lines, tgt_lines, src_dict, tgt_dict):
    src_dataset = MockDataset(src_dict, src_lines)
    tgt_dataset = MockDataset(tgt_dict, tgt_lines)
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
    )


def make_ne_pair(src_lines, tgt_lines, ne_dict):
    return make_lang_pair(src_lines, tgt_lines, ne_dict, ne_dict)


def make_data_set(mode, src_lines, tgt_lines, src_ne, tgt_ne):
    src_dict, tgt_dict = make_lang_dict(), make_lang_dict()
    ne_dict = make_ne_dict()

    lang_pair = make_lang_pair(src_lines, tgt_lines, src_dict, tgt_dict)
    ne_pair = make_ne_pair(src_ne, tgt_ne, ne_dict)

    return EntityTranslationDataset(
        lang_pair,
        ne_pair,
        mode,
        MAX_NE_COUNT,
        LangWithEntityDictionary(src_dict, ne_dict),
        LangWithEntityDictionary(tgt_dict, ne_dict),
    )

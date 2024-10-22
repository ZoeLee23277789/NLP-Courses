import os
import itertools
from collections import defaultdict, namedtuple

import numpy as np
import torch

from fairseq.data import ConcatDataset, LanguagePairDataset, IndexedDataset, IndexedCachedDataset, data_utils, indexed_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationTask
from fairseq import options

from .entity_translation_dataset import EntityTranslationDataset, LangWithEntityDictionary

TypedEntity = namedtuple(
    'TypedEntity',
    [
        'type',
        'entity'
    ]
)

def make_entity_emb_dict(entity_tokens, dictionary):
    return {
        tuple(dictionary.encode_line(entity_tokens[i], add_if_not_exist=False, append_eos=False).tolist()): i
        for i in range(len(entity_tokens))
    }


@register_task('entity_translation')
class EntityTanslationTask(FairseqTask):
    """
    Translate X to NER, Y
    """

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--mode', type=int, required=True)
        parser.add_argument('--max-ne-id', default=100)

        # used for mode 2

        parser.add_argument('--tgt-entity-text', type=str)
        parser.add_argument('--tgt-bert-entity', type=str)
        parser.add_argument('--src-tgt-bert-mapping', type=str)
        parser.add_argument('--bert-sample-count', type=int, default=10000)
        parser.add_argument('--ignore-entity-type', type=str, default='')
        parser.add_argument('--ne-drop-rate', type=float, default=0)

    def __init__(self, args, src_dict, tgt_dict, ne_dict, tgt_entity_text=None, bert_emb_id_dict=None, bert_emb_value=None, entity_mapping=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.ne_dict = ne_dict
        self.tgt_ne_start_id = len(tgt_dict.lang_dict)

        self.tgt_entity_text = tgt_entity_text
        self.bert_emb_id_dict = bert_emb_id_dict
        self.bert_emb_value = bert_emb_value
        self.entity_mapping = entity_mapping

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        paths = args.data.split(':')
        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))

        # we share the source and target NE dict
        ne_dict = cls.load_dictionary(os.path.join(paths[0], f'dict.{args.source_lang}.ne.txt'))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        print('| [ne] dictionary: {} types'.format(len(ne_dict)))

        src_with_ne_dict = LangWithEntityDictionary(src_dict, ne_dict)
        tgt_with_ne_dict = LangWithEntityDictionary(tgt_dict, ne_dict)

        print('| [{}] entity dictionary: {} types'.format(args.source_lang, len(src_with_ne_dict)))
        print('| [{}] entity dictionary: {} types'.format(args.target_lang, len(tgt_with_ne_dict)))

        # This is moses and BPE processed
        if args.mode == 2:
            with open(args.tgt_entity_text, 'r', encoding='utf8') as f:
                tgt_entity_text = f.readlines()

            tgt_bert_entity = torch.from_numpy(np.load(args.tgt_bert_entity))

            bert_emb_id_dict = make_entity_emb_dict(tgt_entity_text, tgt_dict)

            entity_mapping = defaultdict(set)
            with open(args.src_tgt_bert_mapping, 'r', encoding='utf8') as f:
                for line in f:
                    data = line.strip().split('\t')
                    if len(data) == 2:
                        raise Exception("USING OLD MAPPING is NOT WORKS")
                        # key = src_dict.encode_line(data[0], add_if_not_exist=False, append_eos=False).tolist()
                        # value = tgt_dict.encode_line(data[1], add_if_not_exist=False, append_eos=False).tolist()
                        # entity_mapping[TypedEntity('', tuple(key))].add(tuple(value))
                    elif len(data) == 3:
                        entity_type = data[0]
                        key = src_dict.encode_line(data[1], add_if_not_exist=False, append_eos=False).tolist()
                        value = tgt_dict.encode_line(data[2], add_if_not_exist=False, append_eos=False).tolist()
                        entity_mapping[TypedEntity(entity_type, tuple(key))].add(tuple(value))
                    else:
                        raise Exception("bad mapping table")



            print(f'| load {len(bert_emb_id_dict)} entity in the dict')
            print(f'| load {len(entity_mapping)} entity maping keys, {sum((len(x) for x in entity_mapping.values()))} mapping values')

            return cls(args, src_with_ne_dict, tgt_with_ne_dict, ne_dict, [x.strip() for x in tgt_entity_text], bert_emb_id_dict, tgt_bert_entity, entity_mapping)
        else:
            return cls(args, src_with_ne_dict, tgt_with_ne_dict, ne_dict)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            return indexed_dataset.dataset_exists(filename, impl=self.args.dataset_impl)

        src_datasets = []
        tgt_datasets = []
        src_ne_datasets = []
        tgt_ne_datasets = []

        data_paths = self.args.data.split(':')

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                    ne_prefix = os.path.join(data_path, f'{split_k}.{src}.ne-{tgt}.ne.')
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                    ne_prefix = os.path.join(data_path, f'{split_k}.{tgt}.ne-{src}.ne.')
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(data_utils.load_indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict))
                src_ne_datasets.append(data_utils.load_indexed_dataset(ne_prefix + src + '.ne', self.src_dict))
                tgt_ne_datasets.append(data_utils.load_indexed_dataset(ne_prefix + tgt + '.ne', self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

        assert len(src_datasets) == len(tgt_datasets) == len(src_ne_datasets) == len(tgt_ne_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_ne_dataset, tgt_ne_dataset = src_ne_datasets[0], tgt_ne_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            src_ne_dataset = ConcatDataset(src_ne_datasets, sample_ratios)
            tgt_ne_dataset = ConcatDataset(tgt_ne_datasets, sample_ratios)

        option = {
            'left_pad_source': self.args.left_pad_source,
            'left_pad_target': self.args.left_pad_target,
            'max_source_positions': self.args.max_source_positions,
            'max_target_positions': self.args.max_target_positions,
        }

        lang_pair = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict.lang_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict.lang_dict,
            **option
        )

        ne_pair = LanguagePairDataset(
            src_ne_dataset, src_ne_dataset.sizes, self.src_dict.ne_dict,
            tgt_ne_dataset, tgt_ne_dataset.sizes, self.tgt_dict.ne_dict,
            **option
        )

        self.datasets[split] = EntityTranslationDataset(
            lang_pair,
            ne_pair,
            self.args.mode,
            self.args.max_ne_id,
            self.src_dict,
            self.tgt_dict,
            self.args.ignore_entity_type.split(','),
            self.args.ne_drop_rate,
            split=='train')

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        lang_pair = LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)
        return lang_pair

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                    gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

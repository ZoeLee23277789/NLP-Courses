#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

from sklearn import metrics

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from mode_1_sequence import build_generator

from utils import extract_ne_from_text

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000

    use_cuda = torch.cuda.is_available() and not args.cpu
    print('use_cuda', use_cuda)

    for subset in ['train', 'valid', 'test']:
        task = tasks.setup_task(args)
        task.load_dataset(subset)

        # Set dictionaries
        try:
            src_dict = getattr(task, 'source_dictionary', None)
        except NotImplementedError:
            src_dict = None
        tgt_dict = task.target_dictionary

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(args.replace_unk)

        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=task.max_positions(),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

        # Initialize generator
        gen_timer = StopwatchMeter()
        generator = build_generator(task, args)
        # Generate and compute BLEU score
        if args.sacrebleu:
            scorer = bleu.SacrebleuScorer()
        else:
            scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        num_sentences = 0
        has_target = True

        src_entity_count = 0
        tgt_entity_count = 0

        with progress_bar.build_progress_bar(args, itr) as t:
            wps_meter = TimeMeter()
            for sample in t:
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                if 'net_input' not in sample:
                    continue

                ne_source = sample['ne_source']
                ne_target = sample['ne_target']
                source = sample['net_input']['src_tokens']
                target = sample['target']

                ne_source_cur = ne_source[source != src_dict.pad()]
                src_entity_count += (ne_source_cur != 4).sum().item()

                ne_target_cur = ne_target[target != tgt_dict.pad()]
                tgt_entity_count += (ne_target_cur != 4).sum().item()

                # for i, sample_id in enumerate(sample['id'].tolist()):
                #     ne_source_cur = ne_source[i][source[i] != src_dict.pad()]
                #     src_entity_count += (ne_source_cur != 4).sum().item()

                #     ne_target_cur = ne_target[i][target[i] != tgt_dict.pad()]
                #     tgt_entity_count += (ne_target_cur != 4).sum().item()

        print(subset, src_entity_count, tgt_entity_count)

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()

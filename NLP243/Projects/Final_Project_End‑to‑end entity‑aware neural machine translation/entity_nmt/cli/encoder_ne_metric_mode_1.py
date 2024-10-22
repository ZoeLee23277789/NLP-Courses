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
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
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

    ne_true = []
    ne_pred = []

    pred_ne = 0
    src_ne = 0
    total_correct = 0

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos, encoder_ne_pred = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                source_ne_tokens = utils.strip_pad(sample['ne_source'][i,:], src_dict.pad())
                encoder_ne_tok = encoder_ne_pred[0][i].argmax(-1)
                encoder_ne_tok = encoder_ne_tok[sample['ne_source'][i,:] != src_dict.pad()]

                source_ne_text = src_dict.ne_dict.string(source_ne_tokens, args.remove_bpe, escape_unk=True).split()
                encoder_ne_text = src_dict.ne_dict.string(encoder_ne_tok, args.remove_bpe, escape_unk=True).split()

                assert len(source_ne_text) == len(encoder_ne_text)

                ne_true.extend(source_ne_text)
                ne_pred.extend(encoder_ne_text)

                src_ne_list = extract_ne_from_text(source_ne_tokens, source_ne_tokens, src_dict.ne_dict)
                encoder_ne_list = extract_ne_from_text(encoder_ne_tok, encoder_ne_tok, src_dict.ne_dict)

                src_ne += len(src_ne_list)
                pred_ne += len(encoder_ne_list)
                for x in src_ne_list:
                    if x in encoder_ne_list:
                        total_correct += 1

    print("="*20)
    print(metrics.classification_report(ne_true, ne_pred, digits=4))
    print("="*20)

    none_o_ne_true = []
    none_o_ne_pred = []
    for i in range(len(ne_true)):
        if ne_true[i] != 'O':
            none_o_ne_true.append(ne_true[i])
            none_o_ne_pred.append(ne_pred[i])

    print(metrics.classification_report(none_o_ne_true, none_o_ne_pred, digits=4))
    print("="*20)

    p = total_correct/pred_ne
    r = total_correct/src_ne
    f1 = 2 * p * r / (p + r)
    print(f'p={p*100:.2f}, r={r*100:.2f}, f1={f1*100:.2f}. total correct: {total_correct}, pred_ne: {pred_ne}, src_ne: {src_ne}')


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()

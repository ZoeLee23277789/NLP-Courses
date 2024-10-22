#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter

from collections import defaultdict

# TODO: test and fix me
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

    mapping_table = {}
    if args.mapping_table:
        with open(args.mapping_table, 'r', encoding='utf8') as f:
            for line in f:
                data = line.strip().split('\t')
                mapping_table[(data[0], data[1])] = data[2]

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
    dataset = task.dataset(args.gen_subset)
    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=dataset,
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
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
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
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                #src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                src_tokens = utils.strip_pad(sample['origin_src'][i], tgt_dict.pad())
                src_ne_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], src_dict.pad())
                src_ne_tokens = src_ne_tokens[:-1]  # remove EOS
                target_tokens = None
                target_ne_tokens = None
                if has_target:
                    target_ne_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                    target_tokens = utils.strip_pad(sample['origin_tgt'][i], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                        src_ne_str = src_dict.string(src_ne_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                        src_ne_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)
                        target_ne_str = tgt_dict.string(target_ne_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                        print('SN-{}\t{}'.format(sample_id, src_ne_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))
                        print('TN-{}\t{}'.format(sample_id, target_ne_str))

                # Process top predictions
                src_ne_pos = sample['src_ne_pos']
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )
                    # print(hypo['alignment'])
                    # [(2, 0), (4, 1), (3, 2), (3, 3), (4, 4), (5, 5)]  [(src, tgt)]
                    # alignment = hypo['alignment'].transpose(0, 1) # tgt->src => src->tgt
                    alignment_map = defaultdict(list)

                    for a, b in hypo['alignment']:
                        alignment_map[a].append(b)

                    hypo_final_str_attn = [tgt_dict[v] for v in hypo_tokens[:-1]]
                    hypo_final_str_match = [tgt_dict[v] for v in hypo_tokens[:-1]]

                    # For each tok in src (ne combined), if that is an entity, force the translation in target
                    for tok_id, tok in enumerate(src_ne_tokens):
                        if tok >= len(src_dict.lang_dict):
                            entity_type_str = src_dict[tok].split('-')[0]
                            raw_token_pos = src_ne_pos[i][tok_id]
                            if raw_token_pos is None:
                                continue
                            assert isinstance(raw_token_pos, slice), raw_token_pos
                            raw_tok = src_tokens[raw_token_pos]
                            raw_tok_str = src_dict.string(raw_tok, args.remove_bpe)


                            for tgt_pos in alignment_map[tok_id]:
                                # make sure we only replace the entity
                                if hypo_tokens[tgt_pos] >= len(tgt_dict.lang_dict):
                                    hypo_final_str_attn[tgt_pos] = mapping_table.get((entity_type_str, raw_tok_str), raw_tok_str)

                            for tgt_tok_id, tgt_tok in enumerate(hypo_tokens[:-1]):
                                if tgt_tok - len(tgt_dict.lang_dict) == tok - len(src_dict.lang_dict):
                                    hypo_final_str_match[tgt_tok_id] = mapping_table.get((entity_type_str, raw_tok_str), raw_tok_str)


                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('HA-{}\t{}\t{}'.format(sample_id, hypo['score'], ' '.join(hypo_final_str_attn)))
                        print('HM-{}\t{}\t{}'.format(sample_id, hypo['score'], ' '.join(hypo_final_str_match)))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                            ))

                        if args.print_step:
                            print('I-{}\t{}'.format(sample_id, hypo['steps']))

                        if getattr(args, 'retain_iter_history', False):
                            print("\n".join([
                                'E-{}_{}\t{}'.format(
                                    sample_id, step,
                                    utils.post_process_prediction(
                                            h['tokens'].int().cpu(),
                                            src_str, None, None, tgt_dict, None)[1])
                                for step, h in enumerate(hypo['history'])]))

                    # Score only the top hypothesis
                    if has_target and j == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument('--mapping-table', type=str, default='')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()

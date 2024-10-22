#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, search, utils
from fairseq.meters import StopwatchMeter, TimeMeter

from mode_2_sequence import build_generator
from collections import defaultdict

########################

###########################

def extract_ne_with_type(toks, entities, tok_dict, ne_dict):
    ent_count = defaultdict(int)
    result = {}

    last_ent = []
    last_type = ''
    for tok, ent in zip(toks, entities):
        if tok == tok_dict.eos():
            break

        if tok == tok_dict.pad():
            continue

        ent_text = ne_dict[ent]
        if ent_text == 'O':
            if last_ent:
                result[f'{last_type}-{ent_count[last_type]}'] = tok_dict.string(last_ent)
                ent_count[last_type] += 1

            last_ent = []
            last_type = ''
        elif ent_text.startswith('B-'):
            if last_ent:
                result[f'{last_type}-{ent_count[last_type]}'] = tok_dict.string(last_ent)
                ent_count[last_type] += 1
            
            last_ent = [tok]
            last_type = ent_text[2:]
        elif ent_text.startswith('I-'):
            cur_type = ent_text[2:]
            if cur_type == last_type:
                last_ent.append(tok)
            else:
                result[f'{last_type}-{ent_count[last_type]}'] = tok_dict.string(last_ent)
                ent_count[last_type] += 1
                last_ent = [tok]
                last_type = cur_type
        else:
            raise Exception(ent_text)
    return result

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'
    assert args.max_sentences == 1, f'please use batch size 1 for now, until the bug is fixed'

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
                has_original_target = sample['origin_tgt'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                original_target_tokens = None
                target_type_to_entity = {}
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
                if has_original_target:
                    original_target_tokens = utils.strip_pad(sample['origin_tgt'][i], tgt_dict.pad()).int().cpu()
                    ne_target = utils.strip_pad(sample['ne_target'][i], tgt_dict.ne_dict.pad()).int().cpu()
                    tgt_ne_pos = sample['tgt_ne_pos'][i]

                    for j, tok in enumerate(target_tokens.tolist()):
                        if tok > task.tgt_ne_start_id:
                            entity = original_target_tokens[tgt_ne_pos[j]]
                            target_type_to_entity[tok] = tgt_dict.string(entity, args.remove_bpe, escape_unk=True)

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)
                    
                    if has_original_target:
                        original_target_str = tgt_dict.string(original_target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))
                    if has_original_target:
                        print('TO-{}\t{}'.format(sample_id, original_target_str))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    hypo_with_ne_str = []
                    hypo_with_oracle_ne_str = []
                    entity_oracle_and_hyp = []
                    for k, tok in enumerate(hypo_tokens.tolist()):
                        if tok > task.tgt_ne_start_id:
                            hypo_ne_text = ""
                            oracle_ne_text = ""

                            ne_id = hypo['result_entity_id'][k]
                            if ne_id != -1: 
                                hypo_ne_text = task.tgt_entity_text[ne_id]
                            else: ## TODO: why this will happen??? No mapping??
                                if args.copy_unk_entity:
                                    #try copy from src
                                    src_entities = extract_ne_with_type(
                                        sample['net_input']['src_tokens'][i].tolist(),
                                        hypo['src_ne_tokens'].tolist(),
                                        src_dict,
                                        tgt_dict.ne_dict
                                    )

                                    if tgt_dict[tok] in src_entities:
                                        hypo_ne_text = src_entities[tgt_dict[tok]]
                                    else:
                                        hypo_ne_text = "UNE-0"
                                else:
                                    hypo_ne_text = tgt_dict[tok]
                            hypo_with_ne_str.append(hypo_ne_text)

                            if tok in target_type_to_entity:
                                oracle_ne_text = target_type_to_entity[tok]
                            else:
                                oracle_ne_text= tgt_dict[tok]
                            hypo_with_oracle_ne_str.append(oracle_ne_text)
                        
                            entity_oracle_and_hyp.append((oracle_ne_text, hypo_ne_text))
                        elif tok != tgt_dict.eos():
                            hypo_with_ne_str.append(tgt_dict[tok])
                            hypo_with_oracle_ne_str.append(tgt_dict[tok])
                    
                    hypo_with_ne_str = ' '.join(hypo_with_ne_str)
                    hypo_with_oracle_ne_str = ' '.join(hypo_with_oracle_ne_str)

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))
                        print('HN-{}\t{}'.format(sample_id, hypo_with_ne_str))
                        print('HO-{}\t{}'.format(sample_id, hypo_with_oracle_ne_str))
                        print('EM-{}\t{}'.format(sample_id, '\t'.join([ f'{x[0]}|||{x[1]}' for x in entity_oracle_and_hyp])))

                        #!!!! Note, the srouce padding is in the left, not right side.
                        source_ne_tokens = hypo['src_ne_tokens'][sample['net_input']['src_tokens'][i, :] != tgt_dict.pad()]
                        print('SE-{}\t{}'.format(sample_id, src_dict.ne_dict.string(source_ne_tokens, args.remove_bpe, escape_unk=True)))

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
                            scorer.add_string(original_target_str, hypo_with_ne_str)
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

    ### Sometime, we want to copy the entity directly. e.g. en->de.
    ### Sometime, we don't want to do that like en->zh.
    ### So add flag here
    parser.add_argument('--copy-unk-entity', action='store_true')

    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()

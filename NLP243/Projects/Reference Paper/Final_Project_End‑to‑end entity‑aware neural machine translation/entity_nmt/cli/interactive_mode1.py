#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from mode_1_sequence import build_generator

from utils import extract_ne_from_text

from collections import defaultdict

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

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

    # Initialize generator
    generator = build_generator(task, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    mapping_table = {}
    if args.mapping_table:
        ignore_type = set(args.ignore_type.split(';'))
        with open(args.mapping_table, 'r', encoding='utf8') as f:
            for line in f:
                data = line.strip().split('\t')
                if data[0] not in ignore_type:
                    mapping_table[(data[0], data[1])] = data[2]

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations, encoder_ne_pred = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos, encoder_ne_pred[0][i].argmax(-1)[-len(src_tokens_i):]))

        # sort output to match input order
        for id, src_tokens, hypos, encoder_ne_tok in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))
            src_ne_str = src_dict.ne_dict.string(encoder_ne_tok)
            print('SEP-{}\t{}'.format(id, src_ne_str))

            if mapping_table:
                src_entity_id, src_entity_type = extract_ne_from_text(src_tokens, encoder_ne_tok, src_dict.ne_dict, need_type=True)
            else:
                src_entity_id, src_entity_type = [], []
            
            src_entity_type_text = defaultdict(list)
            for se_id, se_type in zip(src_entity_id, src_entity_type):
                src_entity_type_text[se_type].append([src_dict[x] for x in se_id])

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                
                hypo_str = decode_fn(hypo_str)
                print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                print('TEP-{}\t{}'.format(id, tgt_dict.ne_dict.string(hypo['tgt_ne_tokens'])))

                if mapping_table:
                    hypo_fix_list = [tgt_dict[x] for x in hypo_tokens[:-1]]
                    hyp_entity_pos, hyp_entity_type = extract_ne_from_text(hypo_tokens, hypo['tgt_ne_tokens'], tgt_dict.ne_dict, need_type=True, return_pos=True)

                    for hp, ht in zip(hyp_entity_pos, hyp_entity_type):
                        if len(src_entity_type_text[ht]) > 0:
                            se_text = ' '.join(src_entity_type_text[ht][0])
                            src_entity_type_text[ht] = src_entity_type_text[ht][1:] # each only use one time

                            if (ht, se_text) in mapping_table:
                                he_text = mapping_table[(ht, se_text)]
                                s, t = hp
                                hypo_fix_list[s] = he_text
                                for p in range(s+1, t+1):
                                    hypo_fix_list[p] = None

                    hypo_fix_list = filter(lambda x: x is not None, hypo_fix_list)
                    
                    print('HF-{}\t{}'.format(id, ' '.join(hypo_fix_list)))
                if args.print_alignment:
                    alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                    print('A-{}\t{}'.format(
                        id,
                        alignment_str
                    ))

        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    parser.add_argument('--mapping-table', type=str, default='')
    parser.add_argument('--ignore-type', type=str, default='')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()

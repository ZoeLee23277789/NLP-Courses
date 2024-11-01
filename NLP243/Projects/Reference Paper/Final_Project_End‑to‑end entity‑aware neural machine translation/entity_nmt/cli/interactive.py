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


Batch = namedtuple('Batch', 'ids src_tokens src_lengths src_ne_pos')
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

    dataste = EntityTranslationDataset()  # TODO: build dataset
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'],
            src_lengths=batch['net_input']['src_lengths'],
            src_ne_pos=batch['src_ne_pos']
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

    if args.ne_mapping:
        ne_mapping_table = {}
        with open(args.ne_mapping, 'r', encoding='utf8') as f:
            for line in f:
                data = line.rstrip().split('|||')
                # TODO: We need many process here
                # 1. there may have duplicate in the dict
                # 2. the char case from the wiki title is not correct
                # 3. other noramlization
                ne_mapping_table[data[0].strip().lower()] = data[1].strip().lower()
        print(f'| Load {len(ne_mapping_table)} NE mapping items')
    else:
        ne_mapping_table = None
        print(f'| Not loading NE mapping table')

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
    generator = task.build_generator(args)

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
            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

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

                if ne_mapping_table:
                    for i, token in enumerate(hypo_tokens):
                        if token is ne_token:
                            # todo: what if not map to an entity?
                            original_pos = ne_orignal_pos[alignment[i]]
                            if original_pos is not None:
                                origial_token = src_tokens[original_pos]
                                origial_token_text = src_dict.string(origial_token)
                                hypo_tokens[i] = ne_mapping_table.get(origial_token_text, origial_token_text)  # map or copy
                            else:
                                raise Exception('not align to an entity')

                hypo_str = decode_fn(hypo_str)
                print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
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
    parser.add_args('--ne-mapping', default='')

    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()

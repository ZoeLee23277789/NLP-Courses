from .entity_translation_model import ModelOut
import unittest
from collections import namedtuple

import torch

from .entity_translation_criterion import EntityLabelSmoothedCrossEntropyCriterion
from .test_mocks import *

Args = namedtuple('Args', [
    'label_smoothing',
    'mode',
    'sentence_avg',
    'src_ner_loss_weight',
    'tgt_ner_loss_weight',
    'tgt_ne_lookup_weight'
])


class TestCriterion(unittest.TestCase):
    def run_test(self, args, task, model, sample, expected):
        criterion = EntityLabelSmoothedCrossEntropyCriterion(args, task)
        result = criterion.forward(model, sample)

        self.assertAlmostEqual(expected[0], result[0].item(), 2)
        self.assertEqual(expected[1], result[1])
        self.assertDictEqual(expected[2], result[2])

        self.assertListEqual(
            sorted(list(expected[2].keys())),
            sorted(list(EntityLabelSmoothedCrossEntropyCriterion.aggregate_logging_outputs([result[2]]).keys()))
        )

    def validate(self, mode, sample, model_output, expected):
        args = Args(
            label_smoothing=0.1,
            mode=mode,
            sentence_avg=False,
            src_ner_loss_weight=0.6,
            tgt_ner_loss_weight=0.3,
            tgt_ne_lookup_weight=0.5
        )
        task = MockTask(tgt_ne_start_id=2)

        model = MockModel(model_output)

        self.run_test(args, task, model, sample, expected)

    def test_mode_0(self):
        T = 4
        B = 2
        C = 5
        sample = {
            'id': torch.tensor([0, 1]),
            'nsentences': 2,
            'ntokens': 7,
            'net_input': {
                'src_tokens': torch.tensor([[4,    5, 1083,    2],
                                            [1,  777,  778,    2]]),
                'src_lengths': torch.tensor([4, 3]),
                'prev_output_tokens': torch.tensor([[2,    8, 1083,   10],
                                                    [2,  777,  778,    1]])
            },
            'target': torch.tensor([[0, 1, 2, 3],
                                    [1, 2, 3, 4]]),
            'src_ne_pos': [[slice(0, 1, None), slice(1, 2, None), slice(2, 4, None), slice(4, 5, None)], [slice(0, 1, None), slice(1, 3, None), slice(3, 4, None)]],
            'origin_src': [torch.tensor([4, 5, 6, 7, 2]), torch.tensor([4, 5, 6, 2])],
            'origin_tgt': [torch.tensor([8,  9, 10,  2]), torch.tensor([8, 9, 10, 11, 2])]
        }
        decoder_out = torch.zeros((B, T, C))
        for i in range(B):
            for j in range(T):
                decoder_out[i][j][0] = 1.0
        model_output = ModelOut(
            decoder_out=(decoder_out, None),
            encoder_ne_logit=None,
            decoder_ne_logit=None,
            entity_out=None,
            entity_label=None,
            result_entity_id=None,
            encoder_ne=None
        )
        expected_log = {
            'loss': 10.408994674682617,
            'nll_loss': 10.428995132446289,
            'ntokens': 7,
            'nsentences': 2,
            'sample_size': 7,
        }
        self.validate(0, sample, model_output, (10.41, 7, expected_log))

    def test_mode_1(self):
        T = 5
        B = 2
        C = 6
        sample = {
            'id': torch.tensor([0, 1]),
            'nsentences': 2,
            'ntokens': 9,
            'net_input': {
                'src_tokens': torch.tensor([[4, 5, 6, 7, 2],
                                            [1, 4, 5, 6, 2]]),
                'src_lengths': torch.tensor([5, 4]),
                'prev_output_tokens': torch.tensor([[2, 8, 9, 10, 1],
                                                    [2, 8, 9, 10, 11]])
            },
            'target': torch.tensor([[0, 1, 2, 3, 4],
                                    [1, 2, 3, 4, 5]]),
            'ne_pair': [
                {
                    'id': 0,
                    'source': torch.tensor([0, 1, 2, 3]),
                    'target': torch.tensor([1, 2, 3, 4, 5])
                },
                {
                    'id': 1,
                    'source': torch.tensor([0, 1, 2, 3]),
                    'target': torch.tensor([1, 2, 3, 4, 5])
                }
            ],
            'ne_source': torch.tensor([
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5]
            ]),
            'ne_target': torch.tensor([
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5]
            ])
        }
        decoder_out = torch.zeros((B, T, C))
        encoder_ne_logit = torch.zeros((B, T, C))
        decoder_ne_logit = torch.zeros((B, T, C))
        for i in range(B):
            for j in range(T):
                decoder_out[i][j][0] = 1.0
                encoder_ne_logit[i][j][0] = 1.0
                decoder_ne_logit[i][j][0] = 1.0

        model_output = ModelOut(
            decoder_out=(decoder_out, None),
            encoder_ne_logit=encoder_ne_logit,
            decoder_ne_logit=decoder_ne_logit,
            entity_out=None,
            entity_label=None,
            result_entity_id=None,
            encoder_ne=None
        )
        expected_log = {
            'loss': 29.12925910949707,
            'nll_loss': 15.348732948303223,
            'ntokens': 9,
            'nsentences': 2,
            'sample_size': 9,
            'src_ne_loss': 15.348732948303223,
            't_loss': 15.315399169921875,
            'tgt_ne_loss': 15.348732948303223
        }
        self.validate(1, sample, model_output, (29.13, 9, expected_log))

    def test_mode_2(self):
        T = 5
        B = 2
        C = 6
        sample = {
            'id': torch.tensor([0, 1]),
            'nsentences': 2,
            'ntokens': 9,
            'net_input': {
                'src_tokens': torch.tensor([[4, 5, 6, 7, 2],
                                            [1, 4, 5, 6, 2]]),
                'src_lengths': torch.tensor([5, 4]),
                'prev_output_tokens': torch.tensor([[2, 8, 9, 10, 1],
                                                    [2, 8, 9, 10, 11]])
            },
            'target': torch.tensor([[0, 1, 2, 3, 4],
                                    [1, 2, 3, 4, 5]]),
            'ne_pair': [
                {
                    'id': 0,
                    'source': torch.tensor([0, 1, 2, 3]),
                    'target': torch.tensor([1, 2, 3, 4, 5])
                },
                {
                    'id': 1,
                    'source': torch.tensor([0, 1, 2, 3]),
                    'target': torch.tensor([1, 2, 3, 4, 5])
                }
            ],
            'ne_source': torch.tensor([
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5]
            ]),
            'ne_target': torch.tensor([
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5]
            ]),
            'tgt_ne_pos': [
                [slice(0, 1, None), slice(1, 2, None), slice(2, 3, None), slice(3, 4, None)],
                [slice(0, 1, None), slice(1, 4, None), slice(4, 5, None)]],
            'origin_tgt': [torch.tensor([8,  9, 10,  2]), torch.tensor([8,  9, 10, 11,  2])]

        }
        decoder_out = torch.zeros((B, T, C))
        encoder_ne_logit = torch.zeros((B, T, C))
        entity_lookup = torch.zeros((B, T-1, C))
        for i in range(B):
            for j in range(T):
                decoder_out[i][j][0] = 1.0
                encoder_ne_logit[i][j][0] = 1.0

                if j < T-1:
                    entity_lookup[i][j][0] = 1.0

        entity_out = torch.tensor([
            [0.8, -0.1, 0.3],
            [0.1, 0.9, 0.2],
        ])
        entity_label = torch.LongTensor([0, 1])
        model_output = ModelOut(
            decoder_out=(decoder_out, None),
            encoder_ne_logit=encoder_ne_logit,
            decoder_ne_logit=None,
            entity_out=entity_out,
            entity_label=entity_label,
            result_entity_id=None,
            encoder_ne=None
        )
        expected_log = {
            'loss': 18.138107299804688,
            'nll_loss': 5.130775451660156,
            'ntokens': 9,
            'nsentences': 2,
            'sample_size': 9,
            'src_ne_loss': 15.348732948303223,
            't_loss': 5.180775165557861,
            'tgt_ne_loss': 10.217958450317383,
            'tgt_lookup_loss': 1.365407943725586,
            'entity_lookup_count': 2
        }
        self.validate(2, sample, model_output, (18.14, 9, expected_log))

    def test_mode_2_empty_entity(self):
        T = 5
        B = 2
        C = 6
        sample = {
            'id': torch.tensor([0, 1]),
            'nsentences': 2,
            'ntokens': 9,
            'net_input': {
                'src_tokens': torch.tensor([[4, 5, 6, 7, 2],
                                            [1, 4, 5, 6, 2]]),
                'src_lengths': torch.tensor([5, 4]),
                'prev_output_tokens': torch.tensor([[2, 8, 9, 10, 1],
                                                    [2, 8, 9, 10, 11]])
            },
            'target': torch.tensor([[0, 1, 2, 3, 4],
                                    [1, 2, 3, 4, 5]]),
            'ne_pair': [
                {
                    'id': 0,
                    'source': torch.tensor([0, 1, 2, 3]),
                    'target': torch.tensor([1, 2, 3, 4, 5])
                },
                {
                    'id': 1,
                    'source': torch.tensor([0, 1, 2, 3]),
                    'target': torch.tensor([1, 2, 3, 4, 5])
                }
            ],
            'ne_source': torch.tensor([
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5]
            ]),
            'ne_target': torch.tensor([
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5]
            ]),
            'tgt_ne_pos': [
                [slice(0, 1, None), slice(1, 2, None), slice(2, 3, None), slice(3, 4, None)],
                [slice(0, 1, None), slice(1, 4, None), slice(4, 5, None)]],
            'origin_tgt': [torch.tensor([8,  9, 10,  2]), torch.tensor([8,  9, 10, 11,  2])]

        }
        decoder_out = torch.zeros((B, T, C))
        encoder_ne_logit = torch.zeros((B, T, C))
        entity_lookup = torch.zeros((B, T-1, C))
        for i in range(B):
            for j in range(T):
                decoder_out[i][j][0] = 1.0
                encoder_ne_logit[i][j][0] = 1.0

                if j < T-1:
                    entity_lookup[i][j][0] = 1.0

        entity_out = torch.zeros((0, 16))
        entity_label = torch.zeros((0))
        model_output = ModelOut(
            decoder_out=(decoder_out, None),
            encoder_ne_logit=encoder_ne_logit,
            decoder_ne_logit=None,
            entity_out=entity_out,
            entity_label=entity_label,
            result_entity_id=None,
            encoder_ne=None
        )
        expected_log = {
            'loss': 17.455402374267578,
            'nll_loss': 5.130775451660156,
            'ntokens': 9,
            'nsentences': 2,
            'sample_size': 9,
            'src_ne_loss': 15.348732948303223,
            't_loss': 5.180775165557861,
            'tgt_ne_loss': 10.217958450317383,
            'tgt_lookup_loss': 0.0,
            'entity_lookup_count': 0.0
        }
        self.validate(2, sample, model_output, (17.46, 9, expected_log))


if __name__ == "__main__":
    unittest.main()

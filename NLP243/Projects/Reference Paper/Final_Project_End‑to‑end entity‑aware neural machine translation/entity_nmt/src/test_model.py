import unittest

import torch

from .entity_translation_model import *
from .test_mocks import *


class TestModel(unittest.TestCase):

    def validate(self, mode, model_input, expected, bert_emb_id_dict, bert_emb_value, entity_mapping, train=True):
        args = Args({
            'debug': True,
            'encoder_layerdrop': 0,
            'decoder_layerdrop': 0,
            'activation_dropout': 0,
            'mode': mode,
            'bert_sample_count': 16,
            'max_ne_id': 100,
            'concat_ne_emb': False
        })

        transformer_iwslt_de_en(args)

        task = MockTask(bert_emb_id_dict=bert_emb_id_dict, bert_emb_value=bert_emb_value, entity_mapping=entity_mapping)

        model = EntityTransformer.build_model(args, task)
        if not train:
            model.eval()

        model_output = model(**model_input)

        # There is no need to valide the output value from a very complex network
        # Just ensure everything is there
        for i, item in enumerate(expected):
            if item is None:
                self.assertIsNone(model_output[i], i)
            else:
                self.assertIsNotNone(model_output[i], i)

    def test_mode_0_train(self):
        model_input = {
            'src_tokens': torch.tensor([[4,    5, 1083,    2],
                                        [1,  777,  778,    2]]),
            'src_lengths': torch.tensor([4, 3]),
            'prev_output_tokens': torch.tensor([[2,    8, 1083,   10],
                                                [2,  777,  778,    1]])
        }

        expected = ModelOut(
            decoder_out='',
            encoder_ne_logit=None,
            decoder_ne_logit=None,
            entity_out=None,
            entity_label=None,
            result_entity_id=None,
            encoder_ne=None
        )
        self.validate(
            0,
            model_input,
            expected,
            {},
            torch.zeros((0, 0)),
            {},
            True
        )

    def test_mode_0_eval(self):
        model_input = {
            'src_tokens': torch.tensor([[4,    5, 1083,    2],
                                        [1,  777,  778,    2]]),
            'src_lengths': torch.tensor([4, 3]),
            'prev_output_tokens': torch.tensor([[2,    8, 1083,   10],
                                                [2,  777,  778,    1]])
        }

        expected = ModelOut(
            decoder_out='',
            encoder_ne_logit=None,
            decoder_ne_logit=None,
            entity_out=None,
            entity_label=None,
            result_entity_id=None,
            encoder_ne=None
        )
        self.validate(
            0,
            model_input,
            expected,
            {},
            torch.zeros((0, 0)),
            {},
            False
        )

    def test_mode_1_train(self):
        model_input = {
            'src_tokens': torch.tensor([[4, 5, 6, 7, 2],
                                        [1, 4, 5, 6, 2]]),
            'src_lengths': torch.tensor([5, 4]),
            'prev_output_tokens': torch.tensor([[2, 8, 9, 10, 1],
                                                [2, 8, 9, 10, 11]])
        }

        expected = ModelOut(
            decoder_out='',
            encoder_ne_logit='',
            decoder_ne_logit='',
            entity_out=None,
            entity_label=None,
            result_entity_id=None,
            encoder_ne=None
        )
        self.validate(
            1,
            model_input,
            expected,
            {},
            torch.zeros((0, 0)),
            {},
            True
        )

    def test_mode_1_eval(self):
        model_input = {
            'src_tokens': torch.tensor([[4, 5, 6, 7, 2],
                                        [1, 4, 5, 6, 2]]),
            'src_lengths': torch.tensor([5, 4]),
            'prev_output_tokens': torch.tensor([[2, 8, 9, 10, 1],
                                                [2, 8, 9, 10, 11]])
        }

        expected = ModelOut(
            decoder_out='',
            encoder_ne_logit='',
            decoder_ne_logit='',
            entity_out=None,
            entity_label=None,
            result_entity_id=None,
            encoder_ne=None
        )
        self.validate(
            1,
            model_input,
            expected,
            {},
            torch.zeros((0, 0)),
            {},
            False
        )

    def test_mode_2_train(self):
        model_input = {
            'src_tokens': torch.tensor([[4, 5, 6, 7, 2],
                                        [1, 4, 5, 6, 2]]),
            'src_lengths': torch.tensor([5, 4]),
            'prev_output_tokens': torch.tensor([[2,    8, 1083,   10],
                                                [2,  777,  778,    1]]),
            'ne_source': torch.tensor([
                [4, 4, 1025, 1026, 2],
                [1, 719, 719, 720, 2]
            ]),

            'tgt_ne_pos': [
                [slice(0, 1, None), slice(1, 2, None), slice(2, 3, None), slice(3, 4, None)],
                [slice(0, 1, None), slice(1, 4, None), slice(4, 5, None)]],
            'origin_tgt': [torch.tensor([8,  9, 10,  2]), torch.tensor([8,  9, 10, 11,  2])],
            'target': torch.tensor([[8, 1083,   10,    2],
                                    [777,  778,    2,    1]]),
        }

        expected = ModelOut(
            decoder_out='',
            encoder_ne_logit='',
            decoder_ne_logit=None,
            entity_out='',
            entity_label='',
            result_entity_id=None,
            encoder_ne=None
        )

        bert_emb_id_dict = {
            (0, 1): 0,
        }

        bert_emb_value = torch.randn((5, 16))
        self.validate(
            2,
            model_input,
            expected,
            bert_emb_id_dict,
            bert_emb_value,
            {},
            True
        )

    def test_mode_2_eval(self):
        model_input = {
            'src_tokens': torch.tensor([[4, 5, 6, 7, 2],
                                        [1, 4, 5, 6, 2]]),
            'src_lengths': torch.tensor([5, 4]),
            'prev_output_tokens': torch.tensor([[2,    8, 1083,   10],
                                                [2,  777,  778,    1]]),

            'tgt_ne_pos': [
                [slice(0, 1, None), slice(1, 2, None), slice(2, 3, None), slice(3, 4, None)],
                [slice(0, 1, None), slice(1, 4, None), slice(4, 5, None)]],
            'origin_tgt': [torch.tensor([8,  9, 10,  2]), torch.tensor([8,  9, 10, 11,  2])],
            'target': torch.tensor([[8, 1083,   10,    2],
                                    [777,  778,    2,    1]]),
        }

        expected = ModelOut(
            decoder_out='',
            encoder_ne_logit='',
            decoder_ne_logit=None,
            entity_out='',
            entity_label='',
            result_entity_id='',
            encoder_ne=None
        )

        bert_emb_id_dict = {}
        for i in range(4, 8):
            for j in range(i+1, 8):
                bert_emb_id_dict[tuple(range(i, j))] = len(bert_emb_id_dict)

        bert_emb_value = torch.randn((len(bert_emb_id_dict), 16))
        entity_mapping = {
            (6, 7): 0
        }
        self.validate(
            2,
            model_input,
            expected,
            bert_emb_id_dict,
            bert_emb_value,
            entity_mapping,
            False
        )


if __name__ == "__main__":
    unittest.main()

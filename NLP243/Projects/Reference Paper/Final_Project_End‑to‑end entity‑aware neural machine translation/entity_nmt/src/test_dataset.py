import unittest

import torch

from .test_mocks import *
from .entity_translation_dataset import EntityTranslationDataset


class TestDataset(unittest.TestCase):

    def run_test(self, mode, expected_batch):
        src_lines = [
            'a b c d',
            'a b c'
        ]

        src_ne_lines = [
            'O O B-PERSON I-PERSON',
            'B-LOC B-LOC I-LOC'
        ]

        tgt_lines = [
            'e f g',
            'e f g h'
        ]

        tgt_ne_lines = [
            'O B-PERSON O',
            'B-LOC B-LOC I-LOC I-LOC'
        ]

        dataset = make_data_set(mode, src_lines, tgt_lines, src_ne_lines, tgt_ne_lines)

        self.assertEqual(len(src_lines), len(dataset))
        samples = [dataset[i] for i in range(len(src_lines))]
        batch = dataset.collater(samples)

        self.validate(batch, expected_batch)

    def validate(self, a, b):
        self.assertEqual(type(a), type(b))

        if isinstance(a, dict):
            self.assertListEqual(list(a.keys()), list(b.keys()))
            for k, v in a.items():
                self.validate(a[k], b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for i in range(len(a)):
                self.validate(a[i], b[i])
        elif isinstance(a, torch.Tensor):
            self.assertTrue((a == b).all())
        else:
            self.assertEqual(a, b)

    def test_mode_0(self):
        expected = {
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
            'target': torch.tensor([[8, 1083,   10,    2],
                                    [777,  778,    2,    1]]),
            'src_ne_pos': [[slice(0, 1, None), slice(1, 2, None), slice(2, 4, None), slice(4, 5, None)], [slice(0, 1, None), slice(1, 3, None), slice(3, 4, None)]],
            'origin_src': [torch.tensor([4, 5, 6, 7, 2]), torch.tensor([4, 5, 6, 2])],
            'origin_tgt': [torch.tensor([8,  9, 10,  2]), torch.tensor([8, 9, 10, 11, 2])]
        }
        self.run_test(0, expected)

    def test_mode_1(self):
        expected = {
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
            'target': torch.tensor([[8, 9, 10, 2, 1],
                                    [8, 9, 10, 11, 2]]),
            'ne_pair': [
                {
                    'id': 0,
                    'source': torch.tensor([4, 4, 1025, 1026, 2]),
                    'target': torch.tensor([4, 1025, 4, 2])
                },
                {
                    'id': 1,
                    'source': torch.tensor([719, 719, 720, 2]),
                    'target': torch.tensor([719, 719, 720, 720, 2])
                }
            ],
            'ne_source': torch.tensor([
                [4, 4, 1025, 1026, 2],
                [1, 719, 719, 720, 2]
            ]),
            'ne_target': torch.tensor([
                [4, 1025, 4, 2, 1],
                [719, 719, 720, 720, 2]
            ])
        }
        self.run_test(1, expected)

    def test_mode_2(self):
        expected = {
            'id': torch.tensor([0, 1]),
            'nsentences': 2,
            'ntokens': 7,
            'net_input': {
                'src_tokens': torch.tensor([[4, 5, 6, 7, 2],
                                            [1, 4, 5, 6, 2]]),
                'src_lengths': torch.tensor([5, 4]),
                'prev_output_tokens': torch.tensor([[2,    8, 1083,   10],
                                                    [2,  777,  778,    1]])
            },
            'target': torch.tensor([[8, 1083,   10,    2],
                                    [777,  778,    2,    1]]),
            'ne_pair': [
                {
                    'id': 0,
                    'source': torch.tensor([4, 4, 1025, 1026, 2]),
                    'target': torch.tensor([4, 1025, 4, 2])
                },
                {
                    'id': 1,
                    'source': torch.tensor([719, 719, 720, 2]),
                    'target': torch.tensor([719, 719, 720, 720, 2])
                }
            ],
            'ne_source': torch.tensor([
                [4, 4, 1025, 1026, 2],
                [1, 719, 719, 720, 2]
            ]),
            'ne_target': torch.tensor([
                [4, 1025, 4, 2, 1],
                [719, 719, 720, 720, 2]
            ]),
            'tgt_ne_pos': [
                [slice(0, 1, None), slice(1, 2, None), slice(2, 3, None), slice(3, 4, None)],
                [slice(0, 1, None), slice(1, 4, None), slice(4, 5, None)]],
            'origin_tgt': [torch.tensor([8,  9, 10,  2]), torch.tensor([8,  9, 10, 11,  2])]
        }
        self.run_test(2, expected)


if __name__ == "__main__":
    unittest.main()

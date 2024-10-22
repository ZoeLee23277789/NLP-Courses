import unittest

from .test_mocks import *
from .utils import *
from .entity_dictionary import LangWithEntityDictionary


class TestCombineNE(unittest.TestCase):

    def validate_combine(self, seq, combined, alignment):
        ne_dict = make_ne_dict()
        data = ne_dict.encode_line(seq, add_if_not_exist=False, append_eos=True)
        combined_idx, combined_alignment = combine_ne(data, ne_dict, MAX_NE_COUNT)
        self.assertEqual(2, combined_idx[-1].item())
        combined_text = ne_dict.string(combined_idx)

        self.assertEqual(combined, combined_text)
        self.assertEqual(alignment, combined_alignment)

    def test_empty_input(self):
        self.validate_combine('', '', [slice(0, 1)])

    def test_combine_one(self):
        self.validate_combine(
            'O O B-ORG O B-PERSON O',
            'O O ORG-0 O PERSON-0 O',
            [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4), slice(4, 5), slice(5, 6), slice(6, 7)])

    def test_combine_many(self):
        self.validate_combine(
            'B-ORG I-ORG I-ORG B-ORG O B-PERSON I-LOC',
            'ORG-0 ORG-1 O PERSON-0 LOC-0',
            [slice(0, 3), slice(3, 4), slice(4, 5), slice(5, 6), slice(6, 7), slice(7, 8)])


class TestCombineNeWithTest(unittest.TestCase):
    def validate_combine(self, ne_seq, token_seq, combined_ne, combined_text, alignment):
        dictionary = LangWithEntityDictionary(make_lang_dict(), make_ne_dict())

        ne_seq = dictionary.ne_dict.encode_line(ne_seq, add_if_not_exist=False, append_eos=True)
        token_seq = dictionary.lang_dict.encode_line(token_seq, add_if_not_exist=False, append_eos=True)

        combined_tok_idx, combined_ne_idx, combined_alignment = combine_ne_with_text(token_seq, ne_seq, dictionary, MAX_NE_COUNT)

        combined_tok_text = dictionary.string(combined_tok_idx)
        combined_ne_text = dictionary.ne_dict.string(combined_ne_idx)

        # ends with EOS
        self.assertEqual(2, combined_tok_idx[-1].item())
        self.assertEqual(2, combined_ne_idx[-1].item())

        self.assertEqual(combined_text, combined_tok_text)
        self.assertEqual(combined_ne, combined_ne_text)
        self.assertEqual(alignment, combined_alignment)

    def test_empty_input(self):
        self.validate_combine('', '', '', '', [slice(0, 1)])

    def test_combine_one(self):
        self.validate_combine(
            'B-LOC O B-PERSON',
            'a b c',
            'LOC-0 O PERSON-0',
            'LOC-0 b PERSON-0',
            [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4)])

    def test_combine_many(self):
        self.validate_combine(
            'B-LOC O O B-PERSON I-PERSON I-PERSON O',
            'a b c d e f g',
            'LOC-0 O O PERSON-0 O',
            'LOC-0 b c PERSON-0 g',
            [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 6), slice(6, 7), slice(7, 8)])


class TestExtractNeFromText(unittest.TestCase):
    def validate(self, token_seq, ne_seq, expected, need_type=False):
        lang_dict = make_lang_dict()
        ne_dict = make_ne_dict()

        ne_seq = ne_dict.encode_line(ne_seq, add_if_not_exist=False, append_eos=True)
        token_seq = lang_dict.encode_line(token_seq, add_if_not_exist=False, append_eos=True)
        result = extract_ne_from_text(token_seq, ne_seq, ne_dict, need_type)

        if need_type:
            self.assertListEqual(expected[0], result[0])
            self.assertListEqual(expected[1], result[1])
        else:
            self.assertListEqual(expected, result)

    def test_empty_input(self):
        self.validate('', '', [])

    def test_many(self):
        self.validate(
            'a b c d e f g',
            'B-LOC O B-PERSON I-PERSON I-LOC O B-LOC',
            [(4,), (6, 7), (8,), (10,)]
        )

    def test_many_with_type(self):
        self.validate(
            'a b c d e f g',
            'B-LOC O B-PERSON I-PERSON I-LOC O B-LOC',
            ([(4,), (6, 7), (8,), (10,)], ['LOC', 'PERSON', 'LOC', 'LOC']),
            True
        )

class TestTagEntity(unittest.TestCase):
    def validate(self, token_seq, ne_seq, expected):
        lang_dict = make_lang_dict()
        ne_dict = make_ne_dict()
        dictionary = LangWithEntityDictionary(lang_dict, ne_dict)

        ne_seq = ne_dict.encode_line(ne_seq, add_if_not_exist=False, append_eos=True)
        token_seq = lang_dict.encode_line(token_seq, add_if_not_exist=False, append_eos=True)
        expected_seq = dictionary.encode_line(expected, add_if_not_exist=False, append_eos=True)
        result = tag_entity(token_seq, ne_seq, ne_dict)

        self.assertListEqual(result.tolist(), expected_seq.tolist())

    def test_empty_input(self):
        self.validate('', '', [])

    def test_many(self):
        self.validate(
            'a b c d e f g',
            'B-LOC O B-PERSON I-PERSON I-LOC O B-LOC',
            'B-LOC a I-LOC b B-PERSON c d I-PERSON B-LOC e I-LOC f B-LOC g I-LOC'
        )

if __name__ == "__main__":
    unittest.main()

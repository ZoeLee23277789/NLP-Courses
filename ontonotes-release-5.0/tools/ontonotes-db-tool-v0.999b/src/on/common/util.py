# COPYRIGHT 2007-2011 BY BBN TECHNOLOGIES CORP.

# BY USING THIS SOFTWARE THE USER EXPRESSLY AGREES: (1) TO BE BOUND BY
# THE TERMS OF THIS AGREEMENT; (2) THAT YOU ARE AUTHORIZED TO AGREE TO
# THESE TERMS ON BEHALF OF YOURSELF AND YOUR ORGANIZATION; (3) IF YOU OR
# YOUR ORGANIZATION DO NOT AGREE WITH THE TERMS OF THIS AGREEMENT, DO
# NOT CONTINUE.  RETURN THE SOFTWARE AND ALL OTHER MATERIALS, INCLUDING
# ANY DOCUMENTATION TO BBN TECHNOLOGIES CORP.

# BBN GRANTS A NONEXCLUSIVE, ROYALTY-FREE RIGHT TO USE THIS SOFTWARE
# KNOWN AS THE OntoNotes DB Tool v. 0.9 (HEREINAFTER THE "SOFTWARE")
# SOLELY FOR RESEARCH PURPOSES. PROVIDED, YOU MUST AGREE TO ABIDE BY THE
# LICENSE AND TERMS STATED HEREIN. TITLE TO THE SOFTWARE AND ITS
# DOCUMENTATION AND ALL APPLICABLE COPYRIGHTS, TRADE SECRETS, PATENTS
# AND OTHER INTELLECTUAL RIGHTS IN IT ARE AND REMAIN WITH BBN AND SHALL
# NOT BE USED, REVEALED, DISCLOSED IN MARKETING OR ADVERTISEMENT OR ANY
# OTHER ACTIVITY NOT EXPLICITLY PERMITTED IN WRITING.

# NO WARRANTY. THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND.  THE SOFTWARE IS PROVIDED for RESEARCH PURPOSES ONLY. AS SUCH,
# IT MAY CONTAIN ERRORS, WHICH COULD CAUSE FAILURES OR LOSS OF DATA. TO
# THE MAXIMUM EXTENT PERMITTED BY LAW, BBN MAKES NO WARRANTIES, EXPRESS
# OR IMPLIED AS TO THE SOFTWARE, ITS CAPABILITIES OR FUNCTIONALITY,
# INCLUDING WITHOUT LIMITATION THE IMPLIED WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR
# ANY USE OF THE SOFTWARE. THE USER ASSUMES THE ENTIRE COST OF ALL
# NECESSARY REPAIR OR CORRECTION, EVEN IF BBN HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH A DEFECT OR DAMAGES. BBN MAKES NO WARRANTY THAT
# THE SOFTWARE WILL MEET THE USER REQUIREMENTS, OR WILL BE
# UNINTERRUPTED, TIMELY, SECURE, OR ERROR-FREE.

# LIMITATION OF LIABILITY. THE ENTIRE RISK AS TO THE RESULTS AND
# PERFORMANCE OF THE SOFTWARE IS ASSUMED BY THE USER. TO THE MAXIMUM
# EXTENT PERMITTED BY APPLICABLE LAW, BBN SHALL NOT BE LIABLE WITH
# RESPECT TO ANY SUBJECT MATTER OF THIS AGREEMENT UNDER ANY CONTRACT,
# NEGLIGENCE, STRICT LIABILITY OR OTHER THEORY FOR ANY DIRECT,
# CONSEQUENTIAL, RELIANCE, INCIDENTAL, SPECIAL, DIRECT OR INDIRECT
# DAMAGES WHATSOEVER (INCLUDING WITHOUT LIMITATION, DAMAGES FOR LOSS OF
# BUSINESS PROFITS, OR BUSINESS INFORMATION, OR FOR BUSINESS
# INTERRUPTION, PERSONAL INJURY OR ANY OTHER LOSSES) RELATING TO (A)
# LOSS OR INACCURACY OF DATA OR COST OF PROCUREMENT OF SUBSTITUTE
# SYSTEM, SERVICES OR TECHNOLOGY, (B) THE USE OR INABILITY TO USE THE
# SOFTWARE; (C) UNAUTHORIZED ACCESS TO OR ALTERATION OF YOUR
# TRANSMISSIONS OR DATA; (D) ANY PERSONAL INJURY OR INJURY TO PROPERTY;
# OR (E) ANY OTHER USE OF THE SOFTWARE EVEN IF BBN HAS BEEN FIRST
# ADVISED OF THE POSSIBILITY OF ANY SUCH DAMAGES OR LOSSES.

# WITHOUT LIMITATION OF THE FOREGOING, THE USER AGREES TO COMMIT NO ACT
# WHICH, DIRECTLY OR INDIRECTLY, WOULD VIOLATE ANY U.S. LAW, REGULATION,
# OR TREATY, OR ANY OTHER INTERNATIONAL TREATY OR AGREEMENT TO WHICH THE
# UNITED STATES ADHERES OR WITH WHICH THE UNITED STATES COMPLIES,
# RELATING TO THE EXPORT OR RE-EXPORT OF ANY COMMODITIES, SOFTWARE, OR
# TECHNICAL DATA.

# author: sameer pradhan

"""

:mod:`util` -- Utility functions
------------------------------------------

See:

 - Dealing with config file, command line options, etc:

   - :func:`load_options`
   - :func:`load_config`
   - :func:`parse_cfg_args`
   - :class:`FancyConfigParser`
   - :func:`register_config`
   - :func:`make_bool`

 - Buckwalter Arabic encoding:

   - :func:`buckwalter2unicode`
   - :func:`unicode2buckwalter`
   - :func:`devocalize_buckwalter`

 - Chinese Unicode encoding:

   - :func:`fullwidth`
   - :func:`halfwidth`

 - DB:

   - :func:`esc`
   - :func:`insert_ignoring_dups`
   - :func:`is_db_ref`
   - :func:`make_db_ref`
   - :func:`is_not_loaded`
   - :func:`make_not_loaded`

 - SGML (``.name`` and ``.coref`` files):

   - :func:`make_sgml_safe`
   - :func:`make_sgml_unsafe`

 - File System:

   - :func:`matches_an_affix`
   - :func:`mkdirs`
   - :func:`output_file_name`
   - :func:`sopen`
   - :func:`listdir`
   - :func:`listdir_full`
   - :func:`listdir_both`
   - :func:`validate_file_for_utf_8`
   - :func:`validate_paths_for_utf_8`

 - Other:

   - :func:`bunch`
   - :func:`format_time`
   - :func:`quick_clean`
   - :func:`get_max`
   - :func:`get_lemma`


Functions:

  .. autofunction:: format_time
  .. autofunction:: buckwalter2unicode
  .. autofunction:: unicode2buckwalter
  .. autofunction:: fullwidth
  .. autofunction:: halfwidth
  .. autofunction:: register_config
  .. autofunction:: insert_ignoring_dups
  .. autofunction:: matches_an_affix
  .. autofunction:: output_file_name
  .. autofunction:: validate_file_for_utf_8
  .. autofunction:: validate_paths_for_utf_8
  .. autofunction:: get_max
  .. autofunction:: get_lemma
  .. autofunction:: quick_clean
  .. autofunction:: load_config
  .. autofunction:: mkdirs
  .. autofunction:: load_options
  .. autofunction:: parse_cfg_args
  .. autofunction:: listdir
  .. autofunction:: listdir_full
  .. autofunction:: listdir_both
  .. autofunction:: sopen

  .. exception:: NotInConfigError

     Because people might want to use a dictionary in place of a
     :mod:`ConfigParser` object, use a :exc:`NotInConfigError` as
     the error to catch for ``config[section, value]`` call.
     For example:

     .. code-block:: python

        try:
           load_data(config['Data', 'data_location'])
        except on.common.util.NotInConfigError:
           print 'Loading data failed.  Sorry.'

  .. autofunction:: make_bool
  .. autoclass:: bunch
  .. autofunction:: is_db_ref
  .. autofunction:: make_db_ref
  .. autofunction:: is_not_loaded
  .. autofunction:: make_not_loaded
  .. autofunction:: esc
  .. autofunction:: make_sgml_safe
  .. autofunction:: make_sgml_unsafe
  .. autoclass:: FancyConfigParser

"""

#---- standard library imports ----#
from __future__ import with_statement
import string
import sys
import re
import math
import os
import time
import getopt
import zlib
import gzip
import bz2
import base64
import codecs
import xml.dom.minidom
import ConfigParser
from optparse import OptionParser
from collections import defaultdict
import tempfile
import commands
import xml.etree.ElementTree as ElementTree
import difflib
import inspect
import htmlentitydefs

#---- specific imports ----#
import on.common.log
import on.common.callisto_converter

#---- pre-compiled regular expressions ----#
multi_space_re = re.compile(r"\s+")
start_space_re = re.compile(r"^\s+")
end_space_re = re.compile(r"\s+$")

parse2pos_re = re.compile(r"\(([^ ][^ ]*) [^)(][^)(]*\)")
parse2word_re = re.compile(r"\([^ ][^ ]* ([^)(][^)(]*)\)")
parse2word_pos_re = re.compile(r"\(([^ ][^ ]*) ([^)(][^)(]*)\)")


date_re = re.compile(r"<DATE>\s*(.*?)\s*</DATE>", re.MULTILINE|re.DOTALL|re.UNICODE)
doc_re = re.compile(r"<DOC>.*?</DOC>", re.MULTILINE|re.DOTALL|re.UNICODE)
doc_no_re = re.compile(r"<DOCNO>.*?</DOCNO>", re.MULTILINE|re.DOTALL|re.UNICODE)
doc_id_re = re.compile(r'<DOC DOC(?:ID|NO)="([^"]+)">')
sgml_tag_re = re.compile(r"<.*?>", re.UNICODE)
coref_tag_re = re.compile(r"<(/)?COREF", re.UNICODE)
begin_coref_tag_re = re.compile(r"<COREF", re.UNICODE)
doc_tag_re = re.compile(r"<(/)?DOC>", re.UNICODE)
doc_full_re = re.compile(r'<DOC[^>]*>', re.MULTILINE|re.DOTALL|re.UNICODE)

headline_re = re.compile("<HEADLINE>(.*?)</HEADLINE>", re.DOTALL|re.MULTILINE|re.UNICODE)
headline_tag_re = re.compile("<(/)?HEADLINE>", re.DOTALL|re.MULTILINE|re.UNICODE)
sentence_id_para_re = re.compile("<(S|P)(\sID=(.*?))?>", re.DOTALL|re.MULTILINE|re.UNICODE)
header_re = re.compile("<(/)?HEADER>", re.UNICODE)
body_re = re.compile("<(/)?BODY>", re.UNICODE)
text_re = re.compile("<(/)?TEXT>", re.UNICODE)
sentence_re = re.compile("<(/)?S.*?>", re.UNICODE)
paragraph_re = re.compile("<(/)?P>", re.UNICODE)
ne_coref_tag_re = re.compile("\s+COREFID=.*?>", re.UNICODE)
start_pronoun_tag_re = re.compile("<PRONOUN.*?>", re.UNICODE)
end_pronoun_tag_re = re.compile("</PRONOUN>", re.UNICODE)

lcb_space_re = re.compile("\(\s+", re.UNICODE)
rcb_space_re = re.compile("\s+\)", re.UNICODE)

pool_id_specification_re = re.compile("[PE]\d+") # might also want to specify 4 digits instead of just one or more

buck2fsbuck = {">": "O",
               "<": "I",
               "&": "W",
               "'": "L",
               "|": "M",
               "*": "X"}

buck2uni = {"'": u"\u0621", # hamza-on-the-line
            "|": u"\u0622", # madda
            ">": u"\u0623", # hamza-on-'alif
            "&": u"\u0624", # hamza-on-waaw
            "<": u"\u0625", # hamza-under-'alif
            "}": u"\u0626", # hamza-on-yaa'
            "A": u"\u0627", # bare 'alif
            "b": u"\u0628", # baa'
            "p": u"\u0629", # taa' marbuuTa
            "t": u"\u062A", # taa'
            "v": u"\u062B", # thaa'
            "j": u"\u062C", # jiim
            "H": u"\u062D", # Haa'
            "x": u"\u062E", # khaa'
            "d": u"\u062F", # daal
            "*": u"\u0630", # dhaal
            "r": u"\u0631", # raa'
            "z": u"\u0632", # zaay
            "s": u"\u0633", # siin
            "$": u"\u0634", # shiin
            "S": u"\u0635", # Saad
            "D": u"\u0636", # Daad
            "T": u"\u0637", # Taa'
            "Z": u"\u0638", # Zaa' DHaa'
            "E": u"\u0639", # cayn
            "g": u"\u063A", # ghayn
            "_": u"\u0640", # taTwiil
            "f": u"\u0641", # faa'
            "q": u"\u0642", # qaaf
            "k": u"\u0643", # kaaf
            "l": u"\u0644", # laam
            "m": u"\u0645", # miim
            "n": u"\u0646", # nuun
            "h": u"\u0647", # haa'
            "w": u"\u0648", # waaw
            "Y": u"\u0649", # 'alif maqSuura
            "y": u"\u064A", # yaa'
            "F": u"\u064B", # fatHatayn
            "N": u"\u064C", # Dammatayn
            "K": u"\u064D", # kasratayn
            "a": u"\u064E", # fatHa
            "u": u"\u064F", # Damma
            "i": u"\u0650", # kasra
            "~": u"\u0651", # shaddah
            "o": u"\u0652", # sukuun
            "`": u"\u0670", # dagger 'alif
            "{": u"\u0671", # waSla
             }
"""Buckwalter to Unicode conversion table for decoding ASCII-encoded Arabic"""

html_entity_replacement_map = {"&#8217;": "'",
                               "&#8220;": "\"",
                               "&#8221;": "\"",
                                "&#146;": "'",
                                "&#147;": "\"",
                                "&#148;": "\"",
                               "&#8216;": "'",
                               "&#8212;": "--",
                              "&#65533;": "",
                               "&#8211;": "-",
                                "&#145;": "'",
                               "&#8230;": "...",
                               "&#8226;": "*",
                                "&#151;": "--",
                                "&#150;": "-",
                                "&#130;": ",",
                                "&#183;": ".",
                                "&#133;": "...",
                                "&#132;": "\"",
                                "&#149;": "*",
                                 "&amp;": "&",
                                  "&gt;": ">",
                                  "&lt;": "<",
                                "&quot;": "\"",
                                "&apos;": "'"
                                 }
                    


uni2buck = {}
"""Unicode to Buckwalter conversion table for encoding Arabic as ASCII."""


# Iterate through all the items in the buck2uni dict.
for (key, value) in buck2uni.iteritems():
    # The value from buck2uni becomes a key in uni2buck, and vice
    # versa for the keys.
    uni2buck[value] = key

PUNCT=["#", "$", '"', "-LSB-", "-RSB-", "-LRB-", "-RRB-", "-LCB-", "-RCB-",
       "[", "]", "(", ")", "{", "}", "'", ",", ".", ":", "``", "''", "PUNC",
       "NUMERIC_COMMA"]

def format_time( secs ):
    """ Format a duration in a human readable fashion.

    For example, ``format_time(10000)='2:46:40'``.

    """

    hr  = 0
    min = 0
    sec = 0
    rem = secs

    time_string = ""

    if( rem > 3600 ):
        hr  = rem/3600
        rem = rem%3600

    if( rem > 60 ):
        min = rem/60
        rem = rem%60

    sec = rem


    return "%i:%i:%i" % (hr,min,math.ceil(sec))


def usage():
  USAGE_TEXT=""
  print >>sys.stderr, USAGE_TEXT
  sys.exit(1)




def tighten_curly_braces(a_string):
    a_string = lcb_space_re.sub("(", a_string)
    a_string = rcb_space_re.sub(")", a_string)
    return a_string


def compress_space( a_string ):
    a_string = multi_space_re.sub(" ", a_string)
    a_string = start_space_re.sub("", a_string)
    a_string = end_space_re.sub("", a_string)
    return a_string




def clean_gold_parse( a_parse ):
    a_parse = re.sub(r"\(-NONE- [^)]+\)", "", a_parse)
    a_parse = re.sub(r"\([A-Z]+ +\)", "", a_parse)
    a_parse = re.sub(r"\([A-Z]+ +\)", "", a_parse)
    a_parse = re.sub(r"(\([^\=\- ][^\=\- ]*)[\=\-][^\s]+\s", "\g<1> ", a_parse)
    a_parse = re.sub(r"\([^ ][^ ]*\s+\)", "", a_parse)

    a_parse = re.sub(r"\s+", " ", a_parse)
    a_parse = re.sub(r"^\s+", "", a_parse)
    a_parse = re.sub(r"\s+$", "", a_parse)

    return string.strip(a_parse)




def parse2pos( a_parse ):
    pos_list = parse2pos_re.findall(a_parse)
    return string.join(pos_list)




def parse2word( a_parse ):
    word_list = parse2word_re.findall(a_parse)
    return string.join(word_list)




def parse2word_pos( a_parse ):
    word_pos_list = parse2word_pos_re.findall(a_parse)

    a_string = ""
    i=0
    for i in range(0, len(word_pos_list)):
        a_string = a_string + " " + word_pos_list[i][1] + "_" + word_pos_list[i][0]

    return string.strip(a_string)



def indented2flat_trees(file_name):
    #---- open the file ----#
    file = codecs.open(file_name, "r", "utf-8")

    #---- join lines ----#
    one_parse_string = string.join(file.readlines(), "")

    #---- strip any leading, following spaces ----#
    one_parse_string = string.strip(one_parse_string)

    #---- list of parses ----#
    parse_list = string.split(one_parse_string, "\n(")

    #---- reintroduce the ( in the 1 to nth parses (excluding the 0th) ----#
    k=0
    for k in range(1, len(parse_list)):
        parse_list[k] = "(" + parse_list[k]

    l=0
    for l in range(0, len(parse_list)):
        parse_list[l] = string.strip(compress_space(parse_list[l]))
        parse_list[l] = string.strip(re.sub(r"^\( ", "(TOP ", parse_list[l]))
        parse_list[l] = string.strip(re.sub(r"^\(\(", "(TOP (", parse_list[l]))

    return parse_list




def sort_hash_keys_by_value( hash ):

    def compare( tuple_1, tuple_2 ):
        if( float(tuple_1[1]) < float(tuple_2[1]) ):
            return -1
        elif( float(tuple_1[1]) == float(tuple_2[1]) ):
            return 0
        else:
            return 1

    tuple_list = []
    for key in hash.keys():
        tuple = (key, hash[key])
        tuple_list.append(tuple)

    tuple_list.sort(compare)

    first_item_list = []
    for item in tuple_list:
        first_item_list.append(item[0])

    return first_item_list




def sort_hash_keys_by_len_of_value( hash ):

    def compare( tuple_1, tuple_2 ):
        if( float(tuple_1[1]) < float(tuple_2[1]) ):
            return -1
        elif( float(tuple_1[1]) == float(tuple_2[1]) ):
            return 0
        else:
            return 1

    tuple_list = []
    for key in hash.keys():
        tuple = (key, len(hash[key]))
        tuple_list.append(tuple)

    tuple_list.sort(compare)

    first_item_list = []
    for item in tuple_list:
        first_item_list.append(item[0])

    return first_item_list




def remove_doc_tags(a_string):
    a_string = doc_tag_re.sub("", a_string)
    a_string = doc_full_re.sub("", a_string)
    #return a_string.strip()
    return a_string




def find_all_doc_nos(a_string):
    return doc_no_re.findall(a_string)




def find_all_docs(a_string):
    return doc_re.findall(a_string)




def is_sgml_tag(a_string):
    if(sgml_tag_re.findall(a_string) != []):
        return True
    else:
        return False



def is_coref_tag(a_string):
    if(coref_tag_re.findall(a_string) != []):
        return True
    else:
        return False




def is_begin_coref_tag(a_string):
    if(begin_coref_tag_re.findall(a_string) != []):
        return True
    else:
        return False



def remove_sgml_tags(a_string, strip=True):
    a_string = sgml_tag_re.sub("", a_string)
    if(strip == True):
        return a_string.strip()
    else:
        return a_string



def remove_all_non_ne_sgml_tags(a_string):
    #a_string = date_re.sub("", a_string)
    a_string = a_string.replace("<DATE>", "").replace("</DATE>", "")
    a_string = headline_tag_re.sub("", a_string)
    a_string = sentence_re.sub("", a_string)
    a_string = paragraph_re.sub("", a_string)
    a_string = header_re.sub("", a_string)
    a_string = body_re.sub("", a_string)
    a_string = text_re.sub("", a_string)
    a_string = ne_coref_tag_re.sub(">", a_string)
    a_string = start_pronoun_tag_re.sub("", a_string)
    a_string = end_pronoun_tag_re.sub("", a_string)
    a_string = re.sub("\n+", "\n", a_string)

    #return a_string.strip()
    return a_string




def check_overlap(span_0, span_1):
    if( int(span_1[0]) > int(span_0[0]) ):
        #--- i think that the second "and" is not required ----#
        if( (int(span_1[0]) > int(span_0[1])) and (int(span_1[1]) > int(span_0[1])) ):
            #---- no overlap ----#
            return False
        else:
            #---- overlap ----#
            return True
    else:
        if( (int(span_0[0]) > int(span_1[1])) and (int(span_0[1]) > int(span_1[1])) ):
            #---- no overlap ----#
            return False
        else:
            #---- overlap ----#
            return True

def get_attribute(a_element_tree, attribute_name):
    if(a_element_tree.attrib.has_key(attribute_name)):
        return a_element_tree.attrib[attribute_name]
    else:
        on.common.log.warning("attribute \"%s\" not defined for %s" % (attribute_name, str(a_element_tree)))





#---- pre-compiled regular expressions ----#
corpus_from_doc_id_re = re.compile("^(.*?)/")    #---- the top level directory name is the corpus id

def doc_id2corpus_id(a_doc_id):
    return corpus_from_doc_id_re.findall(a_doc_id)[0]


def get_max(a_list):
    """ The maximum value in a list of int strings """
    def compare(a, b):
        return cmp(int(a), int(b))

    a_list.sort(compare)

    return a_list[-1]


# formats the string to remove dot-unfriendly characters
def format_for_dot(a_string):
    a_string = re.sub("\.a\.n", ".n", a_string)
    a_string = re.sub("\.aN", ".N", a_string)
    a_string = re.sub("\.", "\.", a_string)
    a_string = re.sub("\*", "\*", a_string)
    a_string = re.sub("\>", "\>", a_string)
    a_string = re.sub("#", "", a_string)
    a_string = re.sub("\"", "<quote>", a_string)
    return a_string


def matches_pool_id_specification(a_id):
    if(len(pool_id_specification_re.findall(a_id)) > 0):
        return True
    else:
        return False

def buckwalter2fsbuckwalter(b_word):
    """ convert traditional buckwalter to filename-safe buckwalter """
    if b_word is None:
        return b_word

    return "".join(buck2fsbuck.get(b_chr, b_chr) for b_chr in b_word)

def buckwalter2unicode(b_word, sgml_safety=True):
    """Given a string in Buckwalter ASCII encoded Arabic, return the Unicode version. """

    if sgml_safety:
        b_word = make_sgml_unsafe(b_word)


    return "".join(buck2uni.get(b_char, b_char) for b_char in b_word)

def unicode2buckwalter(u_word, sgml_safe=False, devocalize=False):
    """Given a Unicode word, return the Buckwalter ASCII encoded version.

    If ``sgml_safe`` is set, run the output through :func:`make_sgml_safe` before returning.

    If ``devocalize`` is set delete a,u,i,o before returning.

    """

    def to_buck(c):

        if not sgml_safe:
            return uni2buck.get(c,c)

        if c in '<>':
            return c
        return make_sgml_safe(uni2buck.get(c,c))

    s = "".join([to_buck(u_char) for u_char in u_word])
    if devocalize:
        s = devocalize_buckwalter(s)
    return s

def devocalize_buckwalter(buckwalter_s):
    return buckwalter_s.replace("a", "").replace("u", "").replace("i", "").replace("o", "")

def validate_paths_for_utf_8(paths):
    """ Call validate_file_for_utf_8 recursively on all nodes under this one """
    # Loop over the input
    for path in paths:
        # Check the file type
        if os.path.isdir(path):
            # Get the directory contents
            children = [os.path.join(path, child) for child in os.listdir(path)]
            # Continue iterating
            validate_paths_for_utf_8(children)
        elif os.path.isfile(path):
            validate_file_for_utf_8(path)
        else:
            raise RuntimeError("Unknown file type %s" % path)

def validate_file_for_utf_8(path):
    """ Write messages to stderr for each non-unicode character in the file """

    # Track current line number
    n = 0

    # Handle unicode errors
    try:
        for line in open(path, "r").readlines():
            n += 1
            unicode(line, "utf-8")
    except UnicodeDecodeError, e:
        sys.stderr.write("%s, line %d: %s\n" % (path, n, e))
        pass


def quick_clean(s):
    """ Make a pure ASCII version of a string, replacing with '%'

    Should really only be used for debugging code.

    """

    return "".join([c if ord(c) < 128 else '%' for c in s])


def compare_coref_tuple(a_tuple, b_tuple):
    a_start = int(a_tuple[-2])
    a_end = int(a_tuple[-1])

    b_start = int(b_tuple[-2])
    b_end = int(b_tuple[-1])

    if( a_start < b_start ):
        return -1
    elif(a_start == b_start):
        if( a_end < b_end ):
            return -1
        elif(a_end == b_end):
            return 0
        elif(a_end > b_end):
            return 1
    elif(a_start > b_start):
        return 1










def desubtokenize_annotations(a_string, add_offset_notations=False, delete_annotations=False):
    '''  returns new_string, num_annotations_changed

    where

      a_string:   something like pre-<COREF...>Tuesday</COREF>
      new_string: something like <COREF...>pre-Tuesday</COREF>

    or

      new_token: something like <COREF...S_OFF="4">pre-Tuesday</COREF>

    depending on add_offset_notations

    num_annotations_changed: in this case, 1

    '''

    a_string = a_string.replace("<TURN>", "-TURN-")
    a_string = a_string.replace("<POSTER>", "-POSTER-")
    a_string = a_string.replace("<removed_junk>", "-removed_junk-")
    a_string = a_string.replace("<< ", "-- ")
    a_string = a_string.replace("<  ", "-  ")
    a_string = a_string.replace("< J</ENAM", "- J</ENAM")
    a_string = re.sub(r"<(\+[^>]*\+)>", r"[\1]", a_string)
    a_string = re.sub(r"<(http[^><]*?)>", r"[\1]", a_string)
    a_string = re.sub(r"<([^<>@]*?@[^<>@]*?)>", r"[\1]", a_string)

    #sys.stderr.write("%s\n" % (a_string))

    def is_tag(x): return x.startswith("<")
    def is_close(x): return x.startswith("</")
    def is_open(x): return is_tag(x) and not is_close(x)

    a_string = re.sub(r" (</[^>]*>)", r"\1 ", a_string)
    a_string = re.sub(r"(<[^/>][^>]*>) ", r" \1", a_string)

    def spaceseparate(s):
        cur = []
        for c in s:
            if c.isspace():
                yield "".join(cur)
                cur = []
                yield c
            else:
                cur.append(c)
        if cur:
            yield "".join(cur)

    def tokenize(s):
        while s:
            if "<" not in s:
                for ss in spaceseparate(s):
                    yield ss
                return
            if s[0] != "<":
                nontag, s = s.split("<", 1)
                s = "<" + s
                for ss in spaceseparate(nontag):
                    yield ss

            if len(s.split(">")) == 1:
                yield s
                s = ""
            else:
                tag, s = s.split(">", 1)
                tag += ">"
                yield tag
        yield "\n"

    changes = 0

    new_tokens = []
    token_stack = [] # active tags -- form is (tag, token index, character offset)
    token_table = [] # all tags -- form is (open, close) where open and close are (tag, token index, character offset)

    for a_token in tokenize(a_string):
        def middle_of_token():
            if not new_tokens:
                return False
            if a_token.isspace():
                return False
            return not new_tokens[-1].isspace()

        def current_info():
            assert is_tag(a_token)
            if not middle_of_token():
                s_off = 0
                token_index = len(new_tokens)
            else:
                s_off = len(new_tokens[-1])
                token_index = len(new_tokens) - 1
            return a_token, token_index, s_off

        if is_close(a_token):
            token_table.append((token_stack.pop(), current_info()))
        elif is_open(a_token):
            token_stack.append(current_info())
        else:
            if middle_of_token():
                new_tokens[-1] = "%s%s" % (new_tokens[-1], a_token)
            else:
                new_tokens.append(a_token)

    if token_stack:
        for y in [x for x in new_tokens if x.strip()][-10:]:
            print y
    assert not token_stack, token_stack

    def table_sorter(a, b):
        (a_o_tag, a_o_t_idx, a_o_s_off), (a_c_tag, a_c_t_idx, a_c_s_off) = a
        (b_o_tag, b_o_t_idx, b_o_s_off), (b_c_tag, b_c_t_idx, b_c_s_off) = b

        if a_o_t_idx != b_o_t_idx:
            return cmp(b_o_t_idx, a_o_t_idx) # if one start on a later token, do it first
        if a_c_t_idx != b_c_t_idx:
            return cmp(a_c_t_idx, b_c_t_idx) # if one ends on an earlier token, do it first
        return cmp(a_c_s_off, b_c_s_off)     # if one ends on an eralier character, do it first

    token_table.sort(cmp=table_sorter)

    def add_s_off(tag, s_off):
        if s_off == 0:
            return tag

        return '%s-S_OFF="%s">' % (tag[:-1], s_off)

    def add_e_off(tag, e_off):
        if e_off == 0:
            return tag

        return '%s-E_OFF="%s">' % (tag[:-1], e_off)

    converted_token_table = []
    for (o_tag, o_t_idx, o_s_off), (c_tag, c_t_idx, c_s_off) in token_table:
        try:
            c_e_off = len(new_tokens[c_t_idx]) - c_s_off
        except IndexError:
            print list(enumerate(new_tokens))
            print c_tag, c_t_idx, c_s_off
            raise
        converted_token_table.append(((o_tag, o_t_idx, o_s_off), (c_tag, c_t_idx, c_e_off)))

    deletions = 0
    for (o_tag, o_t_idx, o_s_off), (c_tag, c_t_idx, c_e_off) in converted_token_table:
        if (o_s_off or c_e_off) and delete_annotations:
            deletions += 1
            continue

        if add_offset_notations:
            o_tag = add_e_off(add_s_off(o_tag, o_s_off), c_e_off)

        new_tokens[o_t_idx] = "%s%s" % (o_tag, new_tokens[o_t_idx])
        new_tokens[c_t_idx] = "%s%s" % (new_tokens[c_t_idx], c_tag)


    new_string = "".join(new_tokens)
    changes = new_string.count('-S_OFF="') + new_string.count('-E_OFF="') + deletions
    return "".join(new_tokens), changes








def apf2muc(in_file_name, out_file_name, source_file_name, new, chinese,
            maximum_number_of_subtoken_annotations=4, is_serif_output=False):

    s_file_name = source_file_name
    o_file_name = out_file_name
    DEBUG = True
    CHINESE = chinese
    BN = new
    ERROR = False

    s_file = codecs.open(s_file_name, "r", "utf-8")
    s_file_string = "".join(s_file.readlines())
    s_file.close()

    # Strip the BOM from the beginning of the Unicode string, if it exists
    s_file_string = s_file_string.lstrip(unicode(codecs.BOM_UTF8, "utf8"))

    doc_id_re = re.compile("<DOCID>(.*?)</DOCID>")
    date_re = re.compile("<DATE>(.*?)</DATE>")
    sgml_tag_re = re.compile(r"<.*?>")
    header_re = re.compile("^<DOC>.*?<BODY>", re.S)
    footer_re = re.compile("</BODY>.*$", re.S)

    def fix_subtoken_annotations(coref_doc_string, maximum_number_of_subtoken_annotations):
        """ deal with all problematic subtoken annotations

        For each token:
         - make a list of tags
         - make sure tags all go open open open close close close
         - output [open tags] text [close tags]

        """
        sas = 0
        new_lines = []
        for coref_line in coref_doc_string.split('\n'):
            new_line = []
            for a_token in coref_line.split():
                new_token, sas_new = desubtokenize_annotations(a_token, add_offset_notations=True)
                new_line.append(new_token)
                sas += sas_new
            new_lines.append(" ".join(new_line))

        if sas >= maximum_number_of_subtoken_annotations:
            on.common.log.warning("apf2muc: suspiciously large rate of subtoken annotation (%d) in %s; look into this" % (sas, in_file_name))
            on.common.log.report("apf2muc", "suspiciously large rate of subtoken annotation",
                                 "fixes: %d\nfname: %s" % (sas, in_file_name))

        return "\n".join(new_lines)


    try:
        doc_id = doc_id_re.findall(s_file_string)[0]
    except Exception:
        on.common.log.debug("no DOCID found in the source file using the filename as DOCID.", on.common.log.DEBUG, on.common.log.MAX_VERBOSITY)
        doc_id = re.sub(".source", "", re.sub("^.*/", "", s_file_name))
        doc_id = re.sub(".mrg", "", doc_id)

    if is_serif_output:
        pass
    elif(CHINESE == False or BN == True):
        s_file_string = doc_id_re.sub("", s_file_string)
    else:
        date_string = date_re.findall(s_file_string)[0]
        header_string = header_re.findall(s_file_string)[0]
        footer_string = footer_re.findall(s_file_string)[0]

    a_string = s_file_string
    s_file_string = sgml_tag_re.sub("", s_file_string)

    characters = list(s_file_string)
    on.common.log.debug(characters, on.common.log.DEBUG, on.common.log.MIN_VERBOSITY)

    #---- read one file by default ----#

    file = open(in_file_name)

    f_file_string = "".join(file.readlines())
    file.close()

    a_dom = xml.dom.minidom.parseString(f_file_string)
    entities = a_dom.getElementsByTagName("entity")


    coref_tuple = []

    total_mentions = 0
    i=0
    for i in range(0, len(entities)):
        e_id = entities[i].getAttribute("ID")
        e_type = entities[i].getAttribute("TYPE")

        mentions = entities[i].getElementsByTagName("entity_mention")

        if len(mentions) > 1 or is_serif_output:
            j=0
            for j in range(0, len(mentions)):
                sub_tuple = []
                sub_tuple.append(e_id)
                sub_tuple.append(e_type)

                m_extent = mentions[j].getElementsByTagName("extent")
                m_type = mentions[j].getAttribute("TYPE")

                if(e_type == "IDENT"):
                    sub_tuple.append("IDENT")
                elif(e_type == "APPOS"):
                    sub_tuple.append(m_type)
                elif is_serif_output:
                    sub_tuple.append("OTHER")
                else:
                    on.common.log.report("apf2muc", "Multiple mentions but bad coref link type",
                                         "Expected IDENT or APPOS",
                                         entity_id=e_id,
                                         entity_type=e_type,
                                         fname=in_file_name)
                    ERROR = True

                if(len(m_extent) > 1):
                    on.common.log.error("mentions should have only one extent")
                    ERROR = True

                charseq = m_extent[0].getElementsByTagName("charseq")

                if(len(charseq) > 1):
                    on.common.log.error("extent should have only one charseq")
                    ERROR = True

                start = int(charseq[0].getAttribute("START"))
                end = int(charseq[0].getAttribute("END"))

                sub_tuple.append(start)
                sub_tuple.append(end)

                coref_tuple.append(sub_tuple)

        total_mentions = total_mentions + len(mentions)

    coref_tuple.sort(compare_coref_tuple)

    for a_tuple in coref_tuple:
        a_start = a_tuple[-2]
        a_end = a_tuple[-1]
        a_id = a_tuple[0]
        a_type = a_tuple[1]
        a_subtype = a_tuple[2]

        if(a_type == "IDENT"):
            characters[a_start] = "<COREF-ID=\"%s\"-TYPE=\"%s\">%s" % (a_id, a_type, characters[a_start])
        else:
            characters[a_start] = "<COREF-ID=\"%s\"-TYPE=\"%s\"-SUBTYPE=\"%s\">%s" % (a_id, a_type, a_subtype, characters[a_start])

        characters[a_end] = "%s</COREF>" % (characters[a_end])


    coref_doc_string = "".join(characters)

    sentences = coref_doc_string.split("\n")

    new_sentences = []

    for a_sentence in sentences:
        if(a_sentence.strip() == ""):
            continue
        else:
            new_sentences.append(a_sentence.strip())

    coref_doc_string = "\n".join(new_sentences)

    if(ERROR == True):
        if os.path.exists(o_file_name):
                os.remove(o_file_name)
        raise ApfException("Not creating the output file since there were errors in the APF file. Check error log for more information.")

    if(ERROR == False):
        if(o_file_name == None):
            o_file = sys.stdout
        else:
            o_file = codecs.open(o_file_name, "w", "utf-8")


        if not is_serif_output:
            coref_doc_string = fix_subtoken_annotations(coref_doc_string,maximum_number_of_subtoken_annotations)

        if(o_file_name == None or BN == True):

            if(o_file_name == None):
                print "<DOC>"
                print "<DOCNO>%s</DOCNO>" % (re.sub(".sgm.fid.utf8", "", doc_id))
                print coref_doc_string
                print "</DOC>"
            else:
                o_file.write("<DOC>\n")
                o_file.write("<DOCNO>%s</DOCNO>\n" % (re.sub(".sgm.fid.utf8", "", doc_id)))
                o_file.write("%s\n" % (coref_doc_string))
                o_file.write("</DOC>\n")
        else:
            #---- remove first two lines ----#
            coref_doc_lines = string.split(coref_doc_string, "\n")


            if(len(coref_doc_lines[0].split()) != 1):
                error_file = codecs.open("%s.error" % (os.path.basename(in_file_name)), "w", "utf-8")
                error_file.write("%s" % ("\n".join(coref_doc_lines)))
                error_file.close()
                on.common.log.error("there might be a problem, please check (%s)" % in_file_name)
                ERROR = True


            if(doc_id[0:3] != "ART"):
                if(len(coref_doc_lines[1].split("-")) != 3):
                    error_file = codecs.open("%s.error" % (os.path.basename(in_file_name)), "w", "utf-8")
                    error_file.write("%s" % ("\n".join(coref_doc_lines)))
                    error_file.close()
                    on.common.log.error("there might be a problem, please check")
                    ERROR = True

                coref_doc_string = string.join(coref_doc_lines[2:], "\n")
            else:
                coref_doc_string = string.join(coref_doc_lines[1:], "\n")

            o_file.write(header_string)
            o_file.write("\n" + coref_doc_string + "\n")
            o_file.write(footer_string)

        o_file.flush()
        o_file.close()

class ApfException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# section -> value -> (allowed_values, doc, required, section_required, allow_multiple)
__registered_config_options = defaultdict( dict )

def is_config_section_registered(section):
    return section in __registered_config_options

def is_config_registered(section, value, strict=False):
    if section not in __registered_config_options:
        return False
    return value in __registered_config_options[section] or (
        not strict and "__dynamic" in __registered_config_options[section])

def required_config_options(section):
    if not is_config_section_registered(section):
        return []
    return [value for value in __registered_config_options[section]
            if __registered_config_options[section][value][2]] # required

def required_config_sections():
    return [section for section in __registered_config_options if
            [True for value in __registered_config_options[section]
             if __registered_config_options[section][value][3]]] # section_required

def allowed_config_values(section, option):
    if not is_config_registered(section, option, strict=True):
        return []
    return __registered_config_options[section][option][0]

def allow_multiple_config_values(section, option):
    if not is_config_registered(section, option, strict=True):
        return []
    return __registered_config_options[section][option][4]

def print_config_docs(to_string=False):
    p = []
    p.append("")
    p.append("Allowed configuration arguments:")
    for section in sorted(__registered_config_options.iterkeys()):
        p.append("   Section " + section + ":")

        if section in required_config_sections():
            p[-1] += " (required)"

        for value, (allowed_values, doc, required, section_required, allow_multiple) in sorted(__registered_config_options[section].iteritems()):
            if value == "__dynamic":
                value = "note: other dynamically generated config options may be used"

            p.append("      " + value)
            if required:
                p[-1] += " (required)"

            if doc:
                p.append("         " + doc)
            if allowed_values:
                if allow_multiple:
                    p.append("         may be one or more of:")
                else:
                    p.append("         may be one of:")

                for allowed_value in allowed_values:
                    p.append("            " + allowed_value)
        p.append("")
    s = "\n".join(p)
    if to_string:
        return s
    else:
        on.common.log.status(s)

def register_config(section, value, allowed_values=[], doc=None, required=False, section_required=False, allow_multiple=False):
    """ make decorator so funcs can specify which config options they take.

    usage is:

    .. code-block:: python

      @register_config('corpus', 'load', 'specify which data to load to the db in the format lang-genre-source')
      def load_banks(config):
          ...

    The special value '__dynamic' means that some config values are
    created dynamically and we can't verify if a config argument is
    correct simply by seeing if it's on the list.  Documentation is
    also generated to this effect.

    If ``allowed_values`` is non-empty, then check to see that the
    setting the user chose is on the list.

    If ``allow_multiple`` is True, then when checking whether only
    allowed values are being given the key is first split on
    whitespace and then each component is tested.

    If ``required`` is True, then if the section exists it must
    specify this value.  If the section does not exist, it is free to
    ignore this value.  See ``section_required`` .

    If ``section_required`` is True, then issue an error if
    ``section`` is not defined by the user.  Often wanted in
    combination with ``required`` .

    """

    __registered_config_options[section][value] = (allowed_values, doc, required, section_required, allow_multiple)
    return lambda f: f

def load_options(parser=None, argv=[], positional_args=True):
    """ parses sys.argv, possibly exiting if there are mistakes

    If you set parser to a ConfigParser object, then you have control
    over the usage string and you can prepopulate it with options you
    intend to use.  But don't set a ``--config`` / ``-c`` option;
    load_options uses that to find a configuration file to load

    If a parser was passed in, we return ``(config, parser, [args])``.
    Otherwise we return ``(config, [args])``.  Args is only included
    if ``positional_args`` is True and there are positional arguments

    See :func:`load_config` for details on the ``--config`` option.

    """

    def is_config_appender(arg):
        return "." in arg and "=" in arg and arg.find(".") < arg.find("=")

    parser_passed_in=parser
    if not parser:
        parser = OptionParser()

    parser.add_option("-c", "--config", help="the path to a config file to read options from")

    if argv:
        options, args = parser.parse_args(argv)
    else:
        options, args = parser.parse_args()

    config = load_config(options.config, [a for a in args if is_config_appender(a)])

    other_args = [a for a in args if not is_config_appender(a)]

    return_list = [config]
    if parser_passed_in:
        return_list.append(options)
    if other_args:
        if positional_args:
            return_list.append(other_args)
        else:
            raise Exception("Arguments %s not understood" % other_args)
    else:
        if positional_args:
            raise Exception("This program expects one or more positional arguments that are missing")

    if len(return_list) == 1:
        return return_list[0]
    else:
        return tuple(return_list)


class FancyConfigParserError(Exception):
    """ raised by :class:`FancyConfigParser` when used improperly """

    def __init__(self, vals):
        Exception.__init__(self, 'Config usage must be in the form "config[\'section\', \'item\']". '
                           'Given something more like "config[%s]".' % (", ".join("%r"%v for v in vals)))


class FancyConfigParser(ConfigParser.SafeConfigParser):
    """ make a config parser with support for config[section, value]

    raises :class:`FancyConfigParserError` on improper usage.

    """

    def __getitem__(self, vals):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)
        return self.get(section, item)


    def __setitem__(self, vals, value):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)
        return self.set(section, item, value)

    def __delitem__(self, vals):
        try:
            section, item = vals
        except (ValueError, TypeError):
            raise FancyConfigParserError(vals)

        self.remove_option(section, item)

def load_config(cfg_name=None, config_append=[]):
    """ Load a configuration file to memory.

    The given configuration file name can be a full path, in which
    case we simply read that configuration file.  Otherwise, if you
    give 'myconfig' or something similar, we look in the current
    directory and the home directory.  We also look to see if files
    with this name and extension '.conf' exist.  So for 'myconfig' we
    would look in the following places:

     * ./myconfig
     * ./myconfig.conf
     * [home]/.myconfig
     * [home]/.myconfig.conf

    Once we find the configuration, we load it.  We also extend
    ConfigParser to support ``[]`` notation.  So you could look up key
    ``k`` in section ``s`` with ``config[s,k]``.  See
    :func:`FancyConfigParser` .

    If config_append is set we use :func:`parse_cfg_args` and add any
    values it creates to the config object.  These values override any
    previous ones.

    """

    config = FancyConfigParser()

    if cfg_name:
        config_locs = [cfg_name + '.conf',
                       os.path.expanduser('~/.' + cfg_name + '.conf'),
                       cfg_name,
                       os.path.expanduser('~/.' + cfg_name)]
        l = config.read(config_locs)
        if not l:
            raise Exception("Couldn't find config file.  Looked in:" +
                            "".join(["\n - " + c for c in config_locs]) +
                            "\nto no avail.")


    for (section, key_name), value in parse_cfg_args(config_append).iteritems():
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, key_name, value)

    problems = []
    for section in config.sections():
        if not is_config_section_registered(section):
            on.common.log.status("Ignoring unknown configuration section", section)
            continue
        for option in config.options(section):
            if not is_config_registered(section, option):
                problems.append("Unknown configuration variable %s.%s" % (section, option))
                continue

            value = config.get(section, option)
            allowed = allowed_config_values(section, option)
            multiple = allow_multiple_config_values(section, option)

            values = value.split() if multiple else [value]
            for value in values:
                if allowed and not value in allowed:
                    problems.append("Illegal value '%s' for configuration variable %s.%s.  Permitted values are: %s" %
                                    (value, section, option, ", ".join(["'%s'" % x for x in allowed])))

        for option in required_config_options(section):
            if not config.has_option(section, option):
                problems.append("Required configuration variable %s.%s is absent" % (section, option))

    for section in required_config_sections():
        if not config.has_section(section):
            problems.append("Required configuration section %s is absent" % section)

    if problems:
        print_config_docs()

        on.common.log.status("Configuration Problems:")
        for problem in problems:
            on.common.log.status("  " + problem)

        sys.exit(-1)

    return config

def mkdirs(long_path):
    """ Make the given path exist.  If the path already exists, raise an exception. """

    p_dir = os.path.split(long_path)[0]

    if p_dir and not os.path.exists(p_dir):
        mkdirs(p_dir)

    os.mkdir(long_path)

def prop_sgml2plain_string(prop_sgml_string):
    a_plain_string_from_argument_tagged_string = re.sub("^DOMAIN.*?>", "", prop_sgml_string)
    a_plain_string_from_argument_tagged_string = re.sub("<C.*?>", "", a_plain_string_from_argument_tagged_string)
    a_plain_string_from_argument_tagged_string = re.sub("</C>", "", a_plain_string_from_argument_tagged_string)
    a_plain_string_from_argument_tagged_string = re.sub("</S>", "", a_plain_string_from_argument_tagged_string)
    a_plain_string_from_argument_tagged_string = compress_space(a_plain_string_from_argument_tagged_string)
    return a_plain_string_from_argument_tagged_string


def division(numerator, denominator):
    """ if d = 0 return 0, otherwise return n/d """

    return numerator/denominator if denominator else 0

def output_file_name(doc_id, doc_type, out_dir=""):
    """Determine what file to write an X_document to

    doc_id: a document id
    doc_type: the type of the document, like a suffix (parse, prop, name, ...)
    out_dir: if set, make the output as a child of out_dir

    """

    pathname = doc_id.split("@")[0]

    language = {"en": "english",
                "ar": "arabic",
                "ch": "chinese"}[doc_id.split("@")[-2]]

    original_file_name = "%s/%s/%s/%s.%s" % ("data", language, "annotations",
                                             pathname, doc_type)

    if out_dir:
        original_file_name = os.path.join(out_dir, original_file_name)

    view_parent, view_base = os.path.split(original_file_name)

    if not os.path.exists(view_parent):
        try:
            on.common.util.mkdirs(view_parent)
        except OSError:
            pass # race condition

    return original_file_name

def build_word2morph_hash(db_file):
    w2m_hash = {}
    sys.stderr.write(".\n")
    with open(db_file) as f:
        for line in f:
            word, morph = line.strip().split()
            w2m_hash[word] = morph
    return w2m_hash

def build_sensenum_hash_poly(fname):
    return build_sensenum_hash(fname, poly=True)

def build_sensenum_hash_mono(fname):
    return build_sensenum_hash(fname, poly=False)

def build_sensenum_hash(fname, poly):
    sn_l = {}
    with codecs.open(fname, "r", "utf8") as f:
        for i, line in enumerate(f):
            if poly:
                bits = line.split()
                if len(bits) != 2:
                    raise Exception("Bad file format for %s: %s" % (fname, line))
                sn_l[bits[0]] = int(bits[1])
            else:
                line = line.strip()
                if line:
                    sn_l[line] = 1
    return sn_l

def get_lemma(a_leaf, verb2morph, noun2morph, fail_on_not_found=False):
    """ return the lemma for a_leaf's word

    if we have appropriate word2morph hashes, look the work up
    there.  Otherwise just return the word.  Functionally, for
    chinese we use the word itself and for english we have the
    hashes.  When we get to doing arabic we'll need to add a
    case.

    if fail_on_not_found is set, return "" instead of a_leaf.word if
    we don't have a mapping for this lemma.
    """

    if a_leaf.on_sense:
        pos = a_leaf.on_sense.pos
    elif a_leaf.tag.startswith("V"):
        pos = 'v'
    elif a_leaf.tag.startswith('N'):
        pos = 'n'
    else:
        pos = 'other'

    w = a_leaf.word.lower()

    return word2lemma(w, pos, verb2morph, noun2morph, fail_on_not_found)

def word2lemma(a_word, pos, verb2morph, noun2morph, fail_on_not_found=False):
    """ return the lemma for a word given its pos

    if we have appropriate word2morph hashes, look the work up
    there.  Otherwise just return the word.  Functionally, for
    chinese we use the word itself and for english we have the
    hashes.  When we get to doing arabic we'll need to add a
    case.

    if fail_on_not_found is set, return "" instead of a_leaf.word if
    we don't have a mapping for this lemma.
    """
    w = a_word.lower()
    for p, h in [['n', noun2morph],
                 ['v', verb2morph]]:


        if pos == p and h and w in h:
            if(h[w] == "sayyid"):
                return "say"
            return  h[w]

    return "" if fail_on_not_found else w


def parse_cfg_args(arg_list):
    """Parse command-line style config settings to a dictionary.

    If you want to override configuration file values on the command
    line or set ones that were not set, this should make it simpler.
    Given a list in format [section.key=value, ...] return a
    dictionary in form { (section, key): value, ...}.

    So we might have:

    .. code-block:: python

      ['corpus.load=english-mz',
       'corpus.data_in=/home/user/corpora/ontonotes/data/']

    we would then return the dictionary:

    .. code-block:: python

      { ('corpus', 'load') : 'english-mz',
        ('corpus', 'data_in') : '/home/user/corpora/ontonotes/data/' }

    See also :func:`load_config` and :func:`load_options`

    """

    if not arg_list:
        return {}

    config_append = {}

    for arg in arg_list:
        if len(arg.split("=")) != 2 or len(arg.split("=")[0].split('.')) != 2:
            raise Exception("Invalid argument; not in form section.key=value : " + arg)

        key, value = arg.split("=")
        config_append[tuple(key.split("."))] = value

    return config_append


def listdir(dirname):
    """List a dir's child dirs, sorted and without hidden files.

    Basically :func:`os.listdir`, sorted and without hidden (in the
    Unix sense: starting with a '.') files.

    """

    dl = [x for x in os.listdir(dirname)
          if x[0] != "." and x not in ["report", "bad_data"]]

    dl.sort()
    return dl

def listdir_full(dirname):
    """ A full path to file version of :func:`on.common.util.listdir`. """

    return [os.path.join(dirname, d) for d in listdir(dirname)]

def listdir_both(dirname):
    """ return a list of short_path, full_path tuples

    identical to ``zip(listdir(dirname), listdir_full(dirname))``

    """

    return [(d, os.path.join(dirname, d)) for d in listdir(dirname)]

# documentation for this is in common/__init__.rst
NotInConfigError = ConfigParser.NoOptionError


def sopen(filename, mode="r"):
    """Open a file 'smartly'; understanding '-', '.gz', and normal files.

    If you have a command line argument to represent a filename,
    people often want to be able to put in standard input with a '-'
    or use gzipped (.gz and .bz2) files.  So you use :func:`sopen` in
    place of :func:`open`.

    Returns an open file.

    """

    if filename.endswith(".gz"):
        if mode=="r":
            mode = "rb"
        return gzip.open(filename, mode)
    elif filename.endswith(".bz2"):
        return bz2.BZ2File(filename, mode)
    elif filename == "-":
        if(mode != "r"):
            on.common.log.error("can open standard input for reading only")
        else:
            return sys.stdin
    else:
        return open(filename, mode)


def make_bool(string):
    """Extract the boolean encoded in the string"""

    if string.lower() in "true yes 1".split():
        return True
    elif string.lower() in "false no 0".split():
        return False

    raise ValueError("Invalid string to make_bool: %s" % string)

class bunch():
    """
    a simple class for short term holding related variables

    change code like:

    .. code-block:: python

      def foo_some(a_ontonotes, b_ontonotes):
        a_sense_bank = ...
        a_ontonotes.foo(a_sense_bank)
        a_...
        a_...

        b_sense_bank = ...
        b_ontonotes.foo(b_sense_bank)
        b_...
        b_...

        big_func(a_bar, b_bar)

    To:

    .. code-block:: python

      def foo_some():
        a = bunch(ontonotes=a_ontonotes)
        b = bunch(ontonotes=b_ontonotes)

        for v in [a,b]:
          v.sense_bank = ...
          v.ontonotes.foo(v.sense_bank)
          v. ...
          v. ...

        big_func(a.bar, b.bar)

    Or:

    .. code-block:: python

      def  foo_some():
        def foo_one(v):
          v.sense_bank = ...
          v.ontonotes.foo(v.sense_bank)
          v. ...
          v. ...
          return v

        big_func(foo_one(bunch(ontonotes=a_ontonotes)).bar,
                 foo_one(bunch(ontonotes=b_ontonotes)).bar)

    Basically it lets you group similar things.  It's adding hierarchy
    to the local variables.  It's a hash table with more convenient
    syntax.

    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def is_db_ref(a_hash):
    """ Is this hash a reference to the database?

    If a hash (sense inventories, frames, etc) is equal to
    ``{'DB' : a_cursor}`` that means instead of using the hash as
    information we should go look for our information in the database
    instead.

    """

    return a_hash and a_hash.keys() == ['DB']

def make_db_ref(a_cursor):
    """ Create a hash substitute that means 'go look in the db instead'.

    See :func:`is_db_ref`

    """

    return {'DB': a_cursor}

def is_not_loaded(a_hash):
    """ Do we have no intention of loading the data a_hash is supposed to contain?

    If a hash has a single key 'NotLoaded' that means we don't
    intend to load that hash and we shouldn't complain about data
    inconsistency involving the hash.  So if we're loading senses and
    the sense_inventory_hash :func:`is_not_loaded` then we shouldn't
    drop senses for being references against lemmas that don't exist.
    """

    return a_hash and a_hash.keys() == ['NotLoaded']

def make_not_loaded():
    """ Create a hash substitute that means 'act as if you had this information'

    See :func:`is_not_loaded`

    """

    return {'NotLoaded' : 'NotLoaded'}


def esc(*varargs):
    """ given a number of arguments, return escaped (for mysql) versions of each of them """

    try:
        import MySQLdb
    except ImportError:
        raise Exception("Can't import MySQLdb -- why are you trying to call this function if you're not working with mysql?")

    return tuple([MySQLdb.escape_string(str(s)) for s in varargs])

def make_sgml_safe(s, reverse=False, keep_turn=True):
    """ return a version of the string that can be put in an sgml document

    This means changing angle brackets and ampersands to '-LAB-',
    '-RAB-', and '-AMP-'.  Needed for creating ``.name`` and
    ``.coref`` files.

    If keep_turn is set, <TURN> in the input is turned into [TURN], not turned into -LAB-TURN-RAB-

    """

    if not reverse and keep_turn:
        s = s.replace("<TURN>", "[TURN]")

    for f, r in [("<", "-LAB-"),
                 (">", "-RAB-"),
                 ("&", "-AMP-")]:
        if reverse:
            r, f = f, r
        s = s.replace(f, r)

    return s

def make_sgml_unsafe(s):
    """ return a version of the string that has real <, >, and &.

    Convert the 'escaped' versions of dangerous characters back to
    their normal ascii form.  Needed for reading .name and .coref
    files, as well as any other sgml files like the frames and the
    sense inventories and pools.

    See :func:`make_sgml_safe`

    """

    return make_sgml_safe(s, reverse=True)

def insert_ignoring_dups(inserter, a_cursor, *values):
    """ insert values to db ignoring duplicates

    The caller can be a string, another class instance or a class:

      string  : take to be an sql insert statement
      class   : use it's sql_insert_statement field, then proceed as with string
      instance: get it's __class__ and proceed as with class

    So any of the following are good:

    .. code-block:: python

      insert_ignoring_dups(self, a_cursor, id, tag)
      insert_ignoring_dups(cls,  a_cursor, id, tag)
      insert_ignoring_dups(self.__class__.weirdly_named_sql_insert_statement, a_cursor, id, tag)

    """

    import MySQLdb


    if type(inserter) == type(""):
        insert_statement = inserter
    else:
        if hasattr(inserter, "__class__"):
            inserter = inserter.__class__
        insert_statement = inserter.sql_insert_statement

    try:
        a_cursor.executemany("%s" % insert_statement, [esc(*values)])
    except MySQLdb.Error, e:
        if(str(e.args[0]) != "1062"):
            on.common.log.error("{%s, %s} %s %s" % (insert_statement, values, str(e.args[0]), str(e.args[1])))

def uniq(sequence, id_function=None):

    # order preserving
    if id_function is None:
        def id_function(x): return x

    seen = {}
    result = []
    for item in sequence:
        marker = id_function(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


def matches_an_affix(s, affixes):
    """ Does the given id match the affixes?

    Affixes = prefixes, suffixes

    Given either a four digit string or a document id, return whether
    at least one of the prefixes and at least one of the suffixes
    matches it

    """

    def helper(id_bit):

        if not affixes:
            return True

        prefixes, suffixes = affixes

        ok_for_prefixes, ok_for_suffixes = not prefixes, not suffixes

        for prefix in prefixes:
            if id_bit.startswith(prefix):
                ok_for_prefixes = True

        for suffix in suffixes:
            if id_bit.endswith(suffix):
                ok_for_suffixes = True

        return ok_for_prefixes and ok_for_suffixes

    if len(s) == 4:
        return helper(s)

    if "@" in s:
        s, rest = s.split("@", 1)
    elif "." in s:
        s, rest = s.rsplit(".", 1)

    id_bit_start = s.rfind("_")

    return id_bit_start == -1 or helper(s[id_bit_start+1 : id_bit_start + 5])

def make_ansi_bold(s):
    return "\033[1m" + s + "\033[m"

def diff_align(seq_a, seq_b, map_differences=False, use_difflib=False):
    """ use diff to align lists a and b

    Given two sequences, return a hash from indicies in the first
    sequence to indecies in the second sequence.

        >>> diff_align(['a', 'b', 'c', 'd'], ['a', 'c', 'd'])
        {0: 0, 2: 1, 3: 2}

    That is, a hash from indexes on seq_a to indexes on seq_b

    if map_differences, attempt to map substitutions

    is use_difflib, use python's difflib library instead of calling
    diff.  Calling diff only works if you have diff.  Difflib is less
    effective when there are many small changes, but more effective
    when there are large blocks that should match exactly.

    """

    if not seq_a or not seq_b:
        raise Exception("Empty sequence given to diff_align")

    if not all(x for x in seq_a) or not all(x for x in seq_b):
         raise Exception("Partially empty sequence given to diff_align")

    a_to_b = {}

    if use_difflib:
        sm = difflib.SequenceMatcher(None, seq_a, seq_b)
        for a_start, b_start, match_len in sm.get_matching_blocks():
            for i in range(match_len):
                a_to_b[a_start + i] = b_start + i

    else: # shell out to diff

        def clean_token(s):
            """ because diff uses columns, tokens longer than 30 chars need to be truncated and we can't use those chars """
            return s[:30].replace(" ", "_").replace("|", "_").replace("<", "-").replace(">","-")

        a, b = bunch(), bunch()
        a.diff_input = "\n".join(clean_token(x) for x in seq_a) + "\n"
        b.diff_input = "\n".join(clean_token(x) for x in seq_b) + "\n"

        #a.file = open("a.txt","w")
        #b.file = open("b.txt","w")
        for v in [a, b]:
            v.file = tempfile.NamedTemporaryFile()
            v.file.write(v.diff_input)
            v.file.flush()

        status, output = commands.getstatusoutput("diff -y --expand-tabs %s %s" % (a.file.name, b.file.name))

        for v in [a, b]:
            v.file.close()
            v.idx = 0

        for a_diff_line in output.split("\n"):

            a_diff_line = a_diff_line.decode()


            found_insertion =      ">" in a_diff_line
            found_deletion =       "<" in a_diff_line
            found_substitution =   "|" in a_diff_line

            try:
                if found_insertion:
                    inc = [b]
                    debug_change = "None -> " + seq_b[b.idx]
                elif found_deletion:
                    inc = [a]
                    debug_change = seq_a[a.idx] + " -> None"
                else:
                    if map_differences or not found_substitution:
                        a_to_b[a.idx] = b.idx
                        debug_change = seq_a[a.idx] + " -> " + seq_b[b.idx]
                    else:
                        debug_change = seq_a[a.idx] + " -/-> " + seq_b[b.idx]
                    inc = [a, b]
            except IndexError:
                inc = [a, b]
                debug_change = "Error"


            if not found_substitution:
                bad = False

                if not found_deletion and clean_token(seq_b[b.idx]) not in a_diff_line:
                    bad = True
                if not found_insertion and clean_token(seq_a[a.idx]) not in a_diff_line:
                    bad = True

                if bad:
                    print a_diff_line
                    raise Exception("major diff align badness")


            if found_insertion:
                assert not found_deletion
                assert not found_substitution
            elif found_deletion:
                assert not found_substitution

            for v in inc:
                v.idx += 1


    if a_to_b:
        #if not  max(a_to_b.itervalues()) < len(seq_b):
        #    for a,b in a_to_b.itervalues():
        #        print seq_a[a],"->",seq_b[b]

        assert max(a_to_b.iterkeys()) < len(seq_a), (max(a_to_b.iterkeys()), len(seq_a))
        assert max(a_to_b.itervalues()) < len(seq_b), (max(a_to_b.itervalues()), len(seq_b))

    return a_to_b

def assert_equal(a, b):
    assert a == b
    return a


def wrap(val, cols, ind=0, indent_first_line=True):
    """ wrap the string in 'val' to use a maximum of 'cols' columns.  Lines are indented by 'ind'. """

    if val is None:
        return ""

    wrapped = []

    for s in val.split("\n"):
        while len(s) > cols:
            last_good_wrap_point = -1
            for i, c in enumerate(s):
                if c in ' ':
                    last_good_wrap_point = i
                if i >= cols and last_good_wrap_point != -1:
                    break
            if last_good_wrap_point != -1:
                wrapped.append(s[:last_good_wrap_point])
                s = s[last_good_wrap_point+1:]
            else:
                break
        if s:
            wrapped.append(s)

    a_str = ("\n" + " "*ind).join(w for w in wrapped)
    if indent_first_line:
        return " "*ind + a_str
    return a_str

def lpad(s, l):
    """ add spaces to the beginning of s until it is length l """
    s = str(s)
    return " "*max(0, (l - len(s))) + s

def rpad(s, l):
    """ add spaces to the end of s until it is length l """
    s = str(s)
    return  s + " "*max(0, (l - len(s)))


class InvalidSexprException(Exception):
    def __init__(self, sexpr, parent=None):
        self.sexpr = sexpr
        self.parent = parent

    def __str__(self):

        ns = ""
        ns += self.sexpr
        if self.parent:
            ns += "\n\n"
            ns += str(self.parent)
        return ns

def parse_sexpr(s):
    """ turn an s-expression into a tree of lists:

    (a (b c) d) -> [a, [b, c], d]

    uses spaces and parens only -- no way to have a token with a space in it

    """
    s = s.replace("\n", " ").replace("\t"," ").strip()

    if not s.startswith("(") and not s.endswith(")"):
        return s
    elif s.startswith("(") and s.endswith(")"):
        tr = []
        cur = []
        parens = 0
        for c in s[1:-1].strip() + " ":
            if c == "(":
                parens += 1
            elif c == ")":
                parens -= 1
            elif c == " " and cur:
                if parens == 0:
                    try:
                        x = parse_sexpr("".join(cur))
                    except InvalidSexprException, e:
                        raise InvalidSexprException("Parent: %s" % s, e)

                    if x:
                        tr.append(x)
                    cur = []

            cur.append(c)

        if (cur and cur != [" "]) or parens != 0:
            raise InvalidSexprException("Invalid s-expression: " + s + " note: %s" % "".join(cur) + " parens: %s" % parens)

        return tr
    else:
        raise InvalidSexprException("Invalid s-expression: \n" + s)

def unparse_sexpr(l):
    if type(l) == type([]):
        return "(" + " ".join(unparse_sexpr(a) for a in l) + ")"
    return str(l)

def repr_helper(tuple_gen_exp, ind=2):
    """ given a sequence of 2-tuples, return a nice string like:

    .. code_block:: python

       (1, 'hi'), (2, 'there'), (40, 'you') ->

    .. code_block:: python

       [ 1] : hi
       [ 2] : there
       [40] : you

    """


    lines = []

    k_v = list(tuple_gen_exp)

    if not k_v:
        return " "*ind + "(empty)"

    max_len_k = max(len(str(k_vp[0])) for k_vp in k_v)
    for k, v in sorted(k_v):
        lines.append(" "*ind + "[%s] : %s" % (lpad(k, max_len_k), v))
    return "\n".join(lines)



class timer:
    """ a class to help compute timing statistics for a single
    instance or repeatitions of a particular operation.

    just create a new instance (a_timer) and call a_timer.start()
    to start the timer and a_timer.stop() to record a reading, and
    at the end, call a_timer.end() to print the timing statistics.
    """
    def __init__(self, name):
        self.name = name
        self.list_of_deltas = []
        self.start_time = 0.0
        self.stop_time = 0.0
        self.total_time = 0.0
        self.recent_delta = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        delta = self.stop_time - self.start_time
        self.recent_delta = delta
        self.total_time = self.total_time + delta
        self.list_of_deltas.append(delta)

    def end(self):
        print "timer statistics for '%s':" % (self.name)
        print "     total time: %s"  % (self.total_time)
        number_of_deltas = len(self.list_of_deltas)
        print "number of calls: %s" % (number_of_deltas)
        print "  average delta: %s" % (self.total_time/number_of_deltas)


def score_b_cubed(k, r):
    """

    Given k and r as sequences in the form:

      [chain_1, chain_2, chain_2]

    where chains are sequences in the form:

      [link_1, link_2, link_3]

    Compute precision and recall with B-CUBED and return as a pair of doubles.

    See http://www-nlpir.nist.gov/related_projects/muc/proceedings/muc_7_proceedings/upenn.pdf

    """

    def common_elements(a_chain, b_chain):
        return [x for x in a_chain if x in b_chain]


    def merge_chains(chains):
        """ if multiple chains contain the same link, combine those chains

        Right now this is not implemented.  Instead we just raise an exception

        """

        for a_chain in chains:
            for b_chain in chains:
                if a_chain is not b_chain and common_elements(a_chain, b_chain):
                    raise Exception("Some chains have elements in common")
        return chains

    k, r = merge_chains(k), merge_chains(r)


    def containing_chain(chain_seq, a_link):
        """

        return the chain in chain_seq containing a_link

        return the empty list on failure

        """

        for a_chain in chain_seq:
            if a_link in a_chain:
                return a_chain
        return []

    precision = 0
    recall = 0
    for r_chain in r:
        for a_link in r_chain:
            k_chain = containing_chain(k, a_link)
            n_common = float(len(common_elements(k_chain, r_chain)))

            precision += (n_common / len(r_chain) if r_chain else 0)
            recall += (n_common / len(k_chain) if k_chain else 0)

    n_k_entities = len([k_link for k_chain in k for k_link in k_chain])
    n_r_entities = len([r_link for r_chain in r for r_link in r_chain])

    return  precision / n_r_entities, recall / n_k_entities

def score_their_b_cubed(k_lists, r_lists, source_text): #FIXME -- delete from release (uses local perl script)
    from subprocess import Popen, PIPE
    from tempfile import NamedTemporaryFile

    class Token:
        def __init__(self, text):
            self.text="".join(x for x in text if not x.isspace())
            assert self.text
            self._in_chains = []
            self._start_chains = []
            self._end_chains = []

        def in_chain(self, chain_id):
            if chain_id not in self._in_chains:
                self._in_chains.append(chain_id)

        def start_chain(self, chain_id):
            self._start_chains.append(chain_id)

        def end_chain(self, chain_id):
            self._end_chains.append(chain_id)

        def __str__(self):

            def exclude(bad, mixed, reverse):
                """ return a version of mixed that lacks the instances in bad """

                if reversed:
                    mixed = reversed(mixed)

                bad = bad[:]
                good = []

                for x in mixed:
                    if x in bad:
                        bad.remove(x)
                    else:
                        good.append(x)

                assert not bad

                if reversed:
                    good = list(reversed(good))

                return good


            startends = [x for x in self._start_chains if x in self._end_chains]
            starts = exclude(startends, self._start_chains, reverse=True)
            ends = exclude(startends, self._end_chains, reverse=False)

            ann = [] # annotation

            if starts and ends:
                raise Exception("can't both start and end a chain; not ok (%s, %s, %s)" % (starts, ends, self.text))

            if starts:
                ann.extend("(%s" % x for x in starts)

            if startends:
                ann.extend("(%s)" % x for x in startends)

            if ends:
                ann.extend("%s)" % x for x in reversed(ends))

            """ Format is tab delimited:

            ID TOKEN LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL PRED PPRED APREDs PAPREDs COREF

            But we're mocking this and putting self.text in for everything except coref

            """

            return (self.text+"\t")*16 + "|".join(ann)
            # return "%s\t%s" % (self.text, "|".join(ann))

    def annotated_tokens_list(chain_list, source_text):
        charidx_to_tokenidx = {}
        tokens = []
        cur_token = []
        for i, c in enumerate(source_text):
            charidx_to_tokenidx[i] = len(tokens)
            cur_token.append(c)

            if c.isspace() and "".join(cur_token).strip():
                tokens.append(Token(cur_token))
                cur_token = []

        for chain_id, a_chain in enumerate(chain_list):
            for start, end, subtype, string in a_chain:
                tokens[charidx_to_tokenidx[start]].start_chain(chain_id)

                for ci in range(start,end):
                    if ci in charidx_to_tokenidx:
                        tokens[charidx_to_tokenidx[ci]].in_chain(chain_id)

                tokens[charidx_to_tokenidx[end]].end_chain(chain_id)

        return tokens


    k_file = open("/nfs/titan/u75/ontonotes/corpora/on/test/2010-11-22--coref-other-scorer-from-sgml/tmp_k.txt","w") # tempfile.NamedTemporaryFile()
    r_file = open("/nfs/titan/u75/ontonotes/corpora/on/test/2010-11-22--coref-other-scorer-from-sgml/tmp_r.txt","w") # tempfile.NamedTemporaryFile()


    for x_lists, x_file in [[k_lists, k_file], [r_lists, r_file]]:
        x_file.write("#begin document bbn_test\n")
        for a_token in annotated_tokens_list(x_lists, source_text):
            x_file.write(str(a_token) + "\n")
        x_file.write("#end document bbn_test\n")
        x_file.flush()

    stdoutdata, stderrdata = Popen(args=["perl", "/nfs/titan/u75/ontonotes/corpora/on/test/bcubed_testing/scorer/scorer.pl", "bcub", k_file.name, r_file.name],
                                   cwd="/nfs/titan/u75/ontonotes/corpora/on/test/bcubed_testing/scorer/",
                                   stdout=PIPE).communicate()

    def interpret(s):
        """ interpret stuff in the form '(123.961538461538 / 129) 96.09%' """
        n, slash, d, ignore = s.replace("(", "").replace(")", "").split()
        assert slash == "/"
        n = float(n)
        d = float(d)
        if d == 0:
            return 0
        return n/d

    if stderrdata:
        raise Exception("Command failed with error:\n%s" % stderrdata)

    r,p = -1,-1
    for line in stdoutdata.split("\n"):
        if line.startswith("Coreference: Recall:"):
            r, p, f1 = line.replace("Coreference: ", "").split("\t")
            r = r.replace("Recall: ", "")
            p = p.replace("Precision: ", "")
            r, p = interpret(r), interpret(p)

    if r == -1 or p == -1:
        raise Exception("Something went funny in b cubed scoring")

    k_file.close()
    r_file.close()

    return p,r

def score_callisto(fname_key, fname_response, use_our_bcubed=True):
    """

    Given the fnames of a pair of callisto xml files, produce
    precision and recall numbers for their agreement using the B-CUBED
    algorithm.  See :func:`score_b_cubed`

    """

    def bad_data(reason, *args):
        raise Exception("%s or %s --- %s: %s" % (fname_key, fname_response, reason, "; ".join("%s=%s" % (k, v) for k, v in args)))

    k_lists, source_text_k = on.common.callisto_converter.callisto_to_chain_lists(fname_key, include_metadata=False)
    r_lists, source_text_r = on.common.callisto_converter.callisto_to_chain_lists(fname_response, include_metadata=False)
    if source_text_k != source_text_r:
        bad_data("files have different sources -- probably not the same file")

    if use_our_bcubed:
        return score_b_cubed(k_lists, r_lists)
    else:
        return score_their_b_cubed(k_lists, r_lists, source_text_k)



def clean_up_name_string(s):
    """ """

    s = s.replace("-TURN-", "<TURN>")

    for x in ["GPE", "ORG", "Cardinal", "NORP", "LOC",
              "Date", "Money", "Time", "Event", "FAC",
              "Language", "Law", "Ordinal", "Percent",
              "Product", "Quantity", "Work-of-art",
              "Work-of-Art"]:
        for follower in "->":
            s = s.replace('<'+x+follower, '<ENAMEX TYPE="'+x.upper()+'"' + follower)

        s = s.replace('</'+x+'>', '</ENAMEX>')


    s = s.replace('<PER>', '<ENAMEX TYPE="PERSON">')
    s = s.replace('</PER>', '</ENAMEX>')

    for x in ["NUMEX", "TIMEX"]:
        s = s.replace('<%s TYPE="' % x, '<ENAMEX TYPE="')
        s = s.replace('</%s>' % x, '</ENAMEX>')

    for x in ["CARDINAL", "MONEY", "ORDINAL", "PERCENT", "QUANTITY"]:
        s = re.sub('<ENA(MEX TYPE="%s">[^<]*</)ENAMEX>' % x, "<NU\\1NUMEX>", s)
    for x in ["DATE", "TIME"]:
        s = re.sub('<ENA(MEX TYPE="%s">[^<]*</)ENAMEX>' % x, "<TI\\1TIMEX>", s)

    return s

def make_fname_safe(s):
    """ return a version of s with all characters that might be
    dangerous for the file system changed to underscore.  The current
    list of allowed characters is:

     * A-Z
     * a-z
     * 0-9
     * _
     * -
     * .

    """

    return "".join(c if c.isalnum() or c in "-_." else "_"
                   for c in s)

def make_look_like(source, target):
    """

    given two lists of strings, reorder target so that it looks like source and return the reordered target

    """

    import difflib

    def permutations(iterable, r=None):
        """ we're targeting 2.5 and only 2.6 itertools has permutations.

        copied from itertools documentation:
          http://docs.python.org/library/itertools.html#itertools.permutations

        """

        # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
        # permutations(range(3)) --> 012 021 102 120 201 210
        pool = tuple(iterable)
        n = len(pool)
        r = n if r is None else r
        if r > n:
            return
        indices = range(n)
        cycles = range(n, n-r, -1)
        yield tuple(pool[i] for i in indices[:r])
        while n:
            for i in reversed(range(r)):
                cycles[i] -= 1
                if cycles[i] == 0:
                    indices[i:] = indices[i+1:] + indices[i:i+1]
                    cycles[i] = n - i
                else:
                    j = cycles[i]
                    indices[i], indices[-j] = indices[-j], indices[i]
                    yield tuple(pool[i] for i in indices[:r])
                    break
            else:
                return


    best_target_permutation = []
    best_target_score = 0

    sm=difflib.SequenceMatcher()
    sm.set_seq2(" ".join(source))

    def score(target_permutation):
        sm.set_seq1(" ".join(target_permutation))
        return sm.real_quick_ratio()

    for target_permutation in permutations(target):
        tscore = score(target_permutation)
        if tscore > best_target_score:
            best_target_score = tscore
            best_target_permutation = target_permutation

    assert best_target_permutation
    return best_target_permutation

def cannonical_arabic_word(a_word):
    word = on.common.util.unicode2buckwalter(a_word, devocalize=True)

    for f, r in ((">", "A"),
                 ("<", "A"),
                 ("~", ""),
                 ("-", "")):
        word = word.replace(f, r)

    return word

def guess_encoding(fname):

    for encoding in ["utf8", "gb18030"]:
        try:
            with codecs.open(fname, "r", encoding) as inf:
                for line in inf:
                    pass
            return encoding
        except UnicodeDecodeError:
            pass

    on.common.log.report("guess_encoding", "encoding guess failed", fname=fname)
    return "utf8"

def same_except_for_tokenization_and_hyphenization(s_1, s_2):
    return s_1.replace("- -", " ").replace(" ", "") == s_2.replace("- -", " ").replace(" ", "")

def iterate_trees(string_seq):
    """

    given string_seq which is a sequence of strings, read from
    string_seq and produce strings one at a time that represent trees.

    """
    return [x for x in _iterate_trees_helper(string_seq) if x.strip()]


def _iterate_trees_helper(string_seq):

    parens = 0
    cur_parse = []

    for s in string_seq:
        if (s.startswith(";") or s.startswith("<") or s.startswith("*")) and s.endswith("\n"):
            continue # ignore comments and sgml

        for c in s:
            if c == "(" and parens == 0 and cur_parse:
                yield "".join(cur_parse)
                cur_parse = []

            cur_parse.append(c)

            if c == "(":
                parens += 1
            elif c == ")":
                parens -= 1

                if parens == 0:
                    yield "".join(cur_parse).strip()
                    cur_parse = []

    if parens != 0:
        raise Exception("Parens should have been zero at end, were %s" % parens)
    if "".join(cur_parse).strip():
        raise Exception("curparse should have been empty at end, was %s" % cur_parse)

def get_leaves(s):
    """

    given a string representing a tree, return a list of the leaves
    with their parts of speeches.  Given the string:

      (TOP (S (S (NP-SBJ (DT The)
                         (JJ 30-day
                         (JJ simple)
                         (NN yield))))))

    we return:

      [('DT', 'The'),
       ('JJ', '30-day'),
       ('JJ', 'simple'),
       ('NN', 'yield')]

    """
    return re.findall(r"\(\s*([^\s)]*)\s([^)\s]*)\s*\)", s)

def get_phrase_tags(s):
    """

    given a string representing a tree, return a list of the phrase
    tags used in the string.

    given:

      (TOP (S (S (NP-SBJ (DT The)
                         (JJ 30-day
                         (JJ simple)
                         (NN yield))))))

    we return:

      ['TOP', 'S', 'S', 'NP-SBJ']
    """

    return re.findall(r"\(([^\s)(]+)\s*\(", s)

def get_pos_tags(s):
    """

    given a string representing a tree, return a list of the pos tags
    used in the string.

    given:

      (TOP (S (S (NP-SBJ (DT The)
                         (JJ 30-day
                         (JJ simple)
                         (NN yield))))))

    we return:

      ['DT', 'JJ', 'JJ', 'NN']

    """
    return re.findall(r"\(([^\s()]+)\s*[^\s()]+\s*\)", s)



def find_gaps(x_2_y_hash, xs, ys):
    """ given a hash mapping from x to y (like tree_document.align_to makes), a list of x and a list of y, find the gaps

    example:
      x_2_y_hash:
        { I -> [I]
          am -> [am]
          do -> [doing]      # funny alignment
          ing -> [doing]     # funny alignment
          well -> [well]
          *PRO* -/-> None    # not mapped
          how -> [how]
          *T* -/-> *T*       # not mapped
          are -> [are]
          you -> [you]
          doing -> [do, ing] # funny alignment
          john -> [john] }

      xs: [I am do ing well *PRO* how *T* are you doing john]
      ys: [I am doing well how *T* are you do ing john]

      return the gaps:
        [([*PRO*], []),
         ([*T*], [*T*])]

      the union of all gaps should be the tokens not mapped

    """

    xs, ys = list(xs), list(ys)

    gaps = []
    gap_x, gap_y = [], []

    y = x = None
    for mapped_x, mapped_y_list in sorted(x_2_y_hash.iteritems()):
        mapped_y_list.sort()

        #print "anchor: %s -> %s" % (mapped_x, mapped_y_list)

        x = xs.pop(0)

        if y not in mapped_y_list:
            y = ys.pop(0)

        #print "  x=%s" % (x,)
        #print "  y=%s" % (y,)
        while x < mapped_x:
            gap_x.append(x)
            x = xs.pop(0)
            #print "  x=%s" % (x,)
        while y < mapped_y_list[-1]:
            if y not in mapped_y_list:
                gap_y.append(y)
            y = ys.pop(0)
            #print "  y=%s" % (y,)

        if gap_x or gap_y:
            #print "yield", gap_x, gap_y
            gaps.append((gap_x, gap_y))
            gap_x, gap_y = [], []

    gap_x.extend(xs)
    gap_y.extend(ys)

    if gap_x or gap_y:
        gaps.append((gap_x, gap_y))
    return gaps



class CharacterRangeException(Exception):
    """ raised by :func:`fullwidth` and :func:`halfwidth` when they get bad inputs """
    pass

def is_fullwidth(c):
    try:
        halfwidth(c)
        return True
    except CharacterRangeException:
        return False

def fullwidth(ascii_chars, robust=False):
    """ convert ascii characters to their full width equivalents

    When displaying english text alongside chinese text, people often
    want the english letters to take up two columns the way the
    chinese letters do.  A common way to do this is to use the
    'Fullwidth Forms' unicode sub-range, which goes from FF01 to FF5E.
    The symbols here are the same as the ones in range 0021 to 007E
    ('!' to '~') but two bytes and two columns each.

    In ontonotes we sometimes use this to distinguish data from
    metadata in chinese.  If we had input text that looked like::

      word ( word ) word word

    In english we would treebank this as::

      (... (tag word)
           (tag -LRB-)
           (tag word)
           (tag -RRB-)
           (tag word)
           (tag word) ...)

    While in chinese we would use fullwidth forms::

      (... (tag word)
           (tag FULLWIDTH_LEFT_PARENTHESIS)
           (tag word)
           (tag FULLWIDTH_RIGHT_PARENTHESIS)
           (tag word)
           (tag word) ...)

    if robust, out of range characters are left alone.  Otherwise
    they're raised as CharacterRangeException

    """

    def fw(ascii_char):
        if not (ord('!') <= ord(ascii_char) <= ord('~')):
            if robust:
                return ascii_char
            else:
                raise CharacterRangeException("Input char not in plain text range: %s" % (ord(ascii_char)))
        return unichr(ord(u'\uff01')-ord('!')+ord(ascii_char))

    return "".join(fw(ascii_char) for ascii_char in ascii_chars)

def halfwidth(fullwidth_chars, robust=False):
    """ reverse of :func:`fullwidth` """

    def hw(fullwidth_char):
        target_ord = ord('!')-ord(u'\uff01')+ord(fullwidth_char)
        if not (ord('!') <= target_ord <= ord('~')):
            if robust:
                return fullwidth_char
            else:
                raise CharacterRangeException("Input char not in plain text range: %s -> %s" % (ord(fullwidth_char), target_ord))
        return chr(target_ord)

    return "".join(hw(fullwidth_char) for fullwidth_char in fullwidth_chars)



class NotImplementedException(Exception):
  pass



class DummyCursor():
    """ a fake MySqlDB cursor that does nothing when used.  For testing """

    @property
    def rowcount(self):
        raise NotImplementedException

    def fetchone(self):
        raise NotImplementedException

    def fetchall(self):
        raise NotImplementedException

    def execute(self, *args, **kwargs):
        pass

    def executemany(self, *args, **kwargs):
        pass




def pad_items_in_list(a_list, a_character=None):
    """
    this function will return the same list with the right amount of
    padding equal to two spaces on each side of the widest string. it
    will perform right justification.

    if the optional character is specified, then it will do a
    centering around the character in the process of padding.
    left/right justification does not work with this option.
    """

    if(a_character != None):
        for an_item in a_list:
            if(an_item.find(a_character) == -1):
                a_character = None
                break

    if(a_character != None):
        lmax=0
        rmax=0
        for an_item in a_list:
            an_item = an_item.strip()

            lf = an_item.find("*")
            if(lmax < lf):
                lmax = lf

            rf = len(an_item) - an_item.find("*")
            if(rmax < rf):
                rmax = rf



        i=0
        for i in range(0, len(a_list)):
            a_list[i] = a_list[i].strip()

            x = a_list[i].find(a_character)

            len_i=len(a_list[i])

            a_list[i] = " "*(lmax-x+2) + a_list[i]
            a_list[i] = a_list[i] + " "*(rmax-len_i+x+2)

    else:
        max=0
        for an_item in a_list:
            an_item = an_item.strip()
            x = len(an_item)
            if(max < x):
                max = x

        i=0
        for i in range(0, len(a_list)):
            a_list[i] = a_list[i].strip()

            # FIX ME: special case hack
            if(a_list[i].endswith("*") or
               a_list[i].endswith("-") or
               a_list[i][-1] in string.digits ):
                a_list[i] = "%s " % (a_list[i])

            a_list[i] = a_list[i].rjust(max+2)

    return a_list




def rows2columns(matrix):
    columns = []

    for row in matrix:
        c=0
        for cell in row:
            if(c == len(columns)):
                columns.append([])

            columns[c].append(cell)
            c = c + 1

    return columns






def pretty_print_table(rows, separator=None, out_file=None):

    # cells is the matrix
    r_c_matrix = []
    for row in rows:
        r_c_matrix.append(row.split())


    c_r_matrix = on.common.util.rows2columns(r_c_matrix)


    for i in range(0, len(c_r_matrix)):

        if(i==5 or i>10):
            padding_character=separator
        else:
            padding_character=None

        c_r_matrix[i] = on.common.util.pad_items_in_list(c_r_matrix[i], padding_character)

    r_c_matrix = on.common.util.rows2columns(c_r_matrix)

    if(out_file == None):
        for row in r_c_matrix:
            print " ".join(row).strip()
        print
    else:
        for row in r_c_matrix:
            out_file.write("%s\n" % (" ".join(row).strip()))
        out_file.write("\n")
            
        #raise NotImplementedError("this functionality has not yet been implemented")










def remove_punctuation(gi,ci,go,co):
    g=bunch(f=gi)
    c=bunch(f=ci)
    g.of=go
    c.of=co


    for x in [g,c]:
        x.line=x.f.readline().strip()

    while( g.line != "" and c.line != ""):

        g.line = re.sub("_", "-UNDERSCORE-", g.line)
        c.line = re.sub("_", "-UNDERSCORE-", c.line)

        for x in [g,c]:
            x.line = re.sub("\(([^ ]+) *([^()]+)\)", "(\g<1> \g<1>_\g<2>)", x.line)
            x.line=re.sub("\)", ") ", x.line)
            x.tokens=x.line.split()

            if on.common.log.DEBUG:
                print x.line
                print x.tokens


            x.l=len(x.tokens)

            x.o_line=""
            x.i=0



        while( g.i<g.l  and c.i<c.l ):

            for x in [g,c]:
                while ( x.i<x.l
                    and
                    ( x.tokens[x.i].find(")") == -1 or x.tokens[x.i] == ")" )
                    ):

                    x.o_line = x.o_line + " " + x.tokens[x.i]
                    x.i = x.i + 1


            if (g.i < g.l and
                c.i < c.l):

                if on.common.log.DEBUG:
                    print g.tokens[g.i], c.tokens[c.i]


                if(g.tokens[g.i].split("_")[0] not in [":", ",", "``", "''", ".", "-LRB-", "-RRB-"]):
                    for x in [g,c]:
                        x.o_line = x.o_line + " " + x.tokens[x.i].split("_")[1]
                else:
                    for x in [g,c]:
                        x.o_line = " ".join(x.o_line.split()[:-1])

                for x in [g,c]:
                    x.i = x.i+1


        for x in [g,c]:

            # there can sometimes be a node with only punctuation which will remain undeleted at
            # this point, but will have no leaves.
            #
            #
            # example tree: ... (ADJP-PRD (JJ welcome))))) (PRN (-LRB- -LSB-) (, ...) (-RRB-
            # -RSB-)) (CC but) ...

            x.o_line = re.sub("\([^ ]+\s+\)", "", x.o_line)

            # replace the underscores back
            x.o_line = re.sub("-UNDERSCORE-", "_", x.o_line)

            x.of.write("%s\n" % (x.o_line))
            x.line = x.f.readline().strip()

        sys.stderr.write(".")




def __LINE__():
    try:
        raise Exception
    except:
        return sys.exc_info()[2].tb_frame.f_back.f_lineno

def __FILE__():
    return inspect.currentframe().f_code.co_filename




def convert_html_entities(s):
  matches = re.findall("&#\d+;", s)
  if len(matches) > 0:
      hits = set(matches)
      for hit in hits:
              name = hit[2:-1]
              try:
                      entnum = int(name)
                      s = s.replace(hit, unichr(entnum))
              except ValueError:
                      pass

  matches = re.findall("&\w+;", s)
  hits = set(matches)
  amp = "&amp;"
  if amp in hits:
      hits.remove(amp)
  for hit in hits:
      name = hit[1:-1]
      if htmlentitydefs.name2codepoint.has_key(name):
              s = s.replace(hit, unichr(htmlentitydefs.name2codepoint[name]))
  s = s.replace(amp, "&")
  return s


    
    

def normalize_html_entities(s):
    for k, v in html_entity_replacement_map.iteritems():
        s = s.replace(k, v)
    return s


def two_int_tuple_compare(tuple_1, tuple_2):
    if( float(tuple_1[0]) < float(tuple_2[0]) ):
        return -1
    elif( float(tuple_1[0]) == float(tuple_2[0]) ):
        return cmp(float(tuple_1[1]), float(tuple_2[1]))
    else:
        return 1
    

def arabic_pos2penn_pos(arabic_pos):
    if arabic_pos == 'NOUN_QUANT+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'ADJ+NSUFF_FEM_DU_NOM_POSS': return 'JJ' 
    if arabic_pos == 'IVSUFF_DO:1S': return 'PRP' 
    if arabic_pos == 'PVSUFF_DO:2MP': return 'PRP' 
    if arabic_pos == 'PV_PASS+PVSUFF_SUBJ:1P': return 'VBN' 
    if arabic_pos == 'IV3FP+IV+IVSUFF_SUBJ:FP': return 'VBP' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_MASC_PL_NOM' : return 'NNPS' 
    if arabic_pos == 'ADJ_NUM+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN_PROP+NSUFF_FEM_SG+CASE_INDEF_GEN' : return 'NNP' 
    if arabic_pos == 'DET+NOUN_PROP+CASE_DEF_NOM': return 'NNP' 
    if arabic_pos == 'NOUN+NSUFF_FEM_PL+CASE_INDEF_GEN': return 'NNS' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_FEM_SG+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'ADJ.VN+NSUFF_FEM_SG+CASE_INDEF_GEN': return 'JJ' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_SG+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'IV1S+IV' : return 'PRP' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_PL+CASE_DEF_NOM': return 'NNS' 
    if arabic_pos == 'CONNEC_PART': return 'CC' 
    if arabic_pos == 'NOUN_QUANT+NSUFF_FEM_SG+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'IVSUFF_DO:3D': return 'PRP' 
    if arabic_pos == 'DET+NOUN.VN+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'INTERJ': return 'UH' 
    if arabic_pos == 'FOCUS_PART': return 'RP' 
    if arabic_pos == 'NOUN_NUM+NSUFF_FEM_SG+CASE_INDEF_NOM': return 'CD' 
    if arabic_pos == 'ADJ.VN+NSUFF_MASC_PL_NOM': return 'JJ' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_PL+CASE_DEF_NOM': return 'NNS' 
    if arabic_pos == 'ADJ_COMP+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'DEM_PRON_MS': return 'DT' 
    if arabic_pos == 'NOUN+NSUFF_FEM_SG+CASE_INDEF_GEN': return 'NN' 
    if arabic_pos == 'NOUN+NSUFF_MASC_PL_GEN': return 'NNS' 
    if arabic_pos == 'VERB_PART': return 'RP' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_PL+CASE_DEF_GEN': return 'NNS' 
    if arabic_pos == 'ADJ.VN+NSUFF_FEM_SG' : return 'JJ' 
    if arabic_pos == 'DET+ADJ+NSUFF_MASC_PL_NOM': return 'JJ' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_FEM_SG+CASE_DEF_NOM': return 'NNP' 
    if arabic_pos == 'NOUN+NSUFF_FEM_DU_NOM': return 'NNS' 
    if arabic_pos == 'DET+ADJ_COMP+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN_NUM+CASE_INDEF_ACC': return 'CD' 
    if arabic_pos == 'NOUN+CASE_INDEF_ACC': return 'NN' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_DU_ACCGEN': return 'NOFUNC' 
    if arabic_pos == 'IVSUFF_DO:1P': return 'PRP' 
    if arabic_pos == 'NOUN_NUM+NSUFF_MASC_PL_GEN': return 'CD' 
    if arabic_pos == 'POSS_PRON_3FS': return 'PRP$' 
    if arabic_pos == 'NOUN+NSUFF_MASC_DU_NOM': return 'NNS' 
    if arabic_pos == 'NOUN+NSUFF_MASC_PL_GEN_POSS': return 'NNS' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:1P': return 'VBD' 
    if arabic_pos == 'DET+ADJ_COMP+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'PRON_2FP' : return 'PRP' 
    if arabic_pos == 'ADJ+NSUFF_FEM_PL+CASE_INDEF_GEN': return 'JJ' 
    if arabic_pos == 'POSS_PRON_2FS': return 'PRP$' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:3MP': return 'VBD' 
    if arabic_pos == 'NOUN': return 'NN' 
    if arabic_pos == 'ADJ+CASE_INDEF_ACC': return 'JJ' 
    if arabic_pos == 'GRAMMAR_PROBLEM' : return 'JJ' 
    if arabic_pos == 'INTERROG_PART': return 'RP' 
    if arabic_pos == 'ADJ+CASE_INDEF_GEN': return 'JJ' 
    if arabic_pos == 'NOUN+NSUFF_FEM_PL+CASE_INDEF_NOM': return 'NNS' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'DET+ADJ_NUM+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'ADJ+NSUFF_FEM_SG+CASE_INDEF_NOM': return 'JJ' 
    if arabic_pos == 'ADJ.VN+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'NOUN_PROP+CASE_INDEF_GEN': return 'NNP' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_PL+CASE_DEF_ACC': return 'NNS' 
    if arabic_pos == 'NOUN+NSUFF_FEM_SG+CASE_INDEF_NOM': return 'NN' 
    if arabic_pos == 'DET+NOUN_PROP': return 'NNP' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_FEM_SG+CASE_DEF_ACC': return 'NNP' 
    if arabic_pos == 'POSS_PRON_2MS': return 'PRP$' 
    if arabic_pos == 'PRON_3MS': return 'PRP' 
    if arabic_pos == 'IV2MS+IV+IVSUFF_MOOD:S': return 'VBP' 
    if arabic_pos == 'DET+ADJ_NUM+NSUFF_MASC_PL_GEN' : return 'JJ' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_MASC_DU_GEN': return 'JJ' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_DU_GEN': return 'NNS' 
    if arabic_pos == 'ADJ.VN+CASE_DEF_GEN' : return 'JJ' 
    if arabic_pos == 'DEM_PRON': return 'DT' 
    if arabic_pos == 'IVSUFF_DO:3MP': return 'PRP' 
    if arabic_pos == 'PVSUFF_DO:3FS': return 'PRP' 
    if arabic_pos == 'DET+ADJ_NUM+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'NOUN+NSUFF_FEM_PL+CASE_DEF_ACC': return 'NNS' 
    if arabic_pos == 'DET+NOUN.VN' : return 'NN' 
    if arabic_pos == 'ADJ_NUM+CASE_INDEF_GEN': return 'JJ' 
    if arabic_pos == 'DEM_PRON_MP': return 'DT' 
    if arabic_pos == 'EMPHATIC_PART': return 'IN' 
    if arabic_pos == 'IV3MP+IV+IVSUFF_SUBJ:MP_MOOD:I': return 'VBP' 
    if arabic_pos == 'IV': return 'NOFUNC' 
    if arabic_pos == 'DET+ADJ_COMP': return 'JJ' 
    if arabic_pos == 'IV3MS+IV_PASS+IVSUFF_MOOD:J': return 'VBN' 
    if arabic_pos == 'IV3FS+IV+IVSUFF_MOOD:J': return 'VBP' 
    if arabic_pos == 'ADJ+NSUFF_MASC_DU_ACC': return 'JJ' 
    if arabic_pos == 'ADJ.VN+CASE_INDEF_GEN': return 'JJ' 
    if arabic_pos == 'NOUN_QUANT+CASE_INDEF_ACC': return 'NN' 
    if arabic_pos == 'DET+ADJ+NSUFF_MASC_PL_ACC': return 'JJ' 
    if arabic_pos == 'NOUN_PROP+CASE_INDEF_ACC': return 'NNP' 
    if arabic_pos == 'PV_PASS+PVSUFF_SUBJ:1S': return 'VBN' 
    if arabic_pos == 'ADJ+NSUFF_FEM_SG+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'NOUN+NSUFF_MASC_DU_ACC_POSS': return 'NNS' 
    if arabic_pos == 'ADJ.VN+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN_NUM+NSUFF_FEM_SG+CASE_DEF_GEN': return 'CD' 
    if arabic_pos == 'PRON_2MS': return 'PRP' 
    if arabic_pos == 'IV3FS+IV_PASS+IVSUFF_MOOD:I': return 'VBN' 
    if arabic_pos == 'ADJ+NSUFF_FEM_SG+CASE_INDEF_GEN': return 'JJ' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:2FS': return 'VBD' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_PL_GEN': return 'NNS' 
    if arabic_pos == 'ADJ_NUM+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'ADJ_COMP+CASE_INDEF_GEN': return 'JJ' 
    if arabic_pos == 'IV1P+IV_PASS+IVSUFF_MOOD:I' : return 'VBP' 
    if arabic_pos == 'NOUN_QUANT+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'DET+NOUN_QUANT+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'NOUN+NSUFF_FEM_PL+CASE_DEF_NOM': return 'NNS' 
    if arabic_pos == 'PV_PASS+PVSUFF_SUBJ:3FD' : return 'VBP' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_FEM_SG+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:3FS': return 'VBD' 
    if arabic_pos == 'ADJ.VN+NSUFF_MASC_DU_NOM_POSS' : return 'JJ' 
    if arabic_pos == 'NOUN.VN': return 'NN' 
    if arabic_pos == 'DET+ADJ_COMP+NSUFF_MASC_PL_GEN' : return 'NNS' 
    if arabic_pos == 'NOUN+NSUFF_MASC_DU_GEN': return 'NNS' 
    if arabic_pos == 'NOUN_NUM+NSUFF_MASC_DU_GEN': return 'NNS' 
    if arabic_pos == 'DET+ADJ_NUM+NSUFF_FEM_PL+CASE_DEF_NOM': return 'NNS' 
    if arabic_pos == 'POSS_PRON_3MP': return 'PRP$' 
    if arabic_pos == 'IV1S+IV_PASS+IVSUFF_MOOD:I' : return 'VBP' 
    if arabic_pos == 'NOUN+NSUFF_MASC_PL_ACC': return 'NNS' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:1S': return 'VBD' 
    if arabic_pos == 'REL_PRON+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'ADJ.VN+NSUFF_FEM_DU_ACC' : return 'JJ' 
    if arabic_pos == 'NOUN_NUM+NSUFF_MASC_PL_NOM': return 'CD' 
    if arabic_pos == 'PVSUFF_DO:2FS': return 'PRP' 
    if arabic_pos == 'ADJ+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'NOUN_PROP+NSUFF_FEM_PL+CASE_DEF_GEN' : return 'NNPS' 
    if arabic_pos == 'ADJ_NUM+CASE_INDEF_NOM': return 'JJ' 
    if arabic_pos == 'DET+ADJ_NUM+NSUFF_FEM_SG+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'DET+ADJ.VN+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'IV3MS+IV' : return 'VBP' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_SG': return 'NOFUNC' 
    if arabic_pos == 'NOUN_PROP+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NNP' 
    if arabic_pos == 'DET+NOUN+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'NOUN_PROP+CASE_INDEF_NOM': return 'NNP' 
    if arabic_pos == 'INTERROG_PRON': return 'RP' 
    if arabic_pos == 'IV3MP+IV+IVSUFF_SUBJ:MP_MOOD:SJ': return 'VBP' 
    if arabic_pos == 'NOUN_NUM+NSUFF_MASC_DU_ACC_POSS': return 'CD' 
    if arabic_pos == 'ADJ.VN+NSUFF_MASC_PL_ACC': return 'JJ' 
    if arabic_pos == 'ADJ_COMP': return 'JJ' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_SG+CASE_INDEF_ACC': return 'NN' 
    if arabic_pos == 'DET+ADJ+NSUFF_MASC_DU_ACC': return 'JJ' 
    if arabic_pos == 'DET+ADJ_COMP+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'NOUN+CASE_INDEF_NOM': return 'NN' 
    if arabic_pos == 'IV3FS+IV': return 'NOFUNC' 
    if arabic_pos == 'ADJ_NUM+NSUFF_FEM_SG+CASE_INDEF_GEN': return 'JJ' 
    if arabic_pos == 'NOUN_PROP+NSUFF_FEM_SG+CASE_INDEF_ACC' : return 'NNP' 
    if arabic_pos == 'CV': return 'NOFUNC' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:3MS': return 'VBD' 
    if arabic_pos == 'ADJ+NSUFF_MASC_DU_GEN_POSS' : return 'JJ' 
    if arabic_pos == 'IV1S+IV+IVSUFF_MOOD:S': return 'VBP' 
    if arabic_pos == 'DET+ABBREV' : return 'NN' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_SG' : return 'NN' 
    if arabic_pos == 'DET+NOUN_NUM+NSUFF_FEM_SG+CASE_DEF_ACC': return 'CD' 
    if arabic_pos == 'NOUN+NSUFF_FEM_SG': return 'NN' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_SG+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'DET+NOUN+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'NOUN_PROP': return 'NNP' 
    if arabic_pos == 'TYPO': return 'NOFUNC' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_PL+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'CVSUFF_DO:1P': return 'PRP' 
    if arabic_pos == 'NOUN+NSUFF_FEM_DU_GEN_POSS': return 'NNS' 
    if arabic_pos == 'PV+PVSUFF_3MS': return 'RP' 
    if arabic_pos == 'PRON_3MP': return 'PRP' 
    if arabic_pos == 'DET+NOUN_NUM+NSUFF_MASC_PL_GEN': return 'CD' 
    if arabic_pos == 'IV2FP+IV+IVSUFF_SUBJ:FP': return 'VBP' 
    if arabic_pos == 'RC_PART': return 'CC' 
    if arabic_pos == 'PVSUFF_DO:3MS': return 'PRP' 
    if arabic_pos == 'NOUN+NSUFF_MASC_PL_ACCGEN' : return 'NNS' 
    if arabic_pos == 'NOUN_QUANT+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'ADJ_COMP+CASE_INDEF_NOM': return 'JJ' 
    if arabic_pos == 'ADJ+NSUFF_MASC_PL_GEN_POSS': return 'JJ' 
    if arabic_pos == 'NOUN+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'NOUN+NSUFF_FEM_PL': return 'NNS' 
    if arabic_pos == 'POSS_PRON_1P': return 'PRP$' 
    if arabic_pos == 'NOUN_NUM+NSUFF_FEM_SG+CASE_DEF_ACC': return 'CD' 
    if arabic_pos == 'ABBREV': return 'NN' 
    if arabic_pos == 'IV1P+IV' : return 'VBP' 
    if arabic_pos == 'CONJ': return 'CC' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_FEM_DU_ACC' : return 'JJ' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_PL_ACCGEN': return 'NOFUNC' 
    if arabic_pos == 'DET+NOUN': return 'NN' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_DU_GEN': return 'JJ' 
    if arabic_pos == 'ADJ+NSUFF_FEM_DU_GEN': return 'JJ' 
    if arabic_pos == 'PVSUFF_DO:3MP': return 'PRP' 
    if arabic_pos == 'ADJ.VN+CASE_INDEF_ACC': return 'JJ' 
    if arabic_pos == 'IV3MS+IV_PASS+IVSUFF_MOOD:S': return 'VBN' 
    if arabic_pos == 'NOUN+NSUFF_MASC_PL_NOM': return 'NNS' 
    if arabic_pos == 'DET+ADJ.VN+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_PL+CASE_DEF_GEN': return 'NNS' 
    if arabic_pos == 'IV2MS+IV+IVSUFF_MOOD:J': return 'VBP' 
    if arabic_pos == 'NOUN+NSUFF_MASC_PL_ACC_POSS': return 'NNS' 
    if arabic_pos == 'DET+NOUN_QUANT+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'NOUN+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_FEM_DU_GEN': return 'JJ' 
    if arabic_pos == 'IV2D+IV+IVSUFF_SUBJ:D_MOOD:SJ': return 'VBP' 
    if arabic_pos == 'NOUN+NSUFF_MASC_DU_NOM_POSS': return 'NNS' 
    if arabic_pos == 'IV3MS+IV_PASS+IVSUFF_MOOD:I': return 'VBN' 
    if arabic_pos == 'NOUN_QUANT+NSUFF_FEM_SG+CASE_INDEF_GEN': return 'NN' 
    if arabic_pos == 'ADJ+NSUFF_MASC_DU_NOM': return 'JJ' 
    if arabic_pos == 'FOREIGN': return 'NOFUNC' 
    if arabic_pos == 'LATIN' : return 'NN' 
    if arabic_pos == 'NOUN.VN+CASE_INDEF_NOM': return 'NN' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_PL': return 'NOFUNC' 
    if arabic_pos == 'ADJ+NSUFF_FEM_DU_ACC': return 'JJ' 
    if arabic_pos == 'NOUN+NSUFF_MASC_DU_GEN_POSS': return 'NNS' 
    if arabic_pos == 'NOUN.VN+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'IV3FS+IV_PASS+IVSUFF_MOOD:J': return 'VBN' 
    if arabic_pos == 'IV+IVSUFF_SUBJ:3FS' : return 'VBP' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_SG+CASE_INDEF_NOM': return 'NN' 
    if arabic_pos == 'NOUN.VN+CASE_INDEF_GEN': return 'NN' 
    if arabic_pos == 'CV+CVSUFF_SUBJ:2FS': return 'NOFUNC' 
    if arabic_pos == 'DET+ADJ': return 'JJ' 
    if arabic_pos == 'PRON_2FS' : return 'PRP' 
    if arabic_pos == 'PVSUFF_DO:1P': return 'PRP' 
    if arabic_pos == 'PART': return 'RP' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_PL_NOM': return 'NNS' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NNP' 
    if arabic_pos == 'NOUN_NUM+CASE_DEF_GEN': return 'CD' 
    if arabic_pos == 'POSS_PRON_2FP': return 'PRP$' 
    if arabic_pos == 'VERB': return 'IN' 
    if arabic_pos == 'NOUN_QUANT+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'NOUN_PROP+NSUFF_FEM_PL+CASE_INDEF_GEN' : return 'NNPS' 
    if arabic_pos == 'DET+ADJ_NUM+NSUFF_FEM_SG+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_DU_ACC': return 'JJ' 
    if arabic_pos == 'PRON_1S': return 'PRP' 
    if arabic_pos == 'VOC_PART': return 'UH' 
    if arabic_pos == 'DET+NOUN.VN+NSUFF_MASC_PL_ACC' : return 'NNS' 
    if arabic_pos == 'NOUN_PROP+NSUFF_FEM_SG+CASE_DEF_NOM': return 'NNP' 
    if arabic_pos == 'DET+NOUN.VN+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'POSS_PRON_2MP': return 'PRP$' 
    if arabic_pos == 'NOUN+NSUFF_FEM_DU_GEN': return 'NNS' 
    if arabic_pos == 'PRON' : return 'PRP' 
    if arabic_pos == 'IVSUFF_DO:2MS': return 'PRP' 
    if arabic_pos == 'NOUN_NUM+NSUFF_FEM_SG+CASE_INDEF_GEN': return 'CD' 
    if arabic_pos == 'PVSUFF_DO:2MS': return 'IN' 
    if arabic_pos == 'DET+NOUN_NUM+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'DEM_PRON_F': return 'DT' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_FEM_PL+CASE_DEF_NOM' : return 'JJ' 
    if arabic_pos == 'DET+ADJ+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'ADJ_NUM': return 'CD' 
    if arabic_pos == 'ADJ.VN+NSUFF_FEM_SG+CASE_DEF_ACC' : return 'JJ' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_PL+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN+NSUFF_FEM_PL+CASE_INDEF_ACC': return 'NNS' 
    if arabic_pos == 'ADJ+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'PV_PASS+PVSUFF_SUBJ:3MS': return 'VBN' 
    if arabic_pos == 'ADJ+NSUFF_FEM_SG+CASE_INDEF_ACC': return 'JJ' 
    if arabic_pos == 'ADJ+NSUFF_FEM_SG+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'DET+ADJ+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'DET+ADJ.VN+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_DU_NOM': return 'JJ' 
    if arabic_pos == 'NOUN.VN+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'REL_PRON+NSUFF_FEM_SG+CASE_DEF_GEN' : return 'NN' 
    if arabic_pos == 'IV3FS+IV+IVSUFF_MOOD:S': return 'VBP' 
    if arabic_pos == 'NOUN_NUM+NSUFF_MASC_PL_ACC': return 'CD' 
    if arabic_pos == 'CV+CVSUFF_SUBJ:2MP': return 'VB' 
    if arabic_pos == 'IVSUFF_DO:2FS' : return 'PRP' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_FEM_SG+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'ADJ_NUM+NSUFF_FEM_SG+CASE_INDEF_ACC': return 'NN' 
    if arabic_pos == 'DET+NOUN_NUM+NSUFF_MASC_PL_NOM' : return 'NNS' 
    if arabic_pos == 'NOUN_NUM+NSUFF_FEM_DU_ACC': return 'CD' 
    if arabic_pos == 'NOUN+NSUFF_FEM_DU_NOM_POSS': return 'NNS' 
    if arabic_pos == 'DET+ADJ+NSUFF_MASC_PL_GEN': return 'JJ' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:3FP': return 'VBD' 
    if arabic_pos == 'ADJ.VN+CASE_INDEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN_NUM+CASE_DEF_NOM': return 'CD' 
    if arabic_pos == 'DEM_PRON_FD': return 'DT' 
    if arabic_pos == 'DET+NOUN_NUM+CASE_DEF_ACC' : return 'NN' 
    if arabic_pos == 'IV2MS+IV_PASS+IVSUFF_MOOD:I' : return 'VBP' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_DU_ACCGEN_POSS' : return 'JJ' 
    if arabic_pos == 'NOUN_QUANT+CASE_INDEF_GEN': return 'NN' 
    if arabic_pos == 'PRON_3FP': return 'PRP' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_MASC_PL_GEN': return 'NNPS' 
    if arabic_pos == 'PRON_2MP': return 'PRP' 
    if arabic_pos == 'DET+ADJ_NUM+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'SUB_CONJ': return 'IN' 
    if arabic_pos == 'PREP': return 'IN' 
    if arabic_pos == 'NOUN+NSUFF_FEM_SG+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_PL_ACC': return 'NNS' 
    if arabic_pos == 'PV': return 'NOFUNC' 
    if arabic_pos == 'IV1S+IV+IVSUFF_MOOD:J': return 'VBP' 
    if arabic_pos == 'NOUN.VN+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'IV1P+IV+IVSUFF_MOOD:S': return 'VBP' 
    if arabic_pos == 'ADJ+NSUFF_MASC_DU_GEN': return 'JJ' 
    if arabic_pos == 'IV1P+IV+IVSUFF_MOOD:J': return 'VBP' 
    if arabic_pos == 'POSS_PRON_1S': return 'PRP$' 
    if arabic_pos == 'FUT_PART': return 'NN' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_FEM_PL+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_MASC_PL_NOM': return 'JJ' 
    if arabic_pos == 'DET+NOUN_QUANT+NSUFF_FEM_SG+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'ADJ+NSUFF_MASC_PL_NOM': return 'JJ' 
    if arabic_pos == 'IV2MP+IV+IVSUFF_SUBJ:MP_MOOD:I': return 'VBP' 
    if arabic_pos == 'ADJ+NSUFF_MASC_PL_ACCGEN': return 'NOFUNC' 
    if arabic_pos == 'IVSUFF_DO:2MP': return 'PRP' 
    if arabic_pos == 'NOUN+CASE_INDEF_GEN': return 'NN' 
    if arabic_pos == 'ADJ_NUM+CASE_INDEF_ACC': return 'JJ' 
    if arabic_pos == 'PV_PASS': return 'NOFUNC' 
    if arabic_pos == 'ADV': return 'RB' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_MASC_PL_ACCGEN' : return 'JJ' 
    if arabic_pos == 'NOUN_QUANT+NSUFF_FEM_SG+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'DET+NOUN+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_DU_GEN': return 'NNS' 
    if arabic_pos == 'DET+ADJ+NSUFF_MASC_PL_ACCGEN': return 'NOFUNC' 
    if arabic_pos == 'NOUN+NSUFF_MASC_DU_ACC': return 'NNS' 
    if arabic_pos == 'NOUN_PROP+CASE_DEF_GEN': return 'NNP' 
    if arabic_pos == 'NOUN+NSUFF_MASC_DU' : return 'NN' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:3MD': return 'VBD' 
    if arabic_pos == 'JUS_PART': return 'IN' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_MASC_PL_GEN': return 'JJ' 
    if arabic_pos == 'ADJ_NUM+NSUFF_FEM_DU_GEN' : return 'NNS' 
    if arabic_pos == 'ADJ.VN+NSUFF_FEM_SG+CASE_INDEF_NOM': return 'JJ' 
    if arabic_pos == 'CV+CVSUFF_SUBJ:2MS': return 'VB' 
    if arabic_pos == 'IV2D+IV+IVSUFF_SUBJ:D_MOOD:I': return 'VBP' 
    if arabic_pos == 'NOUN.VN+NSUFF_MASC_PL_ACC': return 'NNS' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_SG+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'PRON_2D' : return 'PRP' 
    if arabic_pos == 'DET+ADJ+NSUFF_MASC_DU_GEN': return 'JJ' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_PL+CASE_DEF_ACC' : return 'JJ' 
    if arabic_pos == 'ADJ_NUM+NSUFF_FEM_SG+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN+NSUFF_FEM_SG+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'ADJ_COMP+CASE_INDEF_ACC': return 'JJ' 
    if arabic_pos == 'NOUN+NSUFF_FEM_DU_ACC': return 'NNS' 
    if arabic_pos == 'ADJ+NSUFF_MASC_DU_NOM_POSS': return 'JJ' 
    if arabic_pos == 'IV_PASS': return 'NOFUNC' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_DU' : return 'NNS' 
    if arabic_pos == 'ADJ+NSUFF_FEM_SG+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'DET+ADJ+NSUFF_MASC_DU_NOM': return 'JJ' 
    if arabic_pos == 'DET+ADJ+NSUFF_FEM_SG+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'DET+NOUN.VN+NSUFF_FEM_SG+CASE_DEF_ACC' : return 'NN' 
    if arabic_pos == 'DET+NOUN_NUM+NSUFF_MASC_PL_ACC': return 'CD' 
    if arabic_pos == 'DET+NOUN_QUANT+NSUFF_FEM_SG+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'NOUN+NSUFF_MASC_PL_NOM_POSS': return 'NNS' 
    if arabic_pos == 'DET+TYPO': return 'NOFUNC' 
    if arabic_pos == 'DET+FOREIGN': return 'NOFUNC' 
    if arabic_pos == 'IV1S+IV_PASS+IVSUFF_MOOD:S': return 'VBP' 
    if arabic_pos == 'INTERROG_PRON+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'IV3MP+IV_PASS+IVSUFF_SUBJ:MP_MOOD:SJ': return 'VBN' 
    if arabic_pos == 'DET+NOUN.VN+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_DU_ACC': return 'NNS' 
    if arabic_pos == 'PV_PASS+PVSUFF_SUBJ:3MD': return 'VBN' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_FEM_SG': return 'NNP' 
    if arabic_pos == 'PRON_3D': return 'PRP' 
    if arabic_pos == 'POSS_PRON_3MS': return 'PRP$' 
    if arabic_pos == 'ADJ_COMP+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'POSS_PRON_3D': return 'PRP$' 
    if arabic_pos == 'DET+NOUN_PROP+CASE_DEF_ACC': return 'NNP' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_FEM_DU_NOM': return 'NNP' 
    if arabic_pos == 'ADJ+NSUFF_FEM_DU_NOM': return 'JJ' 
    if arabic_pos == 'ADJ_NUM+NSUFF_FEM_SG+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'DET+ADJ_NUM': return 'JJ' 
    if arabic_pos == 'ADJ+NSUFF_MASC_PL_GEN': return 'JJ' 
    if arabic_pos == 'IV2FS+IV+IVSUFF_SUBJ:2FS_MOOD:I': return 'VBP' 
    if arabic_pos == 'DET+NOUN.VN+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_MASC_DU_ACC' : return 'JJ' 
    if arabic_pos == 'NOUN_NUM+CASE_INDEF_GEN': return 'CD' 
    if arabic_pos == 'IV3MS+IV+IVSUFF_MOOD:I': return 'VBP' 
    if arabic_pos == 'NOUN_PROP+CASE_DEF_ACC': return 'NNP' 
    if arabic_pos == 'RESTRIC_PART': return 'RP' 
    if arabic_pos == 'DET+ADJ.VN': return 'JJ' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_DU_NOM': return 'NNS' 
    if arabic_pos == 'NOUN+NSUFF_FEM_PL+CASE_DEF_GEN': return 'NNS' 
    if arabic_pos == 'DET+ADJ_COMP+NSUFF_MASC_DU_GEN' : return 'NNS' 
    if arabic_pos == 'INTERROG_PRON+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'NOUN_QUANT+CASE_INDEF_NOM': return 'NN' 
    if arabic_pos == 'ADJ_COMP+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN_NUM+NSUFF_FEM_SG+CASE_DEF_NOM': return 'CD' 
    if arabic_pos == 'ADJ_NUM+CASE_DEF_ACC': return 'JJ' 
    if arabic_pos == 'ADJ_NUM+NSUFF_FEM_SG+CASE_INDEF_NOM': return 'JJ' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'IVSUFF_DO:3MS': return 'PRP' 
    if arabic_pos == 'IV2MP+IV+IVSUFF_SUBJ:MP_MOOD:SJ': return 'VBP' 
    if arabic_pos == 'DEM_PRON_MD': return 'DT' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_SG+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'IV2MS+IV+IVSUFF_MOOD:I': return 'VBP' 
    if arabic_pos == 'INTERROG_ADV': return 'RP' 
    if arabic_pos == 'ADJ+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'DET+ADJ_NUM+NSUFF_MASC_DU_ACC' : return 'JJ' 
    if arabic_pos == 'ADJ+NSUFF_FEM_PL+CASE_INDEF_ACC' : return 'JJ' 
    if arabic_pos == 'PSEUDO_VERB': return 'IN' 
    if arabic_pos == 'NOUN+CASE_DEF_ACC': return 'IN' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_FEM_SG': return 'NOFUNC' 
    if arabic_pos == 'REL_PRON+CASE_INDEF_ACC': return 'WP' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_FEM_DU_GEN': return 'NNP' 
    if arabic_pos == 'DIALECT': return 'NOFUNC' 
    if arabic_pos == 'NOUN_NUM+CASE_DEF_ACC': return 'CD' 
    if arabic_pos == 'IVSUFF_DO:3FS': return 'PRP' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_SG+CASE_INDEF_GEN' : return 'NN' 
    if arabic_pos == 'INTERROG_PRON+NSUFF_FEM_SG+CASE_DEF_GEN' : return 'RP' 
    if arabic_pos == 'DET+NOUN_NUM+NSUFF_FEM_SG+CASE_DEF_GEN': return 'NN' 
    if arabic_pos == 'IV1P+IV+IVSUFF_MOOD:I': return 'VBP' 
    if arabic_pos == 'IV3MD+IV_PASS+IVSUFF_SUBJ:D_MOOD:I' : return 'VBN' 
    if arabic_pos == 'ADJ+NSUFF_FEM_SG': return 'NOFUNC' 
    if arabic_pos == 'PV_PASS+PVSUFF_SUBJ:3FS': return 'VBN' 
    if arabic_pos == 'POSS_PRON_3FP': return 'PRP$' 
    if arabic_pos == 'IV1S+IV+IVSUFF_MOOD:I': return 'VBP' 
    if arabic_pos == 'ADJ.VN+NSUFF_MASC_PL_GEN': return 'JJ' 
    if arabic_pos == 'NOUN_PROP+CASE_DEF_NOM': return 'NNP' 
    if arabic_pos == 'NOUN_QUANT': return 'IN' 
    if arabic_pos == 'ADJ': return 'JJ' 
    if arabic_pos == 'NOUN_NUM+NSUFF_MASC_DU_GEN_POSS': return 'CD' 
    if arabic_pos == 'PV_PASS+PVSUFF_SUBJ:3FP' : return 'VBD' 
    if arabic_pos == 'DET+ADJ_NUM+NSUFF_MASC_DU_GEN' : return 'JJ' 
    if arabic_pos == 'IV3FD+IV+IVSUFF_SUBJ:D_MOOD:SJ': return 'VBP' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:2MP': return 'VBD' 
    if arabic_pos == 'DET+NOUN_PROP+NSUFF_FEM_PL+CASE_DEF_GEN' : return 'NNPS' 
    if arabic_pos == 'NOUN.VN+CASE_INDEF_ACC': return 'NN' 
    if arabic_pos == 'PVSUFF_DO:1S': return 'PRP' 
    if arabic_pos == 'IV3MD+IV+IVSUFF_SUBJ:D_MOOD:SJ': return 'VBP' 
    if arabic_pos == 'ADJ+CASE_INDEF_NOM': return 'JJ' 
    if arabic_pos == 'PRON_3FS': return 'PRP' 
    if arabic_pos == 'IV2MS+IV_PASS+IVSUFF_MOOD:J': return 'VBN' 
    if arabic_pos == 'NOUN+NSUFF_FEM_SG+CASE_INDEF_ACC': return 'NN' 
    if arabic_pos == 'IV3MS+IV+IVSUFF_MOOD:S': return 'VBP' 
    if arabic_pos == 'PV_PASS+PVSUFF_SUBJ:3MP': return 'VBN' 
    if arabic_pos == 'IV3MD+IV+IVSUFF_SUBJ:D_MOOD:I': return 'VBP' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_DU_ACC': return 'NNS' 
    if arabic_pos == 'DET+ADJ+CASE_DEF_GEN': return 'JJ' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_PL+CASE_DEF_ACC' : return 'NNS' 
    if arabic_pos == 'NOUN+NSUFF_FEM_DU_ACC_POSS': return 'NNS' 
    if arabic_pos == 'DET': return 'NN' 
    if arabic_pos == 'NOUN+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'NOUN_QUANT+NSUFF_MASC_DU_ACC_POSS': return 'NNS' 
    if arabic_pos == 'NOUN_NUM+CASE_INDEF_NOM': return 'CD' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:3FD': return 'VBD' 
    if arabic_pos == 'DET+NOUN_QUANT+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'ADV+CASE_INDEF_ACC': return 'RB' 
    if arabic_pos == 'NOUN_PROP+NSUFF_FEM_SG': return 'NNP' 
    if arabic_pos == 'ADJ.VN+NSUFF_FEM_SG+CASE_INDEF_ACC': return 'JJ' 
    if arabic_pos == 'IV3FD+IV+IVSUFF_SUBJ:D_MOOD:I': return 'VBP' 
    if arabic_pos == 'NOUN_PROP+NSUFF_FEM_SG+CASE_DEF_ACC': return 'NNP' 
    if arabic_pos == 'ADJ.VN+NSUFF_MASC_DU_ACC' : return 'JJ' 
    if arabic_pos == 'DET+ADJ_NUM+NSUFF_FEM_SG+CASE_DEF_NOM': return 'JJ' 
    if arabic_pos == 'ADJ.VN': return 'NOFUNC' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_SG+CASE_DEF_ACC': return 'NN' 
    if arabic_pos == 'IV3FS+IV+IVSUFF_MOOD:I': return 'VBP' 
    if arabic_pos == 'DET+NOUN+NSUFF_MASC_DU_NOM': return 'NNS' 
    if arabic_pos == 'NEG_PART': return 'RP' 
    if arabic_pos == 'PV+PVSUFF_SUBJ:2MS': return 'VBD' 
    if arabic_pos == 'IV3MS+IV+IVSUFF_MOOD:J': return 'VBP' 
    if arabic_pos == 'NOUN_NUM+NSUFF_MASC_DU_NOM': return 'NNS' 
    if arabic_pos == 'PUNC': return 'PUNC' 
    if arabic_pos == 'DET+NOUN_QUANT+CASE_DEF_NOM': return 'NN' 
    if arabic_pos == 'NOUN_QUANT+NSUFF_MASC_DU_NOM_POSS' : return 'NNS' 
    if arabic_pos == 'NOUN_NUM': return 'CD' 
    if arabic_pos == 'NOUN.VN+NSUFF_MASC_PL_NOM' : return 'NNS' 
    if arabic_pos == 'IV2FS+IV+IVSUFF_SUBJ:2FS_MOOD:SJ' : return 'VBP' 
    if arabic_pos == 'DET+NOUN_NUM+NSUFF_FEM_SG+CASE_DEF_NOM' : return 'CD' 
    if arabic_pos == 'ADJ+NSUFF_FEM_PL+CASE_DEF_NOM' : return 'JJ' 
    if arabic_pos == 'IV3FS+IV_PASS+IVSUFF_MOOD:S': return 'VBN' 
    if arabic_pos == 'DET+NOUN_PROP+CASE_DEF_GEN': return 'NNP' 
    if arabic_pos == 'DEM_PRON_FS': return 'DT' 
    if arabic_pos == 'ADJ+NSUFF_MASC_PL_ACC': return 'JJ' 
    if arabic_pos == 'DET+NOUN.VN+NSUFF_MASC_PL_GEN': return 'NNS' 
    if arabic_pos == 'REL_ADV': return 'WRB' 
    if arabic_pos == 'DET+NOUN+NSUFF_FEM_SG': return 'NOFUNC' 
    if arabic_pos == 'REL_PRON': return 'WP' 
    if arabic_pos == 'PRON_1P': return 'PRP' 
    if arabic_pos == 'IV3MP+IV_PASS+IVSUFF_SUBJ:MP_MOOD:I': return 'VBN' 
    if arabic_pos == 'DET+ADJ.VN+NSUFF_MASC_PL_ACC': return 'JJ' 
    if arabic_pos == 'NOUN_NUM+NSUFF_FEM_SG+CASE_INDEF_ACC': return 'CD' 
    if arabic_pos == 'PVSUFF_DO:3D': return 'PRP' 
    if arabic_pos == 'NOUN.VN+NSUFF_FEM_SG+CASE_DEF_NOM': return 'NN' 



    if arabic_pos == 'DET+ADJ+NSUFF_MASC_PL_ACCGEN_POSS': return 'JJ' 
    if arabic_pos == 'IV3FS+IV+IVSUFF_SUBJ:3FS': return 'VBP' 
    if re.match('CVSUFF_DO', arabic_pos) : return 'PRP' 
    if re.match('ADJ', arabic_pos): return 'JJ' 
    if re.match('NOUN_NUM', arabic_pos): return 'CD' 
    if re.search('IV_PASS', arabic_pos): return 'VBN' 
    if arabic_pos == 'DET+DIALECT': return 'NN' 




    if re.search('NOUN_PROP', arabic_pos) and re.search('_PL', arabic_pos):
      return "NNPS"

    if re.search('NOUN_PROP', arabic_pos) and re.search('_DU', arabic_pos):
      return "NNPS"

    if re.search('NOUN_PROP', arabic_pos) and re.search('_SG', arabic_pos):
      return "NNP"



    if re.search('NOUN', arabic_pos) and re.search('_PL', arabic_pos):
      return "NNS"

    if re.search('NOUN', arabic_pos) and re.search('_DU', arabic_pos):
      return "NNS"

    if re.search('NOUN', arabic_pos) and re.search('_SG', arabic_pos):
      return "NN"



    sys.stderr.write("unknown arabic_pos tag\n")

    return 'NOFUNC'


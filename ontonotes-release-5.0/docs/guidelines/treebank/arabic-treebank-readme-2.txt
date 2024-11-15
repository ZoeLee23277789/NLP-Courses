
Contents:
1. Two Levels of Annotation
2. File Extensions and Directory Structure
3. Changes From Previous Release Structure
4. The UNVOCALIZED form in the after-treebank data
5. A note about multiple trees on one line

============================================================================
1. Two Levels of Annotation
============================================================================

The next section describes in detail the different file extensions and 
directories used to store the annotated data.  The key fact underlying 
this organization is that there are two separate stages of annotation, 
as described under "Annotation Process" in readme.html, and these are 
stored in two separate Annotation Graph files, from which other files 
are generated.

1) POS annotation - This is the selection of a POS tag for a token 
from the source text file.  In previous releases, the Annotation Graph .xml
file containing this information was stored in the data/xml/pos directory.
However, for reasons discussed in Section 3, this file is not being included
in this release.  However, we are still including the text files in 
data/pos/before-treebank which contain the key information from those .xml
files.

2) Treebank annotation - The tokens from the POS Annotation are modified, 
by splitting off clitics, to create the tokens used for Treebank annotation,
and these tokens, and the new tree structure, are stored in different 
Annotation Graph .xml files. These files are stored, as in previous releases,
in data/xml/treebank.  Various other files are generated from this data:
the text files in data/pos/after-treebank and the three different tree files
in data/penntree and its subdirectories.

============================================================================
2. File Extensions and Directory Structure
============================================================================

Each FILE in docs/file.ids has a corresponding file in the 
following directories.   

data/sgm/FILE.sgm  (utf-8)
    Processed source files in sgml format. Please note that
    there is a parallel text and English Treebank corpus that
    has been developed at LDC for these same 599 source files
    and that has been released to the GALE community and will 
    be published soon.

data/pos/before-treebank/FILE.txt (utf-8)
   Information about the tokens used for the original analysis with the
   Buckwalter analyzer.  So this is a listing of each token before
   clitic-separation.
   Each token contains the following information:
-----------------------------------------------------------
INPUT STRING: (utf-8 characters from .sgm file)
    IS_TRANS: (Buckwalter transliteration of previous, used for input to
               analyzer)
     COMMENT: (annotator comment about word)
       INDEX: (automatically assigned index, based on paragraph&word#)
     OFFSETS: (start,end) - pair of integers - Annotation Graph offset into
                            .sgm file, corresponding to INPUT STRING
-----------------------------------------------------------
   Future releases will also include a listing of alternatives for 
   IS_TRANS from the current version of the Buckwalter analyzer, with
   one marked as the correct solution.  For discussion of why this is
   not included in this release, see Section 3 below.

   Both INPUT STRING and IS_TRANS are trimmed so that any leading and
   trailing whitespace pointed to by the OFFSETS are deleted.

data/xml/treebank/FILE.xml
   As discussed in Section 1,  this consists of the result of splitting the
   tokens    used for POS Annotation (and therefore included in
   /data/pos/before-treebank/FILE.txt) for the purposes of treebank
   annotation, and then modified with treebanking information and 
   further POS changes.  

data/pos/after-treebank/FILE.txt
   Information about each token in the corresponding xml/after-treebank .xml
   file.  So this is a listing of each token after clitic-separation.
   Each token contains the following information:
-----------------------------------------------------------
INPUT_STRING: (utf-8 characters from .sgm file)
    IS_TRANS: (Buckwalter transliteration of previous)
     COMMENT: (annotator comment about word)
       INDEX: (automatically assigned index, based on paragraph&word#)
     OFFSETS: (start,end) - pair of integers - Annotation Graph offset into 
                            sgm file
 UNVOCALIZED: (the unvocalized form of the word)
   VOCALIZED: (the vocalized form of the word, taken from the solution)
  VOC_STRING: (the Arabic utf-8 of the vocalized form)
         POS: (the pos tag, taken from the solution)
       GLOSS: (the gloss, taken from the solution)
       LEMMA: (the lemma, taken from the solution)
-----------------------------------------------------------
   For further discussion of these items, see Section 3 below.

data/penntree/without-vowel/FILE.tree
   Penn Treebanking style output, generated from the xml/after-treebank .xml
   file.  Each terminal is of the form (pos word), 
   where pos and word correspond to the POS and UNVOCALIZED values for the 
   corresponding token in pos/after-treebank/FILE.txt, respectively.

data/penntree/with-vowel/FILE.tree
   Penn Treebanking style output, generated from the xml/after-treebank .xml
   file. Each terminal is of the form (pos word), 
   where pos and word correspond to the POS and VOCALIZED values for the 
   corresponding token in pos/after-treebank/FILE.txt, respectively.
   
data/penntree/combined-utf8/FILE.tree
   Also generated from the xml/after-treebank .xml file.  
   We are including this new combined form to make it easier to relate the
   full information about each word to the tree structure, without having to
   work with the data/xml/treebank/FILE.xml or the information in the
   data/pos/after-treebank/FILE.txt and data/penntree/with(out)-vowel/FILE.tree
   files.  
   These trees are not meant to be easy for people to read, but rather to 
   collect in one place all the relevant
   information for further processing as people choose.  

   This file format is a mix of penntree-like representation and a variant
   of the text information in the pos/after-treebank/FILE.txt files.
   The tree contains stand-ins for each of the lexical items.  e.g.:

   (FRAG (NP (NOUN_NUM W1) (NP (NOUN+CASE_INDEF_ACC W2) ....

   and then following the tree the items W1,W2, etc. are listed.
   Each such W item has the following formation on one line,
   with the character U+00B7 used as a delimiter:

   IS_TRANS
   COMMENT
   INDEX
   OFFSET start
   OFFSET end
   UNVOCALIZED
   VOCALIZED
   GLOSS
   LEMMA
   BAMAVOC
   LOOKUP_STATUS

   All except the last two items are exactly as in the pos/after-treebank/FILE.txt
   file.  BAMAVOC and LOOKUPSTATUS are two additional pieces of information,
   found explicitly or implicitly in the xml/treebank/FILE.xml file, and are
   provided for obsessive completeness only.

   BAMAVOC is the relevant substring of the vocalized form produce by
   the BAMA morphological analyzer, vocalized without any segmentation.
   It is included as part of the solution for a word in the .xml file,
   along with the lemma and the usual vocalized form.  (How it is
   different from the vocalized form is beyond the scope of this
   readme.)

   The LOOKUP_STATUS gives some additional information as to whether the token 
   was actually sent through BAMA originally. The most common LOOKUP_STATUS 
   is 3 and indicates that the token was part of a token from the source 
   file that was passed into BAMA. 
   The LOOKUP_STATUS 1 indicates that the word was not sent through BAMA,
   and is usually limited to punctuation or numbers.  The LOOKUP_STATUS 2 
   is used to indicate a word with an empty UNVOCALIZED form, which occurs
   for the few cases in which vocalized form is a suffix consisting entirely
   of diacritics, in which case we use the dummy term "nullp" for the 
   UNVOCALIZED form.


============================================================================
3. Changes From Previous Release Structure
============================================================================

============================================================================
3a. data/xml/pos/FILE.xml not included:
============================================================================

This release contains a substantially modified treebank both in terms of the
tags, the tokenization, and the trees.
These changes were incorporated only into the data/xml/treebank .xml files, 
since those .xml files, not the ones in data/xml/pos, are used for ongoing 
annotation and modification.

The data/xml/pos .xml file contains the alternatives from the Buckwalter 
analyzer at the time the analysis was originally done, which does not include 
the new POS changes.  As a result, the POS information in the .xml files
previously released in the data/xml/pos directory is obsolete and should 
be ignored, and so is not included here. We have instead 
extracted all the relevant information as to the original tokens from the 
.sgm file sent into the BAMA analyzer and included them in the 
data/pos/before-treebank .txt files, as discussed above in Sections 1 and 2.

============================================================================
3b. For the pos/before-treebank/FILE.txt file, the changes are as follows:
============================================================================

1. The LOOK-UP word is now called the IS_TRANS, for "input string
transliteration".  For non-punctuation/number items, this is the same
as what was previously labeled the LOOK-UP word.  Punctuation and 
numbers did not previously have a LOOK-UP word, since they were not sent
through the analyzer.
2. The OFFSETS into the .sgm file are now included.  This is done for two
reasons.  First, on general principle, so that this information can be
obtained without having to go through the .xml files.  Second, to more easily
relate the token information in the pos/before-treebank and the 
pos/after-treebank files, again without having to go through the 
corresponding .xml files as intermediaries.
3. Earlier releases included the various POS alternatives at the time of POS
annotation.  For the same reason that the data/xml/pos .xml file is not
included, as discussed in Section 3a above, these POS alternatives are no
longer included.

============================================================================
3c. For the pos/after-treebank/FILE.txt file, the changes are as follows:
============================================================================

1. What used to be called the LOOKUP-WORD is now called UNVOCALIZED.  The
concept of LOOKUP-WORD for the after-treebank files is actually meaningless,
since the lookup in the Buckwalter analyzer is done only on the 
before-treebank words.   This field is now called UNVOCALIZED to make it
clear that it is the value used in the without-vowel files.  See Section 4
below for some more discussion of this.
2. IS_TRANS is the Buckwalter transliteration of the INPUT_STRING.  This 
information was not included before (although it was derivable from the
INPUT_STRING).  As discussed in the paper cited in Section 4, 
it is *not* necessarily the same as the 
former LOOKUP-WORD, now UNVOCALIZED.
3. OFFSET - offsets into the .sgm file.  Previously, this information was
contained only in the corresponding after-treebank/FILE.xml file.  As
indicated above for the pos/before-treebank/FILE.txt file, it is our hope
that this will make it easier to relate the unsplit tokens in the 
before-treebank and after-treebank pos files.  The OFFSETs in this file
correspond to the INPUT_STRING.
4. VOCALIZED,GLOSS,LEMMA are now included as separate fields.  Previously
they were contained only in the marked (with an asterisk) solution from the
POS alternative.  The POS alternatives are not included, both for the
same reason as they are not included in the pos/before-treebank/FILE.txt
file, and also because in any case the POS alternatives are only relevant
for the tokens as sent through BAMA, not the tokens as used for treebank
annotation.
5. VOC_STRING is the Arabic utf-8 string corresponding to the VOCALIZED
solution.  This information was not previously included (although derivable
from the vocalized information in the solution).  

============================================================================
3d. penntree/combined/FILE.tree:
============================================================================

This tree was not included before.  As discussed above in Section 2, under
data/penntree/combined/FILE.tree, it includes all of the information from
the corresponding after-treebank/FILE.txt file.  Our hope is 
that this will make it easier for users to
utilize the various aspects of the annotation, without needing to spend
time aligning separate files.

============================================================================
4. The UNVOCALIZED form in the after-treebank data
============================================================================

For tokens which are not split, the offset and other information does not
change between the before-treebank and after-treebank files.  (Aside from
the INDEX, which is created on the fly as the POS files are created, and
so can change depending on the number of previous tokens.) 

The situation is more complicated for split tokens, in which the 
UNVOCALIZED form of the word was created by deleting diacritics from 
the relevant segment of the vocalized solution.  The following paper
discusses this issue in detail:

Mohamed Maamouri, Seth Kulick, Ann Bies 
Diacritic Annotation in the Arabic Treebank and Its Impact on Parser
Evaluation; LREC 2008, Marrakech, Morocco, May 28-30, 2008
http://papers.ldc.upenn.edu/LREC2008/Diacritic_Annotation_ATB.pdf

Given the differences between the formerly-called LOOK-UP WORD and the
original INPUT STRING, it is not accurate to refer to this as a
LOOK-UP WORD, which is why we now refer to this as UNVOCALIZED.

In addition to the case of the UNVOCALIZED form in split tokens as 
discussed in the paper, it is currently the case that 
some other words can be "out of sync" between the UNVOCALIZED 
and VOCALIZED form, since it is only the VOCALIZED 
forms that have been modified as part of the revision.  For example,  it is
possible that a token An was earlier analyzed as <in~ but now changed to
>an~, in which case the UNVOCALIZED form is still <n while the vocalized
form is >an~a.


============================================================================
5. A note about multiple trees on one line
============================================================================

It is possible for one line in the .tree files to include more than one
complete tree. The reason for this is that the annotators work 
on one "Paragraph" at a time - e.g., a tree with the root "Paragraph"
node, as can be seen by looking at the "treebanking" feature in the xml
files.  When the trees are generated, the "Paragraph" node is dropped
If the form of the annotation was (Paragraph S1 S2), where S1 and S2
are both complete trees, then they will appear on one line.










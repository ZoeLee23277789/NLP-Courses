"""

This is a collection of functions for dealing with callisto xml
files.  Most of these functions convert between xml and apf or sgml,
but if you want to work with chains in memory, you probably want to
look at :func:`callisto_to_chain_lists` .

"""

from __future__ import with_statement
import os
import re
import codecs
from collections import defaultdict
import base64
import xml.etree.ElementTree as ElementTree

class CallistoConverterException(Exception):
    def __init__(self, a_string):
        self.parameter = a_string
    def __str__(self):
        return str(self.parameter)

class ProgrammingException(CallistoConverterException):
    def __init__(self, issue):
        self.parameter = "This should never happen.  Please report this to the author: %s" % issue

class InvalidApfException(CallistoConverterException):
    def __init__(self, fname, issue):
        self.parameter = "The file %s is not a valid .apf file: %s" % (fname, issue)

class InvalidAifXmlException(CallistoConverterException):
    def __init__(self, fname, issue):
        self.parameter = "The file %s is not valid .aif.xml file: %s" % (fname, issue)

def apf_to_callisto(fname_apf, fname_source, out_xml=None, munge_primary_mentions=False):
    """

    given the file names of an apf file and the appropriate source
    file, produce a callisto xml file.  If out_xml is not given,
    default to fname_apf + .aif.xml

    Overall Structure:
     - read the apf file into several hashes that correspond to the sections of the aif.xml
     - write the hashes out with proper syntax

    fname_apf, fname_source, and out_xml may be file names or open
    files.  If fname_apf is an open file, out_xml must be set.  If
    files are passed in open, they are left open.

    If munge_primary_mentions is true, then the primary mention will
    always be the first mention.

    """

    closeme = []

    if type(fname_apf) != type("") and out_xml is None:
        raise ProgrammingException("if fname_apf is an open file, out_xml must be set")

    try:
        et = ElementTree.parse(fname_apf)
    except Exception, e:
        raise InvalidApfException(fname_apf, "not valid xml: %s" % (e.args, ))

    if type(fname_source) == type(""):
        fname_source = codecs.open(fname_source, "r", "utf8")
        closeme.append(fname_source)

    source_text = fname_source.read().decode("utf8")

    e = et.getroot()

    if e.tag != "source_file":
        raise InvalidApfException(fname_apf, "does not start with 'source file' tag")

    URI = e.attrib["URI"]

    d = e[0]
    assert d.tag == "document"

    document_id = d.attrib["DOCID"]


    type_next_id = defaultdict(int) # id_name -> next id to use -- used by id_maker
    def id_maker(id_name, store_in):
        """
        given an id prefix (like Reg for Region) and a hash to
        store examples in, return a function that will use
        type_next_id to assign the next id in that sequence and store
        the passed in arguments as the key for that id in store_in

        The numeric parts of the IDs are padded with zeros to make them 4 digits long.

        """

        def next_id(*value):
            a_id = id_name + str(type_next_id[id_name]).zfill(4)
            store_in[a_id] = value

            type_next_id[id_name] += 1

            return a_id
        return next_id

    anchors = {} # id -> int
    text_extent_regions = {} # id -> start_anchor, end_anchor
    head_full_regions = {} # id -> head_region, full_region
    ace_entity_regions = {} # id -> annotation_id_primary, annotation_id_secondary, ...
    ace_entity_annotations = {} # id -> primary_id, ace_entity_region, type
    head_full_annotations = {} # id -> mention_id, head_full_region, subtype

    make_anchor = id_maker("Anc", anchors)
    make_text_extent_region = id_maker("Reg", text_extent_regions)
    make_head_full_region = id_maker("Reg", head_full_regions)
    make_ace_entity_region = id_maker("Reg", ace_entity_regions)
    make_ace_entity_annotation = id_maker("Ann", ace_entity_annotations)
    make_head_full_annotation = id_maker("Ann", head_full_annotations)

    def make_text_extent_regions(start_anchor, end_anchor):
        return (make_text_extent_region(start_anchor, end_anchor),
                make_text_extent_region(start_anchor, end_anchor))

    for entity in d:
        """ read all of the entities in the apf and populate the hashes. """

        assert entity.tag == "entity"

        annotation_ids = []

        first_annotation_id, first_annotation_id_start = None, None

        for entity_mention in entity:
            charseq = entity_mention[0][0]
            assert charseq.tag == "charseq"

            c_start = int(charseq.attrib["START"])
            c_end = int(charseq.attrib["END"]) + 1
            text = charseq.text

            if text != source_text[c_start:c_end]:
                raise Exception(fname_apf, "source file doesn't match apf: s[%s:%s]=%r, t=%r" % (c_start, c_end, source_text[c_start:c_end], text))

            ema = entity_mention.attrib
            head_full_annotation_params = [["string", "ace_id", ema["ID"]],
                                           ["boolean", "ldcatr", ema["LDCATR"]],
                                           ["string", "ldctype", ema.get("LDCTYPE", "")],
                                           ["boolean", "metonymy", ema["METONYMY_MENTION"]],
                                           ["string", "reference", ""],
                                           ["string", "role", ""],
                                           ["string", "type", ema["TYPE"]]]

            annotation_id = make_head_full_annotation(make_head_full_region(*make_text_extent_regions(make_anchor(c_start), make_anchor(c_end))),
                                                      head_full_annotation_params)

            if first_annotation_id == None or c_start < first_annotation_id_start:
                first_annotation_id, first_annotation_id_start = annotation_id, c_start

            assert entity_mention.attrib["PRIMARY"] in ["true", "false"]

            if entity_mention.attrib["PRIMARY"] == "true":
                annotation_ids.insert(0, annotation_id)
            else:
                annotation_ids.append(annotation_id)

        if munge_primary_mentions and first_annotation_id is not None:
            annotation_ids.remove(first_annotation_id)
            annotation_ids.insert(0, first_annotation_id)

        ea = entity.attrib

        ace_entity_annotation_params = [["string", "ace_id", ea["ID"]],
                                        ["string", "class", ""],
                                        ["string", "type", ea["TYPE"]],
                                        ["string", "subtype", ea.get("SUBTYPE", "")]]

        make_ace_entity_annotation(make_ace_entity_region(*annotation_ids), ace_entity_annotation_params)

    if out_xml is None:
        out_xml = fname_apf + ".aif.xml"

    if type(out_xml) == type(""):
        out_xml_file = codecs.open(out_xml, "w", "utf8")
        closeme.append(out_xml_file)
    else:
        out_xml_file = out_xml

    w = out_xml_file.write

    w('<?xml version="1.0" encoding="US-ASCII"?>\n')
    w('<!DOCTYPE Corpus SYSTEM "http://www.nist.gov/speech/atlas/aif.dtd">\n\n')

    w('<Corpus xmlns="%s" xmlns:xlink="%s" xmlns:dc="%s" id="Cor6" AIFVersion="1.1" type="ace_corpus" schemeLocation="%s">\n' % (
        "http://www.nist.gov/speech/atlas",
        "http://www.w3.org/1999/xlink",
        "http://www.ukoln.ac.uk/interop-focus/activities/z3950/int_profile/bath/draft/stable1.html",
        "http://callisto.mitre.org/maia/ace2004.maia.xml"))

    w('  <Metadata/>\n')
    w('  <SimpleSignal id="Sig6" type="text" mimeClass="text" mimeType="sgml" xlink:href="%s" encoding="UTF-8" track="ALL" xlink:type="simple">\n' % (URI))
    w('    <body encoding="Base64">%s</body>\n' % base64.b64encode(source_text.encode("utf8")))
    w('  </SimpleSignal>\n')
    w('  <AnchorSet containedType="text-point">\n')

    for anchor_id, anchor_char in sorted(anchors.iteritems()):
        w('    <Anchor id="%s" type="text-point">\n' % anchor_id)
        w('      <Parameter type="char" unit="NULL_UNIT" role="char">%s</Parameter>\n' % anchor_char)
        w('      <SignalRef xlink:href="#Sig6" role="text" xlink:type="simple"/>\n')
        w('    </Anchor>\n')

    w('  </AnchorSet>\n')
    w('  <RegionSet containedType="ace_argument-mention_region"/>\n')
    w('  <RegionSet containedType="ace_argument_region"/>\n')

    w('  <RegionSet containedType="ace_entity_region">\n')
    for region_id, annotation_ids in sorted(ace_entity_regions.iteritems()):
        w('    <Region id="%s" type="ace_entity_region">\n' % region_id)
        w('      <AnnotationRef xlink:href="#%s" role="primary-mention" xlink:type="simple"/>\n' % annotation_ids[0])
        w('      <AnnotationRefSet containedType="ace_entity-mention">\n')

        for annotation_id in annotation_ids:
            w('        <AnnotationRef xlink:href="#%s" role="null" xlink:type="simple"/>\n' % annotation_id)
        w('      </AnnotationRefSet>\n')
        w('    </Region>\n')
    w('  </RegionSet>\n')


    w('  <RegionSet containedType="ace_event-mention_region"/>\n')
    w('  <RegionSet containedType="ace_event_region"/>\n')
    w('  <RegionSet containedType="ace_quantity-mention_region"/>\n')
    w('  <RegionSet containedType="ace_quantity_region"/>\n')
    w('  <RegionSet containedType="ace_relation-mention_region"/>\n')
    w('  <RegionSet containedType="ace_relation_region"/>\n')

    w('  <RegionSet containedType="head-full">\n')
    for region_id, (full_region_id, head_region_id) in sorted(head_full_regions.iteritems()):
        w('    <Region id="%s" type="head-full">\n' % region_id)
        w('      <RegionRef xlink:href="#%s" role="full" xlink:type="simple"/>\n' % full_region_id)
        w('      <RegionRef xlink:href="#%s" role="head" xlink:type="simple"/>\n' % head_region_id)
        w('    </Region>\n')
    w('  </RegionSet>\n')

    w('  <RegionSet containedType="text-extent">\n')
    for region_id, (start_anchor, end_anchor) in sorted(text_extent_regions.iteritems()):
        w('    <Region id="%s" type="text-extent">\n' % region_id)
        w('      <AnchorRef xlink:href="#%s" role="end" xlink:type="simple"/>\n' % end_anchor)
        w('      <AnchorRef xlink:href="#%s" role="start" xlink:type="simple"/>\n' % start_anchor)
        w('    </Region>\n')
    w('  </RegionSet>\n')

    w('  <Analysis id="Ana6" type="ace_annotation-set" role="ace_annotation-set">\n')
    w('    <AnnotationSet containedType="ace_argument"/>\n')
    w('    <AnnotationSet containedType="ace_argument-mention"/>\n')
    w('    <AnnotationSet containedType="ace_entity">\n')

    def write_params(params, ind):
        for type, role, value in params:
            w('%s<Parameter type="%s" unit="NULL_UNIT" role="%s"' % (ind, type, role))
            if value == "":
                w('/>\n')
            else:
                w('>%s</Parameter>\n' % value)

    for annotation_id, (ace_entity_mention_region_id, params) in sorted(ace_entity_annotations.iteritems()):
        w('      <Annotation id="%s" type="ace_entity">\n' % annotation_id)
        w('        <RegionRef xlink:href="#%s" role="ace_entity-mentions" xlink:type="simple"/>\n' % ace_entity_mention_region_id)
        w('        <Content type="ace_entity_content">\n')

        write_params(params, "          ")

        w('        </Content>\n')
        w('      </Annotation>\n')
    w('    </AnnotationSet>\n')

    w('    <AnnotationSet containedType="ace_entity-mention">\n')

    for annotation_id, (head_full_region_id, params) in sorted(head_full_annotations.iteritems()):
        w('      <Annotation id="%s" type="ace_entity-mention">\n' % annotation_id)
        w('        <RegionRef xlink:href="#%s" role="head-full" xlink:type="simple"/>\n' % head_full_region_id)
        w('        <Content type="ace_entity-mention_content">\n')

        write_params(params, "          ")

        w('        </Content>\n')
        w('      </Annotation>\n')

    w('    </AnnotationSet>\n')
    w('    <AnnotationSet containedType="ace_event"/>\n')
    w('    <AnnotationSet containedType="ace_event-mention"/>\n')
    w('    <AnnotationSet containedType="ace_event-mention-extent"/>\n')
    w('    <AnnotationSet containedType="ace_quantity"/>\n')
    w('    <AnnotationSet containedType="ace_quantity-mention"/>\n')
    w('    <AnnotationSet containedType="ace_relation"/>\n')
    w('    <AnnotationSet containedType="ace_relation-mention"/>\n')
    w('    <AnnotationSet containedType="ace_relation-mention-extent"/>\n')
    w('    <AnnotationSet containedType="timex2"/>\n')
    w('  </Analysis>\n')
    w('</Corpus>\n')

    out_xml_file.flush()

    for f in closeme:
        f.close()

class FileContainsBothNameAndCorefAnnotationException(CallistoConverterException):
    def __init__(self, fname):
        self.parameter = "the file %s has both name and coreference annotation" % fname

class NoAnnotationFoundException(CallistoConverterException):
    def __init__(self, fname):
        self.parameter = "no annotation found in %s" % fname

def callisto_to_chain_lists(fname, include_metadata=True):
    """
    Given a filename for a callisto xml file, return the a list of its chains in the format:

      [ [ (chain identifier, chain type), (start, end, subtype, string), (start, end, subtype, string), ... ],
        [ (chain identifier, chain type), (start, end, subtype, string), (start, end, subtype, string), ... ],
        ...], document_source_string

    If include_metadata is false, don't include the (chain identifier,
    chain type) tuple at the beginning of each chain list

    """

    (document_id, primary_to_all, ace_entities, ace_entity_mentions, text_extents, anchors, fulls, source_text_raw) = parse_callisto_xml(fname, stop_at_mentions=True)

    if not primary_to_all:
        raise NoAnnotationFoundException("no annotation found in %s" % fname)

    e_lists = []

    for primary_annotation, all_annotations in sorted(primary_to_all.iteritems()):

        primary_id = ace_entity_mentions[primary_annotation]["id"]

        # clean up the primary id
        if "-" in primary_id and primary_id.split("-")[-1].isdigit():
           primary_id= "-".join(primary_id.split("-")[:-1])

        if primary_id not in ace_entities:
            for x in ["ann1-", "ann2-"]:
                if x + primary_id in ace_entities:
                   primary_id= x + primary_id

        # use primary id to find the type, knowing that automatically
        # generated m_foo entries represent IDENT
        if primary_id.startswith("m_"):
            primary_type = "IDENT"
        else:
            primary_type = ace_entities[primary_id]

        e_list = [(primary_id, primary_type)] if include_metadata else []

        for a_annotation in all_annotations:
            subtype = ace_entity_mentions[a_annotation]["subtype"]

            region = ace_entity_mentions[a_annotation]["region"]
            region = fulls.get(region, region)
            start_anchor, end_anchor = text_extents[region]
            start_char, end_char = anchors[start_anchor], anchors[end_anchor]

            string = source_text_raw[start_char:end_char]

            e_list.append((start_char, end_char, subtype, string))

        e_lists.append(e_list)

    return e_lists, source_text_raw

def callisto_to_apf(fname, out_apf=None, out_source=None):
    """
    given the fname of a callisto xml file, produce an apf and source pair

    If the fnames for the apf and source are not given, then fname.apf
    and fname.source are used.

    fname, out_apf, and out_source may be open files instead of
    filenames, but if fname is an open file then out_apf and
    out_source may not be None.

    """

    closeme = []

    if type(fname) != type("") and (out_apf is None or out_source is None):
        raise ProgrammingException("If fname is an open file, then out_apf and out_source may not be None")

    document_id, primary_to_all, ace_entities, ace_entity_mentions, text_extents, anchors, fulls, source_text_raw = parse_callisto_xml(fname, stop_at_mentions=True)

    if not primary_to_all:
        raise NoAnnotationFoundException("no annotation found in %s" % fname)

    if not out_apf:
        out_apf = fname + ".apf"
    if not out_source:
        out_source = fname + ".source"

    if type(out_apf) == type(""):
        out_apf = open(out_apf, "w")
        closeme.append(out_apf)

    w = out_apf.write

    w('<?xml version="1.0" encoding="UTF-8"?>\n')
    w('<!DOCTYPE source_file PUBLIC "SYSTEM" "apf.v5.1.5.dtd">\n')
    w('\n')
    w('<source_file URI="file://%s" SOURCE="unknown" TYPE="text" VERSION="5.0" AUTHOR="unknown" ENCODING="UTF-8">\n' % fname)
    w('  <document DOCID="%s">\n' % document_id)
    for primary_annotation, all_annotations in sorted(primary_to_all.iteritems()):
        primary_id = ace_entity_mentions[primary_annotation]["id"]

        if "-" in primary_id and primary_id.split("-")[-1].isdigit():
           primary_id= "-".join(primary_id.split("-")[:-1])

        if primary_id not in ace_entities:
            for x in ["ann1-", "ann2-"]:
                if x + primary_id in ace_entities:
                   primary_id= x + primary_id

        if primary_id.startswith("m_"):
            primary_type = "IDENT"
        else:
            primary_type = ace_entities[primary_id]

        w('    <entity ID="%s" TYPE="%s">\n' % (primary_id, primary_type))


        for a_annotation in all_annotations:
            is_primary = str(a_annotation == primary_annotation).lower()
            w('      <entity_mention ID="%s" TYPE="%s" PRIMARY="%s" METONYMY_MENTION="FALSE" LDCATR="FALSE">\n' % (
                ace_entity_mentions[a_annotation]["id"], ace_entity_mentions[a_annotation]["subtype"], is_primary))
            w('        <extent>\n')

            region = ace_entity_mentions[a_annotation]["region"]
            region = fulls.get(region, region)
            if region not in text_extents:
                print "  ".join(sorted(text_extents.keys()))
                print region
            start_anchor, end_anchor = text_extents[region]
            start_char, end_char = anchors[start_anchor], anchors[end_anchor]
            charseq = '          <charseq START="%s" END="%s">%s</charseq>\n' % (
                start_char, end_char-1, source_text_raw[start_char:end_char].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
            try:
                c=charseq.encode("utf8")
                w(c)
                w('        </extent>\n')
                w('        <head>\n')
                w(c)
                w('        </head>\n')
                w('      </entity_mention>\n')
            except UnicodeEncodeError:
                print "%r" % charseq
                raise

        w('    </entity>\n')
    w('  </document>\n')
    w('</source_file>\n')


    if type(out_source) == type(""):
        out_source = open(out_source, "w")
        closeme.append(out_source)

    out_source.write(source_text_raw.encode("utf8"))

    out_apf.flush()
    out_source.flush()

    for f in closeme:
        f.close()

class NotImplementedException(CallistoConverterException):
    def __init__(self, fname, issue):
        self.parameter = "The file %s has %s, something we don't know how to deal with." % (fname, issue)

class BadDataException(CallistoConverterException):
    def __init__(self, fname, issue):
        self.parameter = "The file %s has the problem that %s" % (fname, issue)

def parse_callisto_xml(fname, stop_at_mentions=False):
    """

    fname should be a file in callisto .aif.xml format

    Returns a tuple:
      document_id, annotation_opens, annotation_closes, source_text_raw

    document_id: whatever the .xml says the document id is

    annotation_opens: list of (start_char, type, type_specific, ...)
      start_char: offset into source_text_raw
      type: name or coref
      type_specific:
       -> type == name: annotation type
       -> type == coref: id, type, subtype or None

    annotation_closes: list of (start_char, type, type_specific)
      type_specific:
        -> type == coref: none
        -> type == name: annotation type

    source_text_raw:      string, the raw source text


    If stop_at_mentions, stop computing things early and return:
      document_id, primary_to_all, ace_entities, ace_entity_mentions, text_extents, anchors, fulls, source_text_raw


    """

    annotation_opens = []
    annotation_closes = []

    PREPEND='http://www.nist.gov/speech/atlas'
    PREPEND_B='http://www.w3.org/1999/xlink'

    def tagis(e, s):
        return e.tag == '{%s}%s' % (PREPEND, s)

    def find(e, s):
        return e.find('{%s}%s' % (PREPEND, s))

    def findall(e, s):
        return e.findall('{%s}%s' % (PREPEND, s))


    def get_source_text(e):
        assert tagis(e, "SimpleSignal")

        body = find(e, "body")

        assert body.attrib["encoding"] == "Base64"

        return base64.decodestring(body.text)

    def get_anchors(e):
        assert tagis(e, "AnchorSet")

        mapping = {}

        for anchor in findall(e, "Anchor"):
            mapping[anchor.attrib["id"]] = int(find(anchor, "Parameter").text)

        return mapping

    def pull_reference(e):
        try:
            ref = e.attrib["{%s}href" % PREPEND_B]
        except Exception:
            print e, e.attrib
            raise

        if ref.startswith("#"):
            ref = ref[1:]

        return ref

    def get_text_extents(region_set):
        text_extents = {}
        for region in region_set:
            start = None
            end = None
            for anchor_ref in region:
                if anchor_ref.attrib["role"] == "start":
                    start = pull_reference(anchor_ref)
                elif anchor_ref.attrib["role"] == "end":
                    end = pull_reference(anchor_ref)

            assert start is not None and end is not None
            text_extents[region.attrib["id"]] = (start, end)

        return text_extents

    if not os.path.exists(fname):
        raise InvalidAifXmlException(fname, "file does not exist")

    try:
        et = ElementTree.parse(fname)
    except Exception:
        raise InvalidAifXmlException(fname, "invalid xml; failed to parse")

    e = et.getroot()

    if e.tag == "source_file":
        raise InvalidAifXmlException(fname, "it is apf, not callisto xml")

    assert tagis(e, "Corpus"), e.tag

    simple_signal = find(e, "SimpleSignal")

    document_id = pull_reference(simple_signal).replace(".source", "").replace(".sgml", "").replace(".xml", "").replace(".aif", "").split("/")[-1]

    try:
        source_text_raw = get_source_text(simple_signal).decode("utf8")
    except UnicodeDecodeError:
        print "-"*70
        print "Raw source text:"
        print "----------------"
        print "%r"% list(enumerate(get_source_text(simple_signal)))
        print "-"*70
        raise


    # I really don't understand why you have to remove all the sgml
    # tags from the source before callisto's anchors align with the
    # text.  This really doesn't make sense.  But it seems to be
    # needed.  This also worries me about lines that might contain
    # greater than or less than legitimately
    source_text_raw = re.sub(r"</?[A-Z]+[^>\n]*>", "", source_text_raw)

    anchors = get_anchors(find(e, "AnchorSet")) # Anc -> text index

    primary_to_all = {} # AnnA -> [AnnA, AnnB, ... ]
    text_extents = {} # RegNNN -> (AncSTART, AncEND)
    ace_entities =  {} # id -> {type}
    ace_entity_mentions = {} # AnnA -> {region, id, subtype}
    fulls = {} # RegNNN -> RegNNM

    for region_set in findall(e, "RegionSet"):
        if region_set:
            if region_set.attrib["containedType"] == "ace_entity_region":
                for region in region_set:
                    if len(region) == 1 and not region[0]:
                        continue

                    primary_to_all[pull_reference(region[0])] = [pull_reference(ar) for ar in region[1]]


            elif region_set.attrib["containedType"] == "text-extent":
                text_extents = get_text_extents(region_set)
            elif region_set.attrib["containedType"] == "head-full":
                for region in region_set:
                    fulls[region.attrib["id"]] = pull_reference(region[0])
            else:
                raise NotImplementedException(fname, "value '%s' for 'RegionSet/containedType'" % region_set.attrib["containedType"])


    name_annotations = {} # region -> type
    for analysis in findall(e, "Analysis"):
        if analysis.attrib["type"] == "generic-set":
            for annotation_set in analysis:
                annotation_type = annotation_set.attrib["containedType"]
                for annotation in annotation_set:
                    name_annotations[pull_reference(annotation[0])] = annotation_type

        elif analysis.attrib["type"] == "ace_annotation-set":
            for annotation_set in analysis:
                if annotation_set:
                    if annotation_set.attrib["containedType"] == "ace_entity":
                        for annotation in annotation_set:
                            content = annotation[1]
                            ace_id = None
                            ace_type = None

                            for parameter in content:
                                if parameter.attrib["role"] == "ace_id":
                                    ace_id = parameter.text
                                elif parameter.attrib["role"] == "type":
                                    ace_type = parameter.text

                            if ace_type is None:
                                ace_type = "NOT_SPECIFIED"

                            assert ace_id is not None and ace_type is not None, (ace_id, ace_type, len(annotation))

                            ace_entities[ace_id] = ace_type

                    elif annotation_set.attrib["containedType"] == "ace_entity-mention":
                        for annotation in annotation_set:
                            a_subtype = None
                            a_id = None

                            for a_param in annotation[1]:
                                if a_param.attrib["role"] == "type":
                                    a_subtype = a_param.text
                                elif a_param.attrib["role"] == "ace_id":
                                    a_id = a_param.text

                            ace_entity_mentions[annotation.attrib["id"]] = {"region" : pull_reference(annotation[0]),
                                                                            "id" : a_id,
                                                                            "subtype" : a_subtype}
                    else:
                        raise NotImplementedException(fname, "value '%s' for Analysis/AnnotationSet/containedType" % region_set.attrib["containedType"])
        else:
            raise NotImplementedException(fname, "value '%s' for Analysis/type" % analysis.attrib["type"])


    if stop_at_mentions:
        return document_id, primary_to_all, ace_entities, ace_entity_mentions, text_extents, anchors, fulls, source_text_raw


    if name_annotations:
        names = []
        for region, type in name_annotations.iteritems():
            start, end = text_extents[region]

            names.append([anchors[start], anchors[end], type])

        for start, end, type in sorted(names):
            annotation_opens.append((start, "name", type))
            annotation_closes.append((end, "name", type))

    elif primary_to_all:
        corefs = [] # list of [ start, end, id, type, subtype ]
        for primary_annotation, all_annotations in primary_to_all.iteritems():

            if len(all_annotations) == 1:
                continue

            primary_id = ace_entity_mentions[primary_annotation]["id"]

            if "-" in primary_id and primary_id.split("-")[-1].isdigit():
                primary_id = "-".join(primary_id.split("-")[:-1])

            if primary_id not in ace_entities and "ann1-" + primary_id in ace_entities:
                primary_id = "ann1-" + primary_id

            try:
                type = ace_entities[primary_id]
            except Exception:
                if primary_id.startswith("m_") or "-m_" in primary_id:
                    type = "IDENT"
                else:
                    raise BadDataException(fname, "missing type entry for id %s" % primary_id)

            if type not in ["IDENT", "APPOS"]:
                raise BadDataException(fname, "annotation type '%s' for id '%s' is niether IDENT nor APPOS" % (ace_entities[primary_id], primary_id))

            for annotation in all_annotations:
                subtype = ace_entity_mentions[annotation]["subtype"]

                if type == "IDENT":
                    subtype = None

                sub_id = ace_entity_mentions[annotation]["id"]
                region = ace_entity_mentions[annotation]["region"]

                #assert primary_id in sub_id or (primary_id.startswith("m_") or sub_id.startswith("m_") or primary_id.startswith("ann1-")), (primary_id, sub_id)

                region = fulls.get(region, region)
                start, end = text_extents[region]

                corefs.append([anchors[start], anchors[end], primary_id, type, subtype])


        # check for illegal cross bracketing
        for start_a, end_a, id_a, type_a, subtype_a in corefs:
            for start_b, end_b, id_b, type_b, subtype_b in corefs:
                if start_a < start_b < end_a < end_b:
                    if end_a == end_b-1 and source_text_raw[end_a] == " ":
                        pass
                    elif start_a == start_b-1 and source_text_raw[start_a] == " ":
                        pass
                    else:
                        raise Exception("Cross Bracketing\n    ID_A=%s ID_B=%s\n    text_a=%s\n    text_b=%s\n"
                                        "(%s, %s, %s, %s), (%s, %s, %s, %s)" %
                                        (id_a, id_b,
                                         source_text_raw[start_a:end_a],
                                         source_text_raw[start_b:end_b],
                                         start_a, end_a, type_a, subtype_a,
                                         start_b, end_b, type_b, subtype_b,
                                         ))


        for start, end, id, type, subtype in sorted(corefs):
            annotation_opens.append((start, "coref", id, type, subtype))
            annotation_closes.append((end, "coref"))

    else:
        raise NoAnnotationFoundException(fname)

    return document_id, annotation_opens, annotation_closes, source_text_raw

class OnCommonUtilNeededError(CallistoConverterException):
    def __init__(self, function_needing_on_common_util):
        self.parameter = "The library on.common.util was needed for the function %s, but was not found" % function_needing_on_common_util

class DeSubtokenizationFailedException(CallistoConverterException):
    def __init__(self, fname, e):
        self.parameter = "The file %s failed because of a %s error with message %s" % (fname, type(e), e)

def callisto_to_sgml(fname, out_sgml=None, buckit=False, language="unknown", wrap=True):
    """
    given the fname of a callisto xml file, produce either fname.coref
    or fname.name depending on whether the file represents name or
    coref annotation.

     if buckit, then run everything through unicode2buckwalter before writing out

    if wrap, wrap the whole thing in <DOC ...> </DOC>

    """

    try:
        from on.common.util import unicode2buckwalter, desubtokenize_annotations
    except ImportError:
        raise OnCommonUtilNeededError("callisto_to_sgml")

    document_id, annotation_opens, annotation_closes, source_text_raw = parse_callisto_xml(fname)

    filename = None


    source_text = list(source_text_raw)

    names = corefs = 0


    for open_annotation in annotation_opens:
        idx = open_annotation[0]
        if open_annotation[1] == "name":
            open_annotation_str = "<%s>" % open_annotation[2]
            names += 1
        else:
            open_annotation_str = '<COREF-ID="%s"-TYPE="%s"%s>' % (
                open_annotation[2], open_annotation[3], '-SUBTYPE="%s"' % open_annotation[4] if open_annotation[4] else "")
            corefs += 1

        source_text[idx] = "%s%s" % (open_annotation_str, source_text[idx])

    for close_annotation in annotation_closes:
        idx = close_annotation[0]
        if close_annotation[1] == "name":
            close_annotation_str = "</%s>" % close_annotation[2]
        else:
            close_annotation_str = "</COREF>"

        source_text[idx] = "%s%s" % (close_annotation_str, source_text[idx])

    if corefs and not names:
        ext = "coref"
    elif names and not corefs:
        ext = "name"
    elif names and corefs:
        raise FileContainsBothNameAndCorefAnnotationException(fname)
    else:
        raise NoAnnotationFoundException(fname)

    filename = out_sgml if out_sgml else fname + "." + ext

    with codecs.open(filename, "w", "utf8") as out_f:
        if wrap:
            out_f.write('<DOC DOCNO="%s">\n' % document_id)

        n_text = "".join(source_text)


        try:
            n_text, num_fixed = desubtokenize_annotations(n_text, add_offset_notations=True)
        except Exception, e:
            raise DeSubtokenizationFailedException(fname, e)

        if buckit:
            n_text = unicode2buckwalter(n_text, sgml_safe=True)

        n_text = n_text.replace("\r"," ")

        n_text = n_text.strip()
        out_f.write(n_text)
        out_f.write("\n")

        if wrap:
            out_f.write('</DOC>\n')

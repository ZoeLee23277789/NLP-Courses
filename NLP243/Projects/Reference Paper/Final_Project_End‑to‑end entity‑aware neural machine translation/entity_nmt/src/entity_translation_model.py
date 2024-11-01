import torch

import fairseq
from torch import nn
from torch.nn import functional as F

from fairseq.models import FairseqEncoder, FairseqDecoder
from fairseq.models.fairseq_model import BaseFairseqModel
from fairseq.models.transformer import TransformerModel, EncoderOut, DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
import copy
import random

import numpy as np

from collections import namedtuple

from .utils import *

ModelOut = namedtuple('ModelOut', [
    'decoder_out',  # the (decoder out, extra), same as original,
    'encoder_ne_logit',
    'decoder_ne_logit',
    'entity_out',
    'entity_label',
    'result_entity_id',
    'encoder_ne'
])

NE_PENALTY=1e8

all_types = ['ORG', 'EVENT', 'PRODUCT', 'FAC', 'PERCENT', 'WORK_OF_ART', 'ORDINAL', 'LOC',
                 'LANGUAGE', 'LAW', 'PERSON', 'TIME', 'CARDINAL', 'GPE', 'QUANTITY', 'DATE', 'NORP', 'MONEY']
class EntityEncoderDecoderModel(BaseFairseqModel):

    def __init__(self, args, ne_dict, encoder, decoder, tgt_ne_start_id, bert_emb_id_dict, bert_emb_value, entity_mapping):
        super().__init__()

        self.args = args
        self.ne_dict = ne_dict
        self.encoder = encoder
        self.decoder = decoder
        self.mode = args.mode
        self.tgt_ne_start_id = tgt_ne_start_id
        self.bert_emb_id_dict = bert_emb_id_dict  # tuple ot token -> bert id
        self.bert_emb_value = torch.nn.Parameter(bert_emb_value, requires_grad=False)
        self.entity_mapping = entity_mapping  # tupe of src id -> set (list of tgt id)

        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)
        assert 0 <= self.mode <= 5
        assert 0 <= self.args.bert_lookup_layer <= self.args.decoder_layers # 0 is the input
        assert 1 <= self.args.src_ne_layer <= self.args.encoder_layers

        self.encoder_ne_process_mask = {}
        if self.mode == 1 or self.mode == 2 or self.mode == 4 or self.mode == 5:
            self.src_ne_fc1 = nn.Linear(args.encoder_embed_dim, args.src_ne_project, bias=True)
            self.src_ne_fc2 = nn.Linear(args.src_ne_project, len(ne_dict), bias=True)

        if self.mode == 1 or self.mode == 4 or self.mode == 5:
            self.tgt_ne_fc1 = nn.Linear(args.decoder_embed_dim, args.tgt_ne_project, bias=True)
            self.tgt_ne_fc2 = nn.Linear(args.tgt_ne_project, len(ne_dict), bias=True)
        elif self.mode == 2:
            self.bert_sample_count = min(args.bert_sample_count, len(bert_emb_id_dict))

            self.bert_dim = bert_emb_value.shape[1]
            self.tgt_bert_ne_fc1 = nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim, bias=True)
            self.tgt_bert_ne_fc2 = nn.Linear(args.decoder_embed_dim, self.bert_dim, bias=True)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--src-ne-project', type=int)
        parser.add_argument('--src-ne-project-dropout', type=float, default=0.0)
        parser.add_argument('--tgt-ne-project', type=int)
        parser.add_argument('--concat-ne-emb', action='store_true')
        parser.add_argument('--bert-lookup-layer', type=int) # 0 is the input, and n is the last layer
        parser.add_argument('--bert-lookup-dropout', type=float, default=0.0) #
        parser.add_argument('--src-ne-layer', type=int) # 1 is the first layer output, and n is the last layer

    def to_bert_emb_space(self, x):
        # Is this too simple? Do we need more layers? e.g. attention?
        # Do we need enable dropout here?
        x = F.relu(self.tgt_bert_ne_fc1(x))
        x = F.dropout(x, self.args.bert_lookup_dropout, self.training)
        return self.tgt_bert_ne_fc2(x)

    #@profile
    def encoder_ne_process(self, encoder_out, ne_type, need_logit):
        """
        ne_type:0 => O, B-XX, I-XX
        ne_type:1 => O, XX-0, XX-1
        """
        assert ne_type == 0 or ne_type == 1
        if encoder_out.encoder_states is None:
            entity_input = encoder_out.encoder_out
        else:
            assert len(encoder_out.encoder_states) == self.args.encoder_layers
            entity_input = encoder_out.encoder_states[self.args.src_ne_layer - 1] # T B C
        encoder_ne_emb = F.dropout(F.relu(self.src_ne_fc1(entity_input)), self.args.src_ne_project_dropout, self.training)

        if need_logit:
            encoder_ne_logit = self.src_ne_fc2(encoder_ne_emb)
            encoder_ne_logit = encoder_ne_logit.transpose(0, 1)  # T x B x C => B x T x C

            C = encoder_ne_logit.shape[2]
            max_ne_id = self.args.max_ne_id
            
            if (C,max_ne_id) not in self.encoder_ne_process_mask:
                logit_mask = [False]*C
                for i in range(C):
                    if i < self.ne_dict.nspecial + 1:  # spcial and 'O'
                        if i == self.ne_dict.bos_index or i == self.ne_dict.unk_index:
                            logit_mask[i] = False
                        else:
                            logit_mask[i] = True
                    else:
                        k = (i - (self.ne_dict.nspecial + 1)) % (max_ne_id + 2)
                        if ne_type == 0:
                            logit_mask[i] = k < 2
                        else:
                            logit_mask[i] = k >= 2
                self.encoder_ne_process_mask[(C,max_ne_id)] = logit_mask
            else:
                logit_mask = self.encoder_ne_process_mask[(C,max_ne_id)]
        
            logit_mask = torch.as_tensor(logit_mask, device=encoder_ne_logit.device)
            encoder_ne_logit[:, :, ~logit_mask] = float('-inf')

        else:
            encoder_ne_logit = None

        if self.args.concat_ne_emb:
            combined_encoder_out = torch.cat((encoder_out.encoder_out, encoder_ne_emb), dim=-1) # T x B x C
        else:
            combined_encoder_out = encoder_out.encoder_out + encoder_ne_emb
        encoder_out_with_emb = EncoderOut(
            encoder_out=combined_encoder_out,
            encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
            encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
            encoder_states=encoder_out.encoder_states,  # List[T x B x C]
        )

        return encoder_out_with_emb, encoder_ne_logit
    
    #@profile
    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
                - the encoder ne logit
                - the decoder ne logit
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=(self.mode != 0), **kwargs)
        if self.mode == 0 or self.mode==3:
            return ModelOut(
                decoder_out=self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs),
                encoder_ne_logit=None,
                decoder_ne_logit=None,
                entity_out=None,
                entity_label=None,
                result_entity_id=None,
                encoder_ne=None
            )
        elif self.mode == 1 or self.mode == 4 or self.mode == 5:
            encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=True)

            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, features_only=True, **kwargs)
            if self.decoder.share_input_output_embed:
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_tokens.weight)
            else:
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_out)

            assert (decoder_out_feature == extra['inner_states'][-1].transpose(0, 1)).all()

            ne_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1)

            decoder_ne_emb = F.relu(self.tgt_ne_fc1(ne_input_feature))
            decoder_ne_logit = self.tgt_ne_fc2(decoder_ne_emb)
            return ModelOut(
                decoder_out=(decoder_out, extra),
                encoder_ne_logit=encoder_ne_logit,
                decoder_ne_logit=decoder_ne_logit,
                entity_out=None,
                entity_label=None,
                result_entity_id=None,
                encoder_ne=None
            )
        elif self.mode == 2:
            encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=True)

            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, features_only=True, **kwargs)
            if self.decoder.share_input_output_embed:
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_tokens.weight)
            else:
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_out)

            bert_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1)
            assert (decoder_out_feature == extra['inner_states'][-1].transpose(0,1)).all()
            if self.training:
                """
                During the training, there is no need to find entity from src as we have the ground truch in tgt.
                So, extract from the tgt (via prev_output_tokens) and make negative sampling.
                """
                ne_source = kwargs['ne_source']
                tgt_ne_pos = kwargs['tgt_ne_pos']
                origin_tgt = kwargs['origin_tgt']
                target = kwargs['target']
                B, T, C = decoder_out_feature.shape
                entity_position = (target > self.tgt_ne_start_id)  # B * T. Note we don't check 'O' here. We don't expect that here
                #entity_bert_id = torch.zeros((B, T), dtype=torch.long, device=decoder_out_feature.device)
                entity_bert_id = [[-1] * T for _ in range(B)]
                all_entitiy_ids = set()

                # Some of them are using entity in the same batch. As the batch is random generated, it ok to use them as false label
                # 1. entities from target (ground truth)
                # TODO: optimize
                for i in range(B):
                    origin_tgt_i = origin_tgt[i].tolist()
                    for j in range(T):
                        if entity_position[i, j]:
                            origin_tgt_toks = tuple(origin_tgt_i[tgt_ne_pos[i][j]])
                            if origin_tgt_toks in self.bert_emb_id_dict:
                                ent_id_cur = self.bert_emb_id_dict[origin_tgt_toks]
                                entity_bert_id[i][j] = ent_id_cur
                                all_entitiy_ids.add(ent_id_cur)
                            else:
                                # This means we don't preprocess it in BERT. so cannot train on it.
                                # Just skip for now.
                                entity_position[i, j] = False
                                pass
                entity_bert_id = torch.as_tensor(entity_bert_id, device=decoder_out_feature.device)
                # 2. entityies from src mapping. They are harder than random sampling
                for i in range(B):
                    src_entities, src_entity_types = extract_ne_from_text(src_tokens[i], ne_source[i], self.ne_dict, need_type=True)
                    for entity, entity_type in zip(src_entities, src_entity_types):
                        if (entity_type, entity) in self.entity_mapping:
                            mapped_entities = self.entity_mapping[(entity_type, entity)]
                            for tgt_entity in mapped_entities:
                                if tgt_entity in self.bert_emb_id_dict:
                                    all_entitiy_ids.add(self.bert_emb_id_dict[tgt_entity])


                # 3. Random sample from the bert dict, which in theory should be unlimited large
                if len(all_entitiy_ids) < self.bert_sample_count:
                    # Not exact match, bust much faster
                    all_ids = np.random.choice(len(self.bert_emb_id_dict), self.bert_sample_count)
                    for v in all_ids:
                        all_entitiy_ids.add(v)

                all_entitiy_ids = list(all_entitiy_ids)
                ent_id_to_sample_id = { v:i for i, v in enumerate(all_entitiy_ids)}

                """
                bert_emt_matrix = torch.zeros((self.bert_sample_count, self.bert_dim), device=target.device)

                for i, vec_id in enumerate(all_entitiy_ids):
                    # print(i, vec_id, bert_emt_matrix.shape, self.bert_emb_value.shape)
                    bert_emt_matrix[i, :] = self.bert_emb_value[vec_id]
                """
                bert_emt_matrix = self.bert_emb_value[all_entitiy_ids]

                entity_position = entity_position.view(-1)

                entity_embeddings = self.to_bert_emb_space(bert_input_feature.reshape((B*T, C))[entity_position])

                entity_out = F.linear(entity_embeddings, bert_emt_matrix)  # ( (entity count) B*T, C)
                
                entity_label = entity_bert_id.view(-1)[entity_position]   # (( (entity count)B*T)) # this is the orginal id, need to map to sampled id.
                entity_label_in_sample_id = torch.zeros_like(entity_label, dtype=torch.long ,device=entity_label.device)

                for i in range(len(entity_label)):
                    entity_label_in_sample_id[i] = ent_id_to_sample_id[entity_label[i].item()]

                return ModelOut(
                    decoder_out=(decoder_out, extra),
                    encoder_ne_logit=encoder_ne_logit,
                    decoder_ne_logit=None,
                    entity_out=entity_out,
                    entity_label=entity_label_in_sample_id,
                    result_entity_id=None,
                    encoder_ne=None
                )
            else:
                """
                During test time, we need to extract from the src ne prediction, then find the candidate. 
                This is for speed and accuracy.
                Note: this could be under test, under dev. In dev, the label is still required to compute the loss

                Actually, target will be always available. When inference, it will use 'forward_encoder' and 'forward_decoder' sepreately.
                Not this method. TODO: clean me up.
                """

                encoder_ne_pred = encoder_ne_logit.argmax(axis=-1)  # B x T x C => B x T

                # When dev, that target is avabiable to compute the loss
                target = kwargs.get('target', None)
                origin_tgt = kwargs.get('origin_tgt', None)
                tgt_ne_pos = kwargs.get('tgt_ne_pos', None)

                B, T, C = decoder_out_feature.shape
                entities_in_batch = [[] for _ in range(B)]
                all_entitiy_ids = set()

                if target is not None:
                    target_entity_position = (target > self.tgt_ne_start_id).cpu().numpy()  # B * T. Note we don't check 'O' here. We don't expect that here
                    #target_entity_bert_id = torch.zeros((B, T), dtype=torch.long, device=decoder_out_feature.device)
                    target_entity_bert_id = [ [-1]*T for _ in range(B)]
                    # When target is avaible, we need to add them in the entity ids. So that it can appear in the dictionary to compute loss.
                    for i in range(B):
                        for j in range(T):
                            if target_entity_position[i, j]:
                                origin_tgt_toks = tuple(origin_tgt[i].tolist()[tgt_ne_pos[i][j]])
                                if origin_tgt_toks in self.bert_emb_id_dict:
                                    ent_id_cur = self.bert_emb_id_dict[origin_tgt_toks]
                                    target_entity_bert_id[i][j] = ent_id_cur
                                    all_entitiy_ids.add(ent_id_cur)
                                    entities_in_batch[i].append(ent_id_cur)
                                else:
                                    # This means we don't preprocess it in BERT. so cannot train on it.
                                    # Just ignore here, as we cannot compute loss for unknown entity
                                    # TODO: find a better way to handle it
                                    target_entity_position[i, j] = False
                                    target_entity_bert_id[i][j] = -1
                    target_entity_bert_id = torch.as_tensor(target_entity_bert_id, device=decoder_out_feature.device)

                for i in range(B):
                    src_entities, src_entity_types = extract_ne_from_text(src_tokens[i], encoder_ne_pred[i], self.ne_dict, need_type=True)
                    for entity, entity_type in zip(src_entities, src_entity_types):
                        if (entity_type, entity) in self.entity_mapping:
                            mapped_entities = self.entity_mapping[(entity_type, entity)]
                            for tgt_entity in mapped_entities:
                                if tgt_entity in self.bert_emb_id_dict:
                                    entities_in_batch[i].append(self.bert_emb_id_dict[tgt_entity])
                    
                    all_entitiy_ids = all_entitiy_ids.union(entities_in_batch[i])

                all_entitiy_ids = list(all_entitiy_ids)

                # We need to combine all the entity in a batch, look up then mask by -inf
                # TODO:and we need to allow copy

                if len(all_entitiy_ids) > 0:
                    ent_id_to_sample_id = { v:i for i, v in enumerate(all_entitiy_ids)}
                    
                    """
                    bert_ent_matrix = torch.zeros((len(all_entitiy_ids), self.bert_dim), device=prev_output_tokens.device)
                    for i, vec_id in enumerate(all_entitiy_ids):
                        bert_ent_matrix[i, :] = self.bert_emb_value[vec_id]
                    """
                    bert_ent_matrix = self.bert_emb_value[all_entitiy_ids]

                    entity_position = (decoder_out.argmax(-1) > self.tgt_ne_start_id).cpu().numpy()  # B * T, Note we don't check 'O' here. We don't expect that here

                    entity_embeddings =  self.to_bert_emb_space(bert_input_feature)
                    entity_out = F.linear(entity_embeddings, bert_ent_matrix)  # B * T * C

                    # mask token not in that sentence
                    """
                    # No need to mask now. Only dev need to use it.
                    for i in range(B):
                        masked_entity_id = [True] * len(all_entitiy_ids)

                        for vec_id in entities_in_batch[i]:
                            masked_entity_id[ent_id_to_sample_id[vec_id]] = False

                        entity_out[i, :, masked_entity_id] = float('-inf')
                    """
                    entity_out_max = entity_out.argmax(-1)  # (B * T)

                    # map back to original entity id
                    #result_entity_id = torch.zeros_like(entity_out_max, device=entity_out_max.device)
                    
                    result_entity_id = [[-1]*T for _ in range(B)]
                    for i in range(B):
                        for j in range(T):
                            if entity_position[i, j]:
                                result_entity_id[i][j] = all_entitiy_ids[entity_out_max[i, j]]

                    result_entity_id = torch.as_tensor(result_entity_id, device=entity_out_max.device)

                    if target is not None:
                        target_entity_position = torch.as_tensor(target_entity_position)
                        entity_out = entity_out[target_entity_position]
                        entity_label = target_entity_bert_id[target_entity_position].view(-1)

                        # Map to sample dict id
                        entity_label_in_sample_id = torch.zeros_like(entity_label, dtype=torch.long ,device=entity_label.device)
                        for i in range(len(entity_label)):
                            entity_label_in_sample_id[i] = ent_id_to_sample_id[entity_label[i].item()]
                    else:
                        # Not required as we don't compute loss
                        entity_out, entity_label_in_sample_id = None, None
                else:
                    # There is no entity extracted from src. :(
                    # TODO: shall we do anything to rescue it? Or it just that there is no entity in this batch?
                    result_entity_id = torch.ones((B, T), dtype=torch.long, device=decoder_out_feature.device) * -1
                    if target is not None:
                        entity_out = torch.zeros((0), dtype=torch.long, device=decoder_out_feature.device)
                        entity_label_in_sample_id = torch.zeros((0), dtype=torch.long, device=decoder_out_feature.device)
                    else:
                        entity_out, entity_label_in_sample_id = None, None

                return ModelOut(
                    decoder_out=(decoder_out, extra),
                    encoder_ne_logit=encoder_ne_logit,
                    decoder_ne_logit=None,
                    entity_out=entity_out,
                    entity_label=entity_label_in_sample_id,
                    result_entity_id=result_entity_id,
                    encoder_ne=None
                )
        else:
            raise Exception(f'Bad mode {self.mode}')

    def forward_decoder(self, prev_output_tokens, encoder_out, **kwargs):
        if self.mode == 0 or self.mode == 3:
            return self.decoder(prev_output_tokens, encoder_out, **kwargs)
        elif self.mode == 1 or self.mode == 4 or self.mode == 5:
            # encoder_out_with_emb, _ = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=False)
            # return self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, **kwargs)

            encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=True)

            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, features_only=True, **kwargs)
            if self.decoder.share_input_output_embed:
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_tokens.weight)
            else:
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_out)

            assert (decoder_out_feature == extra['inner_states'][-1].transpose(0, 1)).all()

            ne_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1)

            decoder_ne_emb = F.relu(self.tgt_ne_fc1(ne_input_feature))
            decoder_ne_logit = self.tgt_ne_fc2(decoder_ne_emb)
            return ModelOut(
                decoder_out=(decoder_out, extra),
                encoder_ne_logit=encoder_ne_logit,
                decoder_ne_logit=decoder_ne_logit,
                entity_out=None,
                entity_label=None,
                result_entity_id=None,
                encoder_ne=None
            )

        else:
            src_tokens = kwargs['src_tokens']
            encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, ne_type=0, need_logit=True)
            
            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb, features_only=True, **kwargs)
            if self.decoder.share_input_output_embed:
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_tokens.weight)
            else:
                decoder_out = F.linear(decoder_out_feature, self.decoder.embed_out)
            encoder_ne_pred = encoder_ne_logit.argmax(axis=-1)  # B x T x C => B x T
            bert_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1) 

            #entity_position = (decoder_out.argmax(-1) > self.tgt_ne_start_id).cpu().numpy()  # B * T, Note we don't check 'O' here. We don't expect that here
            entity_pred_type = torch.topk(decoder_out, k=5, dim=-1)[1].cpu()
            entity_position = (entity_pred_type > self.tgt_ne_start_id).any(dim=-1).numpy() # B * T, Note we don't check 'O' here. We don't expect that here
            B, T, C = decoder_out_feature.shape
            entities_in_batch = [[] for _ in range(B)]
            
            all_typed_entities = set()
            all_entitiy_ids = set()

            src_entities = [[] for _ in range(B)]
            src_entity_types = [[] for _ in range(B)]
            for i in range(B):
                src_entities[i], src_entity_types[i] = extract_ne_from_text(src_tokens[i], encoder_ne_pred[i], self.ne_dict, need_type=True)
                for entity, entity_type in zip(src_entities[i], src_entity_types[i]):
                    if (entity_type, entity) in self.entity_mapping:
                        mapped_entities = self.entity_mapping[(entity_type, entity)]
                        for tgt_entity in mapped_entities:
                            if tgt_entity in self.bert_emb_id_dict:
                                entities_in_batch[i].append((entity_type, self.bert_emb_id_dict[tgt_entity]))
                
                # When there is an entity in the hypo, but can't find any mapping from src.
                # We iterate all src sub str, and hope to find sth. It will make the inference even slower...
                if not entities_in_batch[i] and entity_position[i].any():
                    src_token_list = src_tokens[i].tolist()
                    for s in range(0, len(src_token_list)):
                        # assume the src is pad left
                        if src_token_list[s] == self.ne_dict.pad():
                            continue

                        if src_token_list[s] == self.ne_dict.eos():
                            break

                        for e in range(s + 1, len(src_token_list)):
                            entity = tuple(src_token_list[s:e])
                            for t in all_types:
                                if (t, entity) in self.entity_mapping:
                                    mapped_entities = self.entity_mapping[(t, entity)]
                                    for tgt_entity in mapped_entities:
                                        if tgt_entity in self.bert_emb_id_dict:
                                            entities_in_batch[i].append((t, self.bert_emb_id_dict[tgt_entity]))


                all_typed_entities = all_typed_entities.union(entities_in_batch[i])

            all_typed_entities = list(all_typed_entities)
            all_entitiy_ids = [x[1] for x in all_typed_entities]

            # We need to combine all the entity in a batch, look up then mask by -inf
            # TODO:and we need to allow copy

            if len(all_entitiy_ids) > 0:
                ent_id_to_sample_id = { v:i for i, v in enumerate(all_entitiy_ids)}
                
                """
                bert_ent_matrix = torch.zeros((len(all_entitiy_ids), self.bert_dim), device=prev_output_tokens.device)
                for i, vec_id in enumerate(all_entitiy_ids):
                    bert_ent_matrix[i, :] = self.bert_emb_value[vec_id]
                """
                bert_ent_matrix = self.bert_emb_value[all_entitiy_ids]

                
                entity_embeddings =  self.to_bert_emb_space(bert_input_feature)
                entity_out = F.linear(entity_embeddings, bert_ent_matrix)  # B * T * C

                # mask token, if not in that sent, or wrong type
                for i in range(B):
                    for j in range(T):
                        if not entity_position[i, j]:
                            entity_out[i, j, :] = float('-inf')
                        else:
                            pred_target_types = entity_pred_type[i][j][ entity_pred_type[i][j] > self.tgt_ne_start_id ].tolist()
                            pred_target_type_text = set([self.ne_dict[x - self.tgt_ne_start_id].split('-')[0] for x in pred_target_types])
                            
                            assert len(pred_target_types) > 0

                            masked_entity_id = [True] * len(all_entitiy_ids)

                            for (ent_type, vec_id) in entities_in_batch[i]:
                                if ent_type in pred_target_type_text: #vec type is correct:
                                    masked_entity_id[ent_id_to_sample_id[vec_id]] = False

                            entity_out[i, j, masked_entity_id] = float('-inf')

                entity_out_max = entity_out.argmax(-1)  # (B * T)

                # map back to original entity id
                #result_entity_id = torch.zeros_like(entity_out_max, device=entity_out_max.device)
                result_entity_id = [[-1]*T for _ in range(B)]
                for i in range(B):
                    for j in range(T):
                        #if entity_position[i, j]:
                        ent_pred = entity_out_max[i, j].item()
                        if entity_out[i, j, ent_pred] != float('-inf'):
                            result_entity_id[i][j] = all_entitiy_ids[ent_pred]
                        else:
                            result_entity_id[i][j] = -1

                            if entity_position[i, j]:
                                # Here is predicted as an entity, but cannot find anything to fill it.
                                # So reduce the logit so that it will output normal token
                                decoder_out[i][j][self.tgt_ne_start_id:] -= NE_PENALTY

                result_entity_id = torch.as_tensor(result_entity_id, device=entity_out_max.device)
                # Not required as we don't compute loss
                # TODO: maybe add them in output for debug ?
                entity_out, entity_label_in_sample_id = None, None
            else:
                # There is no entity extracted from src. :(
                # TODO: shall we do anything to rescue it? Or it just that there is no entity in this batch?
                result_entity_id = torch.ones((B, T), dtype=torch.long, device=decoder_out_feature.device) * -1

                entity_out, entity_label_in_sample_id = None, None

                ## Prevent to output any entity
                decoder_out[:, :, self.tgt_ne_start_id:] -= NE_PENALTY

            return ModelOut(
                decoder_out=(decoder_out, extra),
                encoder_ne_logit=encoder_ne_logit,
                decoder_ne_logit=None,
                entity_out=entity_out,
                entity_label=entity_label_in_sample_id,
                result_entity_id=result_entity_id,
                encoder_ne = encoder_ne_pred
            )

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


@fairseq.models.register_model('entity_transformer')
class EntityTransformer(EntityEncoderDecoderModel):

    def __init__(self, args, ne_dict, encoder, decoder, tgt_ne_start_id, bert_emb_id_dict, bert_emb_value, entity_mapping):
        super().__init__(args, ne_dict, encoder, decoder, tgt_ne_start_id, bert_emb_id_dict, bert_emb_value, entity_mapping)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        EntityEncoderDecoderModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        ##### Copy from transformer.py ####
        # make sure all arguments are present in older models
        base_architecture(args)

        if getattr(args, 'encoder_layers_to_keep', None):
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if getattr(args, 'decoder_layers_to_keep', None):
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = fairseq.models.transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerModel.build_encoder(args, src_dict, encoder_embed_tokens)

        # The decoder-encoder attion K, V is larger, as we combine the NE emb
        if args.mode != 0 and args.concat_ne_emb:
            new_args = copy.deepcopy(args)
            new_args.encoder_embed_dim = args.encoder_embed_dim + args.src_ne_project
            decoder = TransformerModel.build_decoder(new_args, tgt_dict, decoder_embed_tokens)
        else:
            assert args.mode == 0 or args.encoder_embed_dim == args.src_ne_project, f'mode {args.mode}, {args.encoder_embed_dim} != {args.src_ne_project}'
            decoder = TransformerModel.build_decoder(args, tgt_dict, decoder_embed_tokens)

        tgt_ne_start_id = len(task.tgt_dict.lang_dict)
        return cls(args, task.ne_dict, encoder, decoder, tgt_ne_start_id, task.bert_emb_id_dict, task.bert_emb_value, task.entity_mapping)


@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer')
def base_architecture(args):
    fairseq.models.transformer.base_architecture(args)
    args.concat_ne_emb = getattr(args, 'concat_ne_emb', False)
    args.src_ne_project = getattr(args, 'src_ne_project', args.encoder_embed_dim)
    args.tgt_ne_project = getattr(args, 'tgt_ne_project', args.src_ne_project)
    args.bert_lookup_layer = getattr(args, 'bert_lookup_layer', args.decoder_layers) # Use last layer by default
    args.src_ne_layer = getattr(args, 'src_ne_layer', args.encoder_layers)
    args.tgt_ne_drop_rate = getattr(args, 'tgt_ne_drop_rate', 0.0)


@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    fairseq.models.transformer.transformer_iwslt_de_en(args)
    base_architecture(args)

@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    fairseq.models.transformer.transformer_vaswani_wmt_en_de_big(args)
    base_architecture(args)
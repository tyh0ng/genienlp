#
# Copyright (c) 2018-2019, Salesforce, Inc.
#                          The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import os
from collections import defaultdict
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig, \
    BERT_PRETRAINED_MODEL_ARCHIVE_LIST, XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from typing import NamedTuple, List

from .numericalizer.simple import SimpleNumericalizer
from .numericalizer.transformer import BertNumericalizer, XLMRobertaNumericalizer
from . import word_vectors
from .almond_embeddings import AlmondEmbeddings
from .pretrained_lstm_lm import PretrainedLTSMLM

from ..paraphrase.transformers_utils import BertModelV2, XLMRobertaModelV2

_logger = logging.getLogger(__name__)

EMBEDDING_NAME_TO_NUMERICALIZER_MAP = dict()
EMBEDDING_NAME_TO_NUMERICALIZER_MAP.update(
    {embedding: BertNumericalizer for embedding in BERT_PRETRAINED_MODEL_ARCHIVE_LIST})
EMBEDDING_NAME_TO_NUMERICALIZER_MAP.update(
    {embedding: XLMRobertaNumericalizer for embedding in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST})

class EmbeddingOutput(NamedTuple):
    all_layers: List[torch.Tensor]
    last_layer: torch.Tensor

class WordVectorEmbedding(torch.nn.Module):
    def __init__(self, vec_collection):
        super().__init__()
        self._vec_collection = vec_collection
        self.dim = vec_collection.dim
        self.num_layers = 0
        self.embedding = None

    def init_for_vocab(self, vocab):
        vectors = torch.empty(len(vocab), self.dim, device=torch.device('cpu'))
        for ti, token in enumerate(vocab.itos):
            vectors[ti] = self._vec_collection[token.strip()]

        # wrap in a list so it will not be saved by torch.save and it will not
        # be moved around by .to() and similar methods
        self.embedding = [torch.nn.Embedding(len(vocab.itos), self.dim)]
        self.embedding[0].weight.data = vectors

    def grow_for_vocab(self, vocab, new_words):
        if not new_words:
            return
        new_vectors = []
        for word in new_words:
            new_vector = self._vec_collection[word]

            # charNgram returns  a [1, D] tensor, while Glove returns a [D] tensor
            # normalize to [1, D] so we can concat along the second dimension
            # and later concat all vectors along the first
            new_vector = new_vector if new_vector.dim() > 1 else new_vector.unsqueeze(0)
            new_vectors.append(new_vector)

        if new_vectors:
            self.embedding[0].weight.data = torch.cat([self.embedding[0].weight.data.cpu()] + new_vectors, dim=0)

    def forward(self, input: torch.Tensor, padding=None):
        last_layer = self.embedding[0](input.cpu()).to(input.device)
        return EmbeddingOutput(all_layers=[last_layer], last_layer=last_layer)

    def to(self, *args, **kwargs):
        # ignore attempts to move the word embedding, which should stay on CPU
        kwargs['device'] = torch.device('cpu')
        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        # ignore attempts to move the word embedding
        pass

class TransformerEmbedding(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        model.config.output_hidden_states = True
        self.dim = model.config.hidden_size
        self.num_layers = model.config.num_hidden_layers
        self.model = model

    def init_for_vocab(self, vocab):
        self.model.resize_token_embeddings(len(vocab))

    def grow_for_vocab(self, vocab, new_words):
        self.model.resize_token_embeddings(len(vocab))

    def get_positional_embedding(self, input):
        position_ids = torch.arange(input.size()[1], dtype=torch.long, device=input.device)
        position_ids = position_ids.unsqueeze(0).expand(*input.size())
        position_embeddings = self.model.embeddings.position_embeddings(position_ids)

        return position_embeddings

    def forward(self, input: torch.Tensor, entity_ids=None, entity_masking=None, mask_entities=False, padding=None, output_hidden_states=True):
        inputs = {'input_ids': input, 'attention_mask': (~padding).to(dtype=torch.float), 'mask_entities': mask_entities, 'output_hidden_states': output_hidden_states}
        if entity_ids is not None:
            inputs['entity_ids'] = entity_ids
        if entity_masking is not None:
            inputs['entity_masking'] = entity_masking
        last_hidden_state, _pooled, hidden_states = self.model(**inputs)

        return EmbeddingOutput(all_layers=hidden_states, last_layer=last_hidden_state)

class PretrainedLMEmbedding(torch.nn.Module):
    def __init__(self, model_name, cachedir):
        super().__init__()
        # map to CPU first, we will be moved to the right place later
        pretrained_save_dict = torch.load(os.path.join(cachedir, model_name), map_location=torch.device('cpu'))

        self.itos = pretrained_save_dict['vocab']
        self.stoi = defaultdict(lambda: 0, {
            w: i for i, w in enumerate(self.pretrained_decoder_vocab_itos)
        })
        self.dim = pretrained_save_dict['settings']['nhid']
        self.num_layers = 1
        self.model = PretrainedLTSMLM(rnn_type=pretrained_save_dict['settings']['rnn_type'],
                                      ntoken=len(self.pretrained_decoder_vocab_itos),
                                      emsize=pretrained_save_dict['settings']['emsize'],
                                      nhid=pretrained_save_dict['settings']['nhid'],
                                      nlayers=pretrained_save_dict['settings']['nlayers'],
                                      dropout=0.0)
        self.model.load_state_dict(pretrained_save_dict['model'], strict=True)

        self.vocab_to_pretrained = None

    def init_for_vocab(self, vocab):
        self.vocab_to_pretrained = torch.empty(len(self.vocab), dtype=torch.int64)

        unk_id = self.stoi['<unk>']
        for ti, token in enumerate(vocab.itos):
            if token in self.pretrained_decoder_vocab_stoi:
                self.vocab_to_pretrained[ti] = self.stoi[token]
            else:
                self.vocab_to_pretrained[ti] = unk_id

    def grow_for_vocab(self, vocab, new_words):
        self.init_for_vocab(vocab)

    def forward(self, input: torch.Tensor, padding=None):
        pretrained_indices = torch.gather(self.vocab_to_pretrained, dim=0, index=input)
        rnn_output = self.model(pretrained_indices)
        return EmbeddingOutput(all_layers=[rnn_output], last_layer=rnn_output)

def _name_to_vector(emb_name, cachedir):
    if emb_name == 'glove':
        return WordVectorEmbedding(word_vectors.GloVe(cache=cachedir))
    elif emb_name == 'small_glove':
        return WordVectorEmbedding(word_vectors.GloVe(cache=cachedir, name="6B", dim=50))
    elif emb_name == 'char':
        return WordVectorEmbedding(word_vectors.CharNGram(cache=cachedir))
    elif emb_name == 'almond_type':
        return AlmondEmbeddings()
    elif emb_name.startswith('fasttext/'):
        # FIXME this should use the fasttext library
        return WordVectorEmbedding(word_vectors.FastText(cache=cachedir, language=emb_name[len('fasttext/'):]))
    elif emb_name.startswith('pretrained_lstm/'):
        return PretrainedLMEmbedding(emb_name[len('pretrained_lstm/'):], cachedir=cachedir)
    else:
        raise ValueError(f'Unrecognized embedding name {emb_name}')

def get_embedding_type(emb_name):
    if '@' in emb_name:
        return emb_name.split('@')[0]
    else:
        return emb_name

def load_embeddings(cachedir, context_emb_names, question_emb_names, decoder_emb_names,
                    max_generative_vocab=50000, num_db_types=0, db_unk_id=0, logger=_logger, cache_only=False):
    logger.info(f'Getting pretrained word vectors and pretrained models')

    context_emb_names = context_emb_names.split('+')
    question_emb_names = question_emb_names.split('+')
    decoder_emb_names = decoder_emb_names.split('+')

    all_vectors = {}
    context_vectors = []
    question_vectors = []
    decoder_vectors = []

    numericalizer = None
    numericalizer_type = None
    for emb_name in context_emb_names:
        if not emb_name:
            continue
        if emb_name in all_vectors:
            context_vectors.append(all_vectors[emb_name])
            continue

        emb_type = get_embedding_type(emb_name)
        if emb_type in EMBEDDING_NAME_TO_NUMERICALIZER_MAP:
            if numericalizer is not None and numericalizer_type != emb_type and not cache_only:
                raise ValueError('Cannot specify multiple Transformer embeddings')

            config = AutoConfig.from_pretrained(emb_type, cache_dir=cachedir)
            config.output_hidden_states = True
            if numericalizer is None:
                numericalizer = EMBEDDING_NAME_TO_NUMERICALIZER_MAP[emb_type](emb_type, config, max_generative_vocab,
                                                                              cache=cachedir)
                numericalizer_type = emb_type

            # load the tokenizer once to ensure all files are downloaded
            AutoTokenizer.from_pretrained(emb_type, cache_dir=cachedir)

            if emb_type in BERT_PRETRAINED_MODEL_ARCHIVE_LIST:
                transformer_model = BertModelV2(config, num_db_types, db_unk_id).from_pretrained(emb_type,
                                                                                                 num_db_types=num_db_types,
                                                                                                 db_unk_id=db_unk_id,
                                                                                                 cache_dir=cachedir,
                                                                                                 output_hidden_states=True)
            elif emb_type in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST:
                transformer_model = XLMRobertaModelV2(config, num_db_types, db_unk_id).from_pretrained(emb_type,
                                                                                                     num_db_types=num_db_types,
                                                                                                     db_unk_id=db_unk_id,
                                                                                                     cache_dir=cachedir,
                                                                                                     output_hidden_states=True)
            context_vectors.append(TransformerEmbedding(transformer_model))

        else:
            if numericalizer is not None:
                logger.warning('Combining Transformer embeddings with other pretrained embeddings is unlikely to work')
            vec = _name_to_vector(emb_type, cachedir)
            all_vectors[emb_name] = vec
            context_vectors.append(vec)

    for emb_name in question_emb_names:
        if not emb_name:
            continue
        if emb_name in all_vectors:
            question_vectors.append(all_vectors[emb_name])
            continue

        emb_type = get_embedding_type(emb_name)
        if emb_type in EMBEDDING_NAME_TO_NUMERICALIZER_MAP:
            if numericalizer is not None and numericalizer_type != emb_type:
                raise ValueError('Cannot specify multiple Transformer embeddings')

            config = AutoConfig.from_pretrained(emb_type, cache_dir=cachedir)
            config.output_hidden_states = True
            if numericalizer is None:
                numericalizer = EMBEDDING_NAME_TO_NUMERICALIZER_MAP[emb_type](emb_type, config=config,
                                                                              max_generative_vocab=max_generative_vocab,
                                                                              cache=cachedir)

                numericalizer_type = emb_type

            # load the tokenizer once to ensure all files are downloaded
            AutoTokenizer.from_pretrained(emb_type, cache_dir=cachedir)

            if emb_type in BERT_PRETRAINED_MODEL_ARCHIVE_LIST:
                transformer_model = BertModelV2(config, num_db_types, db_unk_id).from_pretrained(emb_type,
                                                                                                 num_db_types=num_db_types,
                                                                                                 db_unk_id=db_unk_id,
                                                                                                 cache_dir=cachedir,
                                                                                                 output_hidden_states=True)
            elif emb_type in XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST:
                transformer_model = XLMRobertaModelV2(config, num_db_types, db_unk_id).from_pretrained(emb_type,
                                                                                                     num_db_types=num_db_types,
                                                                                                     db_unk_id=db_unk_id,
                                                                                                     cache_dir=cachedir,
                                                                                                     output_hidden_states=True)
            question_vectors.append(TransformerEmbedding(transformer_model))

        else:
            if numericalizer is not None:
                logger.warning('Combining Transformer embeddings with other pretrained embeddings is unlikely to work')
            vec = _name_to_vector(emb_type, cachedir)
            all_vectors[emb_name] = vec
            question_vectors.append(vec)

    for emb_name in decoder_emb_names:
        if not emb_name:
            continue
        emb_type = get_embedding_type(emb_name)
        if emb_name in EMBEDDING_NAME_TO_NUMERICALIZER_MAP:
            raise ValueError('Transformer embeddings cannot be specified in the decoder')

        if emb_name in all_vectors:
            decoder_vectors.append(all_vectors[emb_name])
        else:
            vec = _name_to_vector(emb_type, cachedir)
            all_vectors[emb_name] = vec
            decoder_vectors.append(vec)

    if numericalizer is None:
        numericalizer = SimpleNumericalizer(max_generative_vocab=max_generative_vocab, pad_first=False)

    return numericalizer, context_vectors, question_vectors, decoder_vectors
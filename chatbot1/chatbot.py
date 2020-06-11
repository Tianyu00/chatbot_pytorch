from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import urllib.request
import zipfile
from my_helper import *
from my_classes import *

#### global variables
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print('Device to use:', device)

    ########## variables (no need to change anything else besides these) ##########
    #### folder/file name
    corpus = 'cornell movie-dialogs corpus'
    corpus_name = 'cornell movie-dialogs corpus'
    save_dir = 'save'

    #### vocabulary trimming
    MIN_COUNT = 3  # for trimming rate words

    #### models configuration
    model_name = 'cb_model'
    attn_model = 'dot' # general, concat
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    #### training configuration
    clip = 50.0
    teacher_forcing_ratio = 0.5 # 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 20000   # total iterations
    print_every = 1000
    save_every = 500

    #### continue training or new training
    loadFilename = None
    checkpoint_iter = None
    #### if load from saved model, run the following
    # checkpoint_iter = 20000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
                               '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                               '{}_checkpoint.tar'.format(checkpoint_iter))

    ########## end of variables ##########




    #### download data and do extract conversations
    download_data(corpus)
    corpus = os.path.join(corpus, corpus)
    datafile = extract_conversations(corpus)

    #### load conversations and make tensors
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile)
    # print('\npairs:')
    # for pair in pairs[:10]:
    #     print(pair)

    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    # show_some_tensors()

    #### build models
    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.num_words, hidden_size)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    encoder.train()
    decoder.train()

    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

        voc.__dict__ = checkpoint['voc_dict']
        embedding.load_state_dict(checkpoint['embedding'])
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    print('Starting Training!')
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, hidden_size,
              save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, checkpoint_iter,
              teacher_forcing_ratio)

    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder)

    evaluateInput(encoder, decoder, searcher, voc)


if __name__ == "__main__":
    main()

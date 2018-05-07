import numpy as np
import re
import os
import itertools
from collections import Counter
import argparse
import random
from sklearn.feature_extraction import stop_words

def get_config():
    """
        embedding_dim = 256
        filter_sizes = [3,4,5]
        num_filters = 512
        drop = 0.5

        epochs = 100
        batch_size = 30
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--neg_sample', type=int, default=200000, help='number of neg samepls')	
    parser.add_argument('--embedding_dim', type=int, default=200, help='embedding dimension')
    parser.add_argument('--filter_sizes', type=list, default=[3,4,5], help='convolution filters sizes')
    parser.add_argument('--num_filters', type=int, default=512, help='number of conv filters')
    parser.add_argument('--drop', type=float, default=0.5, help='ratio to be set as zeros')
    parser.add_argument('--vocabulary_size', type=int, default=20000, help='the size of vocabulary')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')

    config = parser.parse_args()
    print 'config is ', config
    
    return config


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(config):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
#     positive_examples = list(open("./data/rt-polarity.pos", "r").readlines())
    positive_examples = list(open("./data/"+config['dataset']+".pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
#     negative_examples = list(open("./data/rt-polarity.neg", "r").readlines())
    negative_examples = list(open("./data/"+config['dataset']+".neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # random select some negatives samples
    random.Random(6).shuffle(negative_examples)
    negative_examples = negative_examples[:config['neg_sample']]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences, config):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(config):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(config)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded, config)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

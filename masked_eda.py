# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
import numpy as np
from math import ceil
from nltk.parse.corenlp import CoreNLPParser
# cleaning up text
import re

random.seed(1)
np.random.seed(100)

# data about available masked models
unmaskers = {'bert': {'mask_token': '[MASK]', 'model': 'bert-base-uncased'},
             'roberta': {'mask_token': '<mask>', 'model': 'roberta-base'},
             'distilbert': {'mask_token': '[MASK]', 'model': 'distilbert-base-uncased'}}

# Placeholder parameteres, DO NOT TOUCH! These will get filled by main.py
unmasker = None
unmasker_token = ''

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
                        'whom', 'this', 'that', 'these', 'those', 'am',
                        'is', 'are', 'was', 'were', 'be', 'been', 'being',
                        'have', 'has', 'had', 'having', 'do', 'does', 'did',
                        'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                        'because', 'as', 'until', 'while', 'of', 'at',
                        'by', 'for', 'with', 'about', 'against', 'between',
                        'into', 'through', 'during', 'before', 'after',
                        'above', 'below', 'to', 'from', 'up', 'down', 'in',
                        'out', 'on', 'off', 'over', 'under', 'again',
                        'further', 'then', 'once', 'here', 'there', 'when',
                        'where', 'why', 'how', 'all', 'any', 'both', 'each',
                        'few', 'more', 'most', 'other', 'some', 'such', 'no',
                        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                        'very', 's', 't', 'can', 'will', 'just', 'don',
                        'should', 'now', '']


def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces

    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


def load_mlm(mask_model):

    unmasker_data = unmaskers.get(mask_model)
    if not unmasker_data:
        print('Masked language model not found!')
        return False

    global unmasker_token
    unmasker_token = unmasker_data['mask_token']
    print('Loading {} model...'.format(mask_model))
    from transformers import pipeline
    global unmasker
    unmasker = pipeline('fill-mask', model=unmasker_data['model'])
    return True


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

def get_unmasked_sr_results(unmasked, target_word):
    if unmasker_token == '[MASK]':
        # bert or distilbert
        return [(re.sub(r'[^\w ]', '', unmasked_data['sequence'][6:-6]), unmasked_data['score']) for unmasked_data in unmasked if unmasked_data['token_str'] != target_word]
    elif unmasker_token == '<mask>':
        # roberta
        return [(re.sub(r'[^\w ]', '', unmasked_data['sequence'][3:-4]), unmasked_data['score']) for unmasked_data in unmasked if unmasked_data['token_str'][1:] != target_word]


def synonym_replacement(words, n):
    if len(words) < 2:
        return words

    new_words = words.copy()
    random_word_list = list(
        set([word for word in words[:-1] if word not in stop_words]))
    random.shuffle(random_word_list)
    random_word_list = random_word_list[:n]
    for random_word in random_word_list:
        new_words = get_synonym_unmasked(new_words, random_word)

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonym_unmasked(current_words, target_word):
    output_sentence = ' '.join(current_words)

    # How many times 'target_word' occures in 'current_words'?
    target_count = current_words.count(target_word)
    for _ in range(target_count):
        masked_sentence = output_sentence.replace(
            target_word, unmasker_token, 1)
        unmasked = unmasker(masked_sentence)
        # unmasked is [ ("sentence one", score_1), ("sentence two", score_2), ... ]
        unmasked = get_unmasked_sr_results(unmasked, target_word)

        candidates = []
        weights = []
        for tup in unmasked:
            if tup[0] != '':
                candidates.append(tup[0])
                weights.append(tup[1])

        if len(candidates) > 0:
            weights = weights / np.sum(weights)
            output_sentence = np.random.choice(
                candidates, size=1, p=weights)[0]

    output_sentence = output_sentence.split(' ')
    return [token for token in output_sentence if token is not '']

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################


def random_deletion(words, p):

    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 4:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################


def get_unmasked_ri_results(unmasked):
    if unmasker_token == '[MASK]':
        # bert or distilbert
        return [(unmasked_data['token_str'], unmasked_data['score']) for unmasked_data in unmasked if str.isalnum(unmasked_data['token_str'])]
    if unmasker_token == '<mask>':
        # roberta
        return [(unmasked_data['token_str'][1:], unmasked_data['score']) for unmasked_data in unmasked if str.isalnum(unmasked_data['token_str'])]


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    unmasked = []
    while (len(unmasked) < 1):
        temp_words = new_words.copy()
        random_idx = random.randint(0, len(temp_words) - 1)
        temp_words.insert(random_idx, unmasker_token)
        masked_sentence = ' '.join(temp_words)
        unmasked = unmasker(masked_sentence)
        unmasked = get_unmasked_ri_results(unmasked)

    candidates = [tup[0] for tup in unmasked]
    weights = [tup[1] for tup in unmasked]
    weights = weights / np.sum(weights)
    new_words.insert(random_idx, np.random.choice(
        candidates, size=1, p=weights)[0])

########################################################################
# main data augmentation function
########################################################################


def eda(sentence, mask_model, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    op_count = ceil(alpha_sr) + ceil(alpha_ri) + ceil(alpha_rs) + ceil(p_rd)
    num_new_per_technique = ceil(num_aug / op_count)

    # sr
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    # ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(
        sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
        # repeat empty sentence if not enough sentences created
        if len(augmented_sentences) < num_aug:
            n_needed = num_aug - len(augmented_sentences)
            augmented_sentences.extend('' for _ in range(n_needed))
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [
            s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    # append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences

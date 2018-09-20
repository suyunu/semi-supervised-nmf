import pandas as pd
import numpy as np
from time import time
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

import pickle
from random import sample


lem = WordNetLemmatizer()


def get_synonyms(word, synsets):
    synonyms = []
    for synset in synsets:
        synonyms += [str(lemma.name()) for lemma in synset.lemmas()]

    synonyms = [synonym.replace("_", " ") for synonym in synonyms]
    synonyms = list(set(synonyms))
    synonyms = [synonym for synonym in synonyms if synonym != word]

    return synonyms

def get_synonym_antonym(word, pos):
    wordnet_pos = {
        "noun": wordnet.NOUN,
        "verb": wordnet.VERB,
        "adj": wordnet.ADJ,
        "adv": wordnet.ADV
    }
    if pos:
        synsets = wordnet.synsets(word, pos=wordnet_pos[pos])
    else:
        synsets = wordnet.synsets(word)

    synonyms = get_synonyms(word, synsets)
    
    return synonyms


def generate_synonyms(main_list, check_list, synonym_count):

    all_synonyms = []
    final_list = []
    syn_check_list = []
    
    for cl in check_list:
        synset_cl = wordnet.synsets(cl)
        if len(synset_cl) > 0:
            syn_check_list.append(synset_cl[0])
    
    for my_word in main_list:
        synonym_cand = []
        synonyms = get_synonym_antonym(my_word, "")
        
        for syn in synonyms:
            for pos in ['n', 'v', 'r', 'a', 's']:
                lm = lem.lemmatize(syn, pos)
                if lm != syn:
                    syn = lm
                    break
            if (syn not in main_list) and (syn not in check_list) and (syn not in all_synonyms) and (len(syn.split()) == 1):
                wn_syn = wordnet.synsets(syn)
                if len(wn_syn) > 0:
                    wn_syn = wn_syn[0]
                    score = 0
                    for scl in syn_check_list:
                        tmp_score = wn_syn.wup_similarity(scl)
                        if tmp_score is not None:
                            score += tmp_score
                    synonym_cand.append((score, syn))

        generate_synonym = []
        synonym_cand.sort(reverse=True)
        for i, sc in enumerate(synonym_cand):
            if i == synonym_count:
                break
            generate_synonym.append(sc[1])
            
        all_synonyms.extend(generate_synonym)
        final_list.append((my_word, generate_synonym))
    return final_list


def important_train_synonyms(df_words, word_count, synonym_count, similarity_span):
    df_list = []
    
    for col in df_words:
        words = list(df_words[col])[:word_count]
        synonym_tuples = generate_synonyms(words, words[:similarity_span], synonym_count)
        df_list.append(pd.DataFrame(synonym_tuples, columns=[col+" - word", col+" - synonyms"]))
        
    synoyms_df = pd.concat(df_list, axis=1)
    return synoyms_df


def find_doc_word_synonyms(docs, synonyms_df, trained_vocabulary):
    syn_to_word = []
    for c_id in range(0, len(synonyms_df.columns), 2):
        tmp_sw = {}
        for idx, row in synonyms_df.iloc[:, c_id:c_id+2].iterrows():
            for syn in row[1]:
                tmp_sw[syn] = row[0]
        syn_to_word.append(tmp_sw)

    word_replacements = []
    for doc in docs:
        # remove words that appear in vocabulary
        doc_miss = [w for w in doc.split() if w not in trained_vocabulary]

        word_repl = []
        for c_id in range(0, len(synonyms_df.columns), 2):
            tmp_wr = {}
            for word in doc_miss:
                if word in syn_to_word[c_id//2]:
                    tmp_wr[word] = syn_to_word[c_id//2][word]
            word_repl.append(tmp_wr)
        word_replacements.append(word_repl)
        
    return word_replacements


def replace_synoyms(doc, word_replacements):
    return ' '.join(str(word_replacements.get(word, word)) for word in doc.split())
import pandas as pd
import numpy as np
from time import time

import nltk
from nltk.corpus import brown
from nltk.tokenize.moses import MosesDetokenizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

import matplotlib.pyplot as plt
from math import pi

from omterms.interface import *

from ipywidgets import interact, fixed

import pickle

import libs.text_preprocess as tp

import warnings
warnings.filterwarnings("ignore")


categories=['universalism', 'hedonism', 'achievement', 'power',
       'self-direction', 'benevolence', 'conformity', 'tradition', 'stimulation',
       'security']

schwartz =['universalism', 'benevolence', 'conformity', 'tradition',
       'security', 'power', 'achievement', 'hedonism', 'stimulation',
       'self-direction']

def plot_radar_chart(doc_topic_cumul, doc):
    # ------- PART 1: Create background
    
    schwartz_dist = []
    for sch in schwartz:
        schwartz_dist.append(doc_topic_cumul[doc][categories.index(sch)])
    
    N = len(schwartz)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8,8))
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], schwartz)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
    plt.ylim(0,100)


    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values = list(schwartz_dist) + list(schwartz_dist[:1])
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    # Add legend
    #plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Schwartz Chart - Doc " + str(doc))
    plt.show()
    
    
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
    
    
def print_top_words(model, theme, tfidf_vectorizer, n_top_words, n_topics=3):
    feature_names = tfidf_vectorizer.get_feature_names()
    print(color.CYAN + color.BOLD + categories[theme] + color.END)
    for topic_idx, topic in enumerate(model[theme].components_):
        if topic_idx / n_topics == 1:
            break
        message = color.BOLD + "Topic #%d: " % topic_idx + color.END
        message += " - ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
def print_cumulative_train_doc_topics(data, doc_topic, doc, n_best):
    test_theme = data.iloc[doc]['theme']
    print(color.BOLD + "Doc " + str(doc) + color.RED +  " (" + test_theme + ")\t: " + color.END, end='')
    dt = doc_topic[doc]
    for i in dt.argsort()[:-n_best - 1:-1]:
        print("(", end='')
        try:
            print(color.CYAN + color.BOLD + categories[i] + color.END, end='')
        except:
            print(color.CYAN + color.BOLD + "General" + color.END, end='')
        print(", %d, %.2lf)  " %(i, dt[i]), end='')    
    print()
    
def print_cumulative_test_doc_topics(doc_topic, doc, n_best):
    print(color.BOLD + "Doc " + str(doc) + "\t: " + color.END, end='')
    dt = doc_topic[doc]
    for i in dt.argsort()[:-n_best - 1:-1]:
        print("(", end='')
        try:
            print(color.CYAN + color.BOLD + categories[i] + color.END, end='')
        except:
            print(color.CYAN + color.BOLD + "General" + color.END, end='')
        print(", %d, %.2lf)  " %(i, dt[i]), end='')    
    print()

def print_doc_topics(doc_topic, doc, n_best):
    print(color.BOLD + "Doc " + str(doc) + "\t: " + color.END, end='')
    for i in doc_topic[doc].argsort()[:-n_best - 1:-1]:
        print("(", end='')
        try:
            print(color.CYAN + color.BOLD + categories[i//3] + color.END, end='')
        except:
            print(color.CYAN + color.BOLD + "General" + color.END, end='')
        print(", %d, %.2lf)  " %(i, doc_topic[doc][i]), end='')    
    print()

def print_train_results(doc_topic, doc, corpus, data):
    print(color.BOLD + "Document " + str(doc) + color.END)
    print()
    print(color.BOLD + "Text: " + color.END)
    print("..." + corpus[doc][len(corpus[doc])//3:len(corpus[doc])//3+500] + "...")
    print()
    print()
    
    print(color.BOLD + "Topic Distribution: " + color.END)
    #print(pd.DataFrame(data=[W_test_norm[doc]], index = [doc], columns=categories+['general']))
    print_cumulative_train_doc_topics(data, doc_topic, doc, 11) 
    print()
    
    plot_radar_chart(doc_topic, doc)
        
## HELPER FUNCTIONS
def cumulate_W(W, n_topics):
    W_cumul = []
    for d in W:
        temp = []
        for i in range(W.shape[1]//n_topics):
            temp.append(d[i*n_topics:(i+1)*n_topics].sum())
        W_cumul.append(temp)

    W_cumul = np.asarray(W_cumul)
    
    return W_cumul

def normalize_W(W):
    W_cumul_norm = W/(W.sum(axis=1).reshape(W.shape[0], 1))
    W_cumul_norm *= 100
    
    return W_cumul_norm

def export_to_excel(W, docs, filepath):
    '''
    Take cumulated W as input.
    Don't forget to put xlsx as file extension '''
    
    df = pd.DataFrame(data=W,index = range(len(W)), columns=categories)
    df['Text'] = docs
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_excel(filepath)
    return df

def export_to_csv(W, docs, filepath):
    '''
    Take cumulated W as input.
    Don't forget to put csv as file extension '''
    
    df = pd.DataFrame(data=W,index = range(len(W)), columns=categories)
    df['Text'] = docs
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(filepath)
    return df

## MAIN FUNCTIONS
# just seperates list
# https://stackoverflow.com/a/35518205
def partition_list2(a, k):
    if k <= 1: return [a]
    if k >= len(a): return [[x] for x in a]
    partition_between = [(i+1)*len(a) // k for i in range(k-1)]
    average_height = float(sum(a[:,0]))/k
    best_score = None
    best_partitions = None
    count = 0

    while True:
        starts = [0] + partition_between
        ends = partition_between + [len(a)]
        partitions = [a[starts[i]:ends[i]] for i in range(k)]
        heights = [np.sum(p[:,0]) for p in partitions]
        #heights = list(map(sum, np.array(partitions)[:,0]))

        abs_height_diffs = list(map(lambda x: abs(average_height - x), heights))
        worst_partition_index = abs_height_diffs.index(max(abs_height_diffs))
        worst_height_diff = average_height - heights[worst_partition_index]

        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1

        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return best_partitions
        count += 1

        move = -1 if worst_height_diff < 0 else 1
        bound_to_move = 0 if worst_partition_index == 0                        else k-2 if worst_partition_index == k-1                        else worst_partition_index-1 if (worst_height_diff < 0) ^ (heights[worst_partition_index-1] > heights[worst_partition_index+1])                        else worst_partition_index
        direction = -1 if bound_to_move < worst_partition_index else 1
        partition_between[bound_to_move] += move * direction

def print_best_partition(a, k):
    print('Partitioning {0} into {1} partitions'.format(a, k))
    p = partition_list(a, k)
    print('The best partitioning is {0}\n    With heights {1}\n'.format(p, list(map(sum, p))))
    
def initialize_H3(X, theme_counts, n_topics, p):
    X_sum = list(np.sum(X>0, axis=1))
    X_sum = [(x, i) for i, x in enumerate(X_sum)]
    #X_sum.sort(reverse=True)
    #X_sum = np.array(X_sum)

    X_sum_list = []
    X_parts_list = []

    tc_sum = 0
    for tc in theme_counts:
        X_sum_list.append(X_sum[tc_sum:tc+tc_sum])
        X_sum_list[-1].sort(reverse=True)
        X_sum_list[-1] = np.array(X_sum_list[-1][:p*n_topics])
        X_parts_list.append(partition_list2(X_sum_list[-1], n_topics))


        tc_sum += tc

    H_list = []
    tc_sum = 0
    for i, tc in enumerate(theme_counts):
        H = np.zeros((n_topics+1, X.shape[1]))

        for j in range(0, n_topics):
            H[j] = np.average(X[X_parts_list[i][j%len(X_parts_list[i])][:, 1]], axis=0)
            #H[j][np.where(H[j]==0)] += np.average(X[tc_sum:tc+tc_sum], axis=0)[np.where(H[j]==0)]

        for j in range(n_topics, n_topics+1):
            bckg_parts = []
            for k in range(len(theme_counts)):
                bckg_parts.extend(X_parts_list[k][(j-n_topics)%len(X_parts_list[k])][:int(p//2), 1])
            H[j] = np.average(X[bckg_parts], axis=0)
            #H[j][np.where(H[j]==0)] += np.average(X, axis=0)[np.where(H[j]==0)]
        tc_sum += tc
            

        H_list.append(H)
    return H_list

def read_data(filepath):
    data = pd.read_json(filepath)
    data = data[data['text']!=""]
    data = data.sort_values('theme.id')
    
    return data
    
def extract_corpus(data):    
    corpus = list(data['text'])
    return corpus

def preprocess_corpus(corpus):
    PPcorpus = [' '.join(list((extract_terms(doc, extra_process = ['stem'])['Stem']+' ')*extract_terms(doc, extra_process = ['stem'])['TF'])) for doc in corpus]
    return PPcorpus

def train_corpus(corpus, data, brown_corpus, n_topics, betaloss, bckg_brown):
    N = len(data)
    
    theme_counts = data.groupby(['theme.id','theme']).count().iloc[:,1]
    pd_theme_counts = pd.DataFrame(theme_counts)
    n_themes = len(theme_counts)
    
    n_top_words = 5
    n_components = n_topics*(n_themes)
    
    
    print("Extracting tf-idf features for NMF...")
    #tfidf_vectorizer= TfidfVectorizer(min_df=1, ngram_range=(1,3), max_features=50000)
    tfidf_vectorizer= CountVectorizer(min_df=1, ngram_range=(1,3), max_features=50000)
    t0 = time()
    
    W_list = []
    
    if bckg_brown:
        tfidf = tfidf_vectorizer.fit_transform(corpus+brown_corpus)
        tc_sum = 0
        for tc in theme_counts:
            W = np.zeros((N+len(brown_corpus),n_topics+1))
            W[N:, n_topics:] = np.random.random((len(brown_corpus),1))
            W[tc_sum:tc_sum+tc, :] = np.random.random((tc,n_topics+1))

            tc_sum += tc
            W_list.append(W)
    else:
        tfidf = tfidf_vectorizer.fit_transform(corpus)
        tc_sum = 0
        for tc in theme_counts:
            W = np.zeros((N,n_topics+1))
            W[:, n_topics:] = np.random.random((N,1))
            W[tc_sum:tc_sum+tc, :n_topics] = np.random.random((tc,n_topics))

            tc_sum += tc
            W_list.append(W)
        
    n_features = tfidf.shape[1]
    #print(n_features)
    #print("done in %0.2fs." % (time() - t0))
    
    X = tfidf 
    nmf_list = []
    H_list = initialize_H3(X.toarray(), theme_counts, n_topics, p=20)

    for i, W in enumerate(W_list):
        #print("Fitting NMF for " + str(theme_counts.index[i][1]))
        t0 = time()
        H = H_list[i]
        #H = np.random.rand(n_topics+1, n_features)

        nmf = NMF(n_components= n_topics+1, solver='mu', beta_loss=betaloss,
                  alpha=.1, l1_ratio=.5, init = 'custom')

        nmf.fit_transform(X=X,W=W,H=H)
        #print("done in %0.2fs." % (time() - t0))

        nmf_list.append(nmf)
    print("Fitting NMF for " + str(categories)[1:-1])
    #print("Fitting NMF for " + str(theme_counts.index[:][1]))
    
    return nmf_list, W_list, tfidf, tfidf_vectorizer
    
def get_pretrained_words(pre_nmf_list, pre_tfidf_vectorizer, word_count, normalized=False):
    n_topics = pre_nmf_list[0].components_.shape[0]-1
    word_list = []
    feature_names = pre_tfidf_vectorizer.get_feature_names()
    
    nmf_comps = []
    for pnmf in pre_nmf_list:
        aa = pnmf.components_
        nmf_comps.append(aa/np.sum(aa,axis=1)[:, np.newaxis])
    
    for theme in range(10):
        #word_topic = cumulate_W(pre_nmf_list[theme].components_.T,n_topics).T[anti]
        for nt in range(n_topics):
            if normalized:
                word_topic = nmf_comps[theme][nt]
            else:
                word_topic = pre_nmf_list[theme].components_[nt]
            tmp_list = []
            for i, idx in enumerate(list(reversed(word_topic.argsort()))):
                if i == word_count:
                    break
                tmp_list.append((feature_names[idx], np.round(word_topic[idx], 3)))
            word_list.append(tmp_list)
    
    schwartz_word_score = []
    for sch in schwartz:
        for nt in range(n_topics):
            schwartz_word_score.append(word_list[n_topics*categories.index(sch)+nt])
        
    df_list = []
    for i, a in enumerate(schwartz_word_score):
        df_list.append(pd.DataFrame(a, columns=[schwartz[i//n_topics]+ " (" + str(i%n_topics)+ ") - word",
                                                schwartz[i//n_topics]+ " (" + str(i%n_topics)+ ") - score"]))
    score_df = pd.concat(df_list, axis=1)
    
    return score_df

def export_pretrained_excel(pre_nmf_list, pre_tfidf_vectorizer, filepath, word_count=-1):
    df = get_pretrained_words(pre_nmf_list, pre_tfidf_vectorizer, word_count)
    df.to_excel(filepath)

def train_model(filepath, n_topics, betaloss, bckg_brown):
    data = read_data(filepath)

    # corpusPP = preprocess_corpus(corpus) ## To use Bulent Ozer's library
    corpusPP = list(data.text.apply(tp.clean_text))

    train_context = {}

    nmf_list, W_list, tfidf, tfidf_vectorizer = train_corpus(corpusPP, data, [], n_topics, betaloss, bckg_brown)
    train_context['nmf_list'] = nmf_list
    train_context['W_list'] = W_list
    train_context['tfidf'] = tfidf
    train_context['tfidf_vectorizer'] = tfidf_vectorizer

    return train_context

def report_training_topics(train_context, n_top_words, n_topics):
    print("\nTopics in NMF model:")
    for i in range(10):
        print_top_words(train_context['nmf_list'], i, train_context['tfidf_vectorizer'], n_top_words, n_topics)

def report_df(train_context, word_count):

    return get_pretrained_words(train_context['nmf_list'], train_context['tfidf_vectorizer'], word_count)

def report_train_excel(train_context, output_file, word_count):

    return export_pretrained_excel(train_context['nmf_list'], train_context['tfidf_vectorizer'], output_file, word_count)

def create_trained_data(train_context, output_file):
    return pickle.dump( [train_context['nmf_list'], train_context['tfidf_vectorizer']], open(output_file, "wb" ) )


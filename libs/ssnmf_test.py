import pandas as pd
import numpy as np
from time import time

import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import brown
from nltk.tokenize.moses import MosesDetokenizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

import pickle
import requests
import json
import re

from newspaper import Article

import matplotlib.pyplot as plt
from math import pi

from omterms.interface import *

from ipywidgets import interact, fixed

from IPython.display import display

from bs4 import BeautifulSoup

import libs.text_preprocess as tp
import libs.synonyms as syn




categories=['universalism', 'hedonism', 'achievement', 'power',
       'self-direction', 'benevolence', 'conformity', 'tradition', 'stimulation',
       'security']

schwartz =['universalism', 'benevolence', 'conformity', 'tradition',
       'security', 'power', 'achievement', 'hedonism', 'stimulation',
       'self-direction']


def plot_radar_chart(doc_topic_cumul, doc, doc_names):
    # ------- PART 1: Create background
 
    # number of variablecategories
    
    
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
    plt.title("Schwartz Chart - " + doc_names[doc])
    plt.savefig("Schwartz_Chart_" + str(doc))
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
        message += " ".join([feature_names[i]
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
    
def print_cumulative_test_doc_topics(W_test_norms, doc, n_best):
    print(color.BOLD + "Doc " + str(doc) + "\t: " + color.END, end='')
    n_topics = W_test_norms.shape[2]
    dt = W_test_norms[:,doc,:].flatten()
    for i in dt.argsort()[:n_best - 1:-1]:
        print("(", end='')

        print(color.CYAN + color.BOLD + categories[i//n_topics] + color.END, end='')

        print(" (%d), %.2lf)  " %(i%n_topics, dt[i]), end='')    
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
    
    
def print_test_results(doc, W_test_high, W_test_norms, tfidf_test, pre_nmf_list, pre_tfidf_vectorizer, word_topic_scores, word_topic_sp,
                       doc_names, pre_trained_doc, purity_score, word_count, only_doc_words):
    print(color.BOLD + "Document " + str(doc) + ": " + doc_names[doc] + color.END)
    #print()
    #print(color.BOLD + "Text: " + color.END)
    #print("..." + corpus[doc][len(corpus[doc])//3:len(corpus[doc])//3+500] + "...")
    print()
    print()
    
    print(color.BOLD + "Topic Distribution: " + color.END)
    
    #print(pd.DataFrame(data=[W_test_norm[doc]], index = [doc], columns=categories+['general']))
    print_cumulative_test_doc_topics(W_test_norms, doc, 11)
    print()
    
    plot_radar_chart(W_test_high, doc, doc_names)
    print()
    
    df_scores = schwartz_word_scores(W_test_norms[:,doc,:], tfidf_test[doc], word_topic_scores[:,:,doc,:], word_topic_sp[:,:,doc,:], pre_tfidf_vectorizer, purity_score, word_count, only_doc_words)    
    
    display(df_scores)
            
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

def prepare_export(W, docs, doc_names, filepath):
    schwartz_dist = []
    for doc in range(len(docs)):
        temp_dist = []
        for sch in schwartz:
            temp_dist.append(W[doc][categories.index(sch)])
        schwartz_dist.append(temp_dist)
    schwartz_dist = np.asarray(schwartz_dist)
    
    df = pd.DataFrame(data=schwartz_dist,index = range(len(schwartz_dist)), columns=schwartz)
    df['Text'] = docs
    df["name"] = doc_names
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    
    return df
    
def export_to_excel(W, docs, doc_names, filepath):
    '''
    Take cumulated W as input.
    Don't forget to put xlsx as file extension '''
    
    df = prepare_export(W, docs, doc_names, filepath)
    df.to_excel(filepath)
    return df

def export_to_csv(W, docs, doc_names, filepath):
    '''
    Take cumulated W as input.
    Don't forget to put csv as file extension '''
    
    df = prepare_export(W, docs, doc_names, filepath)
    df.to_csv(filepath)
    return df

def export_word_scores_excel(test_context, doc_names, pre_trained_doc, filepath, purity_score=False, word_count=10, only_doc_words=False):
    W_test_norms, W_test_list, tfidf_test = test_context['W_test_norms'], test_context['W_test_list'], test_context['tfidf_test']
    writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
    
    pre_nmf_list, pre_tfidf_vectorizer = pickle.load( open( pre_trained_doc, "rb" ) )
    word_topic_scores, word_topic_sp = calculate_word_topic_scores(pre_nmf_list, W_test_list)
    
    for i, dn in enumerate(doc_names):
        df = schwartz_word_scores(W_test_norms[:,i,:], tfidf_test[i], word_topic_scores[:,:,i,:], word_topic_sp[:,:,i,:], pre_tfidf_vectorizer, purity_score, word_count, only_doc_words)
        dn = re.sub('[\\\:/*?\[\]]', '', dn)
        df.to_excel(writer, str(i)+'-'+dn[:25])
        
    writer.save()
    writer.close()
    
def export_doc_tfidf_scores(test_corpusPP, doc_names, pre_trained_doc, filepath = 'tfidf_docs.xlsx', use_model = True):
    writer = pd.ExcelWriter(filepath, engine = 'xlsxwriter')
    
    _, tfidf_vectorizer = pickle.load( open( pre_trained_doc, "rb" ) )
    tf_vectorizer= CountVectorizer(min_df=1, ngram_range=(1,3), max_features=50000)
    
    if use_model:
        tfidf = tfidf_vectorizer.transform(test_corpusPP)
    else:
        tfidf = tfidf_vectorizer.fit_transform(test_corpusPP)
    
    tf = tf_vectorizer.fit_transform(test_corpusPP)

    word_list = []
    df_list = []
    
    tf_feature_names = tf_vectorizer.get_feature_names()
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        
    for i, dn in enumerate(doc_names):
        word_list = []
        tfidf_doc = tfidf[i].toarray()[0]
        tf_doc = tf[i].toarray()[0]
        
        for idx in list(reversed(tf_doc.argsort())):
            if tf_doc[idx] <= 0:
                break
            if tf_feature_names[idx] not in tfidf_feature_names:
                continue
            idy = tfidf_feature_names.index(tf_feature_names[idx])
            word_list.append((tf_feature_names[idx], np.round(tf_doc[idx], 2), np.round(tfidf_doc[idy], 2), len(tf_feature_names[idx].split())))
        
        #df = pd.DataFrame(word_list, columns=["word (" + dn + ")", "tf (" + dn + ")",
        #                                                "tf-idf (" + dn + ")", "ngram (" + dn + ")"])
        df = pd.DataFrame(word_list, columns=["word", "tf", "tf-idf", "ngram"])
        
        dn = re.sub('[\\\:/*?\[\]]', '', dn)
        df.to_excel(writer, str(i)+'-'+dn[:25], index=False)
        
        #df_list.append(df)
        
    #score_df = pd.concat(df_list, axis=1)
    #score_df.to_excel(filepath)
    
    writer.save()
    writer.close()

def getLinksHTMLaref(page):
    """

    :param page: html of web page (here: Python home page) 
    :return: urls in that page 
    """
    start_link = page.find("a href=")
    if start_link == -1:
        return None, 0
    start_quote = page.find('"', start_link)
    end_quote = page.find('"', start_quote + 1)
    url = page[start_quote + 1: end_quote]
    return url, end_quote

def getLinksHTML(page):
    """

    :param page: html of web page (here: Python home page) 
    :return: urls in that page 
    """
    start_link = page.find("href=")
    if start_link == -1:
        return None, 0
    start_quote = page.find('"htt', start_link)
    end_quote = page.find('"', start_quote + 1)
    url = page[start_quote + 1: end_quote]
    return url, end_quote

def getLinksXML(page):
    """

    :param page: html of web page (here: Python home page) 
    :return: urls in that page 
    """
    start_link = page.find("<link/>")
    if start_link == -1:
        return None, 0
    start_quote = page.find('http', start_link)
    end_quote = page.find('<', start_quote )
    url = page[start_quote : end_quote]
    return url, end_quote

def extractFromURL(surl):
    response = requests.get(surl)
    # parse html
    page = str(BeautifulSoup(response.content,"lxml"))
    is_XML = surl.endswith('xml')
    url_list = []
    while True:
        if is_XML:
            url, n = getLinksXML(page)
        else:
            url, n = getLinksHTML(page)
        
        page = page[n:]
        if url:
            if set(url_list).intersection(set(url)) == set() or len(set(url_list).intersection(set(url))) != len(url):
                url_list.append(url)
        else:
            break
        
    page = str(BeautifulSoup(response.content,"lxml"))
    stlink= surl.find("//")
    stlink= surl.find("/",stlink+2 )
    base = surl[0:stlink]
    while True:
        if is_XML:
            break
        else:
            url, n = getLinksHTMLaref(page)
        page = page[n:]
        if url:
            url = base+url
            if set(url_list).intersection(set(url)) == set() or len(set(url_list).intersection(set(url))) != len(url):
                url_list.append(url)
        else:
            break
            
    return url_list

## MAIN FUNCTIONS
# just seperates list
# https://stackoverflow.com/a/35518205
def preprocess_corpus(corpus):
    
    PPcorpus = [' '.join(list((extract_terms(doc, extra_process = ['stem'])['Stem']+' ')*extract_terms(doc, 
                extra_process = ['stem'])['TF'])) if doc != '' else '' for doc in corpus]
    return PPcorpus
    
def evaluate_docs(docs, nmf, tfidf_test, betaloss = 'kullback-leibler'):
    X_test = tfidf_test
    H_test = nmf.components_
    
    # Fit the NMF model
    t0 = time()

    W_test = nmf.transform(X_test)
    
    return W_test

def evaluate_test_corpus(pretrained_filepath, test_corpusPP, word_replacements, betaloss):
    nmf_list, tfidf_vectorizer = pickle.load( open( pretrained_filepath, "rb" ) )
    
    W_test_list = []
    print("Fitting NMF for " + str(categories)[1:-1])
    for i, nmf in enumerate(nmf_list):
        #print("Fitting NMF for " + str(categories[i]))
        if word_replacements == []:
            tfidf_test = tfidf_vectorizer.transform(test_corpusPP)
        else:
            docs = [syn.replace_synoyms(test_corpusPP[idx], word_replacements[idx][i]) for idx in range(len(test_corpusPP))]
            tfidf_test = tfidf_vectorizer.transform(docs)
        
        n_features = tfidf_test.shape[1]
        W_test = evaluate_docs([], nmf, tfidf_test, betaloss = betaloss)
        W_test_list.append(W_test)
        
    
    # Sum up sub topics
    W_test_norms = []
    for W_test in W_test_list:
        temp_docs = []
        for dd in W_test:
            temp = []
            for w in dd[:-1]:
                temp.append(100*w/(w+dd[-1]))
            temp_docs.append(temp)
        W_test_norms.append(temp_docs)

    W_test_norms = np.asarray(W_test_norms)
    W_test_norms = np.nan_to_num(W_test_norms)
    
    W_test_high = W_test_norms.max(axis=2).T
    
    # cumulated-normalized and raw
    return W_test_high, W_test_norms, np.asarray(W_test_list), tfidf_test


def print_training_topics(pretrained_filepath, n_top_words, n_topics):
    nmf_list, tfidf_vectorizer = pickle.load( open( pretrained_filepath, "rb" ) )
    print("\nTopics in NMF model:")
    for i in range(10):
        print_top_words(nmf_list, i, tfidf_vectorizer, n_top_words, n_topics)

def add_corpus_txt(filepath, test_corpus):
    try:
        f = open(filepath, "r")
        txt = f.read()
        test_corpus.append(txt)
        f.close()
    except:
        test_corpus.append("")
        print("File not found - " + filepath)

def url_scraper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        news_info =  {
            "url": url,
            "title": article.title,
            "text": article.text,
            "summary": "",
            #"authors": article.authors,
            "keywords": [],
            "publish_date": article.publish_date.__str__()[:19],
            #"top_image": article.top_image
        }
    except:
        news_info =  {
            "Error": "Content Not Found"
        }

    try:
        article.nlp()
        news_info["summary"] = article.summary,
        news_info["keywords"] = article.keywords
    except:
        pass

    return news_info

def add_corpus_url(url, api_key, test_corpus):
    res = url_scraper(url)
    if "Error" in res:
        test_corpus.append("")
        print(res["Error"] + " - " + url)
    elif "text" in res:
        test_corpus.append(res['text'])
        if res['text'] == "":
            print("Empty text - " + url)
    else:
        test_corpus.append("")
        print("Empty text - " + url)

def print_interactive_test_results(W_test_high, W_test_norms, W_test_list, tfidf_test, doc_names, pre_trained_doc, purity_score, word_count, only_doc_words):
    pre_nmf_list, pre_tfidf_vectorizer = pickle.load( open( pre_trained_doc, "rb" ) )
    word_topic_scores, word_topic_sp = calculate_word_topic_scores(pre_nmf_list, W_test_list)
    
    interact(print_test_results,
             doc = (0, len(W_test_high)-1, 1),
             W_test_high=fixed(W_test_high),
             W_test_norms=fixed(W_test_norms),
             tfidf_test=fixed(tfidf_test),
             pre_nmf_list=fixed(pre_nmf_list),
             pre_tfidf_vectorizer=fixed(pre_tfidf_vectorizer),
             word_topic_scores=fixed(word_topic_scores),
             word_topic_sp=fixed(word_topic_sp),
             doc_names=fixed(doc_names),
             pre_trained_doc=fixed(pre_trained_doc),
             purity_score=fixed(purity_score),
             word_count=fixed(word_count),
             only_doc_words=fixed(only_doc_words))

# scores are multiplied by 100
def calculate_word_topic_scores(pre_nmf_list, W_test_list):
    n_topics = pre_nmf_list[0].components_.shape[0]-1
    H_list = []

    for pnmf in pre_nmf_list:
        aa = pnmf.components_
        H_list.append(aa/np.sum(aa,axis=1)[:, np.newaxis])
        #H_list.append(pnmf.components_)
    H_list = np.asarray(H_list)

    # [value, doc, word]
    word_topic_scores = []
    word_background_scores = []

    for i in range(10):
        temp_ts = []
        temp_bs = []
        for nt in range(n_topics):
            temp_ts.append(np.dot(W_test_list[i][:,nt][:, np.newaxis], H_list[i][nt,:][np.newaxis, :]))
            temp_bs.append(np.dot(W_test_list[i][:,n_topics:], H_list[i][n_topics:,:]))
        word_topic_scores.append(temp_ts)
        word_background_scores.append(temp_bs)

    word_topic_scores = np.asarray(word_topic_scores)
    word_background_scores = np.asarray(word_background_scores)

    word_topic_purity = np.nan_to_num(np.divide(word_topic_scores,word_topic_scores+word_background_scores))
    word_topic_sp = word_topic_scores*word_topic_purity

    word_topic_scores *= 100000
    word_topic_sp *= 100000
    
    return word_topic_scores, word_topic_sp

def find_top_word_scores(pre_tfidf_vectorizer, word_topic, word_count, tfidf_test_doc, only_doc_words):
    word_list = []
    feature_names = pre_tfidf_vectorizer.get_feature_names()
    
    for theme in range(10):
        tmp_word_list = []
        for nt in range(word_topic.shape[1]):
            tmp_list = []
            i = 0 
            for idx in list(reversed(word_topic[theme][nt].argsort())):
                if i == word_count:
                    break
                if not(only_doc_words and (tfidf_test_doc[0,idx] == 0)):
                    tmp_list.append((feature_names[idx], np.round(word_topic[theme][nt][idx], 3)))
                else:
                    i -= 1
                i += 1
            tmp_word_list.append(tmp_list)
        word_list.append(tmp_word_list)
    return word_list

def schwartz_word_scores(W_test_doc, tfidf_test_doc, word_topic_scores, word_topic_sp, pre_tfidf_vectorizer, purity_score, word_count, only_doc_words):
    if purity_score:
        top_scores = find_top_word_scores(pre_tfidf_vectorizer, word_topic_sp, word_count, tfidf_test_doc, only_doc_words)
    else:
        top_scores = find_top_word_scores(pre_tfidf_vectorizer, word_topic_scores, word_count, tfidf_test_doc, only_doc_words)
    
    schwartz_word_score = []
    schwartz_W_test = []
    for sch in schwartz:
        schwartz_word_score.append(top_scores[categories.index(sch)])            
        schwartz_W_test.append((sch.upper(), np.round(W_test_doc[categories.index(sch)], 2)))
        
    df_list = []
    for i, sws in enumerate(schwartz_word_score):
        for j, s in enumerate(sws):
            df_list.append(pd.DataFrame([("% " + schwartz_W_test[i][0]+" (" + str(j) + ")", schwartz_W_test[i][1][j])]+s,
                                        columns=[schwartz[i]+ " (" + str(j)+ ") - word",
                                                 schwartz[i]+ " (" + str(j)+ ") - score"]))
        score_df = pd.concat(df_list, axis=1)
    
    return score_df


def get_pretrained_words(pre_trained_doc, normalized, word_count):
    pre_nmf_list, pre_tfidf_vectorizer = pickle.load( open( pre_trained_doc, "rb" ) )
    
    n_topics = pre_nmf_list[0].components_.shape[0]-1
    word_list = []
    feature_names = pre_tfidf_vectorizer.get_feature_names()
    
    nmf_comps = []
    for pnmf in pre_nmf_list:
        aa = pnmf.components_
        nmf_comps.append(1000*aa/np.sum(aa,axis=1)[:, np.newaxis])
    
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
                tmp_list.append((feature_names[idx], np.round(word_topic[idx], 2)))
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

def export_pretrained_excel(pre_trained_doc, filepath, word_count=-1):
    df = get_pretrained_words(pre_trained_doc,  normalized, word_count)
    df.to_excel(filepath)

def add_corpus_docs(doc_names, test_corpus, insigth_api_key):
    for doc in doc_names:
        if re.match("^(http|https)://", doc) is None:
            add_corpus_txt(doc, test_corpus)
        else:
            add_corpus_url(doc, insigth_api_key, test_corpus)

def prepare_test_docs(test_doc_names, pre_trained_doc_name, betaloss):
    test_corpus = []
    insigth_api_key = "thisisinsightapikey" #needs to be filled

    print("Reading data...")
    add_corpus_docs(test_doc_names, test_corpus, insigth_api_key)

    print("Cleaning data...")
    test_corpusPP = [tp.clean_text(doc) for doc in test_corpus]
    
    test_context = {}

    W_test_high, W_test_norms, W_test_list, tfidf_test = evaluate_test_corpus(pre_trained_doc_name, test_corpusPP, [], betaloss)

    test_context['W_test_high'] = W_test_high
    test_context['W_test_norms'] = W_test_norms
    test_context['W_test_list'] = W_test_list
    test_context['tfidf_test'] = tfidf_test

    return test_corpusPP, test_context

def report_df(pre_trained_doc, normalized, word_count):

    return get_pretrained_words(pre_trained_doc, normalized, word_count)


def report_interactive_result(test_context, test_doc_names, pre_trained_doc_name, purity_score, word_count, only_doc_words):
    
    return print_interactive_test_results(test_context['W_test_high'], test_context['W_test_norms'], test_context['W_test_list'], test_context['tfidf_test'], test_doc_names, pre_trained_doc_name, purity_score, word_count, only_doc_words)

def export_excel(test_context, test_corpusPP, test_doc_names, output_file = "test_result.xlsx"):

    return export_to_excel(test_context['W_test_high'], test_corpusPP, test_doc_names, output_file)
    
def export_csv(test_context, test_corpusPP, test_doc_names, output_file = "test_result.csv"):

    return export_to_csv(test_context['W_test_high'], test_corpusPP, test_doc_names, output_file)
    


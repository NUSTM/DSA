#!/usr/bin/env python
# coding=utf8

'''Dual Sentiment Analysis V0.8

Ref:
Rui Xia et al. Dual Training and Dual Prediction for Polarity Classification. In ACL-2013.
Rui Xia et al. Dual Sentiment Analysis. In IEEE TKDE-2015
'''
import os, sys, re, subprocess, getopt
import pytc
import nltk
import numpy as np
from nltk.corpus import wordnet as wn

######## global variable ##########

FNAME_LIST = ['negative', 'positive']
SAMP_TAG = 'review_text'
TERM_WEIGHT = 'BOOL'
NEGATOR = ['not', 'no', 'without', 'never', 'n\'t', 'don\'t', 'hardly', 'doesn\'t']
CONTRAST = ['but', 'however', 'But', 'However']
END_WORDS = ['.', ',', '!', '?', '...']

#####################POS Tag######################
def pos_tag(doc_str_list):
    pos_doc_str_list = []
    for doc_str in doc_str_list:
        term_list = nltk.word_tokenize(doc_str)
        pos_term_list = nltk.pos_tag(term_list)
        pos_doc_str = ' '.join(x[0] + '_'  + x[1] for x in pos_term_list)
        pos_doc_str_list.append(pos_doc_str)
    return pos_doc_str_list

def pos_tag_task(input_dir, output_dir):
    fname_list, samp_tag =  FNAME_LIST, SAMP_TAG
    print 'Reading text...'
    doc_str_list, doc_class_list = pytc.read_text_f2([input_dir+os.sep+x for x in fname_list], samp_tag)
    print 'Pos tagging...'
    pos_doc_str_list = pos_tag(doc_str_list)
    print 'Write pos text...'
    pytc.save_text_f2(output_dir, samp_tag, pos_doc_str_list, doc_class_list)

################FS Antonym Extraction##################
def save_score_list( term_score_list, fname_score):
    score_str = '\n'.join( (str(term_score[0]) + '\t' + str(term_score[1])) for term_score in term_score_list )
    open(fname_score,'w').write(score_str)

def filter_by_postag(fname_socre, adverse_list, postag_list):
    '''
    函数功能：  解析特征选择得分文件，并利用词性过滤，产生候选词
    param:
    fname_socre: 特征选择得分列表文件(带词性标注 )  
    adverse_list: 否定词列表(过滤否定词)
    postag_list: 词性列表(形容词副词、动词)
    '''    
    filter_postag_dict = {}
    term_score_list =  [ (line.strip().split()) for line in open(fname_socre, 'r').readlines() ] 
    for term_score in term_score_list:
        pos_term, score = term_score[0], float(term_score[1])
        term_list = pos_term.split('_')
        if term_list[1] in postag_list and term_list[0] not in adverse_list:#非否定词，且为指定词性
            #针对一个特征词有多个词性时，选择特征选择得分最高作为其得分
            if filter_postag_dict.has_key(term_list[0]):        
                if filter_postag_dict[term_list[0]] < score:
                    filter_postag_dict[term_list[0]] = score
            else:
                filter_postag_dict[term_list[0]] = score
    return filter_postag_dict


def postag_filter(fname_neg_score, fname_pos_score, adverse_list, postag_list, fs_percent):
    '''
    函数功能：  利用特征词的特征选择得分与词性产生反义词对
    param:
    fname_neg_score: neg类特征词得分列表(带词性标注 )  
    fname_pos_score: pos类特征词得分列表(带词性标注 ) 
    adverse_list: 否定词列表(过滤否定词)
    postag_list: 词性列表(形容词副词、动词)
    fs_percent: 保留词的比例
    '''
    neg_rank_dict, pos_rank_dict = {}, {}
    neg_postag_dict = filter_by_postag(fname_neg_score, adverse_list, postag_list)#构建去除词性的否定词典
    pos_postag_dict = filter_by_postag(fname_pos_score, adverse_list, postag_list)

    #判断特征词的词性，属于negative or positive
    for term in neg_postag_dict:
        if neg_postag_dict[term] < pos_postag_dict[term]:
            pos_rank_dict[term] = pos_postag_dict[term]
        else:
            neg_rank_dict[term] = neg_postag_dict[term] 

        #利用特征选择得分，按序排列组合形成反义词对
    neg_score_list = neg_rank_dict.items()
    neg_score_list.sort(key = lambda x:-x[1])#降序
    neg_set_rank = [x[0] for x in neg_score_list]
    pos_score_list = pos_rank_dict.items()
    pos_score_list.sort(key = lambda x:-x[1])
    pos_set_rank = [x[0] for x in pos_score_list]
    dict_len= int(min(len(neg_set_rank), len(pos_set_rank)) * fs_percent)#

    return neg_set_rank[:dict_len], pos_set_rank[:dict_len]

########## WordNet Antonym Extraction ##########
def get_antonym_word_list(word,tag_list):
    antonym_word_list = []
    for syn in wn.synsets(word):
        if syn.pos in tag_list:
            for term in syn.lemmas:
                    if term.antonyms():
                        antonym_word_list.append(term.antonyms()[0].name)
        for sim_syn in syn.similar_tos():
            if syn.pos in tag_list:
                for term in sim_syn.lemmas:
                    if term.antonyms():
                        antonym_word_list.append(term.antonyms()[0].name)
    return list(set(antonym_word_list))

def get_wn_antonym(word_and_pos):     
    adj_pos_list = ['JJ' ,'JJS' ,'JJR']#形容词
    rb_pos_list = ['RB' , 'RBS' , 'RBR'] #副词
    verb_pos_list = ['VB' ,'VBZ' , 'VBD' , 'VBN' , 'VBG' , 'VBP'] #动词
    antonym_word = []
    if word_and_pos[1] in adj_pos_list:
        param = ['s','a']
    elif word_and_pos[1] in rb_pos_list:
        param = ['r']
    elif word_and_pos[1] in verb_pos_list:
        param = ['v']
    else:
        return antonym_word
    
    antonym_word_list = get_antonym_word_list(word_and_pos[0],param)
    return antonym_word_list
def filter_wn_antonym(antonym_word_set, term_set_rank):
    '''利用特征选择排序（带词性），对候选反义词（一对多情形）进行排序筛选    
    '''
    fs_antonym_word_list = []
    for term in term_set_rank:
        if term in antonym_word_set:
            fs_antonym_word_list.append(term)
    fs_antonym_word_list.extend(list(antonym_word_set - set(fs_antonym_word_list)))
    return fs_antonym_word_list

def build_wn_dict(term_post_set, term_set_rank, antonym_dict_fname):
    '''读入term_post_set备选特征词集（带词性），输出对应的wn_dict
    '''
    wn_dict = {}
    for term_post in term_post_set:
        word_and_post = term_post.split('_')
        antonym_word_list = get_wn_antonym(word_and_post)
        if not antonym_word_list:#若未查到反义词，则取下一个词
            continue
        if antonym_word_list:
            if wn_dict.has_key(word_and_post[0]):
                wn_dict[word_and_post[0]] = set(wn_dict[word_and_post[0]]) | set(antonym_word_list)
            else:
                wn_dict[word_and_post[0]] = set(antonym_word_list)
    for key in wn_dict:
        antonym_word_list_fs = filter_wn_antonym(wn_dict[key], term_set_rank)
        wn_dict[key] = antonym_word_list_fs
        
    wn_dict_file = open(antonym_dict_fname, 'w')
    
    for pos_term in term_set_rank:
        item = pos_term.split('_')[0]
        if wn_dict.has_key(item):
            wn_dict_file.writelines(str(item) + '\t' + wn_dict[item][0]+'\n')    
    for item in wn_dict.keys():
        if item not in term_set_rank:
            wn_dict_file.writelines(str(item) + '\t' + wn_dict[item][0]+'\n')

def load_antonym_dict(antonym_dict_fname):
    antonym_dict = {}
    for line in open(antonym_dict_fname):
        word_and_antonym = line.split()
        antonym_dict[word_and_antonym[0]] = word_and_antonym[1]
        if not antonym_dict.has_key(word_and_antonym[1]):
            antonym_dict[word_and_antonym[1]] = word_and_antonym[0]
    return antonym_dict

########## Review Reversion ##########
def reverse_doc(doc_terms, antonym_dict):
    '''文档的反义样本生成算法，针对样本的文本翻转规则
    输入:
        doc_terms: 文档特征词序列
    输出:
        reverse_doc_terms: 反义样本特征词序列
    '''
    rev_doc_terms = []
    i = 0
    while i != len(doc_terms):
        if doc_terms[i] in NEGATOR: # 存在否定结构，对应论文中翻转规则2）
            i += 1
            while i != len(doc_terms) and (doc_terms[i] not in END_WORDS) and (doc_terms[i] not in NEGATOR):
                rev_doc_terms.append(doc_terms[i])
                i += 1
        elif doc_terms[i] in antonym_dict: # 当前词不在否定结构内，对应论文中翻转规则1)
            antonym_word = antonym_dict[doc_terms[i]]
            rev_doc_terms.append(antonym_word)
            i += 1
        else:
            rev_doc_terms.append(doc_terms[i])
            i += 1
    return rev_doc_terms

def reverse_dataset(doc_terms_list, doc_class_list, antonym_dict, rev_all=True):
    '''生成反义样本集(调用reverse_doc)
    参数:          
    doc_terms_list: 数据集文档特征词序列集合
    doc_class_list：数据集文档类别集合(需要转化为0或1)
    antonym_dict: 反义字典
    fname_pair_txt: 保存翻转前与翻转后样本
    rev_all: 反义样本是否完全翻转（部分样本翻转前后一样）。注：测试样本为一一配对，必须rev_all=true
    rev_doc_id_list: 保存反转样本在原样本中对应的id号
    '''    
    rev_cnt = 0
    rev_doc_class_list = []
    rev_doc_terms_list = []
    rev_doc_id_list = []
    for k in range(len(doc_class_list)):
        doc_terms = doc_terms_list[k]
        doc_class = doc_class_list[k]
        samp_class = FNAME_LIST.index(doc_class) # samp_class should be 0 or 1
        intersection_list = set(doc_terms_list[k]) & (set(antonym_dict.keys()) | set(NEGATOR))
        if len(intersection_list) != 0:
            # 如果当前评论存在否定词与情感词，则翻转该句
            rev_doc_terms = reverse_doc(doc_terms, antonym_dict)
            rev_samp_class = 1 - samp_class
            rev_doc_class = FNAME_LIST[rev_samp_class]
            rev_doc_terms_list.append(rev_doc_terms)
            rev_doc_class_list.append(rev_doc_class)
            rev_doc_id_list.append(k)
            rev_cnt += 1
        elif rev_all:
            # rev_all判断是否将无需翻转评论加到到翻转样本集中，同时样本类别需要反转
            rev_samp_class = 1 - samp_class
            rev_doc_class = FNAME_LIST[rev_samp_class]
            rev_doc_terms_list.append(doc_terms)
            rev_doc_class_list.append(rev_doc_class)
            rev_doc_id_list.append(k)
    return rev_doc_terms_list, rev_doc_class_list, rev_doc_id_list
 
def select_reverse_dataset(doc_terms_list, doc_class_list, antonym_dict, id_list):
    '''生成反义样本集(调用reverse_doc)
    参数:          
    doc_terms_list: 数据集文档特征词序列集合
    doc_class_list：数据集文档类别集合(需要转化为0或1)
    antonym_dict: 反义字典
    fname_pair_txt: 保存翻转前与翻转后样本
    id_list: 按照id_list进行反义样本翻转
    '''
    rev_cnt = 0
    rev_doc_class_list = []
    rev_doc_terms_list = []
    rev_doc_id_list = []
    for k in id_list:
        doc_terms = doc_terms_list[k]
        doc_class = doc_class_list[k]
        samp_class = FNAME_LIST.index(doc_class) # samp_class should be 0 or 1
        intersection_list = set(doc_terms_list[k]) & (set(antonym_dict.keys()) | set(NEGATOR))
        if len(intersection_list) != 0:
            # 如果当前评论存在否定词与情感词，则翻转该句
            rev_doc_terms = reverse_doc(doc_terms, antonym_dict)
            rev_samp_class = 1 - samp_class
            rev_doc_class = FNAME_LIST[rev_samp_class]
            rev_doc_terms_list.append(rev_doc_terms)
            rev_doc_class_list.append(rev_doc_class)
            rev_doc_id_list.append(k)
            rev_cnt += 1
        else: # 对于id_list里的每一个样本，无论是否反转前后一样，都进行反转
            rev_samp_class = 1 - samp_class
            rev_doc_class = FNAME_LIST[rev_samp_class]
            rev_doc_terms_list.append(doc_terms)
            rev_doc_class_list.append(rev_doc_class)
            rev_doc_id_list.append(k)
    return rev_doc_terms_list, rev_doc_class_list, rev_doc_id_list
 
def save_reverse_dataset(ori_doc_terms_list, ori_doc_class_list, rev_doc_terms_list, rev_doc_class_list, rev_doc_id_list, rev_data_dir_f2):
    if not os.path.exists(rev_data_dir_f2):
        os.makedirs(rev_data_dir_f2)
    rev_str_class = {}.fromkeys(FNAME_LIST, '')
    for rev_k in range(len(rev_doc_id_list)):
        ori_k = rev_doc_id_list[rev_k]
        ori_doc_str = ' '.join(ori_doc_terms_list[ori_k])
        rev_doc_str =  ' '.join(rev_doc_terms_list[rev_k])
        rev_doc_class = rev_doc_class_list[rev_k]
        rev_str_class[rev_doc_class] += ('\n<Ori_ID>' + str(ori_k) + '</Ori_ID>\n')
        rev_str_class[rev_doc_class] += ('<Ori_Review>\n' + ori_doc_str + '\n</Ori_Review>\n')
        rev_str_class[rev_doc_class] += ('<' + SAMP_TAG + '>\n' + rev_doc_str + '\n</' + SAMP_TAG + '>\n')
    for class_name in FNAME_LIST:
        open(rev_data_dir_f2 + os.sep + class_name, 'w').write(rev_str_class[class_name])

########## Dual Training ##########
def dual_nb_exe(dual_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, 
                dual_model_fname, d2o_output_fname, d2r_output_fname, learn_opt, classify_opt): 
    print '\nDual Prediction via NB exe...'
    sp = subprocess.Popen(pytc.NB_LEARN_EXE + ' ' +  learn_opt + ' ' + dual_samp_fname_train + ' ' + dual_model_fname, shell=True)
    sp.wait()
    sp = subprocess.Popen(pytc.NB_CLASSIFY_EXE + ' ' + classify_opt + ' ' + ori_samp_fname_test + ' ' + dual_model_fname + ' ' + d2o_output_fname, shell=True)
    sp.wait()
    sp = subprocess.Popen(pytc.NB_CLASSIFY_EXE + ' ' + classify_opt + ' ' + rev_samp_fname_test + ' ' + dual_model_fname + ' ' + d2r_output_fname, shell=True)
    sp.wait()
    # parsing d2o reslut:
    ori_samp_class_list_test = [x.split()[0] for x in open(ori_samp_fname_test).readlines()]
    ori_samp_class_list_prd = [x.split()[0] for x in open(d2o_output_fname).readlines()]
    d2o_acc = pytc.calc_acc(ori_samp_class_list_test, ori_samp_class_list_prd)
    # parsing d2r reslut:
    rev_samp_class_list_test = [x.split()[0] for x in open(rev_samp_fname_test).readlines()]
    rev_samp_class_list_prd = [x.split()[0] for x in open(d2r_output_fname).readlines()]
    d2r_acc = pytc.calc_acc(rev_samp_class_list_test, rev_samp_class_list_prd) 
    return d2o_acc, d2r_acc

def dual_libsvm_exe(dual_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, 
                dual_model_fname, d2o_output_fname, d2r_output_fname, learn_opt, classify_opt): 
    print '\nDual Prediction via LIBSVM exe...'
    sp = subprocess.Popen(pytc.LIBSVM_LEARN_EXE + ' ' +  learn_opt + ' ' + dual_samp_fname_train + ' ' + dual_model_fname, shell=True)
    sp.wait()    
    sp = subprocess.Popen(pytc.LIBSVM_CLASSIFY_EXE + ' ' + classify_opt + ' ' + ori_samp_fname_test + ' ' + dual_model_fname + ' ' + d2o_output_fname, shell=True)
    sp.wait()    
    sp = subprocess.Popen(pytc.LIBSVM_CLASSIFY_EXE + ' ' + classify_opt + ' ' + rev_samp_fname_test + ' ' + dual_model_fname + ' ' + d2r_output_fname, shell=True)
    sp.wait()
    # parsing d2o reslut:
    ori_samp_class_list_test = [x.split()[0] for x in open(ori_samp_fname_test).readlines()]
    ori_samp_class_list_prd = [x.split()[0] for x in open(d2o_output_fname).readlines()[1:]]
    d2o_acc = pytc.calc_acc(ori_samp_class_list_test, ori_samp_class_list_prd)
    # parsing d2r reslut:
    rev_samp_class_list_test = [x.split()[0] for x in open(rev_samp_fname_test).readlines()]
    rev_samp_class_list_prd = [x.split()[0] for x in open(d2r_output_fname).readlines()[1:]]
    d2r_acc = pytc.calc_acc(rev_samp_class_list_test, rev_samp_class_list_prd) 
    return d2o_acc, d2r_acc   

def dual_liblinear_exe(dual_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, 
                dual_model_fname, d2o_output_fname, d2r_output_fname, learn_opt, classify_opt): 
    print '\nDual Prediction via LibLinear exe...'
    sp = subprocess.Popen(pytc.LIBLINEAR_LEARN_EXE + ' ' +  learn_opt + ' ' + dual_samp_fname_train + ' ' + dual_model_fname, shell=True)
    sp.wait()    
    sp = subprocess.Popen(pytc.LIBLINEAR_CLASSIFY_EXE + ' ' + classify_opt + ' ' + ori_samp_fname_test + ' ' + dual_model_fname + ' ' + d2o_output_fname, shell=True)
    sp.wait()    
    sp = subprocess.Popen(pytc.LIBLINEAR_CLASSIFY_EXE + ' ' + classify_opt + ' ' + rev_samp_fname_test + ' ' + dual_model_fname + ' ' + d2r_output_fname, shell=True)
    sp.wait()
    # parsing d2o reslut:
    ori_samp_class_list_test = [x.split()[0] for x in open(ori_samp_fname_test).readlines()]
    ori_samp_class_list_prd = [x.split()[0] for x in open(d2o_output_fname).readlines()[1:]]
    d2o_acc = pytc.calc_acc(ori_samp_class_list_test, ori_samp_class_list_prd)
    # parsing d2r reslut:
    rev_samp_class_list_test = [x.split()[0] for x in open(rev_samp_fname_test).readlines()]
    rev_samp_class_list_prd = [x.split()[0] for x in open(d2r_output_fname).readlines()[1:]]
    d2r_acc = pytc.calc_acc(rev_samp_class_list_test, rev_samp_class_list_prd) 
    return d2o_acc, d2r_acc
    
def dsa_o2or(ori_data_dir, rev_data_dir, result_dir, classifier, bigram = False):
    fname_class_set = result_dir + os.sep + 'class.set'
    fname_term_set = result_dir + os.sep + 'term.set'
    ori_samp_fname_train = result_dir + os.sep + 'ori.train.samp'
    ori_samp_fname_test = result_dir + os.sep + 'ori.test.samp'
    rev_samp_fname_test = result_dir + os.sep + 'rev.test.samp'
    ori_model_fname = result_dir + os.sep + 'ori.model'
    o2o_output_fname = result_dir + os.sep + 'o2o.out'
    o2r_output_fname = result_dir + os.sep + 'o2r.out'
    print '\nReading original and reversed text...'
    ori_doc_str_list_train, ori_doc_class_list_train = pytc.read_text_f2([ori_data_dir + os.sep + 'train' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    ori_doc_str_list_test, ori_doc_class_list_test = pytc.read_text_f2([ori_data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    rev_doc_str_list_test, rev_doc_class_list_test = pytc.read_text_f2([rev_data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST[::-1]], SAMP_TAG)
    ori_doc_terms_list_train = pytc.get_doc_terms_list(ori_doc_str_list_train)
    ori_doc_terms_list_test = pytc.get_doc_terms_list(ori_doc_str_list_test)
    rev_doc_terms_list_test = pytc.get_doc_terms_list(rev_doc_str_list_test)
    
    if bigram == True:
        ori_doc_bigrams_terms_list_train = pytc.get_doc_bis_list(ori_doc_str_list_train)
        ori_doc_bigrams_terms_list_test = pytc.get_doc_bis_list(ori_doc_str_list_test)
        rev_doc_bigrams_terms_list_test = pytc.get_doc_bis_list(rev_doc_str_list_test)
        ori_doc_terms_list_train = pytc.get_joint_sets(ori_doc_terms_list_train, ori_doc_bigrams_terms_list_train)
        ori_doc_terms_list_test = pytc.get_joint_sets(ori_doc_terms_list_test, ori_doc_bigrams_terms_list_test)
        rev_doc_terms_list_test = pytc.get_joint_sets(rev_doc_terms_list_test, rev_doc_bigrams_terms_list_test)
    
    term_set = pytc.get_term_set(ori_doc_terms_list_train)
    class_set = FNAME_LIST
    print 'Building samples...'
    if classifier in ['liblinear','libsvm']:
        term_dict = dict(zip(term_set, range(1, len(term_set) + 1)))
        class_dict = dict(zip(class_set, range(len(class_set)))) # class id must be 0 or 1
    elif classifier in ['nb']:
        term_dict = dict(zip(term_set, range(1, len(term_set) + 1)))
        class_dict = dict(zip(class_set, range(1, len(class_set)+1)))
    ori_samp_list_train, ori_class_list_train = pytc.build_samps(term_dict, class_dict, ori_doc_terms_list_train, ori_doc_class_list_train, TERM_WEIGHT)
    ori_samp_list_test, ori_class_list_test = pytc.build_samps(term_dict, class_dict, ori_doc_terms_list_test, ori_doc_class_list_test, TERM_WEIGHT)
    rev_samp_list_test, rev_class_list_test = pytc.build_samps(term_dict, class_dict, rev_doc_terms_list_test, rev_doc_class_list_test, TERM_WEIGHT)
    pytc.save_samps(ori_samp_list_train, ori_class_list_train, ori_samp_fname_train)
    pytc.save_samps(ori_samp_list_test, ori_class_list_test, ori_samp_fname_test)
    pytc.save_samps(rev_samp_list_test, rev_class_list_test, rev_samp_fname_test)
    print 'o2o and o2r classification'
    if classifier == 'liblinear':
        # LibLinear
        learn_opt = '-s 7 -c 1'
        classify_opt = '-b 1'
        o2o_acc, o2r_acc = dual_liblinear_exe(ori_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, ori_model_fname, o2o_output_fname, o2r_output_fname, learn_opt, classify_opt)
    elif classifier == 'libsvm':
        # LibSVM
        learn_opt = '-s 0 -t 0 -b 1 -c 1'
        classify_opt = '-b 1'
        o2o_acc, o2r_acc = dual_libsvm_exe(ori_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, ori_model_fname, o2o_output_fname, o2r_output_fname, learn_opt, classify_opt)
    elif classifier == 'nb':
        #Naive Bayes 
        learn_opt = ''
        classify_opt = '-f 2'
        o2o_acc, o2r_acc = dual_nb_exe(ori_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, ori_model_fname, o2o_output_fname, o2r_output_fname, learn_opt, classify_opt)
    else:
        raise Exception("The input value of classifier is illegal.")
    
    print 'o2o_acc:', o2o_acc
    print 'o2r_acc:', o2r_acc
    return o2o_acc, o2r_acc

def dsa_r2or(ori_data_dir, rev_data_dir, result_dir, classifier, bigram = False):
    fname_class_set = result_dir + os.sep + 'class.set'
    fname_term_set = result_dir + os.sep + 'term.set'
    ori_samp_fname_train = result_dir + os.sep + 'ori.train.samp'
    rev_samp_fname_train = result_dir + os.sep + 'rev.train.samp'
    ori_samp_fname_test = result_dir + os.sep + 'ori.test.samp'
    rev_samp_fname_test = result_dir + os.sep + 'rev.test.samp'
    rev_model_fname = result_dir + os.sep + 'rev.model'
    r2o_output_fname = result_dir + os.sep + 'r2o.out'
    r2r_output_fname = result_dir + os.sep + 'r2r.out'
    print '\nReading original and reversed text...'
    rev_doc_str_list_train, rev_doc_class_list_train = pytc.read_text_f2([rev_data_dir + os.sep + 'train' + os.sep + x for x in FNAME_LIST[::-1]], SAMP_TAG)
    rev_doc_str_list_test, rev_doc_class_list_test = pytc.read_text_f2([rev_data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST[::-1]], SAMP_TAG)
    ori_doc_str_list_test, ori_doc_class_list_test = pytc.read_text_f2([ori_data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    rev_doc_terms_list_train = pytc.get_doc_terms_list(rev_doc_str_list_train)
    rev_doc_terms_list_test = pytc.get_doc_terms_list(rev_doc_str_list_test)
    ori_doc_terms_list_test = pytc.get_doc_terms_list(ori_doc_str_list_test)
    
    if bigram ==True:
        rev_doc_bis_terms_list_train = pytc.get_doc_bis_list(rev_doc_str_list_train)
        rev_doc_bis_terms_list_test = pytc.get_doc_bis_list(rev_doc_str_list_test)
        ori_doc_bis_terms_list_test = pytc.get_doc_bis_list(ori_doc_str_list_test)
        rev_doc_terms_list_train = pytc.get_joint_sets(rev_doc_terms_list_train, rev_doc_bis_terms_list_train)
        rev_doc_terms_list_test = pytc.get_joint_sets(rev_doc_terms_list_test, rev_doc_bis_terms_list_test)
        ori_doc_terms_list_test = pytc.get_joint_sets(ori_doc_terms_list_test, ori_doc_bis_terms_list_test)
        
    term_set = pytc.get_term_set(rev_doc_terms_list_train)
    class_set = FNAME_LIST
    print 'Building samples...'
    if classifier in ['liblinear','libsvm']:
        term_dict = dict(zip(term_set, range(1, len(term_set) + 1)))
        class_dict = dict(zip(class_set, range(len(class_set)))) # class id must be 0 or 1
    elif classifier in ['nb']:
        term_dict = dict(zip(term_set, range(1, len(term_set) + 1)))
        class_dict = dict(zip(class_set, range(1, len(class_set)+1)))
    rev_samp_list_train, rev_class_list_train = pytc.build_samps(term_dict, class_dict, rev_doc_terms_list_train, rev_doc_class_list_train, TERM_WEIGHT)
    ori_samp_list_test, ori_class_list_test = pytc.build_samps(term_dict, class_dict, ori_doc_terms_list_test, ori_doc_class_list_test, TERM_WEIGHT)
    rev_samp_list_test, rev_class_list_test = pytc.build_samps(term_dict, class_dict, rev_doc_terms_list_test, rev_doc_class_list_test, TERM_WEIGHT)
    pytc.save_samps(rev_samp_list_train, rev_class_list_train, rev_samp_fname_train)
    pytc.save_samps(ori_samp_list_test, ori_class_list_test, ori_samp_fname_test)
    pytc.save_samps(rev_samp_list_test, rev_class_list_test, rev_samp_fname_test)
    print 'r2o and r2r classification'

    if classifier == 'liblinear':
        # LibLinear
        learn_opt = '-s 7 -c 1'
        classify_opt = '-b 1'
        r2o_acc, r2r_acc = dual_liblinear_exe(rev_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, rev_model_fname, r2o_output_fname, r2r_output_fname, learn_opt, classify_opt)
    elif classifier == 'libsvm':
        # LibSVM
        learn_opt = '-s 0 -t 0 -b 1 -c 1'
        classify_opt = '-b 1'
        r2o_acc, r2r_acc = dual_libsvm_exe(rev_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, rev_model_fname, r2o_output_fname, r2r_output_fname, learn_opt, classify_opt)        
    elif classifier == 'nb': 
        # Naive Bayes 
        learn_opt = ''
        classify_opt = '-f 2'
        r2o_acc, r2r_acc = dual_nb_exe(rev_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, rev_model_fname, r2o_output_fname, r2r_output_fname, learn_opt, classify_opt)        
    else:
        raise Exception("The input value of classifier is illegal.")
    
    print 'r2o_acc:', r2o_acc
    print 'r2r_acc:', r2r_acc
    return r2o_acc, r2r_acc

def dsa_d2or(ori_data_dir, rev_data_dir, result_dir, classifier, bigram = False):
    fname_class_set = result_dir + os.sep + 'class.set'
    fname_term_set = result_dir + os.sep + 'term.set'
    ori_samp_fname_test = result_dir + os.sep + 'ori.test.samp'
    rev_samp_fname_test = result_dir + os.sep + 'rev.test.samp'     
    ori_samp_fname_train = result_dir + os.sep + 'ori.train.samp'
    dual_samp_fname_train = result_dir + os.sep + 'dual.train.samp'
    dual_model_fname = result_dir + os.sep + 'dual.model'
    d2o_output_fname = result_dir + os.sep + 'd2o.out'
    d2r_output_fname = result_dir + os.sep + 'd2r.out'
    print '\nReading original text...'
    ori_doc_str_list_train, ori_doc_class_list_train = pytc.read_text_f2([ori_data_dir + os.sep + 'train' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    ori_doc_str_list_test, ori_doc_class_list_test = pytc.read_text_f2([ori_data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    ori_doc_terms_list_train = pytc.get_doc_terms_list(ori_doc_str_list_train)
    ori_doc_terms_list_test = pytc.get_doc_terms_list(ori_doc_str_list_test)
    print 'Reading reversed text...'
    rev_doc_str_list_train, rev_doc_class_list_train = pytc.read_text_f2([rev_data_dir + os.sep + 'train' + os.sep + x for x in FNAME_LIST[::-1]], SAMP_TAG)
    rev_doc_str_list_test, rev_doc_class_list_test = pytc.read_text_f2([rev_data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST[::-1]], SAMP_TAG)
    rev_doc_terms_list_train = pytc.get_doc_terms_list(rev_doc_str_list_train)
    rev_doc_terms_list_test = pytc.get_doc_terms_list(rev_doc_str_list_test)
    
    if bigram == True:
        ori_doc_bis_terms_list_train = pytc.get_doc_bis_list(ori_doc_str_list_train)
        ori_doc_bis_terms_list_test = pytc.get_doc_bis_list(ori_doc_str_list_test)
        ori_doc_terms_list_train = pytc.get_joint_sets(ori_doc_terms_list_train, ori_doc_bis_terms_list_train)
        ori_doc_terms_list_test = pytc.get_joint_sets(ori_doc_terms_list_test, ori_doc_bis_terms_list_test)

        rev_doc_bis_terms_list_train = pytc.get_doc_bis_list(rev_doc_str_list_train)
        rev_doc_bis_terms_list_test = pytc.get_doc_bis_list(rev_doc_str_list_test)
        rev_doc_terms_list_train = pytc.get_joint_sets(rev_doc_terms_list_train, rev_doc_bis_terms_list_train)
        rev_doc_terms_list_test = pytc.get_joint_sets(rev_doc_terms_list_test, rev_doc_bis_terms_list_test)

    print 'Combining dual training data...'
    dual_doc_terms_list_train = ori_doc_terms_list_train + rev_doc_terms_list_train
    dual_doc_class_list_train = ori_doc_class_list_train + rev_doc_class_list_train   
    term_set = pytc.get_term_set(dual_doc_terms_list_train)
    class_set = FNAME_LIST

    print 'Building samples...'
    if classifier in ['liblinear','libsvm']:
        term_dict = dict(zip(term_set, range(1, len(term_set) + 1)))
        class_dict = dict(zip(class_set, range(len(class_set)))) # class id must be 0 or 1
    elif classifier in ['nb']:
        term_dict = dict(zip(term_set, range(1, len(term_set) + 1)))
        class_dict = dict(zip(class_set, range(1, len(class_set)+1)))
    dual_samp_list_train, dual_class_list_train = pytc.build_samps(term_dict, class_dict, dual_doc_terms_list_train, dual_doc_class_list_train, TERM_WEIGHT)
    ori_samp_list_test, ori_class_list_test = pytc.build_samps(term_dict, class_dict, ori_doc_terms_list_test, ori_doc_class_list_test, TERM_WEIGHT)
    rev_samp_list_test, rev_class_list_test = pytc.build_samps(term_dict, class_dict, rev_doc_terms_list_test, rev_doc_class_list_test, TERM_WEIGHT)    
    pytc.save_samps(dual_samp_list_train, dual_class_list_train, dual_samp_fname_train)
    pytc.save_samps(ori_samp_list_test, ori_class_list_test, ori_samp_fname_test)
    pytc.save_samps(rev_samp_list_test, rev_class_list_test, rev_samp_fname_test)
    print 'd2o and d2r classification'
    if classifier == 'liblinear':
        # LibLinear
        learn_opt = '-s 7 -c 1'
        classify_opt = '-b 1'
        d2o_acc, d2r_acc = dual_liblinear_exe(dual_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, dual_model_fname, d2o_output_fname, d2r_output_fname, learn_opt, classify_opt)
    elif classifier == 'libsvm':
        # LibSVM
        learn_opt = '-s 0 -t 0 -b 1 -c 1'
        classify_opt = '-b 1'
        d2o_acc, d2r_acc = dual_libsvm_exe(dual_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, dual_model_fname, d2o_output_fname, d2r_output_fname, learn_opt, classify_opt)
    elif classifier == 'nb': 
        # Naive Bayes    
        learn_opt = ''
        classify_opt = '-f 2'
        d2o_acc, d2r_acc = dual_nb_exe(dual_samp_fname_train, ori_samp_fname_test, rev_samp_fname_test, dual_model_fname, d2o_output_fname, d2r_output_fname, learn_opt, classify_opt)
    else:
        raise Exception("The input value of classifier is illegal.")
   
    print 'd2o_acc:', d2o_acc
    print 'd2r_acc:', d2r_acc
    return d2o_acc,  d2r_acc

########## Dual Prediction (Ensemble) ##########
def dual_prediction_2comb(ori_samp_class_list_prd, ori_samp_prb_list_prd, rev_samp_class_list_prd, rev_samp_prb_list_prd, weight):
    '''训练集不翻转，组合测试集翻转的两种结果：
    testn - testp > prb_conf 时，支持testn，否则支持testp
    '''
    dp_samp_class_list = []
    dp_samp_prb_list = []
    for k in range(len(ori_samp_class_list_prd)):
        ori_samp_prb = ori_samp_prb_list_prd[k]
        rev_samp_prb = rev_samp_prb_list_prd[k]
        prb_list = [((1 - weight) * ori_samp_prb[i] + weight * rev_samp_prb[1 - i]) for i in range(2)]
        dp_samp_prb = dict(zip(range(2), prb_list))
        dp_samp_prb_list.append(dp_samp_prb)
        if dp_samp_prb[0] > dp_samp_prb[1]:
            dp_samp_class_list.append(0)
        else:
            dp_samp_class_list.append(1)
    return dp_samp_class_list, dp_samp_prb_list

def dual_prediction_3conf(o2o_samp_class_list_prd, o2o_samp_prb_list_prd, d2o_samp_class_list_prd, d2o_samp_prb_list_prd, d2r_samp_class_list_prd, d2r_samp_prb_list_prd, weight, prb_conf):
    '''训练集翻转训练得到混合系统，组合测试集翻转的两种结果、及原系统 这三种结果。
    1) trainpn_testn、trainpn_testp 加权组合得到 trainpn_testpn,
    2) trainpn_testpn - trainp_testp > prb_conf 时，支持trainpn_testpn，否则支持trainp_testp
    '''
    d2d_samp_class_list_2comb, d2d_samp_prb_list_2comb = dual_prediction_2comb(d2o_samp_class_list_prd, d2o_samp_prb_list_prd, d2r_samp_class_list_prd, d2r_samp_prb_list_prd, weight)
    dp_samp_class_list_3conf = [x for x in o2o_samp_class_list_prd]
    for k in range(len(o2o_samp_class_list_prd)):
            if o2o_samp_class_list_prd[k] != d2d_samp_class_list_2comb[k]:
                if (max(d2d_samp_prb_list_2comb[k].values()) - max(o2o_samp_prb_list_prd[k].values())) > prb_conf:
                    dp_samp_class_list_3conf[k] = d2d_samp_class_list_2comb[k]

    return dp_samp_class_list_3conf

def dual_prediction_4comb(o2o_samp_class_list_prd, o2o_samp_prb_list_prd, r2r_samp_class_list_prd, r2r_samp_prb_list_prd, d2o_samp_class_list_prd, d2o_samp_prb_list_prd, d2r_samp_class_list_prd, d2r_samp_prb_list_prd, weights):
    dp_samp_class_list_4comb = []
    dp_samp_prb_list_4comb = []

    for k in range(len(o2o_samp_class_list_prd)):
        o2o_samp_prb = o2o_samp_prb_list_prd[k]
        r2r_samp_prb = r2r_samp_prb_list_prd[k]
        d2o_samp_prb = d2o_samp_prb_list_prd[k]
        d2r_samp_prb = d2r_samp_prb_list_prd[k]
        prb_list = [(weights[0] * o2o_samp_prb[i] + weights[1] * r2r_samp_prb[1 - i] + weights[2] * d2o_samp_prb[i] + weights[3] * d2r_samp_prb[1-i]) for i in range(2)]
        dp_samp_prb = dict(zip(range(2), prb_list))
        dp_samp_prb_list_4comb.append(dp_samp_prb)
        if dp_samp_prb[0] > dp_samp_prb[1]:
            dp_samp_class_list_4comb.append(0)
        else:
            dp_samp_class_list_4comb.append(1)
    return dp_samp_class_list_4comb
  
########## Demo ##########
def post_tag_demo(token_dir, post_dir):
    for fold in ['train','test']:
        if not os.path.exists(post_dir + os.sep + fold):
            os.makedirs(post_dir + os.sep + fold)
        token_data_dir, post_data_dir = token_dir + os.sep + fold, post_dir + os.sep + fold
        pos_tag_task(token_data_dir, post_data_dir)

def wn_dict_demo(post_data_dir, token_train_dir, antonym_dict_fname, dataset_list):
    '''原始数据集上，生成反义wordnet反义字典
    post_data_dir: 带Part-of-speech的数据集目录
    token_train_dir: token 的 test set目录，用于词典一对多时的过滤
    antonym_dict_fname: 反义字典输出
    '''
    print '\nBuilding wordnet antonym dict...'
    print 'Reading post and token text...'
    doc_str_list_post, doc_class_list_post, doc_str_list_token, doc_class_list_token=[],[],[],[]
    for fold in dataset_list:
        doc_str_list_post_temp, doc_class_list_post_temp = pytc.read_text_f2([post_data_dir + os.sep + fold+ os.sep + x for x in FNAME_LIST], SAMP_TAG)#post
        doc_str_list_token_temp, doc_class_list_token_temp = pytc.read_text_f2([token_train_dir + os.sep + fold+ os.sep + x for x in FNAME_LIST], SAMP_TAG)#token
        doc_str_list_post.extend(doc_str_list_post_temp)
        doc_class_list_post.extend(doc_class_list_post_temp)
        doc_str_list_token.extend(doc_str_list_token_temp)
        doc_class_list_token.extend(doc_class_list_token_temp)
    
    print 'Extracting features on post and token data...'
    doc_terms_list_post = pytc.get_doc_terms_list(doc_str_list_post)
    term_post_set = pytc.get_term_set(doc_terms_list_post)
    doc_terms_list_token = pytc.get_doc_terms_list(doc_str_list_token)
    term_set = pytc.get_term_set(doc_terms_list_token)
    print 'Selecting features on token test data...'

    tf_term_dict = pytc.stat_tf_term(term_set, doc_terms_list_token)
    tf_term_list = tf_term_dict.items()
    tf_term_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in tf_term_list]
    
    print 'Building WN-antonym dict...'
    build_wn_dict(term_post_set, term_set_rank, antonym_dict_fname)
    
def build_fs_dict(post_data_dir, result_dir, fs_method, dataset_list, fs_percent=1):
    '''
    函数功能：  基于特征选择方法和词性过滤反义词典构建主函数
    param:
    post_data_dir:  词性标注训练数据文件夹 
    result_dir:  字典存储目录
    fs_method:     特征选择方法
    dataset_list:   辅助生成词典的数据集
    fs_percent:     反义词在 特征词中的比例
    '''
    jj_postag_list = ['JJ' ,'JJS' ,'JJR', 'JJ' ,'JJS' ,'JJR','RB' , 'RBS' , 'RBR'] 
    v_postag_list = ['VB' ,'VBZ' , 'VBD' , 'VBN' , 'VBG' , 'VBP']
    postag_list = jj_postag_list + v_postag_list
    fname_neg_score, fname_pos_score, fname_fs_dict = result_dir + os.sep +'neg.score', result_dir + os.sep + 'pos.score', result_dir + os.sep + 'antonym.dict'
    doc_str_list_post, doc_class_list_post, doc_str_list_token, doc_class_list_token=[],[],[],[]
    print 'Reading text...'
    for fold in dataset_list:
        doc_str_list_post_temp, doc_class_list_post_temp = pytc.read_text_f2([post_data_dir + os.sep + fold+ os.sep + x for x in FNAME_LIST], SAMP_TAG)#post
        doc_str_list_post.extend(doc_str_list_post_temp)
        doc_class_list_post.extend(doc_class_list_post_temp)
    print ' train size:', len(doc_str_list_post)

    print 'Extracting features...'
    doc_terms_list_train = pytc.get_doc_terms_list(doc_str_list_post)
    class_set = pytc.get_class_set(doc_class_list_post)
    term_set = pytc.get_term_set(doc_terms_list_train)

    print 'Selecting features...' 
    df_term = pytc.stat_df_term(term_set, doc_terms_list_train)#词频字典
    df_class = pytc.stat_df_class(class_set, doc_class_list_post)#类别频率字典
    df_term_class = pytc.stat_df_term_class(term_set, class_set, doc_terms_list_train, doc_class_list_post)#词类别频率词典

    neg_term_set_fs, neg_term_score_list = pytc.supervised_feature_selection(df_class, df_term_class, fs_method, 0, fs_class = 0)#得分序列，得分列表
    pos_term_set_fs, pos_term_score_list = pytc.supervised_feature_selection(df_class, df_term_class, fs_method, 0, fs_class = 1)

    save_score_list(neg_term_score_list.items(), fname_neg_score)
    save_score_list(pos_term_score_list.items(), fname_pos_score)

    print 'Building fs dict...'
    filter_list_neg, filter_list_pos = postag_filter(fname_neg_score, fname_pos_score, NEGATOR, postag_list, fs_percent)#构造词典

    #按照 wn_dict格式写入文件中
    fout = open(fname_fs_dict, 'w')
    for k in range(len(filter_list_neg)):
        fout.writelines(str(filter_list_neg[k])+'\t'+str(filter_list_pos[k])+'\n')
    for k in range(len(filter_list_pos)):
        fout.writelines(str(filter_list_pos[k])+'\t'+str(filter_list_neg[k])+'\n')
    fout.close()
    
def reverse_review_demo(ori_data_dir, rev_data_dir, antonym_dict_fname):
    '''反义样本生成示例
    '''
    if not os.path.exists(rev_data_dir + os.sep + 'train'):
        os.makedirs(rev_data_dir + os.sep + 'train')
    if not os.path.exists(rev_data_dir + os.sep + 'test'):
        os.makedirs(rev_data_dir + os.sep + 'test')
    print '\nCreating polarity-opposite reviews...'
    print 'Reading text...'
    ori_doc_str_list_train, ori_doc_class_list_train = pytc.read_text_f2([ori_data_dir + os.sep + 'train' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    ori_doc_str_list_test, ori_doc_class_list_test = pytc.read_text_f2([ori_data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    ori_doc_terms_list_train = pytc.get_doc_terms_list(ori_doc_str_list_train)
    ori_doc_terms_list_test = pytc.get_doc_terms_list(ori_doc_str_list_test)
    print 'Constructing reverse samples...'
    antonym_dict = load_antonym_dict(antonym_dict_fname)
    rev_doc_terms_list_train, rev_doc_class_list_train, rev_doc_id_list_train = reverse_dataset(ori_doc_terms_list_train, ori_doc_class_list_train, antonym_dict, False)
    rev_doc_terms_list_test, rev_doc_class_list_test, rev_doc_id_list_test = reverse_dataset(ori_doc_terms_list_test, ori_doc_class_list_test, antonym_dict, True)
    print 'Saving reverse samples...'
    print len(rev_doc_terms_list_train), len(rev_doc_class_list_train)
    print len(rev_doc_terms_list_test), len(rev_doc_class_list_test)
    save_reverse_dataset(ori_doc_terms_list_train, ori_doc_class_list_train, rev_doc_terms_list_train, rev_doc_class_list_train, rev_doc_id_list_train, rev_data_dir + os.sep + 'train')
    save_reverse_dataset(ori_doc_terms_list_test, ori_doc_class_list_test, rev_doc_terms_list_test, rev_doc_class_list_test, rev_doc_id_list_test, rev_data_dir + os.sep + 'test')

def select_reverse_review_demo(ori_data_dir, rev_data_dir, result_dir, antonym_dict_fname, perc, bigram=False):
    '''部分反义样本生成示例
    '''
    antonym_dict = load_antonym_dict(antonym_dict_fname)
    print '\nCreating polarity-opposite reviews...'
    print 'Reading text...'
    ori_doc_str_list_train, ori_doc_class_list_train = pytc.read_text_f2([ori_data_dir + os.sep + 'train' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    ori_doc_str_list_test, ori_doc_class_list_test = pytc.read_text_f2([ori_data_dir + os.sep + 'test' + os.sep + x for x in FNAME_LIST], SAMP_TAG)
    ori_doc_terms_list_train = pytc.get_doc_terms_list(ori_doc_str_list_train)
    ori_doc_terms_list_test = pytc.get_doc_terms_list(ori_doc_str_list_test)
    
    if bigram == True:
        ori_doc_bigrams_terms_list_train = pytc.get_doc_bis_list(ori_doc_str_list_train)
        ori_doc_terms_list_train = pytc.get_joint_sets(ori_doc_terms_list_train, ori_doc_bigrams_terms_list_train)

    term_set = pytc.get_term_set(ori_doc_terms_list_train)
    class_set = pytc.get_class_set(ori_doc_class_list_train)
    print 'Building samples...'
    ori_samp_train_fname = result_dir + os.sep + 'select.dt.samp'
    model_fname = result_dir + os.sep + 'select.dt.mod'
    output_fname = result_dir + os.sep + 'select.dt.out'
    
    term_dict = dict(zip(term_set, range(1, len(term_set) + 1)))
    class_dict = dict(zip(class_set, range(len(class_set)))) # class id must be 0 or 1

    ori_samp_list_train, ori_class_list_train = pytc.build_samps(term_dict, class_dict, ori_doc_terms_list_train, ori_doc_class_list_train, TERM_WEIGHT)
    pytc.save_samps(ori_samp_list_train, ori_class_list_train, ori_samp_train_fname)

    print 'Classification on the training data...'
    
    learn_opt = '-s 7 -c 1'
    classify_opt = '-b 1'

    dt_acc = pytc.liblinear_exe(ori_samp_train_fname, ori_samp_train_fname, model_fname, output_fname, learn_opt, classify_opt)
    dt_samp_class_list, dt_samp_prb_list = pytc.load_predictions_liblinear(output_fname)
    
    # Select important samples for reversion
    print 'Detecting polarity shift information...'
    # rank samples   
    id_list_rank = range(len(dt_samp_prb_list))
    id_list_rank.sort(key=lambda i: -max(dt_samp_prb_list[i].values())) # 按后验概率置信度排序
    id_list = id_list_rank[:int(perc * len(id_list_rank))] 
    print str(len(id_list)) + ' samples are selected for reversion...'

    print 'Constructing reverse samples...'

    rev_doc_terms_list_train, rev_doc_class_list_train, rev_doc_id_list_train = select_reverse_dataset(ori_doc_terms_list_train, ori_doc_class_list_train, antonym_dict, id_list)
    rev_doc_terms_list_test, rev_doc_class_list_test, rev_doc_id_list_test = reverse_dataset(ori_doc_terms_list_test, ori_doc_class_list_test, antonym_dict, True)
    print 'Saving reverse samples...'
    print len(rev_doc_terms_list_train), len(rev_doc_class_list_train)
    print len(rev_doc_terms_list_test), len(rev_doc_class_list_test)
    save_reverse_dataset(ori_doc_terms_list_train, ori_doc_class_list_train, rev_doc_terms_list_train, rev_doc_class_list_train, rev_doc_id_list_train, rev_data_dir + os.sep + 'train')
    save_reverse_dataset(ori_doc_terms_list_test, ori_doc_class_list_test, rev_doc_terms_list_test, rev_doc_class_list_test, rev_doc_id_list_test, rev_data_dir + os.sep + 'test')

def dual_training_demo(ori_data_dir, rev_data_dir, result_dir,classifier, bigram = False):
    print '\nDual Training and base classification...'
    dsa_o2or(ori_data_dir, rev_data_dir, result_dir, classifier, bigram)
    dsa_r2or(ori_data_dir, rev_data_dir, result_dir, classifier, bigram)
    dsa_d2or(ori_data_dir, rev_data_dir, result_dir, classifier, bigram)

def dual_prediction_demo(result_dir, param_list, classifier):
#    print '\nDual prediction and system ensemble...'
    ori_samp_fname_test = result_dir + os.sep + 'ori.test.samp'
    rev_samp_fname_test = result_dir + os.sep + 'rev.test.samp'
    o2o_output_fname = result_dir + os.sep + 'o2o.out'
    o2r_output_fname = result_dir + os.sep + 'o2r.out'
    r2o_output_fname = result_dir + os.sep + 'r2o.out'
    r2r_output_fname = result_dir + os.sep + 'r2r.out'
    d2o_output_fname = result_dir + os.sep + 'd2o.out'
    d2r_output_fname = result_dir + os.sep + 'd2r.out'
    ensemble_out_fname = result_dir + os.sep + 'ensemble.out'

    if classifier == 'liblinear':
        ori_samp_class_list_test = [int(x.split()[0]) for x in open(ori_samp_fname_test).readlines()]
        rev_samp_class_list_test = [int(x.split()[0]) for x in open(rev_samp_fname_test).readlines()]        
        o2o_samp_class_list_prd, o2o_samp_prb_list_prd = pytc.load_predictions_liblinear(o2o_output_fname)
        o2r_samp_class_list_prd, o2r_samp_prb_list_prd = pytc.load_predictions_liblinear(o2r_output_fname)
        r2o_samp_class_list_prd, r2o_samp_prb_list_prd = pytc.load_predictions_liblinear(r2o_output_fname)
        r2r_samp_class_list_prd, r2r_samp_prb_list_prd = pytc.load_predictions_liblinear(r2r_output_fname)    
        d2o_samp_class_list_prd, d2o_samp_prb_list_prd = pytc.load_predictions_liblinear(d2o_output_fname)
        d2r_samp_class_list_prd, d2r_samp_prb_list_prd = pytc.load_predictions_liblinear(d2r_output_fname)
    elif classifier == 'libsvm':
        ori_samp_class_list_test = [int(x.split()[0]) for x in open(ori_samp_fname_test).readlines()]
        rev_samp_class_list_test = [int(x.split()[0]) for x in open(rev_samp_fname_test).readlines()]
        o2o_samp_class_list_prd, o2o_samp_prb_list_prd = pytc.load_predictions_libsvm(o2o_output_fname)
        o2r_samp_class_list_prd, o2r_samp_prb_list_prd = pytc.load_predictions_libsvm(o2r_output_fname)
        r2o_samp_class_list_prd, r2o_samp_prb_list_prd = pytc.load_predictions_libsvm(r2o_output_fname)
        r2r_samp_class_list_prd, r2r_samp_prb_list_prd = pytc.load_predictions_libsvm(r2r_output_fname)    
        d2o_samp_class_list_prd, d2o_samp_prb_list_prd = pytc.load_predictions_libsvm(d2o_output_fname)
        d2r_samp_class_list_prd, d2r_samp_prb_list_prd = pytc.load_predictions_libsvm(d2r_output_fname)
    elif classifier == 'nb':
        ori_samp_class_list_test = [int(x.split()[0])-1 for x in open(ori_samp_fname_test).readlines()]
        rev_samp_class_list_test = [int(x.split()[0])-1 for x in open(rev_samp_fname_test).readlines()]
        o2o_samp_class_list_prd, o2o_samp_prb_list_prd = pytc.load_predictions_nb(o2o_output_fname)
        o2r_samp_class_list_prd, o2r_samp_prb_list_prd = pytc.load_predictions_nb(o2r_output_fname)
        r2o_samp_class_list_prd, r2o_samp_prb_list_prd = pytc.load_predictions_nb(r2o_output_fname)
        r2r_samp_class_list_prd, r2r_samp_prb_list_prd = pytc.load_predictions_nb(r2r_output_fname)
        d2o_samp_class_list_prd, d2o_samp_prb_list_prd = pytc.load_predictions_nb(d2o_output_fname)
        d2r_samp_class_list_prd, d2r_samp_prb_list_prd = pytc.load_predictions_nb(d2r_output_fname)
    
    acc_o2o = pytc.calc_acc(o2o_samp_class_list_prd, ori_samp_class_list_test)
    acc_r2r = pytc.calc_acc(r2r_samp_class_list_prd, rev_samp_class_list_test)
    acc_d2o = pytc.calc_acc(d2o_samp_class_list_prd, ori_samp_class_list_test)
    acc_d2r = pytc.calc_acc(d2r_samp_class_list_prd, rev_samp_class_list_test)
    
    if len(param_list) == 2:
        weight, conf = param_list[:2]
        dp_samp_class_list = dual_prediction_3conf(o2o_samp_class_list_prd, o2o_samp_prb_list_prd, d2o_samp_class_list_prd, d2o_samp_prb_list_prd, d2r_samp_class_list_prd, d2r_samp_prb_list_prd, weight, conf)
    elif len(param_list) == 4:
        dp_samp_class_list = dual_prediction_4comb(o2o_samp_class_list_prd, o2o_samp_prb_list_prd, r2r_samp_class_list_prd, r2r_samp_prb_list_prd, d2o_samp_class_list_prd, d2o_samp_prb_list_prd, \
                                                   d2r_samp_class_list_prd, d2r_samp_prb_list_prd, param_list)
    else:
        raise Exception("The number of input parameter is invalid.")
    acc_dsa = pytc.calc_acc(dp_samp_class_list, ori_samp_class_list_test)
    save_predictions_result(dp_samp_class_list, ensemble_out_fname)
    return acc_o2o, acc_r2r, acc_d2o, acc_d2r, acc_dsa

def start_demo():
    bigram, nltk = False, False
    options, args = getopt.getopt(sys.argv[1:], "hbnt:r:o:c:s:f:", ["help", "bigram", 'nltk', "token=", "reverse=", "output=", "classifier=", "select=", "fs_method="])
    for name,value in options:
        if name in ("-h","--help"):
            print usage()
            return
        if name in ("-b","--bigram"):
            bigram = True
        if name in ("-n","--nltk"):
            nltk = True
        if name in ("-t","--token"):
            token_data_dir = value
        if name in ("-r","--reverse"):
            rev_data_dir = value
        if name in ("-o","--output"):
            result_dir = value
        if name in ("-c","--classifier"):
            classifier = value
        if name in ("-s","--select"):
            perc = float(value)
        if name in ("-f","--fs_method"):
            fs_method = value
    param_list = map(float, args)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(rev_data_dir):
        os.makedirs(rev_data_dir)
    post_data_dir = os.getcwd() + os.sep + 'post'
    antonym_dict_fname = result_dir + os.sep + 'wn.dict'
    dataset_list = ['train', 'test']
    if nltk == True:
        post_tag_demo(token_data_dir, post_data_dir)
    else:
        pytc.pos_tag_reviews(token_data_dir, post_data_dir, FNAME_LIST)
    if 'fs_method' not in dir():
        wn_dict_demo(post_data_dir, token_data_dir, antonym_dict_fname, dataset_list)
    else:
        build_fs_dict(post_data_dir, result_dir, fs_method, dataset_list)
    if 'perc' in dir():
        select_reverse_review_demo(token_data_dir, rev_data_dir, result_dir, antonym_dict_fname, perc, bigram)
    else:
        reverse_review_demo(token_data_dir, rev_data_dir, antonym_dict_fname)
    dual_training_demo(token_data_dir, rev_data_dir, result_dir, classifier, bigram)

    acc_o2o, acc_r2r, acc_d2o, acc_d2r, acc_ensemble = dual_prediction_demo(result_dir, param_list, classifier)
    print '\nDone!\nacc_o2o:',acc_o2o, 'acc_r2r:', acc_r2r, 'acc_d2o:', acc_d2o, 'acc_d2r:', acc_d2r, 'acc_ensemble:', acc_ensemble
########## Other Functions ##########
def usage():
    usage = '''
    Usage: dsa.py [options] [paramaters]
    Options:  -h, --help, display the usage of the DSA commands
              -b, --bigram, if this paramater is set, it means use the unigram and
                    bigram features for sentiment classification, otherwise only use the
                    unigram features
              -n, --nltk, when this paramater is to set, it means using nltk as the POS tagging
                    tool, if not means POS tagging with the stanford-postagger.
              -t, --token path, the token data directory
              -r, --reverse path, the reverse samples directory
              -o, --output path, the directory to save the output files
              -c, --classifier [libsvm|liblinear|nb], the classifier toolkits used for sentiment classification, the value 
                   'libsvm', 'liblinear' and 'nb', correspond to libsvm classifier, logistic regression classifier and
                    Naive Bayes classifier respectively
              -s, --select ratio, the ratio of token samples selected to reverse. If not set, it means
                    to reverse all token samples
              -f, --fs_method [CHI|IG|LLR|MI|WLLR] The feature-selecting methods to constructing the pseudo-antonym
                    dictionary. If this paramater is not set, it means construct a antonym dictionary with wordnet
    paramaters:
           weight conferdence, two paramaters mean essemble a system with 3conf DSA
           weight weight weight weight, four paramaters mean essemble with four system(o2o, o2r, d2o, d2r)
           '''
    return  usage
def save_predictions_result(dp_samp_class_list, ensemble_out_fname):
    fout = open(ensemble_out_fname, 'w')
    for line in dp_samp_class_list:
        fout.write(str(line)+'\n')

if __name__ == '__main__':
    start_demo()
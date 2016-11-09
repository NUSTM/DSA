# coding: utf-8
''' PyTC Functions V4.40
Author: Rui Xia (rxia.cn@gmail.com)
Date: Last updated on 2016-11-01 by Leyi Wang
'''

import os, re, sys, random, math, subprocess
from nltk.stem import WordNetLemmatizer

is_win32 = (sys.platform == 'win32')
########### Global Parameters ###########
if is_win32:
    TOOL_PATH = 'F:\\NJUST\\Toolkits'
    NB_LEARN_EXE = TOOL_PATH + '\\openpr-nb_v1.16\\windows\\nb_learn.exe'
    NB_CLASSIFY_EXE = TOOL_PATH + '\\openpr-nb_v1.16\\windows\\nb_classify.exe'
    SVM_LEARN_EXE = TOOL_PATH + '\\svm_light\\svm_learn.exe'
    SVM_CLASSIFY_EXE = TOOL_PATH + '\\svm_light\\svm_classify.exe'
    LIBSVM_LEARN_EXE = TOOL_PATH + '\\libsvm-3.10\\windows\\svm-train.exe'
    LIBSVM_CLASSIFY_EXE = TOOL_PATH + '\\libsvm-3.21\\windows\\svm-predict.exe' 
    LIBLINEAR_LEARN_EXE = TOOL_PATH + '\\liblinear-2.1\\windows\\train.exe'
    LIBLINEAR_CLASSIFY_EXE = TOOL_PATH + '\\liblinear-2.1\\windows\\predict.exe'

else:
    TOOL_PATH = '/home/lywang/Toolkits'
    NB_LEARN_EXE = TOOL_PATH + '/openpr-nb_v1.16/nb_learn'
    NB_CLASSIFY_EXE = TOOL_PATH + '/openpr-nb_v1.16/nb_classify'
    SVM_LEARN_EXE = TOOL_PATH + '/svm_light/svm_learn'
    SVM_CLASSIFY_EXE = TOOL_PATH + '/svm_light/svm_classify'
    LIBSVM_LEARN_EXE = TOOL_PATH + '/libsvm-3.21/svm-train'
    LIBSVM_CLASSIFY_EXE = TOOL_PATH + '/libsvm-3.21/svm-predict' 
    LIBLINEAR_LEARN_EXE = TOOL_PATH + '/liblinear-2.1/train'
    LIBLINEAR_CLASSIFY_EXE = TOOL_PATH + '/liblinear-2.1/predict'

STANFORD_POSTAGGER_DIR = TOOL_PATH + os.sep + "stanford-postagger-2013-04-04"
LOG_LIM = 1E-300

########## PosTag Reviews ##########
def save_reviews(file_name, data_list, samp_tag):
    fout = open(file_name, 'w')
    fout.writelines(['<' + samp_tag + '>\n' + x + '\n</' + samp_tag + \
                     '>\n' for x in data_list])
    fout.close()

def format_post_data(post_data_fname, post_data_str, samp_tag = 'review_text'):
    patn = '<' + samp_tag + '>_[A-Z]+?\s(.*?)</' + samp_tag + '>_[A-Z]+?\s'
    post_data_list= re.findall(patn, post_data_str, re.S)
    save_reviews(post_data_fname, post_data_list, samp_tag)

def pos_tag_reviews(token_dir, post_dir, fname_list):
    print 'Building Pos-tagging Reviews ...'
    for fold in ['train','test']:
        if not os.path.exists(post_dir + os.sep + fold):
            os.makedirs(post_dir + os.sep + fold)
        for fname in fname_list:
            token_data_fname = token_dir + os.sep + fold + os.sep + fname
            post_data_fname = post_dir + os.sep + fold + os.sep + fname
            cmd = 'java -mx300m -cp ' + STANFORD_POSTAGGER_DIR + os.sep + 'stanford-postagger.jar' + os.pathsep + STANFORD_POSTAGGER_DIR+os.sep+'lib' + os.sep + '* edu.stanford.nlp.tagger.maxent.MaxentTagger -model '\
                    + STANFORD_POSTAGGER_DIR + os.sep+'models' + os.sep + 'english-left3words-distsim.tagger -textFile ' + token_data_fname
            pop = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell = True)
            stdout, stderr = pop.communicate()
            if pop.wait() != 0:
                    raise Exception('Check the path of STANFORD_POSTAGGER_DIR, and make sure the java path environment variable is set!')
            format_post_data(post_data_fname, stdout)

########## File Access Fuctions ##########
def read_text_f2(fname_list, samp_tag):
    '''text format 2: one class one file, docs are sperated by samp_tag
    '''
    doc_class_list = []
    doc_str_list = []
    for fname in fname_list: # for fname in sorted(fname_list):
        # print 'Reading', fname
        doc_str = open(fname, 'r').read()
        patn = '<' + samp_tag + '>(.*?)</' + samp_tag + '>'
        str_list_one_class = re.findall(patn, doc_str, re.S)
        class_label = os.path.basename(fname)
        doc_str_list.extend(str_list_one_class)
        doc_class_list.extend([class_label] * len(str_list_one_class))
    doc_str_list = [x.strip() for x in doc_str_list]
    return doc_str_list, doc_class_list

def save_text_f2(save_dir, samp_tag, doc_str_list, doc_class_list):
    '''text format 2: one class one file, docs are sperated by samp_tag
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    class_set = sorted(list(set(doc_class_list)))
    doc_str_class = [''] * len(class_set)
    for k in range(len(doc_class_list)):
        class_id = class_set.index(doc_class_list[k])
        #doc_str = ' '.join(doc_terms_list[k])
        doc_str = doc_str_list[k]
        doc_str_class[class_id] += ('<' + samp_tag + '>\n' + doc_str + \
            '</' + samp_tag + '>\n')
    for class_id in range(len(class_set)):
        class_label = class_set[class_id]
        fobj = open(save_dir + os.sep + class_label, 'w')
        fobj.write(doc_str_class[class_id])
        fobj.close()

########## Feature Extraction Fuctions ##########
def get_doc_terms_list(doc_str_list):
    return [x.split() for x in doc_str_list]

def get_doc_bis_list(doc_str_list):
    unis_list = [x.split() for x in doc_str_list]
    doc_bis_list = []
    for k in range(len(doc_str_list)):
        unis = unis_list[k]
        if len(unis) <= 1:
            doc_bis_list.append([])
            continue
        unis_shift = unis[1:] + [unis[0]]
        bis = [unis[j] + '<w-w>' + unis_shift[j] for j in \
            range(len(unis))][0:-1]
        doc_bis_list.append(bis)
    return doc_bis_list

def get_joint_sets(doc_terms_list1, doc_terms_list2):
    joint_list = []
    for k in range(len(doc_terms_list1)):
        doc_terms1 = doc_terms_list1[k]
        doc_terms2 = doc_terms_list2[k]
        joint_list.append(doc_terms1 + doc_terms2)
    return joint_list

########## Text Statistic Fuctions ##########
def get_class_set(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    return class_set

def get_term_set(doc_terms_list):
    term_set = set()
    for doc_terms in doc_terms_list:
        term_set.update(doc_terms)
    return sorted(list(term_set))

def stat_df_term(term_set, doc_terms_list):
    '''
    df_term is a dict
    '''
    df_term = {}.fromkeys(term_set, 0)
    for doc_terms in doc_terms_list:
#        cand_terms = set(term_set) & set(doc_terms) # much more cost!!!
        for term in set(doc_terms):
            if df_term.has_key(term):
                df_term[term] += 1
    return df_term

def stat_tf_term(term_set, doc_terms_list):
    '''
    tf_term is a dict
    '''
    tf_term = {}.fromkeys(term_set, 0)
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            if tf_term.has_key(term):
                tf_term[term] += 1
    return tf_term

def stat_df_class(class_set, doc_class_list):
    '''
    df_class is a list
    '''
    df_class = [doc_class_list.count(x) for x in class_set]
    return df_class

def stat_df_term_class(term_set, class_set, doc_terms_list, doc_class_list):
    '''
    df_term_class is a dict-list

    '''
    class_id_dict = dict(zip(class_set, range(len(class_set))))
    df_term_class = {}
    for term in term_set:
        df_term_class[term] = [0]*len(class_set)
    for k in range(len(doc_class_list)):
        class_label = doc_class_list[k]
        class_id = class_id_dict[class_label]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            if df_term_class.has_key(term):
                df_term_class[term][class_id] += 1
    return df_term_class

########## Feature Selection Functions ##########
def supervised_feature_selection(df_class, df_term_class, fs_method='IG',
                                 fs_num=0, fs_class=-1):
    if fs_method == 'MI':
        term_set_fs, term_score_dict = feature_selection_mi(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'IG':
        term_set_fs, term_score_dict = feature_selection_ig(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'CHI':
        term_set_fs, term_score_dict = feature_selection_chi(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'WLLR':
        term_set_fs, term_score_dict = feature_selection_wllr(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'LLR':
        term_set_fs, term_score_dict = feature_selection_llr(df_class, \
            df_term_class, fs_num, fs_class)
    return term_set_fs, term_score_dict

def feature_selection_mi(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            p_c_t = (cap_a + 1.0) / (cap_a + cap_b + class_set_size)
            p_c = float(cap_a+cap_c) / cap_n
            score = math.log(p_c_t / p_c)
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def feature_selection_ig(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            p_c = float(cap_a + cap_c) / cap_n
            p_t = float(cap_a + cap_b) / cap_n
            p_nt = 1 - p_t
            p_c_t = (cap_a + 1.0) / (cap_a + cap_b + class_set_size)
            p_c_nt = (cap_c + 1.0) / (cap_c + cap_d + class_set_size)
            score = - p_c * math.log(p_c) + p_t * p_c_t * math.log(p_c_t) + \
                p_nt * p_c_nt * math.log(p_c_nt)
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def feature_selection_chi(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            cap_nu = float(cap_a * cap_d - cap_c * cap_b)
            cap_x1 = cap_nu / ((cap_a + cap_c) * (cap_b + cap_d))
            cap_x2 = cap_nu / ((cap_a+cap_b) * (cap_c+cap_d))
            score = cap_nu * cap_x1 * cap_x2
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def feature_selection_wllr(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
#            doc_set_size = len(df_class)
        cap_n = sum(df_class)
        term_set_size = len(df_term_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            p_t_c = (cap_a + 1E-6) / (cap_a + cap_c + 1E-6 * term_set_size)
            p_t_not_c = (cap_b + 1E-6)/(cap_b + cap_d + 1E-6 * term_set_size)
            score = p_t_c * math.log(p_t_c / p_t_not_c)
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def feature_selection_llr(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            p_c_t = (cap_a + 1.0)/(cap_a + cap_b + class_set_size)
            p_nc_t = 1 - p_c_t
            p_c = float(cap_a + cap_c)/ cap_n
            p_nc = 1 - p_c
            score = math.log(p_c_t * p_nc / (p_nc_t * p_c))
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

def save_term_score(term_score_dict, fname):
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    fout = open(fname, 'w')
    for term_score in term_score_list:
        fout.write(term_score[0] + '\t' + str(term_score[1]) + '\n')
    fout.close()

def load_term_score(fname):
    term_score_dict = {}
    for line in fname:
        term_score = line.strip().split('\t')
        term_score_dict[term_score[0]] = term_score[1]
    return term_score_dict


########## Sample Building ##########
def build_samps(term_dict, class_dict, doc_terms_list, doc_class_list,
                term_weight, idf_term=None):
    '''Building samples with sparse format
    term_dict -- term1: 1; term2:2; term3:3, ...
    class_dict -- negative:1; postive:2; unlabel:0
    '''
    samp_dict_list = []
    samp_class_list = []
    for k in range(len(doc_class_list)):
        doc_class = doc_class_list[k]
        samp_class = class_dict[doc_class]
        samp_class_list.append(samp_class)
        doc_terms = doc_terms_list[k]
        samp_dict = {}
        for term in doc_terms:
            if term_dict.has_key(term):
                term_id = term_dict[term]
                if term_weight == 'BOOL':
                    samp_dict[term_id] = 1
                elif term_weight == 'TF':
                    if samp_dict.has_key(term_id):
                        samp_dict[term_id] += 1
                    else:
                        samp_dict[term_id] = 1
                elif term_weight == 'TFIDF':
                    if samp_dict.has_key(term_id):
                        samp_dict[term_id] += idf_term[term]
                    else:
                        samp_dict[term_id] = idf_term[term]
        samp_dict_list.append(samp_dict)
    return samp_dict_list, samp_class_list

def save_samps(samp_dict_list, samp_class_list, fname, feat_num=0):
    length = len(samp_class_list)
    fout = open(fname, 'w')
    for k in range(length):
        samp_dict = samp_dict_list[k]
        samp_class = samp_class_list[k]
        fout.write(str(samp_class) + '\t')
        for term_id in sorted(samp_dict.keys()):
            if feat_num == 0 or term_id < feat_num:
                fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
        fout.write('\n')
    fout.close()

def load_samps(fname, fs_num=0):
    fsample = open(fname, 'r')
    samp_class_list = []
    samp_dict_list = []
    for strline in fsample:
        samp_class_list.append(strline.strip().split()[0])
        if fs_num > 0:
            samp_dict = dict([[int(x.split(':')[0]), float(x.split(':')[1])] \
                for x in strline.strip().split()[1:] if int(x.split(':')[0]) \
                < fs_num])
        else:
            samp_dict = dict([[int(x.split(':')[0]), float(x.split(':')[1])] \
                for x in strline.strip().split()[1:]])
        samp_dict_list.append(samp_dict)
    fsample.close()
    return samp_dict_list, samp_class_list


########## Classification Functions ##########
def nb_exe(fname_samp_train, fname_samp_test, fname_model, fname_output,
           learn_opt='', classify_opt='-f 2'):
    print '\nNB executive classifing...'
    pop = subprocess.Popen(NB_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model, shell=True)
    pop.wait()
    pop = subprocess.Popen(NB_CLASSIFY_EXE + ' ' + classify_opt + ' ' + \
        fname_samp_test + ' ' + fname_model + ' ' + fname_output, shell=True)
    pop.wait()
    samp_class_list_test = [x.split()[0] for x in \
        open(fname_samp_test).readlines()]
    samp_class_list_nb = [x.split()[0] for x in \
        open(fname_output).readlines()]
#    neg_scores = sorted([float(x.split()[1]) for x in \
#        open(fname_output).readlines()])
#    pos_scores = sorted([float(x.split()[2]) for x in \
#        open(fname_output).readlines()])
#    print 'NEG\n', neg_scores
#    print 'POS\n', pos_scores
    acc = calc_acc(samp_class_list_nb, samp_class_list_test)
    return acc

def libsvm_exe(fname_samp_train, fname_samp_test, fname_model, fname_output,
               learn_opt='-t 0 -c 1 -b 1', classify_opt='-b 1'):
    print '\nLibSVM executive classifing...'
    pop = subprocess.Popen(LIBSVM_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model, shell=True)
    pop.wait()
    pop = subprocess.Popen(LIBSVM_CLASSIFY_EXE + ' ' + classify_opt + ' ' + \
        fname_samp_test + ' ' + fname_model + ' ' + fname_output, shell=True)
    pop.wait()
    samp_class_list_test = [x.split()[0] for x in \
        open(fname_samp_test).readlines()]
    samp_class_list_svm = [x.split()[0] for x in \
        open(fname_output).readlines()[1:]]
    acc = calc_acc(samp_class_list_svm, samp_class_list_test)
    return acc

def liblinear_exe(fname_samp_train, fname_samp_test, fname_model, fname_output,
                  learn_opt='-s 7 -c 1', classify_opt='-b 1'):
    print '\nLiblinear executive classifing...'
    pop = subprocess.Popen(LIBLINEAR_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model, shell=True)
    pop.wait()
    pop = subprocess.Popen(LIBLINEAR_CLASSIFY_EXE + ' ' + classify_opt + ' ' \
        + fname_samp_test + ' ' + fname_model + ' ' + fname_output, shell=True)
    pop.wait()
    samp_class_list_test = [x.split()[0] for x in \
        open(fname_samp_test).readlines()]
    samp_class_list_svm = [x.split()[0] for x in \
        open(fname_output).readlines()[1:]]
    acc = calc_acc(samp_class_list_svm, samp_class_list_test)
    return acc

def load_predictions_nb(prd_fname):
    samp_class_list = []
    samp_prb_list = []
    for line in open(prd_fname):
        samp_class_list.append(int(line.split()[0])-1)
        samp_prb = dict()        
        for term in line.split()[1:]:
            samp_prb[int(term.split(':')[0])] = float(term.split(':')[1])
        samp_prb_list.append(samp_prb)
    return samp_class_list, samp_prb_list

def load_predictions_liblinear(prd_fname):
    samp_class_list = []
    samp_prb_list = []
    class_id = [int(x) for x in open(prd_fname).readlines()[0].split()[1:]]
    for line in open(prd_fname).readlines()[1:]:
        samp_class_list.append(int(line.split()[0]))
        samp_prb = dict(zip(class_id, [float(x) for x in line.split()[1:]]))
        samp_prb_list.append(samp_prb)
    return samp_class_list, samp_prb_list

def load_predictions_libsvm(prd_fname):
    samp_class_list = []
    samp_prb_list = []
    class_id = [int(x) for x in open(prd_fname).readlines()[0].split()[1:]]
    for line in open(prd_fname).readlines()[1:]:
        samp_class_list.append(int(line.split()[0]))
        samp_prb = dict(zip(class_id, [float(x) for x in line.split()[1:]]))
        samp_prb_list.append(samp_prb)
    return samp_class_list, samp_prb_list
########## Evalutation Functions ##########
def calc_acc(labellist1, labellist2):
    if len(labellist1) != len(labellist2):
        print 'Error: different lenghts!'
        return 0
    else:
        samelist = [int(x == y) for (x, y) in zip(labellist1, labellist2)]
        acc = float((samelist.count(1)))/len(samelist)
        return acc
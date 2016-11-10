# Dual Sentiment Analysis Toolkit

By Rui Xia, Nanjing University of Science & Technology, China

# Table of Contents

- Introduction
- Citation
- Configuration
- Data Structures
- Usage
- Examples

# Introduction

This code is designed to implement the approach DTDP, an noval and effective method proposed to solve the polarity shift problem of sentiment classfication. For details of DTDP, please refer to [1]. 
This system uses the WordNet to generate an antonym dictionary on the training data with POS tags, and then we use this dictionary to create sentiment-reverse reviews for data expasion according to the reversed
rules. In DT, the classifier is learnt to maximizing a combination of likehoods of the original and reversed
training data. In DP, predictions are made by considering two side of one review.

# Citation

If you use this package, please cite the following work:

[1] Rui Xia, Feng Xu, Chengqing Zong, Qianmu Li, Yong Qi, and Tao Li. Dual Sentiment Analysis: Considering Two Sides of One Review. IEEE Transactions on Knowledge and Data Engineering, vol. 27, no. 8, pp. 2120-2133, 2015.

[2] Rui Xia, Tao Wang, Xuelei Hu, Shoushan Li, and Chengqing Zong. Dual Training and Dual Prediction for Polarity Classification. Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (ACL), pp. 521-525, 2013.

# Configuration

This system can be used on Linux/Unix or Windows:

Toolkits in need:

- Classifier
  1. Logistic regression classfier: liblinear <http://www.csie.ntu.edu.tw/~cjlin/libsvm>
  2. SVM: libsvm <http://www.csie.ntu.edu.tw/~cjlin/libsvm>
  3. Navie bayes: OpenPR_NB  <http://msrt.njust.edu.cn/staff/rxia/>


- NLTK

  You can generate the antonym by the wordnet module of nltk. In our experiment, version nltk-2.0.4 is in use.

  1. First Pos tagger the token data with nltk. 
  2. Generate the antonym by the wordnet module of nltk: Install nltk first.

- Stanford Log-linear Part-Of-Speech Tagger

  The pos tagger with nltk may take a long of time if your token data in huge. So we provide an interface to use the Stanford Part-Of-Speech Tagger.

  The Speech Tagging tool:http://nlp.stanford.edu/software/tagger.shtml

Try to configure the path of these tookits above before executing:

- In the module 'pytc.py': configure the path of these tookits metioned above.

# Data Structures

## Data

Both in training data and test data, every review should start with the tagger '\<review_text>' and ends with '\</review_text>', just like '\<review_text>' sampl1 '\</review_text>';
-Token data without POS tags: Tranning data and test data without POS tags.

### Directory

- Token data directiory: In this dictionary, there are two subdirectories named "train", "test" respectively.

  Each of these subdirectories,contains two files, file named "negative" for negative reviews and "postive" for postive reviews. And all of these reviews are without POS tags.


- Reversed data directiory: This dictionary is for reversed data. 
- Result directiory: The output path of the result.

Note:

Labels in the testing file are only used to calculate accuracy or errors. 

If they are unknown, just put all of the data in 'negative' file and 'positive' files separately.

# Usage

```powershell
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
          -c, --classifier [libsvm|liblinear|nb], the classifier toolkits used for sentiment
                classification, the value 'libsvm', 'liblinear' and 'nb', correspond to libsvm
                classifier, logistic regression classifier and Naive Bayes classifier
                respectively
          -s, --select ratio, the ratio of token samples selected to reverse. If not set, it
                means to reverse all token samples
          -f, --fs_method [CHI|IG|LLR|MI|WLLR] The feature-selecting methods to constructing the
                pseudo-antonym dictionary. If this paramater is not set, it means construct a
                antonym dictionary with wordnet
Paramaters:
       weight conferdence, two paramaters mean essemble a system with 3conf DSA
       weight weight weight weight, four paramaters mean essemble with four system(o2o, o2r, d2o, d2r)
```

# Examples

Code module:dsa.py 

On Dos of windows system or shell of linux system, input the follow command:

- 3Conf_system:

  ```powershell
  $ python dsa.py -n -t data/kitchen -r reverse -o result -c liblinear -s 0.95 0.8 -0.1
  $ python dsa.py -b -n -t data/kitchen -r reverse -o result -c liblinear -s 0.95 0.8 -0.1
  ```
  Note:

  ```powershell
   data/kitchen: The directiory of token data without POS tags
    reverse: The directiory of token data reversed 
    result: The directory to save the output files
    liblinear: The classifier used for sentiment classification
    0.8: The weight of d2o(To use original training data and reversed data to predict original test data),d2r(To use original data and reversed training data to predict reversed test data)
    0.1: The confidence of d2d(To use the ensemble prediction of d2o and d2r), o2o 
  ```

In addition, we generalize dsa to a four system ensemble, using o2o, o2r, d2o, d2r predictions:

- 4Com_system:

  ```powershell
  $ python dsa.py -n -t data/kitchen -r reverse -o result -c liblinear -s 0.95 0.4 0.1 0.1 0.4
  $ python dsa.py -b -n -t data/kitchen -r reverse -o result -c liblinear -s 0.95 0.4 0.1 0.1 0.4
  ```

  Note: 

  ```powershell
  data/kitchen: Data without POS tags directiory
  reverse: Reversed data directiory
  result: Result directiory
  liblinear: Representation of classfier, you can choose 'libsvm', 'liblinear' or 'nb', corresponding libsvm classifier, logistic regression classifier and Naive Bayes classifier respectively
  0.4: The weight of o2o prediction
  0.1: The weight of r2r prediction
  0.1: The weight of d2o prediction
  0.4: The weight of d2r prediction
  ```

  ​



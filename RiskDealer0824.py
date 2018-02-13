# -*- coding: utf-8 -*-

import os
os.chdir('C:/DAI/442BMWDealershipRiskManagement/rawdata')
print os.getcwd()

# read labeled data from local disk
import pandas as pd

# raw1 = pd.read_excel('filtered.xlsx',sheetname='sheet1')
# raw2 = pd.read_excel('filtered.xlsx',sheetname='sheet2')
# labeled_data = pd.concat([raw1[['id', 'post_content', 'dealer','is_risk']],raw2[['id', 'post_content', 'dealer','is_risk']]])
# labeled_data.to_csv("sampledata.csv", encoding='utf-8', index= False )

labeled_data = pd.read_csv("sampledata.csv")
# print(labeled_data.head(10))

# write txt files which contain all the risk and not risk post_content
riskdata = labeled_data[ labeled_data ['is_risk']== 1]
noriskdata = labeled_data[ labeled_data ['is_risk']!= 1]
# riskpost=riskdata[['post_content']].reset_index(drop=True) #339
# noriskpost=noriskdata[['post_content']].reset_index(drop=True) #4567
# riskpost.to_csv("riskpost.txt", sep= '\t',  encoding='utf-8', header= False, index= False )
# noriskpost.to_csv("noriskpost.txt", sep= '\t',  encoding='utf-8', header= False, index= False )

# read unlabeled data / new posts info from local file
new_data = pd.read_csv('C:/DAI/442BMWDealershipRiskManagement/rawdata/autocrawl_posts_201704110856.csv')
unlabeled_data = new_data[['id', 'post_date', 'post_content',  'dealer']]
udataP=unlabeled_data.iloc[ :40,:] # only take first 40 rows of new posts to save some time
# newpost = unlabeled_data[['post_content']].reset_index(drop=True) # all rows of new posts
# # newpost.to_csv("newpost.txt", sep='\t', encoding='utf-8', header=False, index=False)

#
# Step 1 : Feature Extraction and Selection
##### 1.1.Feature Extraction using jieba cut and drop stop words
# 1.定义结巴分词函数，返回分词列表如：[['我','爱','北京','天安门'],['你','好'],['hello']]，list ['我','爱','北京','天安门']代表一条评论
import jieba
def read_post(df):
    stop = [line.strip().decode('utf-8') for line in open('C:/DAI/Sentiment analysis/RiskyDealers/labeled data/stop.txt', 'r').readlines()]  # 停用词列表
    posts = list(df.post_content)
    str = []  # 空白列表
    for post in posts:
        fenci = jieba.cut((post+'\n').decode('utf-8'), cut_all=False)
        str.append(list(set(fenci) - set(stop)))
    return str
# posts=read_post(riskdata)
# print(len(posts))
# for post in posts:
#     print(u" " + " ".join(post))

#### 1.2  Feature Selection using information gain
# 一般来说，太多的特征会降低分类的准确度，所以需要使用一定的方法，来“选择”出信息量最丰富的特征，再使用这些特征来分类。
# 特征选择遵循如下步骤：
# 1. 计算出整个语料里面每个词/token的信息量
# 2. 根据信息量进行倒序排序，选择排名靠前的信息量的词
# 3. 把这些词作为特征
from nltk.probability import  FreqDist,ConditionalFreqDist
from nltk.metrics import  BigramAssocMeasures
# 获取信息量最高(前number个)的特征(卡方统计, chi-square)
def jieba_feature(number):
     riskWords = []
     noriskWords = []
     for items in read_post(riskdata): ## 循环遍历每一条评论的list, e.g., ['我','爱','北京','天安门']
         for item in items: # 循环遍历本评论中的每一个分词
            riskWords.append(item) # 把token读入riskWords(list)
     for items in read_post(noriskdata):
         for item in items:
            noriskWords.append(item)
     word_fd = FreqDist() # 统计所有词的词频, 建立 FreqDist 对象，,FreqDist继承自dict，所以我们可以像操作字典一样操作FreqDist对象
     cond_word_fd = ConditionalFreqDist() # 统计risk文本中的词频和norisk文本中的词频，
     for word in riskWords:
         word_fd[word] += 1 # word is the key of the dict, the value is the freq of key(word)
         cond_word_fd['risk'][word] += 1
     for word in noriskWords:
         word_fd[word] += 1 # overall frequency
         cond_word_fd['norisk'][word] += 1 # frequency within each class
     risk_word_count = cond_word_fd['risk'].N() # risky词的数量
     norisk_word_count = cond_word_fd['norisk'].N() # noRisky词的数量
     total_word_count = risk_word_count + norisk_word_count
     word_scores = {}# 包括了每个词和这个词的信息量dict
     risk_scores = {}
     for word, freq in word_fd.items():# overall frequency
         risk_score = BigramAssocMeasures.chi_sq(cond_word_fd['risk'][word],  (freq, risk_word_count), total_word_count) # risk词的卡方统计量，这里也可以计算互信息等其它统计量
         norisk_score = BigramAssocMeasures.chi_sq(cond_word_fd['norisk'][word],  (freq, norisk_word_count), total_word_count) # 同理
         word_scores[word] = risk_score + norisk_score # 一个词的信息量等于risk卡方统计量加上norisk卡方统计量，word_scores包括了每个词和这个词的信息量
         risk_scores[word] = risk_score
  # 字典类型的排序，可用lambda表达式来排序，用sorted函数的key= 参数排序：按照value进行排序 item[1]
     best_vals = sorted(word_scores.items(), key=lambda item:item[1],  reverse=True)[:number] # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
     best_words = set([w for w,s in best_vals])
     return dict([(word, True) for word in best_words])

# riskywords=jieba_feature(60)
# print(u" " + " ".join(riskywords.keys()))

#
## Step 2 : 构建训练需要的数据格式：
#[[{'买': 'True', '京东': 'True', '物流': 'True', '包装': 'True', '\n': 'True', '很快': 'True', '不错': 'True', '酒': 'True', '正品': 'True', '感觉': 'True'},  1],
# [{'买': 'True', '\n':  'True', '葡萄酒': 'True', '活动': 'True', '澳洲': 'True'}, 1],
# [{'\n': 'True', '价格': 'True'}, 1]]
def build_features(fnumber):
     feature = jieba_feature(fnumber)# 结巴分词, 取1500个features 以字典形式存储
     riskFeatures = [] # [{'买': 'True', '\n':  'True', '葡萄酒': 'True', '活动': 'True', '澳洲': 'True'}, 'risk']
     for items in read_post(riskdata): ## 循环遍历每一个评论的向量，e.g., ['我','爱','北京','天安门']
         a = {}
         for item in items: ## 遍历每一个分词
            if item in feature.keys():
                a[item]='True'
         riskWords = [a,1] # 为risk文本/评论赋予"risk"
         riskFeatures.append(riskWords)
     noriskFeatures = []
     for items in read_post(noriskdata):
         a = {}
         for item in items:
            if item in feature.keys():
                a[item]='True'
         noriskWords = [a, 0] #为norsik文本赋予"norisk"
         noriskFeatures.append(noriskWords)
     newpostFeatures = []
     for items in read_post(udataP):
        a = {}
        for item in items:
           if item in feature.keys():
               a[item] = 'True'
        postWords = [a, 'unknown']  # 为norsik文本赋予"norisk"
        newpostFeatures.append(postWords)
     # return riskFeatures, noriskFeatures, newpostFeatures
     labeledpostFeatures = []
     for items in read_post(labeled_data):
        a = {}
        for item in items:
           if item in feature.keys():
               a[item] = 'True'
        postWords = [a, 'unknown']  # 为norsik文本赋予"norisk"
        labeledpostFeatures.append(postWords)
     return riskFeatures, noriskFeatures, newpostFeatures, labeledpostFeatures
# print(noriskFeatures[:10])
# print(len(riskFeatures))#339
# print(len(noriskFeatures))#4567
# print(newpostFeatures)

# Step 3 : split data into training and testing data
riskFeatures, noriskFeatures, newpostFeatures,labeledpostFeatures  = build_features(1500)
from random import shuffle
shuffle(riskFeatures) #把文本的排列随机化
shuffle(noriskFeatures) #把文本的排列随机化
# ---- case1: risk:norisk=1:5, class imbalance
train = riskFeatures[:255] + noriskFeatures[:255] # 训练集(75%)
test = riskFeatures[255: ]+ noriskFeatures[255:340] # 验证集(25%)+ noriskFeatures[255*5:340*5]
 # test = riskFeatures[250:]+ noriskFeatures[1250: ]
# ----case2: risk:norisk=1:5, class imbalance
# train = riskFeatures[:238] + noriskFeatures[:238*5] # train (70%)
# devtest = riskFeatures[238:289]+ noriskFeatures[238*5:(238+51)*5 ] # validation(15%)
# test = riskFeatures[289:]+ noriskFeatures[(238+51)*5:340*5 ] # test(15%)
# dev, tag_dev = zip(*devtest)
data, tag = zip(*test)
# print(len(tag))
newdata = newpostFeatures # 新的预测集
data1, tag1 = zip(*newdata)
# print(len(data1))
# labeledpostFeatures = riskFeatures + noriskFeatures
orgdata= labeledpostFeatures # 原始数据集
data2, tag2 = zip(*orgdata)

# Step 4 : build text classifier and calculate accuracy
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score

# #----------------------------------model and feature no selection block-------------------------------------
# def score(classifier):
#     classifier = SklearnClassifier(classifier)  # 在nltk中使用scikit-learn的接口
#     classifier.train(train)  # 训练分类器
#     pred = classifier.classify_many(dev)  # 对测试集的数据进行分类，给出预测的标签
#     return (roc_auc_score(tag_dev, pred))
# feature_no=[500,1000,1500,2000]
# for fnumber in feature_no:
#     riskFeatures, noriskFeatures, newpostFeatures = build_features(fnumber)  # 获得合适格式的训练数据
#     labeledpostFeatures = riskFeatures + noriskFeatures
#     train = riskFeatures[:255] + noriskFeatures[:255*5] # 训练集(75%)
#     test = riskFeatures[255: ]+ noriskFeatures[:340*5 ] # 验证集(25%)
#     dev, tag_dev = zip(*test)
#     print"Feature no %f" %fnumber
#     print('BernoulliNB`s auc is %f' % score(BernoulliNB()))  # 先验为伯努利分布的朴素贝叶斯
#     print('MultinomiaNB`s auc is %f' % score(MultinomialNB()))  # 先验为多项式分布的朴素贝叶斯
#     print('LinearSVC`s auc is %f' % score(LinearSVC()))  # linear support vector classifier
#     print('RandomForest`s auc is %f' % score(RandomForestClassifier()))
#     print('AdaBoost`s auc is %f' % score(AdaBoostClassifier()))
# #----------------------------------model and feature no selection block-------------------------------------

# def score(classifier):
#      classifier = SklearnClassifier(classifier) # 在nltk中使用scikit-learn的接口
#      classifier.train(train)  # 训练分类器
#      pred = classifier.classify_many(data)  # 对测试集的数据进行分类，给出预测的标签
#      scores=[roc_auc_score(tag, pred), accuracy_score(tag, pred),precision_score(tag, pred), recall_score(tag, pred),f1_score(tag, pred)]
#      riskProb=[pdist.prob(1) for pdist in classifier.prob_classify_many(data)]
#      riskinfo=[pred,  riskProb ]
#      pred1 = classifier.classify_many(data1)  # 对新数据（post）进行分类，给出预测的标签
#      riskProb1=[pdist.prob(1) for pdist in classifier.prob_classify_many(data1)]
#      predinfo=[pred1,  riskProb1 ]
#      return (scores, riskinfo, predinfo)#

# # print(score(MultinomialNB())[1][0])
# # print(score(MultinomialNB())[1][1])
# # print(score(MultinomialNB())[2][0])
# # print(score(MultinomialNB())[2][1])
#
# # udataP['is_risk']=score(MultinomialNB())[2][0]
# # udataP['risk_confidence']=score(MultinomialNB())[2][1]
# # print(udataP.head())
#
#
# #step 5: model evaluation measures and curves

# print('Auc is %f' %(score(MultinomialNB())[0][0]) )
# print('Accuracy is %f' %(score(MultinomialNB())[0][1]) )
# print('Precision is %f' %(score(MultinomialNB())[0][2]) )
# print('Recall/True Positive rate is %f' %(score(MultinomialNB())[0][3]) )
# print('F1 measure is %f' %(score(MultinomialNB())[0][4]) )

# #-------------------------plot ROC curve  only for balanced dataset-----------------------------
# import matplotlib.pyplot as plt
# classifier = SklearnClassifier(MultinomialNB())  # 在nltk中使用scikit-learn的接口
# classifier.train(train)
# pred = classifier.classify_many(data)#Target
# probs=[pdist.prob(1) for pdist in classifier.prob_classify_many(data)]#Predicted probability column of event (interesting class)
# # calculate the fpr and tpr for all thresholds of the classification
# fpr, tpr, threshold = roc_curve(tag, probs)
# roc_auc = auc(fpr, tpr)
# # print(len(threshold))#50
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# #------------------ plot K-S curve only for balanced dataset-------------------
# #http://www.360doc.com/content/17/0408/09/20558639_643798853.shtml
# import matplotlib.pyplot as plt
# classifier = SklearnClassifier(MultinomialNB())  # 在nltk中使用scikit-learn的接口
# classifier.train(train)
# pred = classifier.classify_many(data)#Target
# probs=[pdist.prob(1) for pdist in classifier.prob_classify_many(data)]
# from sklearn import metrics
# KSdata = {'1': tag, '2': probs}
# df = pd.DataFrame(KSdata)
# df.columns = ['actual',  'probability']
# df.sort_values('probability', ascending=False)
# # print(type(df.shape))tuple
# actual = list(df.actual)
# probability = list(df.probability)
# predict = [0]*len(actual)
# df['predict'] = predict
# TPRs = [0]
# FPRs = [0]
# rows = [0]
# for k in range(9):
#     threshold=0.1*(k+1)
#     # predict = [0] * len(actual)
#     df.predict = predict
#     # print(threshold)
#     df.ix[df.probability>threshold, 'predict'] = 1
#     confusion= metrics.confusion_matrix(df.predict, df.actual)
#     # print(confusion)
#     tn, fp, fn, tp = confusion.ravel() # binary class
#     TPR = float(tp)/(tp+ fp)
#     TPR = metrics.recall_score(df.predict, df.actual)
#     FPR = float(fp)/(fp+tn)
#     TPRs.append(TPR)
#     FPRs.append(FPR)
#     rows.append(0.1*(k+1))
# TPRs.append(1)
# FPRs.append(1)
# rows.append(1)
# # print(TPRs)
# # print(FPRs)
# # print(rows)
#
# plt.title('K-S curve')
# plt.plot(rows,TPRs, 'g-',label='TPR')
# plt.plot(rows,FPRs,'r-',label='FPR')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.legend()
# plt.show()
#
# # ks = max( TPRs-FPRs )
# # print('ks statictic is %f' %ks)



# def score(classifier):
#     classifier = SklearnClassifier(classifier)
#     classifier.train(train)
#     data0, tag0 = zip(*train)
#     pred0 = classifier.classify_many(data0)
#     n = 0
#     s = len(pred0)
#     for i in range(0,s):
#          if pred0[i]==tag0[i]:
#             n = n+1
#     return float(n)/s
# print(score(MultinomialNB())) # estimation accuracy on itself(train) is around 0.94

# def score(classifier):
#     classifier = SklearnClassifier(classifier)
#     classifier.train(train)  # 训练分类器
#     pred2 = classifier.classify_many(data2)  # 对原始数据（labeled_post）进行分类，给出预测的标签
#     return (pred2)
# labeled_data['risk_pred']=score(MultinomialNB())
# lpost=labeled_data [['id','is_risk','risk_pred']].reset_index(drop=True)
# print float(len(lpost[lpost.is_risk==lpost.risk_pred]))/len(lpost) # estimation accuracy on labeled data is 0.96
# # lpost.to_csv("predlabeled.csv", sep= ',', encoding='utf-8' )

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('spam.csv')


# In[3]:


df=pd.read_csv('spam.csv')


# In[4]:


df.sample(5)


# In[5]:


df.shape


# In[6]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evalutaion
# 6. Improvement
# 7. Website
# 8. Deploy


# ## 1. Data cleaning

# In[7]:


df.info()


# In[8]:


# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[9]:


df.sample(5)


# In[10]:


# renaming columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[11]:


# label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[12]:


# 1 is assign for spam and 0 for ham
df['target']=encoder.fit_transform(df['target'])


# In[13]:


df.head()


# In[14]:


# missing values
df.isnull().sum()


# In[15]:


# check for duplicate values
df.duplicated().sum()


# In[16]:


# remove duplicates
df.drop_duplicates(keep='first')


# In[17]:


df=df.drop_duplicates(keep='first')


# In[18]:


# now  check for duplicated it shows 0
df.duplicated().sum()


# In[19]:


df.shape


# # 2.EDA

# In[20]:


# check how many percentage is spam and ham
df['target'].value_counts()


# In[21]:


# better  representation in piechart
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[22]:


# it clearly show data is imbalance
# one analysis  is how many words,alaphabets,sentences


# In[23]:


# deeper analysis
import nltk


# In[24]:


nltk.download('punkt')


# In[25]:


# the no of character is used(len)
df['num_characters']=df['text'].apply(len)


# In[26]:


df.head()


# In[27]:


# number of words
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[30]:


df.head()


# In[31]:


# check desccribe
df[['num_characters','num_words','num_sentences']].describe()


# In[32]:


# analyese ham and spam different


# In[33]:


# this is for ham
df[df['target']==0][['num_characters','num_words','num_sentences']].describe()


# In[34]:


# this is for spam
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[35]:


# histogram plot for both hamm and spam
import seaborn as sns


# In[36]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')


# In[37]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')


# In[38]:


# correlation coeffiectnt 
# relation between words,sentence,alphabet
sns.pairplot(df,hue='target') 


# In[39]:


# correlation coeffiectnt 
sns.heatmap(df.corr())


# 3. Data preprocessing
#   . Lower case
# <!-- tokennization means break the words -->
#   . Tokennization
#   . Removing special charaters
#   . Removing stop words and punctutaion
#   . Stemming

# In[40]:


def transform_text(text):
    text=text.lower()
#   Tokennization
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    return text


# In[41]:


transform_text('HI HOW Are You')


# In[42]:


# Removing special charaters

def transform_text(text):
    text=text.lower()
#   Tokennization
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    return y        


# In[43]:


# removing stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')


# In[44]:


import string 
string.punctuation


# In[45]:


transform_text('Hi r u at %% eg')


# In[46]:


df['text'][2000]


# In[47]:



# . Removing stop words and punctutaion
def transform_text(text):
    text=text.lower()
#   Tokennization
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if  i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    return y        


# In[48]:


transform_text('hi how are u Nitesh?')


# In[49]:


df['text'][2000]


# In[50]:


# stemming
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('singing')


# In[51]:


# . stemming
def transform_text(text):
    text=text.lower()
#   Tokennization
    text=nltk.word_tokenize(text)

    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if  i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y) 


# In[52]:


transform_text('I love teaching?')


# In[53]:


df['text'].apply(transform_text)


# In[54]:


df['transformed_text']=df['text'].apply(transform_text)


# In[55]:


df.head()


# In[59]:


# word cloud 
from wordcloud import WordCloud  
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[60]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[61]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[63]:


# for ham message
ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[64]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[65]:


# top 30 words use of ham and spam
df.head()


# In[66]:


df[df['target']==1]


# In[68]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[69]:


len(spam_corpus)


# In[72]:


# most common 30 words that is used in spam 
from collections import Counter
pd.DataFrame(Counter(spam_corpus).most_common(30))


# In[74]:


# Model interpertibilty
from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[75]:


ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[76]:


len(ham_corpus)


# In[77]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[78]:


# Model buliding based on Naïve Bayes


# 4. Model Building

# In[ ]:


# bag of words to use text to vectorize


# In[79]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()


# In[80]:


x=cv.fit_transform(df['transformed_text']).toarray()


# In[81]:


# 5169 is email message and 6677 is words
x.shape


# In[82]:


y=df['target'].values


# In[83]:


y


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[88]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[89]:


gnb= GaussianNB()
mnb= MultinomialNB()
bnb= BernoulliNB()


# In[90]:


# Here Accuracy_score is 87% through GaussianNB and Precision_score is 52% 
gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[91]:


# Here Accuracy_score is 96% through MultinomialNB and Precision_score is 84% 
mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[93]:


# Here Accuracy_score is 97% through BernouliNB and Precision_score is 98% 
bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[95]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer()


# In[96]:


x=tfidf.fit_transform(df['transformed_text']).toarray()


# In[97]:


x.shape


# In[98]:


y=df['target'].values


# In[99]:


y


# In[100]:


from sklearn.model_selection import train_test_split


# In[101]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[102]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[103]:


gnb= GaussianNB()
mnb= MultinomialNB()
bnb= BernoulliNB()


# In[104]:


# Here Accuracy_score is 87% through GaussianNB and Precision_score is 51% 
gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[105]:


# Here Accuracy_score is 95% through MultinomialNB and Precision_score is 100% 
mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[106]:


# Here Accuracy_score is 97% through BernouliNB and Precision_score is 98% 
bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[ ]:


# so After the conclusion is that CountVectorizer or TfidfVectorizer is used to choose between which is used as MultinomialNB
# or  BernouliNB
# tfidf---->MNb


# In[110]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[112]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[113]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
}


# In[118]:


def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[119]:


train_classifier(svc,x_train,y_train,x_test,y_test)


# In[121]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, x_train,y_train,x_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[122]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[123]:


performance_df


# In[124]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[125]:


performance_df1


# In[126]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[127]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[129]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000')


# In[130]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling')


# In[133]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[134]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[135]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars')


# In[136]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[137]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[143]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[150]:


voting.fit(x_train,y_train)


# In[145]:


y_pred = voting.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[146]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[147]:


from sklearn.ensemble import StackingClassifier


# In[148]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[149]:


clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





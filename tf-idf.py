
# coding: utf-8

# Corpus exploration

# In[2]:


# terms in product description
terms = dict()
uids = product_description['product_uid']
docs = product_description['product_description']
for i in range(0, len(product_description)):
    terms[uids[i]] = re.split("\W+", docs[i])


# In[3]:


import re
def calculateTf(uid, t):
    term = terms[uid]
    tf = term.count(t)
    return tf

def calculateIdf(uid, t):
    term = terms[uid]
    N = len(product_description)
    idf = np.log(N/(getDf(t) + 1))
    return idf

def getDf(term):
    cnt = 0
    for t in terms.values():
        if term in t:
            cnt += 1
    return cnt


# In[4]:


# scores can be represent as tf*idf
def getScore(uid):
    return calculateIdf(uid)*calculateTf(uid)


# In[5]:


# explore attributes
attributes = attributes[attributes['value'] != 'No']
brands = attributes[attributes['name'] == "MFG Brand Name"]
material = attributes[attributes['name'] == "Material"]


# In[6]:


material.head()


# In[7]:


brands.head()


# In[8]:


# Feature engineering
# merge train and test with description
train = train_data.merge(product_description, on = "product_uid", how = 'left')
test = test_data.merge(product_description, on = "product_uid", how = 'left')


# In[9]:


train.head()


# In[10]:


test.head()


# In[11]:


# merge train and test with brand and material
train = train.merge(brands, on = "product_uid", how = 'left')
test = test.merge(brands, on = "product_uid", how = 'left')
train = train.merge(material,on = "product_uid", how = 'left')
test = test.merge(material, on = "product_uid", how = 'left')


# In[12]:


train.head()


# In[13]:


test.head()


# Calculate AND and OR operator's result of each term and save

# In[16]:


# calculate AND score, OR score and add new column to train
search_terms = []
for st in train['search_term']:
    search_terms.append(re.split("\W+", st))
AND_score = []
OR_score = []
idx = 0
for ts in search_terms:
    cur_uid = train['product_uid'][idx]
    cur_and = 1 # and , multiply all terms' score
    cur_or = 0 # or, find max one 
    for t in ts:
        cur_tf = calculateTf(cur_uid, t)
        cur_idf = calculateIdf(cur_uid, t)
        cur_and = cur_and * cur_tf*cur_idf
        cur_or = max(cur_or, cur_tf*cur_idf)
    AND_score.append(cur_and)
    OR_score.append(cur_or)
    idx += 1
train['AND_score'] = AND_score
train['OR_score'] = OR_score


# In[17]:


# calculate AND score, OR score and add new column to test
search_terms_t = []
for st in test['search_term']:
    search_terms_t.append(re.split("\W+", st))
AND_score_t = []
OR_score_t = []
idx = 0
for ts in search_terms_t:
    cur_uid = test['product_uid'][idx]
    cur_and = 1 # and , multiply all terms' score
    cur_or = 0 # or, find max one 
    for t in ts:
        cur_tf = calculateTf(cur_uid, t)
        cur_idf = calculateIdf(cur_uid, t)
        cur_and = cur_and * cur_tf*cur_idf
        cur_or = max(cur_or, cur_tf*cur_idf)
    AND_score_t.append(cur_and)
    OR_score_t.append(cur_or)
    idx += 1
test['AND_score'] = AND_score_t
test['OR_score'] = OR_score_t


# In[18]:


# process with product title, the percent of terms that appear in title 
train_title = []
test_title = []
idx = 0
for ts in train['search_term']:
    cur_terms = set(re.split("\W+", ts))
    cur_title = set(re.split("\W+", train['product_title'][idx]))
    train_title.append(float(len(cur_terms.intersection(cur_title)))/len(cur_title))
    idx += 1
train['title_contain'] = train_title


# In[19]:


idx = 0
for ts in test['search_term']:
    cur_terms = set(re.split("\W+", ts))
    cur_title = set(re.split("\W+", test['product_title'][idx]))
    test_title.append(float(len(cur_terms.intersection(cur_title)))/len(cur_title))
    idx += 1
test['title_contain'] = test_title


# In[20]:


train.head()


# In[21]:


test.head()


# In[22]:


train.head()


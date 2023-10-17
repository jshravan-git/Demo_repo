#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import pandas as pd
import re
import nltk
import random


# In[2]:


# build Vocabulary
from nltk import word_tokenize


# In[ ]:





# In[3]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam


# In[4]:


#project_path1 = 'C:/Users/poorn/CyberSecData/'
#file_path1 = project_path1 + 'GL_Bot_New.json'
file_path = 'CyberSecControl.json'


# In[5]:


# Read the json corpus data
import json
json_data = json.loads( open( file_path ).read( ) )


# In[30]:


print(json_data)


# In[6]:


corps_voc = set()
corps_tags = []
corps_word_tag = []


# {"intents": [
#         {"tag": "Intro",
#          "patterns": ["hi", 
#                       "how are you", 
#                       "online",
#                       "i am from",
#                       "hey ya",
#                       "talking to you for first time"],
#          "responses": ["Hello! how can i help you ?"],
#          "context_set": ""
#         }

# In[7]:


# Iterate JSOn data and load contents (Corpus has Intents that has a structure
# tag - High level group
# patterns - possible inputs from user for training the model
# response -  List of responses that can be provided randomly if more than one response is available
# 

for intent_itm in json_data['intents']:
    tag = intent_itm['tag']
    corps_tags.append( tag ) #  add tag value
    for pattern in intent_itm['patterns']:
        words = word_tokenize( pattern )   #  Break the pattern row words
        corps_voc.update( words )          #  collect unique word
        corps_word_tag.append( ( words, tag ) )   
        


# In[8]:


corps_voc = list( corps_voc )  # change from set to list


# In[9]:


from nltk.stem import WordNetLemmatizer


# In[10]:


# use WordNet alongside the NLTK module to find the meanings of words, synonyms, antonyms, and more. 
nltk.download('wordnet') 
nltk.download('omw-1.4')


# In[11]:


# Create a list of spl characters
spl_chars = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/']
wnl = WordNetLemmatizer()
corps_voc = [ wnl.lemmatize(word)  for word in corps_voc if word not in spl_chars]


# In[12]:


from nltk.corpus import wordnet
wnl.lemmatize("organizations")


# In[13]:


training_data = []
output_empty = [0] * len(corps_tags)
output_empty


# In[14]:


print(len(corps_word_tag)) # No of sentences in [patterns]
print(len(corps_voc))  #  # of words across all sentences in [patterns]
print(len(corps_tags))


# In[15]:


for docu in corps_word_tag:
    print(docu[0], docu[1])
    


# In[ ]:





# In[16]:


for document in corps_word_tag:
  bag = []
  word_patterns = document[0]
  #word_patterns = [ wnl.lemmatize (word.lower())  for word in word_patterns]
  word_patterns = [ wnl.lemmatize ( x.lower() )  for x in word_patterns ]
  #print(word_patterns)
  for word in corps_voc:
    bag.append(1) if word in word_patterns else bag.append(0)
  output_row = list(output_empty)
  output_row[ corps_tags.index(document[1])] = 1
  #print(output_row)
  training_data.append( [ bag,  output_row])


# In[17]:


# Prepare training data 
training_data = []
output_empty = [0] * len(corps_tags)
for document in corps_word_tag:
  bag = []
  word_patterns = document[0]
  word_patterns = [wnl.lemmatize (word.lower()) for word in word_patterns]
  #print(word_patterns)
  for word in corps_voc:
    bag.append(1) if word in word_patterns else bag.append(0)
    if word in word_patterns:
      print('match :', word)
  output_row = list (output_empty)
  output_row[corps_tags.index(document[1])] = 1
  training_data.append ([bag, output_row])


# In[18]:


#print(len(training_data))
#for td in training_data:
    #print(len(td[0]),  len(td[1]))


# In[19]:


#print(training_data[0][0])


# In[20]:


#random.shuffle(training_data)
Data_type = str
training_data = np.array(training_data, dtype=object)
X_train = list(training_data[:, 0])
y_train = list(training_data[:, 1])


# In[21]:


print(training_data.shape)
print(len(X_train[0]))
print(len(y_train[0]))


# In[22]:


# define the model
model = Sequential()

model.add (Dense (128, input_shape=(len(X_train[0]),), activation='relu'))
model.add (Dropout(0.5))
model.add( Dense( 64, activation='relu' ) )
model.add (Dropout(0.5))
model.add( Dense( 32, activation='relu' ) )

model.add( Dense( len(y_train[0]), activation='softmax') )


# In[23]:


opt_sgd = SGD(learning_rate=0.01)
model.compile( loss='categorical_crossentropy', optimizer=opt_sgd , metrics=['accuracy'])


# In[24]:


# train the model by fitting the training data
history=model.fit(np.array(X_train), np.array(y_train), epochs=200, verbose=1, batch_size=5)


# In[25]:


def clean_up_sentence (sentence):
  #print('Clean up ' , sentence)
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [wnl.lemmatize(word) for word in sentence_words]
  return sentence_words


def bag_of_words (sentence):
  #print('bag_of_words ' , sentence)
  sentence_words = clean_up_sentence (sentence)
  bag = [0] * len (corps_voc)
  for w in sentence_words:
    for i, word in enumerate(corps_voc) :
      if word == w:
        bag[i] = 1
  return np.array (bag)


# In[26]:


def predict_response(sentence):
  bow = bag_of_words (sentence.lower())
  res = model.predict(np.array([bow]), verbose=0)[0]
  #print(res)
  ERROR_THRESHOLD = 0.05
  results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
  #results = [[i, r] for i, r in enumerate(res) ]
  results.sort(key=lambda x: x[1], reverse=True)
  return_list= []
  for r in results:
    return_list.append({'intent': corps_tags[r[0]], 'probability' : str(r[1])})
  return return_list


# In[ ]:





# In[27]:


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = []
    intro_res = " "
    #result.append('I cannot understand,  please re-phrase the question')
    for predictn in intents_list:
        tag = predictn['intent']
        #print(tag)       
        if tag == 'Exit':
            continue
        for i in list_of_intents:
            if i['tag'] == tag:
                if tag == 'Intro':
                    intro_res = random.choice(i['responses'])
                else:
                    result.append(random.choice(i['responses']))
    
    if len(result) == 0:
        result.append(intro_res)                                          
    #print(result)
    return result



# In[28]:


p_r = predict_response('Hi')
print(p_r)
print(get_response (p_r, json_data))


# In[34]:


print ("Bot Running")
print("Welcome , Hi I am the Chatbot to find , preditct matching Common Security Framework using keywords, type 'Exit' to quit ")

exit_intent = 'Exit'
chat_alive = True
while chat_alive:
  message = input("")
  if message == 'Exit':
    chat_alive = False
    continue
  else:
    ints = predict_response(message.lower())
    #print('predicted....', ints)
    if( len(ints) > 1 and ints[0]['intent'] == 'Intro'):
        ints.pop(0)
    res = get_response (ints, json_data)
    chk_below_msg_printed = False
    if(len(res) == 0):
      print("Cannot comprehend,  please rephrase")
    if(ints[0]['intent'] != 'Intro'):
      print('Please refer the below Sub category and Functions for ## ',  message , ' ##')  
      print('the top one has high match probabaility')
    #print(res)
    for pred_msg in res:
      pred_msg = pred_msg.strip()
      if(pred_msg == ''):
        print("Please Rephrase the search string")
      else:
        print('---- Prediction Start ------')
        print(pred_msg)
        print('---- Prediction end ------\n')
   

print('Good bye')  


# In[ ]:





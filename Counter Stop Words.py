
# coding: utf-8

# In[1]:


#import necessary packages
import pandas as pd
import nltk
import operator
nltk.download('stopwords')
from nltk.corpus import stopwords

import re

def remove_numerical(txt):
   txt = re.sub('[0-9]+', '', txt)
   return txt


# In[2]:


#grab all the common English stopwords
stopWords = stopwords.words('english')
stopWords1 = set([word.replace("'", "") for word in stopWords])
moreStopWords = set(["pt", "md", "md1", "md2", "so", "i", "know", "okay", "ok", "yeah", "would", "like", "and", "um", "think", "you", "go", "going", "kind", "oth", "mean", "things", "the", "umm", "uh", "mhmm", "get", "legend", "na", "nah", "mm", "hm", "hmm", "ah", "phi", "right", "it", "oh", "that", "mmhmm", "umhmm", "em", "of",'"you', "one?", "'em","uhhuh", "kay", "md2>md", "mmm", "e", "umm", "othumhmm", "si", "pt/so", "m", "mi", "my", "gee", "just", "iii", "mmhmm", "dr", "laughs", "umhmm", "inaudible", "t", "mmhmmm", "+", "ii", "uhhmm","whatnot", "mmmhmmm", "uhmmm", "its", "youre", "im", "were", "thats", "ill", "theyre", "ive", "youve", "or", "itd", "\"i", "have", "youll", "youd", "one", "dont", "see", "say", "it’s", "that’s", "gonna","alright", "actually", "two", "don't", "gonna", "you're", "i'm", "three", "it's", "ya", "lets", "'i", 'that…that', 'um…', "theres", "got"])
moreStopWords2 = set(["us","cant","five","yep", "id", "whats", "ten","hes", "huh", "four", "seven", "second", "theyll", "wed", "weve", "gotta", "TRUE", "nope", "hi", "theyve", "hey", "yup", "nine", "itll", "x", "u", "cuz", "mdmd", "mmhm", "mhm", "p", "ahh", "gotcha", "hello", "c", "mr", "everybodys", "tc", "gosh", "bye", "nd", "th", "l", "mmmhmm", "hed", "eh", "st", "b", "maam", "r", "uhh", "er", "yea", "gi", "onto", "etc", "geez", "thered", "eleven", "alot", "blah", "w", "whatd", "sevens", "doin", "ohh", "ha", "mightve", "yall", "n", "FALSE", "cm", "g", "se", "heh", "h", "teacher", "ahhh", "ohhh", "nn", "wha", "f", "pr", "aah", "et", "ms", "tur", "fifth", "ish", "biop", "ro", "sur", "gal", "oops", "yada", "de", "fella", "ca", "err", "surg", "whoa", "okp", "chf", "hdr", "pvr", "henry", "sixes", "dude", "ac", "polyp", "aaah"])
stopWords2 = stopWords1.union(moreStopWords).union(moreStopWords2)
stopWords3 = set([word.lower() for word in stopWords2])

stopWords4 = sorted(list(stopWords3))
print(len(stopWords4))


# In[3]:


#use conversation 1 transcripts as my basis #added in convo 2 transcripts later for more fine-tuning
transcript_df = pd.read_csv("/Users/samanthagarland/Downloads/processed_transcripts1.csv")
#should have "..." "-" and numbers taken out

t1 = transcript_df["Convo_1"]


#assemble all conversations into one string
all_conversations_string1 = ""
for s in t1:
    if type(s) is str:
        all_conversations_string1 += s
print(len(all_conversations_string1))

all_conversations_string = ""
for char in all_conversations_string1:
    if char.isalpha() or char == " ":
        all_conversations_string += char
print(len(all_conversations_string))


# In[4]:


all_words = []
for word in all_conversations_string.split(" "):
    word = remove_numerical(word)
    if word is not "" and word is not " ":
        all_words.append(word.lower())

#all_uniq_words = set(all_words)
#uniq_words = all_uniq_words.difference(stopWords3)

#print("All words:", len(all_words))
#print("All unique words:", len(all_uniq_words))
#print("All stop words:", len(stopWords3))
#print("All unique (non-stop) words:", len(uniq_words))


# In[10]:


from collections import Counter
counts = Counter(all_words)

most_common = counts.most_common(5300)
top5000_uniq = []
for word, num in most_common:
    if word not in stopWords3:
        top5000_uniq.append((word, num))


# In[11]:


print(len(top5000_uniq))


# In[12]:


for word, num in top5000_uniq:
    print(word, num)


# In[ ]:


#more stop words: "unh", "ps", "onc", "bam", "ho", "mrs", "afib", "gimme", "uti", "kennedy","ohhhh", "whod", "anybodys", "bi", "ts", "uhhh", "whew", "hah", "walsh"


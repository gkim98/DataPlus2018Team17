
# coding: utf-8

# Here, we're trying to find more stopwords from our files. Please note that finding frequencies of each unique (non-stop) word takes a long time, as there are 32,000+ words (now narrowed down to 29,000+), and they then are sorted by frequency.

# In[12]:


#import necessary packages
import pandas as pd
import nltk
import operator
nltk.download('stopwords')
from nltk.corpus import stopwords


# In[36]:

#grab all the common English stopwords
stopWords = stopwords.words('english')
stopWords1 = set([word.replace("'", "") for word in stopWords])
moreStopWords = set(["pt", "md", "md1", "md2", "so", "i", "know", "okay", "ok", "yeah", "would", "like", "and", "um", "think", "you", "go", "going", "kind", "oth", "mean", "things", "the", "umm", "uh", "mhmm", "get", "legend", "na", "nah", "mm", "hm", "hmm", "ah", "phi", "right", "it", "oh", "that", "mmhmm", "umhmm", "em", "of",'"you', "one?", "'em","uhhuh", "kay", "md2>md", "mmm", "e", "umm", "othumhmm", "si", "pt/so", "m", "mi", "my", "gee", "just", "iii", "mmhmm", "dr", "laughs", "umhmm", "inaudible", "t", "mmhmmm", "+", "ii", "uhhmm","whatnot", "mmmhmmm", "uhmmm", "its", "youre", "im", "were", "thats", "ill", "theyre", "ive", "youve", "or", "itd", "\"i", "have", "youll", "youd", "one", "dont", "see", "say", "it’s", "that’s", "gonna","alright", "actually", "two", "don't", "gonna", "you're", "i'm", "three", "it's", "ya", "lets", "'i", 'that…that', 'um…'])
stopWords2 = stopWords1.union(moreStopWords)
stopWords3 = set([word.lower() for word in stopWords2])

stopWords4 = sorted(list(stopWords3))

#use conversation 1 transcripts as my basis
transcript_df = pd.read_csv("/Users/samanthagarland/Downloads/all_transcripts.csv")
t1 = transcript_df["Convo_1"]

#assemble all conversations into one string
all_conversations_string = ""
for s in t1:
    if type(s) is str:
        all_conversations_string += s

all_words = []
for word in all_conversations_string.split(" "):
    if word is not "" and word is not " ":
        all_words.append(word.lower())

all_uniq_words = set(all_words)
uniq_words = all_uniq_words.difference(stopWords3)

print("All words:", len(all_words))
print("All unique words:", len(all_uniq_words))
print("All stop words:", len(stopWords3))
print("All unique (non-stop) words:", len(uniq_words))

#print(stopWords3)


# In[30]:


#create and sort words by frequency
d = []
for word in uniq_words:
    count = all_words.count(word) 
    freq = count / len(all_words)
    d.append((word, count, freq))

d.sort(key=operator.itemgetter(2), reverse = True)

#grab 50 most frequent words, see how many of them are useful
print(d[:50])


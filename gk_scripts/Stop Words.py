
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
stopWords1 = set(stopwords.words('english'))
moreStopWords = set(["PT", "MD", "MD1", "MD2", "SO", "I", "know", "Okay", "OK", "Yeah", "So", "would", "like", "And", "and", "um", "think", "you", "You", "go", "going", "kind", "OTH", "mean", "things", "The", "umm", "uh", "mhmm", "get", "legend", "na", "nah", "mm", "hm", "hmm", "ah", "go", "PHI", "right", "it", "oh", "that", "mm-hmm", "um-hmm", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "10", "...", "em", "of", "the...the", "i...i",'"you', "one?", "'em","uhhuh", "kay", "md2>md", "umthe", "mmm-", "e", "umm", "umm...", "ifif", "othum-hmm", "s...i", "pt/so", "m...i", "mi", "si", "gee", "justi", "just...i", "theythey", "iii", "mymy", "ai", "umif", "youif", "thinki"])
stopWords2 = stopWords1.union(moreStopWords)
stopWords3 = set([word.lower() for word in stopWords2])

#use conversation 1 transcripts as my basis
transcript_df = pd.read_csv("/Users/samanthagarland/Downloads/all_transcripts.csv")
t1 = transcript_df["Convo_1"]

#assemble all conversations into one string
all_conversations_string = ""
for s in t1:
    if type(s) is str:
        s.replace("[", "")
        s.replace(":", "")
        s.replace("]", "")
        s.replace("(", "")
        s.replace(")", "")
        all_conversations_string += s

all_words = []
for word in all_conversations_string.split(" "):
    word = word.replace("[", "")
    word = word.replace(":", "")
    word = word.replace("]", "")
    word = word.replace("(", "")
    word = word.replace(")", "")
    word = word.replace("...", "")
    word = word.replace("-", "")
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
    freq = all_words.count(word) / len(all_words)
    d.append((word, freq))

d.sort(key=operator.itemgetter(1), reverse = True)

#grab 50 most frequent words, see how many of them are useful
print(d[:50])


# In[20]:


#output before changes was: [('PT:', 0.018464405876857302), ('MD2:', 0.014988531964653181), ('[PT:', 0.006142902570784529), ('prostate', 0.005963472322884261), ('cancer', 0.005466986008668078), (']', 0.005253266734336608), ('MD:', 0.004866223477085771), ('surgery', 0.004589092989491118), ('radiation', 0.004144744783957996), ('Um', 0.0033941439039982414), ('one', 0.002765668323656469), ('It', 0.0023974135740391675), ('see', 0.0021085402691735544), ('good', 0.0020131885929333604), ('SO:', 0.002008021753334138), ('back', 0.0020061429025707844), ('Right', 0.001958232208105268), ('want', 0.001956353357341915), ('risk', 0.001905624386731368), ('little', 0.0019046849613496913), ('treatment', 0.001888714729861186), ('No', 0.001878381050662741), ('((PHI', 0.0018323492069605785), ('time', 0.0017788019602050014), ('OTH:', 0.0017393460941745762), ('say', 0.0017365278180295459), ('biopsy', 0.0017111633327242725), ('people', 0.0017102239073425957), ('removed))', 0.0017041176423616966), ('years', 0.0016172207945565936), ('okay', 0.001597492861541381), ('take', 0.0015965534361597042), ('Well', 0.0015956140107780273), ('That', 0.0015932654473238355), ('yeah', 0.0015909168838696434), ('[MD:', 0.0015561581447476023), ('something', 0.001499792621846995), ('probably', 0.0014786555507592671), ('really', 0.0014716098603966912), ('Oh', 0.0014659733081066305), ('[SO:', 0.0014593973304348929), ('well', 0.0014077289344426695), ('thing', 0.0013626365161221836), ('[MD2:', 0.0013616970907405067), ('PSA', 0.001358409101904638), ('make', 0.0013447874338703245), ('But', 0.0013306960531451726), ('side', 0.0013259989262367888), ('We', 0.00132271093740092), ('talk', 0.0012672848398819893)]
#output after changes was: [('pt:', 0.018464405876857302), ('md2:', 0.014988531964653181), ('[pt:', 0.006142902570784529), ('prostate', 0.006014201293494808), ('cancer', 0.00550738130008018), (']', 0.005253266734336608), ('md:', 0.004866223477085771), ('surgery', 0.004672232135769514), ('radiation', 0.0042983408338621515), ('one', 0.003029646855907647), ('well', 0.003003342945220697), ('good', 0.0024528396715580983), ('see', 0.0021752394712726067), ('back', 0.0020155371563875526), ('so:', 0.0020084914660249767), ('want', 0.0019619899096319756), ('little', 0.0019131397897847824), ('risk', 0.0019107912263305903), ('treatment', 0.0018957604202237616), ('((phi', 0.0018323492069605785), ('time', 0.001783029374422547), ('say', 0.0017576648891172737), ('people', 0.0017492100606821825), ('removed))', 0.001742634083010445), ('oth:', 0.0017393460941745762), ('okay?', 0.0017252547134494244), ('biopsy', 0.0017177393103960101), ('take', 0.0016547978098236651), ('oh', 0.0016543280971328268), ('years', 0.0016228573468466544), ('[md:', 0.0015561581447476023), ('probably', 0.0015561581447476023), ('something', 0.0015199902675530458), ('really', 0.0015007320472286716), ('[so:', 0.0014593973304348929), ('psa', 0.00138706157604578), ('make', 0.0013649850795763754), ('thing', 0.0013626365161221836), ('[md2:', 0.0013616970907405067), ('side', 0.0013344537546718799), ('talk', 0.0012776185190804341), ('sure', 0.0012499054703209687), ('two', 0.0012282986865424026), ('could', 0.0012081010408363517), ('got', 0.0011911913839661695), ('lot', 0.0011747514397868256), ('mm-hmm', 0.0011667663240425728), ('um-hmm', 0.001163478335206704), ('let', 0.0011390532752831076), ('come', 0.001125431607248794)]
#best one so far: [('prostate', 0.006018898420403192), ('', 0.005871878348170774), ('cancer', 0.005513017852370241), ('surgery', 0.004678338400750413), ('radiation', 0.00430444709884305), ('one', 0.003034343982816031), ('well', 0.00301414633710998), ('good', 0.0024796132949358868), ('see', 0.002189800564688597), ('back', 0.002018355432532583), ('want', 0.0019619899096319756), ('removed', 0.0019300494466549646), ('little', 0.0019136095024756207), ('risk', 0.0019107912263305903), ('treatment', 0.0018981089836779537), ('time', 0.001865229095319266), ('say', 0.0017590740271897888), ('people', 0.001751088911445536), ('okay?', 0.0017323004038120003), ('biopsy', 0.0017210272992318788), ('take', 0.0016552675225145036), ('years', 0.0016252059103008462), ('probably', 0.0015575672828201175), ('something', 0.0015256268198431066), ('really', 0.0015087171629729243), ('psa', 0.0013894101394999721), ('thing', 0.0013673336430305675), ('make', 0.0013659245049580523), ('side', 0.001335862892744395), ('sure', 0.0013311657658360112), ('talk', 0.00128466420944301), ('two', 0.0012301775373057561), ('could', 0.001209510178908867), ('got', 0.0011911913839661695), ('lot', 0.0011761605778593408), ('let', 0.0011390532752831076), ('come', 0.0011259013199396326), ('alright', 0.0011155676407411878), ('gleason', 0.0011075825249969352), ('bit', 0.0011047642488519048), ('way', 0.0011000671219435207), ('said', 0.0010911425808175913), ('need', 0.001090672868126753), ('3', 0.0010831574650733385), ('low', 0.0010770512000924395), ('2', 0.001061080968603934), ('men', 0.0010577929797680652), ('months', 0.0010328982071536304), ('give', 0.0010178674010468017), ('blank', 0.0009910937776690131)]


# In[31]:


print(d[50:100])


# In[32]:


print(d[100:150])


# In[33]:


print(d[150:200])


# In[34]:


print(d[1000:1500])


# In[35]:


print(d[3000:5000])


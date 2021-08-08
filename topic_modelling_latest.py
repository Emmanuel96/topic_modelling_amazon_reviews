import nltk
nltk.download('stopwords')
import re 
import numpy as np 
import pandas as pd 
from pprint import pprint 

import gensim 
import gensim.corpora as corpora 
from gensim.utils import simple_preprocess 
from gensim.models import CoherenceModel

import spacy

import pyLDAvis 
import pyLDAvis.gensim
import matplotlib.pyplot as pyplot

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from','nice', 'like', 'Like','so','be',\
    'razor', 'stand', 'great', 'well', 'shave','good','product','put','everything','something',\
    'buy', 'ago','suggest', 'however', 'negro', 'subject', 're', 'edu', 'use', 'from', 'my', 'we',\
    'i', 've', 'buy', 'set', 'lot', 'decide', 'give', 'add', 'get'])

import string 
from nltk import PorterStemmer
from nltk import SnowballStemmer

data = []

data = pd.read_json('reviews_Electronics_5.json',lines = True, nrows = 1000)

df = pd.DataFrame.from_dict(data)

# CLEANING DATASET 
# Remove new line characters 
# Remove Emojis 
# Remove Punctuations marks 
# Remove extra spaces 
# remove emails
# Remove stop words 

# Remove emojis by removing the patterns that match the emojis first 
def remove_emojis(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def clean_text(text): 
    # # initializing my stemmers
    # ps = PorterStemmer()

    # remove the emojis
    text = remove_emojis(text)

    # remove punctuation
    text_cleaned = "".join([x for x in text if x not in string.punctuation]) 
        
    # Remove extra white space 
    text_cleaned = re.sub(' +', ' ', text_cleaned)
    text_cleaned = text_cleaned.lower()
    tokens = text_cleaned.split(" ")

    # Taking only those words which are not stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # text_cleaned = " ".join([ps.stem(token) for token in tokens])
    # text_cleaned = " ".join([token for token in tokens])

    return text_cleaned

# apply the clean text function on the reviewText column 
df['cleaned_reviews'] = df['reviewText'].apply(lambda x:clean_text(x))

# pprint(df['cleaned_reviews'])

cleaned_reviews = df['cleaned_reviews']

# # next we tokenize
# # Tokenize words and clean up text 
def tokenize(sentences): 
    for sentence in sentences: 
        # deacc true removes punctuations 
        yield(gensim.utils.simple_preprocess(str(sentence), deacc= True))

# # 
cleaned_review_words = list(tokenize(cleaned_reviews))

# # Next we create our bigram and trigram words 
bigram = gensim.models.Phrases(cleaned_review_words)
trigram = gensim.models.Phrases(bigram[cleaned_review_words])  

# # Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts): 
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts): 
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts: 
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out 

# # form Bigrams 
data_words_bigrams = make_bigrams(cleaned_review_words)
# pprint(data_words_bigrams)

# # form trigrams
data_words_trigrams = make_trigrams((data_words_bigrams))

# # load spacy with english as the language
nlp = spacy.load('en_core_web_sm')
# # lemmatization
data_lemmatized= lemmatization(data_words_trigrams)

# # the two main inputs for the LDA are the dictionary and the Corpus 
# 1- create dictionary 
dictionary = corpora.Dictionary(data_lemmatized)

# 2- Create corpus 
# it's important we understand that this converts the list of words to matching integers for our LDA algorithm
texts = data_lemmatized
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# # Compute Coherence Score
# if __name__ == "__main__":
#     coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=dictionary, coherence='c_v')
#     coherence_lda = coherence_model_lda.get_coherence()
#     print('\nCoherence Score: ', coherence_lda)

# Visualize the topics: 
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
# pyLDAvis.show(vis)
pyLDAvis.save_html(vis, 'hehe.html')

# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=5, y=5)
plt.tight_layout()
plt.show()
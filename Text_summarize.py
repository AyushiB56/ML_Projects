from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords

def tldr(text_to_summarize):
    sent_token= np.array(sent_tokenize(text_to_summarize))
    stop_word_eng = set(stopwords.words("english"))
    tfid= TfidfVectorizer(stop_words=stop_word_eng)
    tf_idf_score= tfid.fit_transform(sent_token)
    tf_idf_score= tf_idf_score.toarray()
    sentence_tf_idf = tf_idf_score.sum(axis=1)
    sent_to_pick = np.where(sentence_tf_idf>3)
    summary=' '.join(sent_token[sent_to_pick])
    return summary
                          

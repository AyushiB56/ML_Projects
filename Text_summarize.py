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
                          
###########

#Another method

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords

def tldr(text_to_summarize):
    # Write your code her
    sentence= sent_tokenize(text_to_summarize)
    stop_word_eng= set(stopwords.words("english"))
    tfidf= TfidfVectorizer(lowercase=True,stop_words=stop_word_eng)
    result= tfidf.fit_transform(sentence)

    result = result.toarray()
    tfidf_score={}
    for idx in range(len(result)):
        tfidf_score[idx]= sum(result[idx]).item()
    threshold=3
    intermediate=[]
    for key,value in tfidf_score.items():
      if value>threshold:
        
        intermediate.append(sentence[key])

    summarize=" ".join(intermediate)
    limit = int(0.5*len(text_to_summarize))
    if len(summarize)>limit:
          summarize=summarize[:limit]

    return summarize
        
        
    

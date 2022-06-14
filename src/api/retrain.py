import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings("ignore")


useless_words = stopwords.words("english")
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
personality_type = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                   "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"  ]
filenames = ["IE.joblib","NS.joblib","FT.joblib","JP.joblib"]
result = []

def translate_personality(personality):
    return [b_Pers[l] for l in personality]
def translate_back(personality):
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s
def pre_process_text():
    list_posts = []
    list_personality = []
    data = pd.read_csv('C:/Users/luisf/Escritorio/TODO/FIB/TFG/ML/dataP.csv')

    for row in data.iterrows():
        posts = row[1].posts
        list_posts.append(posts)
        type_labelized = translate_personality(row[1].type) 
        list_personality.append(type_labelized)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality  = pre_process_text()

cntizer = CountVectorizer(analyzer="word", 
                             max_features=1000,  
                             max_df=0.7,
                             min_df=0.1) 
X_cnt = cntizer.fit_transform(list_posts)
feature_names = list(enumerate(cntizer.get_feature_names()))
tfizer = TfidfTransformer()
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
X = X_tfidf


for l in range(len(personality_type)):
    Y = list_personality[:,l]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)
    model = SVC(random_state = 1)
    model.fit(X_train, y_train)
    filename = filenames[l]
    joblib.dump(model, filename)

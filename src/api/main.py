from fastapi import FastAPI
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from fastapi.middleware.cors import CORSMiddleware
import joblib
import csv  

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IE = joblib.load("../../models/IE.joblib")
NS = joblib.load("../../models/NS.joblib")
FT = joblib.load("../../models/FT.joblib")
JP = joblib.load("../../models/JP.joblib")

list_posts = []
modelnames= [IE,NS,FT,JP]
personality_type = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"  ]
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]
allTypes = ["ISTJ","ISFJ","INFJ","INTJ","ISTP","ISFP","INFP","INTP","ESTP","ESFP","ENFP","ENTP","ESTJ","ESFJ","ENFJ","ENTJ"]
allStats= [2.19,7.78,2.66,7.90,0.48,0.55,0.45,1.03,16.95,21.12,12.58,15.03,1.91,3.12,2.36,3.88]

def translate_back(personality):
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

def translate_personality(personality):
    return [b_Pers[l] for l in personality]

def pre_process_text1():
    list_posts = []
    data = pd.read_csv('../data/processed/dataP.csv')

    for row in data.iterrows():
        posts = row[1].posts
        list_posts.append(posts)

    list_posts = np.array(list_posts)
    return list_posts

def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
        lemmatiser = WordNetLemmatizer()

        useless_words = stopwords.words("english")

        unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
            'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
        unique_type_list = [x.lower() for x in unique_type_list]
        list_posts = []
        for row in data.iterrows():
            posts = row[1].posts
            # Elimina URL 
            temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
            # Elimina caracteres que no son palabras
            temp = re.sub("[^a-zA-Z]", " ", temp)
            # Quita espacios innecesarios
            temp = re.sub(' +', ' ', temp).lower()
            # Eliminar las palabras que se repiten con varias letras
            temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)
            # Elimina las stopwords
            if remove_stop_words:
                temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
            else:
                temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
            # Eliminar las palabras de la personalidad MBTI de los mensajes
            if remove_mbti_profiles:
                for t in unique_type_list:
                    temp = temp.replace(t,"")

            list_posts.append(temp)

        list_posts = np.array(list_posts)
        return list_posts

def treat_post(msg:str):

    list_posts= pre_process_text1()
    cntizer = CountVectorizer(analyzer="word", 
                                max_features=1000,  
                                max_df=0.7,
                                min_df=0.1) 
    X_cnt = cntizer.fit_transform(list_posts)
    tfizer = TfidfTransformer()
    X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
    reverse_dic = {}
    for key in cntizer.vocabulary_:
        reverse_dic[cntizer.vocabulary_[key]] = key
    top_10 = np.asarray(np.argsort(np.sum(X_cnt, axis=0))[0,-10:][0, ::-1]).flatten()
    [reverse_dic[v] for v in top_10]

    my_posts = msg
    mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})
    my_posts= pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)
    my_X_cnt = cntizer.transform(my_posts)
    my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()
    return my_X_tfidf

@app.get("/")
def index():
    return{'Hello'}

@app.get("/predict")
def predict(msg:str):
    prediction= []
    final_msg= treat_post(msg)
    for l in range(len(personality_type)):
        y_pred = modelnames[l].predict(final_msg)
        prediction.append(y_pred[0])

    return{
        'prediction': translate_back(prediction),
        'newPost' : msg,
    }

@app.get("/typestats")
def Stadistics(type:str):
    n= allTypes.index(type)
    stats = allStats[n]
    return{
        'type_stats': stats,
    }

@app.get("/allstats")
def Stadistics():
    return{
        'all_stats': allStats,
    }


@app.post("/updatemodel")
def update(type:str, msg:str):
    
    my_posts = msg
    mydata = pd.DataFrame(data={'type': ['type'], 'posts': [my_posts]})
    new_post= pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)

    fields=[type,new_post[0]]
    with open(r'../data/processed/dataP.csv', 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

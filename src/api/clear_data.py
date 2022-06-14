from fastapi import FastAPI

# Analisis de datos
import pandas as pd
import numpy as np

# Procesamiento de texto
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Paquetes de Machine Learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Guardar modelo
import joblib

app = FastAPI()
global modelnames
global list_personality
global list_posts



list_personality = []
list_posts = []

personality_type = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"  ]
b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def translate_back(personality):
    # transformar el vector binario en personalidad mbti
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s
def translate_personality(personality):
    # transformar mbti a un vector binario
    return [b_Pers[l] for l in personality]

data = pd.read_csv('data/raw/mbti_1.csv')
def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
    
    lemmatiser = WordNetLemmatizer()

    # Identificar stopwords
    useless_words = stopwords.words("english")

    # Eliminarlas de los posts
    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
        'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    unique_type_list = [x.lower() for x in unique_type_list]
    list_personality = []
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
        # transformar mbti a vector binario
        type_labelized = translate_personality(row[1].type) 
        list_personality.append(type_labelized)
        # Datos ya limpios
        list_posts.append(temp)
        row[1].posts = temp
    dataP = pd.DataFrame(data)
    dataP.to_csv("data/processed/dataP.csv", index=False)
    print(dataP)

    # Resultado
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


list_posts, list_personality  = pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True)



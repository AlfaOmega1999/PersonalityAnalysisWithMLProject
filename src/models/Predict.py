# Analisis de datos
import pandas as pd
import numpy as np

# Visualizacion de datos
import seaborn as sns
import matplotlib.pyplot as plt

# Procesamiento de texto
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Paquetes de Machine Learning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Guardar modelo
import joblib

# Ignorar warnings (ruido)
import warnings
warnings.filterwarnings("ignore")

IE = joblib.load("models/IE.joblib")
NS = joblib.load("models/NS.joblib")
FT = joblib.load("models/FT.joblib")
JP = joblib.load("models/JP.joblib")


modelnames= [IE,NS,FT,JP]

# MODELO FINAL
data = pd.read_csv('data/raw/mbti_1.csv')

lemmatiser = WordNetLemmatizer()

# Identificar stopwords
useless_words = stopwords.words("english")

# Eliminarlas de los posts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def translate_personality(personality):
    # transformar mbti a un vector binario
    return [b_Pers[l] for l in personality]

#Mostrar el resultado de la prediccion de la personalidad
def translate_back(personality):
    # transformar el vector binario en personalidad mbti
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

list_personality_bin = np.array([translate_personality(p) for p in data.type])

# Limpiar los posts
def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
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


    # Resultado
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality  = pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True)


# Vectorizar los posts de la base de datos a una matriz de conteo de tokens para el modelo
cntizer = CountVectorizer(analyzer="word", 
                            max_features=1000,  
                            max_df=0.7,
                            min_df=0.1) 
                            
# la característica debe ser de n-grama de la palabra 
# Aprender el diccionario de vocabulario y devolver la matriz término-documento
X_cnt = cntizer.fit_transform(list_posts)

#El objeto enumerate devuelve pares que contienen un recuento y un valor
feature_names = list(enumerate(cntizer.get_feature_names()))

# Transformar la matriz de recuento en una representacion normalizada tf o tf-idf
tfizer = TfidfTransformer()

# Aprender el vector idf (fit) y transformar una matriz de recuento a una representacion tf-idf
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()

#contando las 10 primeras palabras
reverse_dic = {}
for key in cntizer.vocabulary_:
    reverse_dic[cntizer.vocabulary_[key]] = key
top_10 = np.asarray(np.argsort(np.sum(X_cnt, axis=0))[0,-10:][0, ::-1]).flatten()
[reverse_dic[v] for v in top_10]


personality_type = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"  ]

# Posts con la representacion tf-idf
X = X_tfidf





'''
filenames = ["ML/modelos/IE.joblib","ML/modelos/NS.joblib","ML/modelos/FT.joblib","ML/modelos/JP.joblib"]




filenames = ["C:/Users/luisf/Escritorio/PersonalityAnalysisWithML/ML/modelos/IE.joblib","C:/Users/luisf/Escritorio/PersonalityAnalysisWithML/ML/modelos/NS.joblib","C:/Users/luisf/Escritorio/PersonalityAnalysisWithML/ML/modelos/FT.joblib","C:/Users/luisf/Escritorio/PersonalityAnalysisWithML/ML/modelos/JP.joblib"]
modelnames = ["IE","NS","FT","JP"]


for l in range(len(personality_type)):
    filename = filenames[l]
    modelnames[l] = joblib.load(filename)
#names = ["my_X_tfidf","my_X_tfidf2","my_X_tfidf3"]
'''
print("Modelos Cargados")

my_posts  = """ Hi I am 21 years, currently, I am pursuing my graduate degree in computer science and management (Mba Tech CS ), It is a 5-year dual degree.... My CGPA to date is 3.8/4.0 . I have a passion for teaching since childhood. Math has always been the subject of my interest in school. Also, my mother has been one of my biggest inspirations for me. She started her career as a teacher and now has her own education trust with preschools schools in Rural and Urban areas. During the period of lockdown, I dwelled in the field of blogging and content creation on Instagram.  to spread love positivity kindness . I hope I am able deliver my best to the platform and my optimistic attitude helps in the growth that is expected. Thank you for the opportunity. """

# The type is just a dummy so that the data prep function can be reused
mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})
my_posts, dummy  = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)
my_X_cnt = cntizer.transform(my_posts)
my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()
result = []
for l in range(len(personality_type)):
    # hacer predicciones para los datos
    y_pred = modelnames[l].predict(my_X_tfidf)
    result.append(y_pred[0])

print("El resultado es: ", translate_back(result))

my_posts  = """ Hi I am 21 years, currently, I am pursuing my graduate degree in computer science and management (Mba Tech CS ), It is a 5-year dual degree.... My CGPA to date is 3.8/4.0 . I have a passion for teaching since childhood. Math has always been the subject of my interest in school. Also, my mother has been one of my biggest inspirations for me. She started her career as a teacher and now has her own education trust with preschools schools in Rural and Urban areas. During the period of lockdown, I dwelled in the field of blogging and content creation on Instagram.  to spread love positivity kindness . I hope I am able deliver my best to the platform and my optimistic attitude helps in the growth that is expected. Thank you for the opportunity. """

# The type is just a dummy so that the data prep function can be reused
mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})
my_posts, dummy  = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)
my_X_cnt = cntizer.transform(my_posts)
my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()
result = []
for l in range(len(personality_type)):
    # hacer predicciones para los datos
    y_pred = modelnames[l].predict(my_X_tfidf)
    result.append(y_pred[0])

print("El resultado es: ", translate_back(result))

my_posts  = """ Hi I am 21 years, currently, I am pursuing my graduate degree in computer science and management (Mba Tech CS ), It is a 5-year dual degree.... My CGPA to date is 3.8/4.0 . I have a passion for teaching since childhood. Math has always been the subject of my interest in school. Also, my mother has been one of my biggest inspirations for me. She started her career as a teacher and now has her own education trust with preschools schools in Rural and Urban areas. During the period of lockdown, I dwelled in the field of blogging and content creation on Instagram.  to spread love positivity kindness . I hope I am able deliver my best to the platform and my optimistic attitude helps in the growth that is expected. Thank you for the opportunity. """

# The type is just a dummy so that the data prep function can be reused
mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})
my_posts, dummy  = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)
my_X_cnt = cntizer.transform(my_posts)
my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()
result = []
for l in range(len(personality_type)):
    # hacer predicciones para los datos
    y_pred = modelnames[l].predict(my_X_tfidf)
    result.append(y_pred[0])

print("El resultado es: ", translate_back(result))


def test():
    return "hello how r u"
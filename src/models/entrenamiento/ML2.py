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

# Entrenamiento del modelo y evaluacion
from sklearn.model_selection import train_test_split

# Modelo
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Precision
from sklearn.metrics import accuracy_score

# Ignorar warnings (ruido)
import warnings
warnings.filterwarnings("ignore")

# Leer dataset
data_set = pd.read_csv('C:/Users/luisf/Escritorio/TODO/FIB/TFG/ML/mbti_1.csv')

# Tamaño dataset
nRow, nCol = data_set.shape
print(f'There are {nRow} rows and {nCol} columns')

# Tipos Dataset
types = np.unique(np.array(data_set['type']))
print(types)

total = data_set.groupby(['type']).count()
print(total)

#visualizar el num de posts
plt.figure(figsize = (12,4))
plt.bar(np.array(total.index), height = total['posts'],)
print(total['posts'])
plt.xlabel('Tipos de personalidad', size = 14)
plt.ylabel('No. de posts disponibles', size = 14)
plt.title('Posts totales de cada personalidad')
plt.show()

# Tipos ordenados por num de posts
cnt_srs = data_set['type'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.xlabel('Tipos de personalidad', fontsize=14)
plt.ylabel('No. de posts disponibles', fontsize=14)
plt.show()

df = data_set.copy()

# Swarmplot de palabras por post
# Cuenta el numero de palabras en cada mensaje de un usuario
def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)

# Cuenta el numero de palabras por mensaje del total de 50 mensajes en toda la fila
df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
df['variance_of_word_counts'] = df['posts'].apply(lambda x: var_row(x))

plt.figure(figsize=(15,10))
sns.swarmplot("type", "words_per_comment", data=df)
plt.show()

plt.figure(figsize=(15,10))
sns.violinplot(x='type', y='words_per_comment', data=df, inner=None, color='lightgray')
sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
plt.ylabel('Palabras por comentario')
plt.show()

# Joinplot de palabras por post / variacion por post
plt.figure(figsize=(15,10))
sns.jointplot("variance_of_word_counts", "words_per_comment", data=df, kind="hex")
plt.show()

'''
def plot_jointplot(mbti_type, axs, titles):
    df_1 = df[df['type'] == mbti_type]
    sns.jointplot("variance_of_word_counts", "words_per_comment", data=df_1, kind="hex", ax = axs, title = titles)

plt.figure(figsize=(24, 5))    
i = df['type'].unique()
k = 0

for m in range(1,3):
  for n in range(1,7):
    df_1 = df[df['type'] == i[k]]
    sns.jointplot("variance_of_word_counts", "words_per_comment", data=df_1, kind="hex" )
    plt.title(i[k])
    k+=1
#plt.show()
'''

# Palabras mas usadas
words = list(df["posts"].apply(lambda x: x.split()))
words = [x for y in words for x in y]
Counter(words).most_common(40)

# Convertir la personalidad del MBTI a forma numerica utilizando la codificacion de etiquetas
enc = LabelEncoder()
df['type of encoding'] = enc.fit_transform(df['type'])
target = df['type of encoding'] 

##nltk.download() en caso de que salte error
print(stopwords.words('english'))
# Vectorizacion de los posts para el modelo y filtrado de las palabras de parada
vect = CountVectorizer(stop_words='english') 
# Convertir los posts a forma numerica mediante la vectorizacion del recuento
train =  vect.fit_transform(df["posts"])

print(train.shape)


#FINAL MODEL
data = pd.read_csv('C:/Users/luisf/Escritorio/TODO/FIB/TFG/ML/mbti_1.csv')

def get_types(row):
    t=row['type']

    I = 0; N = 0
    T = 0; J = 0
    
    if t[0] == 'I': I = 0
    elif t[0] == 'E': I = 1
    else: print('I-E no concluyente') 
        
    if t[1] == 'N': N = 0
    elif t[1] == 'S': N = 1
    else: print('N-S no concluyente')
        
    if t[2] == 'T': T = 0
    elif t[2] == 'F': T = 1
    else: print('T-F no concluyente')
        
    if t[3] == 'J': J = 0
    elif t[3] == 'P': J = 1
    else: print('J-P no concluyente')
    return pd.Series( {'IE':I, 'NS':N , 'TF': T, 'JP': J }) 

data = data.join(data.apply (lambda row: get_types (row),axis=1))
print(data.head(5))

print ("Introversion (I) /  Extroversion (E):\t", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
print ("Intuition (N) / Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
print ("Thinking (T) / Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
print ("Judging (J) / Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])

 
#Plot distribucion de cada indicador de tipo de personalidad
N = 4
bottom = (data['IE'].value_counts()[0], data['NS'].value_counts()[0], data['TF'].value_counts()[0], data['JP'].value_counts()[0])
top = (data['IE'].value_counts()[1], data['NS'].value_counts()[1], data['TF'].value_counts()[1], data['JP'].value_counts()[1])

ind = np.arange(N)
width = 0.7

p1 = plt.bar(ind, bottom, width, label="I, N, T, F")
p2 = plt.bar(ind, top, width, bottom=bottom, label="E, S, F, P") 

plt.title('Distribucion por tipos de indicadores')
plt.ylabel('Cuenta')
plt.xticks(ind, ('I / E',  'N / S', 'T / F', 'J / P',))
plt.legend()
plt.show()

data[['IE','NS','TF','JP']].corr()
cmap = plt.cm.RdBu
corr = data[['IE','NS','TF','JP']].corr()
plt.figure(figsize=(12,10))
plt.title('Mapa de correlacion de caracteristicas', size=15)
sns.heatmap(corr, cmap=cmap,  annot=True, linewidths=1)
plt.show()


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
print("Lista de MBTI (Binarizada): \n%s" % list_personality_bin)

# Limpiar los posts
def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
  list_personality = []
  list_posts = []
  len_data = len(data)
  i=0
  
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


# Vectorizing the database posts to a matrix of token counts for the model
cntizer = CountVectorizer(analyzer="word", 
                             max_features=1000,  
                             max_df=0.7,
                             min_df=0.1) 
# la caracteristica debe ser de n-grama de la palabra 
# Aprender el diccionario de vocabulario y devolver la matriz termino-documento
print("Usando CountVectorizer :")
X_cnt = cntizer.fit_transform(list_posts)

#El objeto enumerate devuelve pares que contienen un recuento y un valor
feature_names = list(enumerate(cntizer.get_feature_names()))
print("A continuacion se pueden ver 10 nombres de caracteristicas")
print(feature_names[0:10])

# Transformar la matriz de recuento en una representacion normalizada tf o tf-idf
tfizer = TfidfTransformer()

# Aprender el vector idf (fit) y transformar una matriz de recuento a una representacion tf-idf
print("\nUsando Tf-idf :")

print("Ahora el tamaño del conjunto de datos es el siguiente")
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
print(X_tfidf.shape)

#contando las 10 primeras palabras
reverse_dic = {}
for key in cntizer.vocabulary_:
    reverse_dic[cntizer.vocabulary_[key]] = key
top_10 = np.asarray(np.argsort(np.sum(X_cnt, axis=0))[0,-10:][0, ::-1]).flatten()
[reverse_dic[v] for v in top_10]


personality_type = [ "IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) / Sensing (S)", 
                   "FT: Feeling (F) / Thinking (T)", "JP: Judging (J) / Perceiving (P)"  ]

for l in range(len(personality_type)):
    print(personality_type[l])

print("X: 1er post en la representacion tf-idf\n%s" % X_tfidf[0])

print("Para el tipo de personalidad MBTI: %s" % translate_back(list_personality[0,:]))
print("Y: 1a fila del MBTI binarizado: %s" % list_personality[0,:])

# Posts con la representacion tf-idf
X = X_tfidf

#Modelo de Random Forest para el conjunto de datos MBTI
#Entrenamiento individual de cada tipo de personalidad mbti
for l in range(len(personality_type)):
    
    Y = list_personality[:,l]

    # dividir los datos en conjuntos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)

    # ajustar el modelo a los datos de entrenamiento
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # hacer predicciones para los datos de prueba
    y_pred = model.predict(X_test)
    
    predictions = [round(value) for value in y_pred]
    # evaluar las predicciones
    accuracy = accuracy_score(y_test, predictions)
    
    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))

# Stocastic Gradient Descent for MBTI dataset
#Entrenamiento individual de cada tipo de personalidad mbti
for l in range(len(personality_type)):

    Y = list_personality[:,l]

    # dividir los datos en conjuntos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)

    # ajustar el modelo a los datos de entrenamiento
    model = SGDClassifier() 
    model.fit(X_train, y_train)

    # hacer predicciones para los datos de prueba
    y_pred = model.predict(X_test)
    
    predictions = [round(value) for value in y_pred]
    # evaluar las predicciones
    accuracy = accuracy_score(y_test, predictions)
    
    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))

# Logistic Regression for MBTI dataset
#Entrenamiento individual de cada tipo de personalidad mbti
for l in range(len(personality_type)):

    Y = list_personality[:,l]

    # dividir los datos en conjuntos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)

    # ajustar el modelo a los datos de entrenamiento
    model = LogisticRegression() 
    model.fit(X_train, y_train)

    # hacer predicciones para los datos de prueba
    y_pred = model.predict(X_test)
    
    predictions = [round(value) for value in y_pred]
    # evaluar las predicciones
    accuracy = accuracy_score(y_test, predictions)
    
    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))

#2 KNN model for MBTI dataset
#Entrenamiento individual de cada tipo de personalidad mbti
for l in range(len(personality_type)):

    Y = list_personality[:,l]

    # dividir los datos en conjuntos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)

    # ajustar el modelo a los datos de entrenamiento
    model = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
    model.fit(X_train, y_train)

    # hacer predicciones para los datos de prueba
    y_pred = model.predict(X_test)
    
    predictions = [round(value) for value in y_pred]
    # evaluar las predicciones
    accuracy = accuracy_score(y_test, predictions)
   
    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))

#XGBoost model for MBTI dataset 
#Entrenamiento individual de cada tipo de personalidad mbti
for l in range(len(personality_type)):
    
    Y = list_personality[:,l]

    # dividir los datos en conjuntos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)

    # ajustar el modelo a los datos de entrenamiento
    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # hacer predicciones para los datos de prueba
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluar las predicciones
    accuracy = accuracy_score(y_test, predictions)
    
    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))

# SVM model for MBTI dataset
#Entrenamiento individual de cada tipo de personalidad mbti
for l in range(len(personality_type)):
    
    Y = list_personality[:,l]

    # dividir los datos en conjuntos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)

    # ajustar el modelo a los datos de entrenamiento
    model = SVC(random_state = 1)
    model.fit(X_train, y_train)

    # hacer predicciones para los datos de prueba
    y_pred = model.predict(X_test)
    
    predictions = [round(value) for value in y_pred]
    # evaluar las predicciones
    accuracy = accuracy_score(y_test, predictions)
    
    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))


my_posts  = """ Hi I am 21 years, currently, I am pursuing my graduate degree in computer science and management (Mba Tech CS ), It is a 5-year dual degree.... My CGPA to date is 3.8/4.0 . I have a passion for teaching since childhood. Math has always been the subject of my interest in school. Also, my mother has been one of my biggest inspirations for me. She started her career as a teacher and now has her own education trust with preschools schools in Rural and Urban areas. During the period of lockdown, I dwelled in the field of blogging and content creation on Instagram.  to spread love positivity kindness . I hope I am able deliver my best to the platform and my optimistic attitude helps in the growth that is expected. Thank you for the opportunity. """

# The type is just a dummy so that the data prep function can be reused
mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

my_posts, dummy  = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True)

my_X_cnt = cntizer.transform(my_posts)
my_X_tfidf =  tfizer.transform(my_X_cnt).toarray()

# setup parameters for xgboost
param = {}
param['n_estimators'] = 200
param['max_depth'] = 2
param['nthread'] = 8
param['learning_rate'] = 0.2
param['eval_metric'] = 'mlogloss'
#Entrenamiento individual de cada tipo de personalidad mbti
for l in range(len(personality_type)):
    Y = list_personality[:,l]

    # dividir los datos en conjuntos de entrenamiento y de prueba
    seed = 7
    test_size = 0.30
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # ajustar el modelo a los datos de entrenamiento
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    # hacer predicciones para los datos de prueba
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluar las predicciones
    accuracy = accuracy_score(y_test, predictions)
    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))

#XGBoost model for MBTI dataset
result = []
#Entrenamiento individual de cada tipo de personalidad mbti
for l in range(len(personality_type)):
    print("Clasificando %s ...m" % (personality_type[l]))
    
    Y = list_personality[:,l]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)

    # ajustar el modelo a los datos de entrenamiento
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    
   # hacer predicciones para los datos
    y_pred = model.predict(my_X_tfidf)
    result.append(y_pred[0])

print("El resultado es: ", translate_back(result))
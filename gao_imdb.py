#- Charge et vectorise ces données
#- Entraîne et compare des classifieurs sur ce jeu de données
#*features* : La longueur moyenne des mots, 
# #le nombre ou le type  d'adjectifs,la présence d'entités nommées, …
#les importations nécessaire:
# Les librairies pour préparer les données 
import pandas as pd 
import os 
#Visualisation
import matplotlib.pyplot as plt 

#text prétraitement 
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_lg")
stopwords = list(STOP_WORDS)
import re, string
import numpy as np

#dataset traitement 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# modèles et evaluation
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.simplefilter("ignore")
# #pour charger les données 
# # 1 - récupérer tous les textes dans les fichiers txt dans le dossier neg et pos 
# # 2 - créer un tableaux et mettre en colonne ces données 
# # 3 - ajouter une colonne de neg ou pos aux données correspondant 
def txt_prep(full_txt,chemin):
    """ cette fonction prend un nom du fichier et son répertoire comme arguments. 
    la sortie est l'ensemble des contenus dans tous les fichiers txt dans ce répertoire """
    f = os.listdir(chemin)
    with open(full_txt,'w') as outfile:
        for file in f:
            with open(chemin+'/'+ file) as contenu:
                outfile.write(contenu.read())
            outfile.write("\n")
        return full_txt

def full_txt_prep ():
    """ cette fonction regroupe les deux fichiers txt contenant les contenus du répertoire neg et pos"""
    full_neg = txt_prep('./neg/full_neg.txt','./neg') 
    full_pos =txt_prep('./pos/full_pos.txt','./pos') 
    return full_neg,full_pos


def txt2df(file,cat):
    """ cette fonction prend un nom du fichier txt et la valeur à donner à la colonne du dataframe arguments. 
    la sortie est un dataframe contenant deux colonnes : 
    contenu dans le fichier txt (séparé par ligne), et la nature de sa catégorie """
    f = open (file,'r')
    f = f.read()
    f2list = f.split('\n')
    df = pd.DataFrame(f2list,columns =['texte'])
    df['cat'] = cat
    return df

def full_dataframe (full_neg,full_pos):
    """ cette fonction prend les deux dataframes neg et pos comme arguments. 
    la sortie est un ensemble de dataframe combiné par les deux petits dataframe: """
    dfneg = txt2df(full_neg,'neg')
    dfpos = txt2df(full_pos,'pos')
    df_total = pd.concat([dfneg, dfpos], ignore_index = True)
    df_total.drop_duplicates(keep = 'first', inplace=True)
    return df_total

# Text cleaning (remove punctuations and special characters)
def clean_text(texte):
    """ cette function prend un string comme argument. 
  la sortie est lémmatizé et nettoyée"""
    doc = nlp(texte)
    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    clean_tokens = []
    for token in tokens:
        if token not in string.punctuation and token not in stopwords:
            clean_tokens.append(token)               
    return clean_tokens
#df_total['texte'] = df_total['texte'].apply(lambda x: clean_text(x))
#print(df_total)



#grid search for all models 


def fit_model(model, X_train, y_train, X_test, y_test):
    """ cette function prend en entrée model, X_train, y_train, X_test, y_test comme arguments. 
  la sortie s'agit de l'évaluation de ses performance: f1-score avec sa classification report pour train and test
 et la matrice de confusion. """

    print ('\n\n',model,":","\n")
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    cm=confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    print('F1 score sur le train :', f1_score(y_train, y_pred_train, average='micro'))
    print(f"{model} --> {model.score(X_train, y_pred_train)}")
    print(classification_report(y_train, y_pred_train))
    print ("***********************\n")
    print('F1 score sur le test :',f1_score(y_test, y_pred_test, average='micro'))
    print(f"{model} --> {model.score(X_test, y_test)}")
    print(classification_report(y_test, y_pred_test))
    disp.plot()
    plt.show()



def main():
    full_neg,full_pos = full_txt_prep()
    df_total = full_dataframe(full_neg,full_pos) 
    #print(df_total)
    #print(df_total['cat'].value_counts())
    fig,ax = plt.subplots(ncols = 2, figsize = (15,10))
    ax[0].hist([len(x) for x in df_total['texte']])
    ax[1].hist([len(x) for x in df_total['texte']], bins = 500)
    plt.suptitle('Histogram of string lengths')
    plt.show()
    X = df_total['texte']
    y = df_total['cat'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify = y)
    tfidf = TfidfVectorizer(tokenizer =clean_text,max_features= 500, min_df= 1)
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)
    #---- grid search paramètres ----------------------
    param_grid_rforest = {
      'n_estimators': [10,20,50,100], 'max_depth' : [3,5,7], 
      'criterion' :['gini', 'entropy'], 'max_features':['sqrt'],}
    param_grid_svc = {
      'C':[1,10,100], 'gamma':[1,0.1,0.001], 'kernel':['linear','rbf','sigmoid']}
    param_grid_lr = {
      'C':np.logspace(-3,3,7), 'penalty':['l1','l2']}
    param_grid_mnb = {
      'alpha': [1,0.5, 0.1, 0.01, 0.001, 0.0001]}
     ## #---- MODÈLES AVEC LES HYPERPARAMÈTRES TROUVÉS PAR GRID SEARCH----------------------
    models_tuned = [
    RandomForestClassifier(random_state= 42,criterion='entropy', max_depth=7, max_features='sqrt', n_estimators= 50),
    MultinomialNB(alpha= 0.1),
    LogisticRegression(random_state= 42,C= 10.0, penalty= 'l2'),
    SVC(C=10, gamma=1, kernel='rbf')
    ]
    for model in models_tuned:
        fit_model(model, X_train, y_train, X_test, y_test)
        print("Cross Validation Score :",model, cross_val_score(model,X_train, y_train).mean())
        print ("\n\n\n")
    # #---- modèles----------------------
# rdf= RandomForestClassifier(random_state= 42)
# mnb= MultinomialNB()
# lr= LogisticRegression(random_state= 42)
# svc = SVC()

#  #---- RANDOM FOREST---------------------
# grid_rforest = GridSearchCV(rdf, param_grid_rforest, cv=5, verbose=3, n_jobs=-1)
# grid_rforest.fit(X_train, y_train)
# best_Hyperparameters = grid_rforest.best_params_
# print ("Best Hyperparameters rdf :",best_Hyperparameters )
# #---- mnb= MultinomialNB----------------------
# grid_mnb = GridSearchCV(mnb, param_grid_mnb, cv=5, verbose=3, n_jobs=-1)
# grid_mnb.fit(X_train, y_train)
# best_Hyperparameters = grid_mnb.best_params_
# print ("Best Hyperparameters mnb :",best_Hyperparameters )
# #---- LogisticRegression----------------------
# grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, verbose=3, n_jobs=-1)
# grid_lr.fit(X_train, y_train)
# best_Hyperparameters = grid_lr.best_params_
# print ("Best Hyperparameters lr :",best_Hyperparameters )
# #---- SVC----------------------
# grid_svc = GridSearchCV(svc, param_grid_svc, cv=5, verbose=3, n_jobs=-1)
# grid_svc.fit(X_train, y_train)
# best_Hyperparameters = grid_svc.best_params_
# print ("Best Hyperparameters svc :",best_Hyperparameters )



if __name__ == "__main__":
  main()
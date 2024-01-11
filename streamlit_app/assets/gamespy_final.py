# -*- coding: utf-8 -*-

from re import sub
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from joblib import load
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from time import sleep
from datetime import date
from datetime import datetime
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import io

df = pd.read_csv('04_GameSpy_FinalDataset.csv')
df_init = pd.read_csv('vgsales.csv',sep=',')
df_model4 = pd.read_csv("df_tomodel4.csv",sep=",",index_col='Unnamed: 0')
df_model6 = pd.read_csv("df_tomodel6.csv",sep=",",index_col='Unnamed: 0')

st.set_option('deprecation.showPyplotGlobalUse', False)

radio = st.sidebar.radio("Menu",("Contexte", "Description des données", "Analyse du marché","Prédictions des ventes","Testez vos données","A propos de l'équipe"))

if radio == "Contexte":
    st.header("GameSpy")
    st.title(":video_game: Analyse et prédictions des ventes de jeux vidéos")
    st.image("kelly-sikkema.jpg")
    st.write("Au cours des dernières années, l’**industrie des jeux vidéo a connu une croissance rapide**. Dès les années 90, l’**essor des consoles domestiques** (par rapport aux jeux d’arcades) ont fait leur entrée sur le marché. C’est alors le début de l’âge d’or des jeux vidéo avec la sortie de consoles qui resteront plusieurs années sur le marché (Playstation, Xbox, Wii, etc.)...")
    st.write("Loin de stopper cette dynamique, la **pandémie de Covid-19 l’a amplifiée**. L'année 2020 a, en effet, été une année record, puisque les mesures de confinement prises par la plupart des pays du monde ont stimulé les activités virtuelles. En **France**, par exemple, l’industrie des jeux vidéo a réalisé un **chiffre d’affaires de 5,3 milliards d’euros** cette année-là, selon le Syndicat des éditeurs de logiciels de loisirs (SELL).")
    image=Image.open('gaming-history-revenue.jpg')
    st.image(image,caption="L'évolution des ventes de jeux vidéo - Source Visual Capitalist")
    st.write("A la vue de cette importante croissance, **beaucoup d’entreprises investissent** d’énormes sommes d’argent dans cette industrie. Comme tout autre entreprise, elles souhaitent maintenir une connaissance plus ou moins précise du risque lors de la sortie d’un nouveau produit.")
    st.write("**Ce projet a pour objectif d'aider les éditeurs de jeux vidéos à prédire les ventes d'un nouveau jeu vidéo.**")

if radio == "Description des données":
    st.header("GameSpy")
    st.title(":mag_right: Quels sont les éléments qui composent la vente de jeux vidéos ?")
    st.image("rangees-de-utilise-pour-les-consoles-de-jeux.jpg")
    st.header("**Data disponibles**")
    with st.container():
        st.subheader("Dataset initial")
        st.write("Nous avons récupéré un dataset initial avec **16598 lignes**.") 
        st.write("Dataset fourni : https://www.kaggle.com/gregorut/videogamesales")
        st.write("Ce dataset propose un classement des jeux vidéos en fonction de leur nombre de ventes, dont voici le détail et les informations globales qu’il contient : ")
        image=Image.open('dataset_initial_vgsales.jpg')
        st.image(image,caption="Description du dataset initial")
        st.write("La **clé d’entrée** de chaque ligne est le **nom du jeu** et une **plateforme de jeu**.")
        st.write("Les jeux présentés dans le dataset ont une **date de lancement** entre **1980** à **2020**.")
        st.write("Voici un aperçu :")
        st.dataframe(df_init)
        st.write("Le dataset est composé de très peu de données et notamment **peu de données** pouvant expliquer le niveau des ventes. **Nous les avons donc complétées**")

    with st.container():
        st.subheader("Enrichissement de données via scraping")
        st.write("Nous avons ciblé 3 sites populaires, spécialisés dans les revues de jeux vidéo : **Gamekult, Metacritic et jeuxvideo.com**.")
        st.write("Notre objectif était de récupérer des **données pouvant expliquer le volume des ventes** d’un jeu par rapport à un autre afin d’enrichir le dataset initial avant l’analyse approfondie.")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("logo_gamekult.jpg")
            st.write("**Gamekult**")
            st.write("Pour chaque jeu présenté sur le site, récupération : **nom du jeu**, **note des joueurs**, **nombre d’avis des joueurs**, **note attribuée par Gamekult**, **distributeur**, **développeur**, **éditeur** et **licence**.")

        with col2:
            st.image("logo_metacritic.jpg")
            st.write("**Metacritic**")
            st.write("Pour chaque jeu présenté sur le site, récupération : **nom du jeu**, **plateforme**, **studio**, **note Metacritic**, **note des utilisateurs** et **la date de sortie du jeu**")

        with col3:
            st.image("logo_jv.jpg")
            st.write("**jeuxvideo.com**")
            st.write("Pour chaque jeu présenté sur le site, récupération : **nom de jeu**, **plateformes**, **nombre d’avis**, **note des joueurs par plateforme**, **note journaliste** et **la date de lancement du jeu**.")
        
        st.write("")
        st.write("En scrapant **Wikipédia**, nous avons également pu récupérer les informations suivantes :")
        st.write("- Les jeux faisant **partie d'une série** de plusieurs jeux : https://fr.wikipedia.org/wiki/Liste_de_s%C3%A9ries_de_jeux_vid%C3%A9o")
        st.write("- Les jeux faisant **partie d'une série les plus vendues** : https://fr.wikipedia.org/wiki/Liste_des_s%C3%A9ries_de_jeux_vid%C3%A9o_les_plus_vendues")
        st.write("- Les jeux ayant **été cités lors du festival E3** : https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Electronic_Entertainment_Expo")
        st.write("- Les jeux ayant **été lancés en simultané d'une console** : https://fr.wikipedia.org/wiki/Liste_de_jeux_au_lancement_de_consoles_de_jeux_vid%C3%A9o")

    
    with st.container():
        st.subheader("Enrichissement de données via des moyennes mobiles")
        st.write("Pour préparer nos données au machine learning, nous devons écarter certains biais qui peuvent avoir une influence sur l’algorithme d’apprentissage comme les **notes** (des joueurs et des testeurs) et le **nombre d’avis** laissés par les joueurs qui **sont disponibles qu'après lancement.**")
        st.write("Bien évidemment, kes **ventes sont également des données que nous ne pourrons pas garder** pour la construction de notre modèle prédictif.")
        st.write("Malgré tout, les dimensions cités ci-dessus des **jeux précédents sur une même série** ou encore des **jeux précédents publiés par le même studio** pouvaient nous dire beaucoup...")
        st.write("Nous avons donc créé des **moyennes mobiles** dont voici le fonctionnement :")
        st.image("moyennes_mobiles.png")
        
    with st.container():
        st.subheader("Enrichissement de données depuis le dataset initial")
        st.write("Depuis le dataset inital, nous avons identifié les **20 studios** réalisant le plus de ventes.")
        st.write("Nous avons **enrichi tous les jeux ayant été lançé par un de ces top 20 studios** avec une valeur spécifique par rapport aux autres jeux (1 vs 0).")
        st.write("Nous avions en tête qu'un jeu lançé par un des studios du top 20 pouvait voir **ses ventes influencées par la renommé du studio**.")
        
    st.header("**Gestion des NaNs**")
    st.write("Nous avons géré les NaNs des variables Publisher, Studio et Licence en remplaçant d'aboard les valeurs manquantes des **licences par les noms** des jeux puis nous avons **complété entre elles les valeurs manquantes des publishers, studios, licences**.")
    st.write("Nous avons également complété entre elles les **notes des journalistes et les notes des joueurs**:")
    st.image("gestion_nan.png")
    st.write("Pour remplir nos **dernières valeurs manquantes** et pour ne pas faire fuiter nos données, nous avons calculé pour chaque jeu des **moyennes sur les jeux lancés antérieurement**")
        
if radio == "Analyse du marché":
    st.header("GameSpy")
    st.title(":bar_chart: Analyse du marché des jeux vidéos")
    st.image("digital-marketing.jpg")
    selectbox = st.selectbox(
     "Choisissez le type d'analyse que vous souhaitez faire ?",
     ('Aucun choix','Evolution des ventes globales par an et par mois','Répartition des ventes globales par région','Genre de jeux les plus vendus par région','Comparaison des ventes et lancements de jeux par plateformes',
      'Comparaison des ventes et lancements de jeux par genre','Comparaison du nombre de ventes et des notes moyennes par genre', 'Comparaison des jeux les plus vendus et de leurs notes', 'Caractéristiques des jeux les plus vendus',"Analyse des notes attribués et des lancements de jeux par éditeur",
      "Analyse des notes attribués et des lancements de jeux par studio"))
    
    if selectbox == 'Evolution des ventes globales par an et par mois':
        st.header("Croissance des ventes à partir des années 90")
        st.image("graph1.jpg")
        st.write("**L'essor des revenus à partir des années 90** se retrouve bien dans notre dataset.")
        st.write("Dans notre dataset, les ventes de jeux vidéo atteignent leur paroxysme en 2010 et décroissent par la suite. Mais cela s’explique par le fait qu’il y a beaucoup moins de jeux présents dans le dataset à partir de ces années-là.")
        st.write("Nous observons déjà grâce à cette première visualisation que **l’Amérique du Nord a une part importante des ventes mondiales**.")
        st.header("La fin d'année est une période forte pour le marché des jeux vidéos")
        st.image("graph2.jpg")
        st.write("Nous voyons que la **fin d’année** est propice à la sortie de nouveaux jeux vidéo et cela est certainement lié aux fêtes de fin d’année.")
        st.write("**Mars et Juin** sont des mois intéressants pour le marché car les ventes sont élevées par rapport au nombre de jeux lancés.")
        
    if selectbox == 'Répartition des ventes globales par région':
        st.header("La répartition des ventes par région évolue au fur et à mesure des décennies")
        st.image("graph3.jpg")
        st.write("Ce que nous avons observé dans le premier graphique se confirme bien ici. **L’Amérique du Nord possède la majorité du marché mondial** des jeux vidéo.")
        st.write("Cependant, nous voyons bien ici qu’elle avait quasiment un monopole sur le marché du jeu vidéo en 1980 et qu’en **2015**, **sa part de marché est égale à celle de l'Europe**, qui a évolué au fur et à mesure des années. ")

    if selectbox == 'Genre de jeux les plus vendus par région':
        st.header("Le genre des jeux les plus vendus n’est pas le même en fonction des régions du monde")
        st.image("graph4.jpg")
        st.write("Les **jeux d’action, de sport et de tirs** se retrouvent donc dans le **top des ventes** mondiales. Seul le **Japon** se démarque, en préférant largement les **jeux de rôles** parmi tous les autres genres.")
        
    if selectbox == 'Comparaison des ventes et lancements de jeux par plateformes':
        st.header("Comparaison des ventes et lancements de jeux par plateformes")
        st.image("graph5.jpg")
        st.write("Nous voyons qu'il n'y a **pas de relations** entre ces deux variables. Par exemple la Xbox et la PS3 (respectivement plateformes top 2 et top 3 sur le niveau de ventes de jeux) ont eu bien moins de jeux compatibles (⅓ de moins) que la DS qui se retrouve en 5ème position au regard des ventes de jeux par plateforme.")
    
    if selectbox == 'Comparaison des ventes et lancements de jeux par genre':
        st.header("Comparaison des ventes et lancements de jeux par genre")
        st.image("graph6.jpg")
        st.write("Sur l’échelle des genres de jeux, il n’y a **pas non plus de corrélation entre le nombre de ventes par genre et le nombre de jeux lancés**. Comme l’indique le graphique ci-dessus, certains genres se retrouvent dans les top ventes alors que peu de jeux associés ont été lancés.")
        
    if selectbox == 'Comparaison du nombre de ventes et des notes moyennes par genre':
        st.header("Comparaison du nombre de ventes et des notes moyennes par genre")
        st.image("graph7.jpg")
        st.write("L’échelle des notes oscille entre **6.4 et 7**, ce qui montre qu’il y a **peu de variations** entre les notes moyennes et qu’il n’y a pas un grand écart entre les genres.")
        st.write("Cependant, nous voyons que certains **genres sont un peu mieux notés que d’autres** et que cela n’a **pas de lien avec le niveau de ventes** du genre en question.")
    
    if selectbox == 'Comparaison des jeux les plus vendus et de leurs notes':
        st.header('Comparaison des jeux les plus vendus et de leurs notes')
        st.image("graph8.jpg")
        st.write("Les notes de **catégorie “bien”**, c’est à dire entre 5 et 7,5 sur 10, **sont majoritaires** sur toutes les catégories de ventes.")
        st.write("Nous observons parfois peu de différence sur les notes données entre deux catégories de ventes.")
        st.write("Globalement, nous pouvons quand même dire que **les jeux les plus vendus ont plus de très bonnes notes et moins de mauvaises et moyennes notes**.")
    
    if selectbox == 'Caractéristiques des jeux les plus vendus':
        st.header("Les jeux qui font partie d’une licence sont les plus vendus")
        st.image("graph9.jpg")
        st.write("Nous pouvons clairement conclure ici que les **jeux rattachés à une licence (série de jeux) sont mieux vendus** que des jeux indépendants car parmi **le top 10 des jeux** avec les meilleures ventes de notre dataset, **tous font partie d’une licence**.")
        st.header("Les jeux qui comptabilisent le plus de ventes ont été lancés par les meilleurs studios")
        st.image("graph10.jpg")
        st.write("En reprenant le top 10 des jeux les plus vendus du dataset et en regardant s’ils sont rattachés au 20 studios comptabilisant les meilleures ventes (encore une fois du dataset), nous voyons que **9 jeux sur 10 sont rattachés à un top studio**. Seul le jeu Tétris a été lancé par un studio comptabilisant moins de ventes.")
        
    if selectbox == "Analyse des notes attribués et des lancements de jeux par éditeur":
        st.header("Comparaison du nombre de jeux produits par éditeur et ses ventes associées")
        st.image("graph11.jpg")
        st.write("Nous voyons qu’il y a pour **certains éditeurs de grandes disparités entre le nombre de jeux produits et les ventes** cumulées au globales")
        st.write("Nous voyons donc par ce graphique que le **nombre de jeux produit par un éditeur n’est pas représentatif du nombre de ventes générées**.")
        st.header("Comparaison du nombre de jeux produits par éditeur et des notes moyennes associées")
        st.image("graph12.jpg")
        st.write("L’échelle des notes varie entre **6 et 7.4 sur 10**. Ce qui veut dire qu’il n’y a pas de grandes différences sur les notes moyennes globales en fonction des éditeurs.")
        st.write("Dans le détail, nous pouvons quand même voir qu’**aucun lien est établi entre l’éditeur du jeu et la note moyenne donnée** par les joueurs et les journalistes vu le comportement de la courbe.")
        
    if selectbox == "Analyse des notes attribués et des lancements de jeux par studio":
        st.header("Comparaison du nombre de jeux lancés par studio et ses ventes associées")
        st.image("graph12.jpg")
        st.write("La **relation nombre de jeux lancés vs nombre de ventes semble plus forte** pour les studios par rapport aux éditeurs comme nous l’avons vu précédemment.")
        st.write("Malgré cela, nous ne pouvons **pas affirmer qu’il existe une vraie corrélation** car nous voyons que pour plusieurs studios, il **existe un décalage entre le nombre de jeux lancés et le niveau de ventes générées**.")
        st.header("Comparaison du nombre de jeux lancés par studio et des notes moyennes associées")
        st.image("graph13.jpg")
        st.write("Comme pour les éditeurs, l’échelle de notes est serrée avec des notes entre **6.5 et 8.5 sur 10**. Ici aussi, nous voyons donc qu’il n’y a pas une grande différence entre les studios et que les notes moyennes sont plutôt bonnes peu importe le niveau de ventes.")
        st.write("En analysant plus précisément la courbe, nous voyons que **la note moyenne et le niveau de ventes vivent de manière indépendante** et qu’aucune des deux variables peut nous aider à prédire l’autre.")
        
if radio == "Prédictions des ventes":
    st.header("GameSpy")
    st.title(":moneybag: Prédictions des ventes de jeux vidéos")
    st.image("pietro-jeng.jpg")
    st.write("Nous vous proposons 2 catégorisations (en millions d’exemplaires vendus) :")
    st.write("**4 catégories de vente** :")
    st.write("- **de 0 à 0.1 millions** d'exemplaires")
    st.write("- **de 0.1 à 0.249 millions** d'exemplaires")
    st.write("- **de 0.25 à 1 millions** d'exemplaires")
    st.write("- **supérieur à 1 millions** d'exemplaires")
    st.write("--> Plus grande échelle entre les catégories donc moins de précisions mais un meilleur score de prédictions")
    st.write("**6 catégories de vente** :")
    st.write("- **de 0 à 0.1 millions** d'exemplaires")
    st.write("- **de 0.1 à 0.249 millions** d'exemplaires")
    st.write("- **de 0.25 à 0.499 millions** d'exemplaires")
    st.write("- **de 0.5 à 0.999 millions** d'exemplaires")
    st.write("- **de 1 à 5 millions** d'exemplaires")
    st.write("- **et supérieur à 5 millions** d'exemplaires")
    st.write("--> Catégories plus ressérées donc plus précises mais avec un taux d'erreur plus élevé")
    categorie=st.radio("Choisissez le nombre de catégories de ventes sur lequel vous souhaitez réaliser les prédictions",("4 catégories","6 catégories"))
    
    st.header("Il est temps d'entrainer le modèle !")
    st.write("Choisissez les hyperparamètres que vous voulez appliquer au modèle")
        
    if categorie == "4 catégories":
        target=df_model4["Sales_cat"]
        data=df_model4.drop("Sales_cat",axis=1)

        X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2)
        
        st.dataframe(X_test)
        
        depth=st.slider("Quel est la profondeur maximale que vous voulez donner au modèle (max_depth)",min_value=10,max_value=100,value=20,step=10)
        criter=st.selectbox("Quel critère voulez-vous appliquer ? (criterion)",options=["gini","entropy"])
        features=st.selectbox("Combien de variables voulez-vous prendre en compte ? (max_features)",options=["auto","sqrt","log2"])
        
        rfc=RandomForestClassifier(max_depth=depth,criterion=criter,max_features=features)
        
        rfc.fit(X_train,y_train)
        y_pred_test=rfc.predict(X_test)
        
        st.subheader("Score du modèle est")
        st.write(" RF score :" , rfc.score(X_test,y_test))
        
        st.subheader("Matrice de confusion")
        plot_confusion_matrix(rfc,X_test,y_test,labels=[1,2,3,4])
        st.pyplot()
        
        st.subheader("Les features qui ont le plus influencées le modèle sont...")
        
        def plot_feature_importance(importance,names,model_type):

            feature_importance = np.array(importance)
            feature_names = np.array(names)

            data={'feature_names':feature_names,'feature_importance':feature_importance}
            fi_df = pd.DataFrame(data)

            fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

            fig, ax = plt.subplots(figsize=(10,8))
            ax=sns.barplot(x=fi_df['feature_names'], y=fi_df['feature_importance'])
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
        plot_feature_importance(rfc.feature_importances_,data.columns,'RANDOM FOREST')
        
    if categorie == "6 catégories":
        target=df_model6["Sales_cat"]
        data=df_model6.drop("Sales_cat",axis=1)

        X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2)
        
        depth=st.slider("Quel est la profondeur maximale que vous voulez donner au modèle (max_depth)",min_value=10,max_value=100,value=20,step=10)
        criter=st.selectbox("Quel critère voulez-vous appliquer ? (criterion)",options=["gini","entropy"])
        features=st.selectbox("Combien de variables voulez-vous prendre en compte ? (max_features)",options=["auto","sqrt","log2"])
        
        rfc=RandomForestClassifier(max_depth=depth,criterion=criter,max_features=features)
        
        rfc.fit(X_train,y_train)
        y_pred_test=rfc.predict(X_test)
        
        st.subheader("Score du modèle est...")
        st.write(" RF score :" , rfc.score(X_test,y_test)) 
        
        st.subheader("Matrice de confusion")
        plot_confusion_matrix(rfc,X_test,y_test,labels=[1,2,3,4])
        st.pyplot()
        
        st.subheader("Les features qui ont le plus influencées le modèle sont...")
        
        def plot_feature_importance(importance,names,model_type):

            feature_importance = np.array(importance)
            feature_names = np.array(names)

            data={'feature_names':feature_names,'feature_importance':feature_importance}
            fi_df = pd.DataFrame(data)

            fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

            fig, ax = plt.subplots(figsize=(10,8))
            ax=sns.barplot(x=fi_df['feature_names'], y=fi_df['feature_importance'])
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
        plot_feature_importance(rfc.feature_importances_,data.columns,'RANDOM FOREST')
        
if radio == "Testez vos données":
    st.header("GameSpy")
    st.title(":rocket: Testez vos données")
    st.image("spacex.jpg")
    st.write("Nos prédictions sont effectuées sur 4 catégories de vente grâce au modèle de classification Random Forrest")


    st.subheader("Vous avez juste le nom d'un jeu...")


    def getMonthNumber(mois):
        monthNb = None
        if mois == 'janvier':
            monthNb = 1
        elif mois == 'février':
            monthNb = 2
        elif mois == 'mars':
            monthNb = 3
        elif mois == 'avril':
            monthNb = 4
        elif mois == 'mai':
            monthNb = 5
        elif mois == 'juin':
            monthNb = 6
        elif mois == 'juillet':
            monthNb = 7
        elif mois == 'août':
            monthNb = 8
        elif mois == 'septembre':
            monthNb = 9
        elif mois == 'octobre':
            monthNb = 10
        elif mois == 'novembre':
            monthNb = 11
        elif mois == 'décembre':
            monthNb = 12
        return monthNb


    merged = pd.read_csv('GameSpy_FinalDataset_merged.csv', index_col=0)

    # st.write(merged.Platform.unique())
    model = load('dt_class4.joblib')

    # Formulaire pour la demo simple
    with st.form(key='form_simple',clear_on_submit =True):
        name = st.text_input('Renseignez un jeu et validez')
        submit_button_simple = st.form_submit_button(label='Valider')
     
    
    if submit_button_simple:
        driver = webdriver.Chrome(ChromeDriverManager().install())
        sleep(1)
        driver.get('https://www.google.com/')
        driver.find_element_by_id('L2AGLb').click()
        barre = driver.find_element_by_name('q')
        barre.send_keys(name)
        barre.send_keys(Keys.ENTER)
        try:
            info_box = driver.find_element_by_id('kp-wp-tab-overview').text
            info_box_split = info_box.split('\n')
            driver.close()
        except:
            info_box = None
            driver.close()
    
        year = None
        allMonths = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
        mois = None
        studio = None
        licence = None
        publisher = None
        platform = None
        cle = None
        
        if info_box:
            for element in info_box_split:
                if "Date de sortie initiale : " in element:
                    year = element[-4:]
                    try:
                        mois = element.replace("Date de sortie initiale : ", "").split(' ')[1]
                        if mois in allMonths:
                            mois = getMonthNumber(mois)
                    except:
                        mois = None
                if "Développeur : " in element:
                    studio = element.replace("Développeur : ", "")
        
                if "Développeurs : " in element:
                    try:
                        studio = element.replace("Développeurs : ", "").split(', ')[0]
                    except:
                        studio = element.replace("Développeurs : ", "")
                
                if "Série : " in element:
                    try:
                        licence = element.replace("Série : ", "").split(', ')[0]
                    except:
                        licence = element.replace("Série : ", "")
                
                if "Éditeur : " in element:
                    publisher = element.replace("Éditeur : ", "")
            
                if "Éditeurs : " in element:
                    try:
                        publisher = element.replace("Éditeurs : ", "").split(', ')[0]
                    except:
                        publisher = element.replace("Éditeurs : ", "")    
                
                if "Plate-forme : " in element:
                    platform = element.replace("Plate-forme : ", "")
                
                if "Plate-formes : " in element:
                    try:
                        platform = element.replace("Plate-formes : ", "").split(', ')[0]
                    except:
                        platform = element.replace("Plate-formes : ", "")
            
            if licence:
                if licence in merged['GK_licence'].unique():
                    cle = "licence"
                else:
                    if studio:
                        if studio in merged['Studio'].unique():
                            cle = "studio"
                        else:
                            if publisher:
                                if publisher in merged['Publisher'].unique():
                                    cle = "publisher"
                                else:
                                    cle = None
                            else:
                                cle = None
            elif studio:
                if studio in merged['Studio'].unique():
                    cle = "studio"
                else:
                    if publisher:
                        if publisher in merged['Publisher'].unique():
                            cle = "publisher"
                        else:
                            cle = None
                    else:
                        cle = None
            elif publisher:
                if publisher in merged['Publisher'].unique():
                    cle = "publisher"
                else:
                    cle = None


            if cle == 'licence':
                try: 
                    search_df = merged[merged['GK_licence']==licence].sort_values(by=['Date_Sortie']).iloc[-2:,:]
                except:
                    search_df = merged[merged['GK_licence']==licence].sort_values(by=['Date_Sortie']).iloc[-1,:]
            elif cle == 'studio':
                try: 
                    search_df = merged[merged['Studio']==studio].sort_values(by=['Date_Sortie']).iloc[-2:,:]
                except:
                    search_df = merged[merged['Studio']==studio].sort_values(by=['Date_Sortie']).iloc[-1,:]
            elif cle == 'publisher':
                try: 
                    search_df = merged[merged['Publisher']==publisher].sort_values(by=['Date_Sortie']).iloc[-2:,:]
                except:
                    search_df = merged[merged['Publisher']==publisher].sort_values(by=['Date_Sortie']).iloc[-1,:]
            else:
                st.markdown("Nous n'avons pas assez d'informations dans notre dataset pour prédire les ventes.")
        
        else:
            st.markdown("Le scraping n'a pas pu ramener d'informations.")
                
                
        if cle:
            if platform == 'PlayStation 4':
                platform = 11.0
            elif platform == 'Xbox One':
                platform = 18.0
            else:
                platform = 32.0
            
            if year:
                year=year
            else:
                year=2022
            if mois:
                mois=mois
            else:
                mois = merged['Mois'].mode()[0]
            genre = round(search_df['Genre_label'].mean())
            publisher_label = round(search_df['Publisher_label'].mean())
            studio_label = round(search_df['Studio_label'].mean())
            licence_label = round(search_df['GK_licence_label'].mean())
            rm_publisher = search_df['RM_Publisher'].mean()
            rm_publisher_score = search_df['RM_Publisher_score'].mean()
            rm_publisher_rate = search_df['RM_Publisher_rate'].mean()
            rm_publisher_reviews = search_df['RM_Publisher_reviews'].mean()
            rm_studio = search_df['RM_Studio'].mean()
            rm_studio_score = search_df['RM_Studio_score'].mean()
            rm_studio_rate = search_df['RM_Studio_rate'].mean()
            rm_studio_reviews = search_df['RM_Studio_reviews'].mean()
            rm_licence = search_df['RM_Licence'].mean()
            rm_licence_score = search_df['RM_Licence_score'].mean()
            rm_licence_rate = search_df['RM_Licence_rate'].mean()
            rm_licence_reviews = search_df['RM_Licence_reviews'].mean()
            is_serie = round(search_df['is_serie'].mean())
            is_top_serie = round(search_df['is_top_serie'].mean())
            is_top_studio = round(search_df['is_top_studio'].mean())
            is_e3 = round(search_df['is_e3'].mean())
            is_launch_plateform_associated = round(search_df['is_launch_plateform_associated'].mean())
            
            game_to_search = pd.DataFrame({
            'Platform': platform, 
            'Year': year, 
            'Genre': genre,
            'Publisher': publisher_label,
            'Studio': studio_label,
            'GK_licence': licence_label,
            'Mois': mois,
            'RM_Publisher': rm_publisher, 
            'RM_Publisher_score': rm_publisher_score, 
            'RM_Publisher_rate': rm_publisher_rate,
            'RM_Publisher_reviews': rm_publisher_reviews, 
            'RM_Studio' : rm_studio,
            'RM_Studio_score': rm_studio_score,
            'RM_Studio_rate': rm_studio_rate, 
            'RM_Studio_reviews': rm_studio_reviews, 
            'RM_Licence': rm_licence, 
            'RM_Licence_score': rm_licence_score,
            'RM_Licence_rate': rm_licence_rate, 
            'RM_Licence_reviews': rm_licence_reviews,
            'is_serie': is_serie, 
            'is_top_serie': is_top_serie,
            'is_top_studio': is_top_studio, 
            'is_e3': is_e3, 
            'is_launch_plateform_associated': is_launch_plateform_associated
            }, index=['a'])
        
            name = name.upper()
            pred = model.predict(game_to_search)
            if pred[0]==1:
                pred = "à moins de 100 000 exemplaires"
            elif pred[0] == 2:
                pred = "entre 100 000 et 250 000 exemplaires"
            elif pred[0] == 3:
                pred = "entre 250 000 et 1 million d'exemplaires"
            elif pred[0] == 4:
                pred = "à plus d'1 million d'exemplaires"
            st.success(f"Le jeu ***{name}*** devrait se vendre " + pred)
            if licence == search_df['GK_licence'].to_list()[-1]:
                st.markdown(f" ➡️ La moyenne mobile des ventes de la licence ***{search_df['GK_licence'].to_list()[-1]}*** est de {round(rm_licence,2)} millions d'explaires")
            if studio == search_df['Studio'].to_list()[-1]:
                st.markdown(f" ➡️ La moyenne mobile des ventes du studio ***{search_df['Studio'].to_list()[-1]}*** est de {round(rm_studio,2)} millions d'exmplaires" )
            if publisher == search_df['Publisher'].to_list()[-1]:
                st.markdown(f" ➡️ La moyenne mobile des ventes de l'éditeur ***{search_df['Publisher'].to_list()[-1]}*** est de {round(rm_publisher,2)} millions d'exemplaires")
            # name = placeholder.text_input('Renseignez un jeu et appuyer sur Enter', value="", key="2")
            # if '1' not in st.session_state:
                
    st.write("**OU**")
    st.subheader("Vous avez plus d'informations sur le jeu...")

    
    #On ne sort pas avant d'afficher pour garder le même ordre que dans le dataset d'entrainement
    plateformes = list(merged['Platform'].unique())
    plateformes.extend(["PS5","Xbox Series","Switch"])
    genres = list(merged['Genre'].unique())
    studios = list(merged['Studio'].unique())
    publishers = list(merged['Publisher'].unique())
    licences = list(merged['GK_licence'].unique())

    #Labeliser 
    def labelize(liste):
        res = []
        i=1
        for occ in liste:
            res.append(i)
            i+=1
        return res

    plateformes_lab = labelize(plateformes)
    genres_lab = labelize(genres)
    studios_lab = labelize(studios)
    publishers_lab = labelize(publishers)
    licences_lab = labelize(licences)


    # Formulaire pour la demo avancée
    with st.form(key='form_advanced'):
        plateforme = st.selectbox('Selectionnez les plateformes', np.sort(plateformes))
        genre = st.selectbox('Selectionnez le genre', np.sort(genres))
        licence = st.text_input(label='Licence')
        studio = st.text_input(label='Studio')
        publisher = st.text_input(label='Publisher')
        dateDeSortie = datetime.combine(st.date_input("Date de sortie"), datetime.min.time())
        serie = st.checkbox('Ce jeu fait-il partie \'une série ?')
        top_serie = st.checkbox('Cette série a t-elle du succès actuellement ?')
        top_studio = st.checkbox('Ce jeu est-il développé par un studio renommé ?')
        e3 = st.checkbox('Sera t-il présenté au festival E3 ?')
        launched_cons = st.checkbox('Une console sortira t-elle dans le même temps ?')
        submit_button = st.form_submit_button(label='Valider')
                
        

    if submit_button:
        year = dateDeSortie.year
        month = dateDeSortie.month
        nameGameAdvanced = None
        review = None
        na_sales = None
        eu_sales = None
        jp_sales = None
        other_sales = None
        global_sales = None

        # Supprimer les espaces dans champs demandés
        publisher_value = publisher if len(publisher.strip())> 0  else None
        studio_value = studio if len(studio.strip())> 0  else None
        licence_value = licence if len(licence.strip())> 0  else None

        game_to_predict_dict = {
            'Name':nameGameAdvanced,
            'Platform':plateformes_lab[plateformes.index(plateforme)],
            'Year':year,
            'Genre':genres_lab[genres.index(genre)],
            'Publisher':publisher_value,
            'Studio' : studio_value,
            'Review' : None,
            'NA_Sales': na_sales,
            'EU_Sales' : eu_sales,
            'JP_Sales' : jp_sales,
            'Other_Sales': other_sales,
            'Global_Sales' : global_sales,
            'GK_licence' : licence_value,
            'GK_distributeur':None,
            'Mois':month,
            'DateSortie': dateDeSortie,
            'RM_Publisher': None,
            'RM_Publisher_score': None,
            'RM_Publisher_rate': None,
            'RM_Publisher_reviews': None,
            'RM_Studio': None,
            'RM_Studio_score': None,
            'RM_Studio_rate': None,
            'RM_Studio_reviews': None,
            'RM_Licence': None,
            'RM_Licence_score': None,
            'RM_Licence_rate': None,
            'RM_Licence_reviews': None,
            'is_serie': serie,
            'is_top_serie': top_studio,
            'is_top_studio': top_studio,
            'is_e3': e3,
            'is_launch_plateform_associated': launched_cons
        }

        merged = pd.read_csv('04_GameSpy_FinalDataset.csv', index_col=0)
        merged['Date_Sortie'] = pd.to_datetime(merged['Date_Sortie'])


        #Les jeux avec le même publisher qui sont sortis avant le jeu pour lequel on veut prédire les ventes

        df_publisher = merged[(merged['Publisher']==publisher_value) & (merged['Date_Sortie'] <= dateDeSortie)].sort_values(by='Date_Sortie',ascending=False)

        if (df_publisher.shape[0] >=2):
            game_to_predict_dict['RM_Publisher'] = np.mean(df_publisher['RM_Publisher'].head(2)) 
            game_to_predict_dict['RM_Publisher_score'] = np.mean(df_publisher['RM_Publisher_score'].head(2)) 
            game_to_predict_dict['RM_Publisher_rate'] = np.mean(df_publisher['RM_Publisher_rate'].head(2)) 
            game_to_predict_dict['RM_Publisher_reviews'] = np.mean(df_publisher['RM_Publisher_rate'].head(2)) 
        elif(df_publisher.shape[0] ==1):
            game_to_predict_dict['RM_Publisher'] = df_publisher['RM_Publisher'].head(1)
            game_to_predict_dict['RM_Publisher_score'] = df_publisher['RM_Publisher_score'].head(1)
            game_to_predict_dict['RM_Publisher_rate'] = df_publisher['RM_Publisher_rate'].head(1) 
            game_to_predict_dict['RM_Publisher_reviews'] = df_publisher['RM_Publisher_rate'].head(1)

        #Les jeux avec le même studio qui sont sortis avant le jeu pour lequel on veut prédire les ventes

        df_studio = merged[(merged['Studio']==studio_value) & (merged['Date_Sortie'] <= dateDeSortie)].sort_values(by='Date_Sortie',ascending=False)

        if (df_studio.shape[0] >=2):
            game_to_predict_dict['RM_Studio'] = np.mean(df_studio['RM_Studio'].head(2)) 
            game_to_predict_dict['RM_Studio_score'] = np.mean(df_studio['RM_Studio_score'].head(2)) 
            game_to_predict_dict['RM_Studio_rate'] = np.mean(df_studio['RM_Studio_rate'].head(2)) 
            game_to_predict_dict['RM_Studio_reviews'] = np.mean(df_studio['RM_Studio_rate'].head(2)) 
        elif(df_studio.shape[0] ==1):
            game_to_predict_dict['RM_Studio'] = df_studio['RM_Studio'].head(1)
            game_to_predict_dict['RM_Studio_score'] = df_studio['RM_Studio_score'].head(1)
            game_to_predict_dict['RM_Studio_rate'] = df_studio['RM_Studio_rate'].head(1) 
            game_to_predict_dict['RM_Studio_reviews'] = df_studio['RM_Studio_rate'].head(1)

        #Les jeux avec le même studio qui sont sortis avant le jeu pour lequel on veut prédire les ventes

        df_licence = merged[(merged['GK_licence']==licence_value) & (merged['Date_Sortie'] <= dateDeSortie)].sort_values(by='Date_Sortie',ascending=False)

        if (df_licence.shape[0] >=2):
            game_to_predict_dict['RM_Licence'] = np.mean(df_licence['RM_Licence'].head(2)) 
            game_to_predict_dict['RM_Licence_score'] = np.mean(df_licence['RM_Licence_score'].head(2)) 
            game_to_predict_dict['RM_Licence_rate'] = np.mean(df_licence['RM_Licence_rate'].head(2)) 
            game_to_predict_dict['RM_Licence_reviews'] = np.mean(df_licence['RM_Licence_rate'].head(2)) 
        elif(df_studio.shape[0] ==1):
            game_to_predict_dict['RM_Licence'] = df_licence['RM_Licence'].head(1)
            game_to_predict_dict['RM_Licence_score'] = df_licence['RM_Licence_score'].head(1)
            game_to_predict_dict['RM_Licence_rate'] = df_licence['RM_Licence_rate'].head(1) 
            game_to_predict_dict['RM_Licence_reviews'] = df_licence['RM_Licence_rate'].head(1)

       # Labeliser les Valeurs des studios , licence et publisher
        try:
           game_to_predict_dict['Studio'] = studios_lab[studios.index(studio)]
        except:
            game_to_predict_dict['Studio'] = studios_lab[-1] + 1
        
        try:
           game_to_predict_dict['GK_licence'] = licences_lab[licences.index(licence)]
        except:
            game_to_predict_dict['GK_licence'] = licences_lab[-1] + 1
        
        try:
           game_to_predict_dict['Publisher'] = publishers_lab[publishers.index(publisher)]
        except:
            game_to_predict_dict['Publisher'] = publishers_lab[-1] + 1

        df_game_to_predict = pd.DataFrame(game_to_predict_dict,index=[1])                    

        #Remplacer les valeurs 
        df_game_to_predict['is_serie'].replace({True : 1, False : 0},inplace=True)
        df_game_to_predict['is_top_serie'].replace({True : 1, False : 0},inplace=True)
        df_game_to_predict['is_top_studio'].replace({True : 1, False : 0},inplace=True)
        df_game_to_predict['is_e3'].replace({True : 1, False : 0},inplace=True)
        df_game_to_predict['is_launch_plateform_associated'].replace({True : 1, False : 0},inplace=True)


        #Gestion des NaN : remplacement des NaN Licence par les moyennes mobiles des publisher
        df_game_to_predict["GK_licence"] = df_game_to_predict["GK_licence"].fillna(df_game_to_predict["Publisher"])
        df_game_to_predict["RM_Licence"]=df_game_to_predict["RM_Licence"].fillna(df_game_to_predict["RM_Publisher"])
        df_game_to_predict["RM_Licence_score"]=df_game_to_predict["RM_Licence_score"].fillna(df_game_to_predict["RM_Publisher_score"])
        df_game_to_predict["RM_Licence_rate"]=df_game_to_predict["RM_Licence_rate"].fillna(df_game_to_predict["RM_Publisher_rate"])
        df_game_to_predict["RM_Licence_reviews"]=df_game_to_predict["RM_Licence_reviews"].fillna(df_game_to_predict["RM_Publisher_reviews"])

        #Gestion des NaN : remplacement des NaN Studio par les moyennes mobiles des Licences
        df_game_to_predict["Studio"] = df_game_to_predict["Studio"].fillna(df_game_to_predict["GK_licence"])
        df_game_to_predict["RM_Studio"]=df_game_to_predict["RM_Studio"].fillna(df_game_to_predict["RM_Licence"])
        df_game_to_predict["RM_Studio_score"]=df_game_to_predict["RM_Studio_score"].fillna(df_game_to_predict["RM_Licence_score"])
        df_game_to_predict["RM_Studio_rate"]=df_game_to_predict["RM_Studio_rate"].fillna(df_game_to_predict["RM_Licence_rate"])
        df_game_to_predict["RM_Studio_reviews"]=df_game_to_predict["RM_Studio_reviews"].fillna(df_game_to_predict["RM_Licence_reviews"])

        #Gestion des NaN : remplacement des NaN Publisher par les moyennes mobiles des Studios
        df_game_to_predict["Publisher"] = df_game_to_predict["Publisher"].fillna(df_game_to_predict["Studio"])
        df_game_to_predict["RM_Publisher"]=df_game_to_predict["RM_Publisher"].fillna(df_game_to_predict["RM_Studio"])
        df_game_to_predict["RM_Publisher_score"]=df_game_to_predict["RM_Publisher_score"].fillna(df_game_to_predict["RM_Studio_score"])
        df_game_to_predict["RM_Publisher_rate"]=df_game_to_predict["RM_Publisher_rate"].fillna(df_game_to_predict["RM_Studio_rate"])
        df_game_to_predict["RM_Publisher_reviews"]=df_game_to_predict["RM_Publisher_reviews"].fillna(df_game_to_predict["RM_Studio_reviews"])

        #Copie du df game à prédire
        merged_tosplit = df_game_to_predict

        #Suppression des colonnes à présent inutiles
        merged_tosplit = merged_tosplit.drop(['Name','GK_distributeur','DateSortie', 'NA_Sales','Review', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'], axis=1)

        #Labelisation des variables catégorielles
        # cat_cols = merged_tosplit.select_dtypes(include=['object']).columns.to_list()
        # for col in cat_cols:
        #     i = 1
        #     occurences = merged_tosplit[col].unique()
        #     for occ in occurences:
        #         merged_tosplit.loc[merged_tosplit[col]==occ, col] = i
                # i+=1
                
        # merged_tosplit[cat_cols] = merged_tosplit[cat_cols].astype('float')

        #Classes de ventes 
        sales =  {
            1:"entre 0 et 100.000 exemplaires",
            2: "entre 100.000 et 249.000 exemplaires",
            3: "entre 250.000 et 1 million d'exemplaires",
            4: "au délà d' 1 million d'exemplaires ",
        }

        
        cat_cols = merged_tosplit.select_dtypes(include=['object']).columns.to_list()
        merged_tosplit[cat_cols] = merged_tosplit[cat_cols].astype('float')
        
        if(merged_tosplit.isna().sum().sum() == 0):
            # model = load_model()
            pred = model.predict(merged_tosplit)
            pred_text = "Ce jeu se vendra " + sales[pred[0]]
            st.success(pred_text)
        else:
            st.error("Oups ! Pas de données suffisantes pour la prédiction !\n Il se pourrait que vous n'ayez pas renseigné tous les champs ou que nous ne parvenons pas à les retrouver dans notre echantillon d'entraînement")
            st.markdown("\n Veuillez corriger les informations suivantes : ")
            if(df_game_to_predict['RM_Publisher'].isna()[1]==True):
                st.text("- Publisher")
            if (df_game_to_predict['RM_Studio'].isna()[1]==True):
                st.text("- Studio")
            if (df_game_to_predict['RM_Licence'].isna()[1]==True):
                st.text("- Licence")

                            
                            
if radio == "A propos de l'équipe":
    st.header("GameSpy")
    st.title(":sparkles: A propos de l'équipe")
    st.image("nathan-dumlao.jpg")
    st.write("Ce projet a été mené par **3 apprentis data analysts** :")
    st.write("- **Jérémy Piris** - [Linkedin](https://www.linkedin.com/in/jeremy-piris/)")
    st.write("- **Jean-Luc Djeke** - [Linkedin](https://www.linkedin.com/in/jean-luc-djeke-49805286/)")
    st.write("- **Constance Martina** - [Linkedin](https://www.linkedin.com/in/constance-martina-324b338a/)")
    
    

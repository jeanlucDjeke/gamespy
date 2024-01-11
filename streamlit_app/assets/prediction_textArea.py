#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:26:20 2022

@author: jeremy
"""
import streamlit as st
import pandas as pd
from joblib import load
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from time import sleep

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


st.title('Prédiction')
merged = pd.read_csv('/Users/jeremy/Desktop/DATASCIENTEST/02_Projet_GameSpy/00_Bon_Streamlit/data/GameSpy_FinalDataset_merged.csv', index_col=0)
model = load('/Users/jeremy/Desktop/DATASCIENTEST/02_Projet_GameSpy/00_Bon_Streamlit/rfc_grid4.joblib')
name = st.text_input('Renseignez un jeu et appuyer sur Enter')

if name:
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
        st.markdown(f"Le jeu ***{name}*** devrait se vendre " +pred)
        if licence == search_df['GK_licence'].to_list()[-1]:
            st.markdown(f" ➡️ La moyenne mobile des ventes de la licence ***{search_df['GK_licence'].to_list()[-1]}*** est de {round(rm_licence,2)}")
        if studio == search_df['Studio'].to_list()[-1]:
            st.markdown(f" ➡️ La moyenne mobile des ventes du studio ***{search_df['Studio'].to_list()[-1]}*** est de {round(rm_studio,2)}" )
        if publisher == search_df['Publisher'].to_list()[-1]:
            st.markdown(f" ➡️ La moyenne mobile des ventes de l'éditeur ***{search_df['Publisher'].to_list()[-1]}*** est de {round(rm_publisher,2)}")

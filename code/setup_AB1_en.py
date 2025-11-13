# -*- coding: utf-8 -*-
import pandas as pd 
import os 
import numpy as np
from datetime import datetime
import plotly.express as px
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.mlab as mlab

RGBs = {
    "white": [255,255,255],
    "black": [0,0,0],
    "green": [0,150,130],
    "blue": [70,100,170],
    "maygreen": [140,182,60],
    "yellow": [252,229,0],
    "orange": [223,155,27],
    "brown": [167,130,46],
    "red": [162,34,35],
    "purple": [163,16,124],
    "cyan": [35,161,224]
}

LANGUAGE_ENG = True

def load_full_netflix_data(verbose):
    
    I = np.load('../data/npys/I.npy',allow_pickle=True) # Movie ID
    J = np.load('../data/npys/J.npy',allow_pickle=True) # User ID
    V = np.load('../data/npys/V.npy',allow_pickle=True) # Rating of user j for movie i.
    # print("Die User, die Movies und die zugehörigen Ratings werden geladen. Die Daten werden in einer Tabelle / Matrix abgespeichert.\nJede Zeile steht für einen User und jede Spalte für einen Film.")
    
    print("\nDas durchschnittliche Rating des Datensatzes lautet:")
    print(V.mean())
        
    if verbose:
        print("Die Matrix hat folgende Eigenschaften:\n")
        # Show some stats
        print("Nicht-Null Einträge in der Ratingmatrix:")
        print(I.size)
        print("\nAnzahl Filme in der Ratingmatrix, d.h. Anzahl der Spalten:")
        print(np.unique(I).size)
        print("\nAnzahl User in der Ratingmatrix, d.h. Anzahl der Zeilen:")
        print(np.unique(J).size)
        #print("Die höchste User-ID lautet:")
        #print(np.amax(J))
        
    else:
    
        return I, J, V



def load_summary_data(include_genre = False):
    # FIXME: Do we want to return the same df twice?
    df = pd.read_csv("../data/summary.csv")
    df_table = df.copy()
    
    if not LANGUAGE_ENG:
        df_table = df_table.rename(
        columns={
            "title": "Filmtitel",
            "genres": "Genres",
            "releaseyear": "Erscheinungsjahr",
            "idx": "ID",
            "count": "Anzahl Ratings",
            "avg": "durchschnittl. Rating",
            "std": "Standardabweichung",
        })
    else:
        df_table = df_table.rename(
        columns={
            "title": "Titles",
            "genres": "Genres",
            "releaseyear": "Release year",
            "idx": "ID",
            "count": "Number of ratings",
            "avg": "Average rating",
            "std": "Standard deviation",
        })
        
    if not include_genre: 
        del df_table['Genres']
    
    df_table = df_table.convert_dtypes()
    if not LANGUAGE_ENG:
        print("Der Netflixdatensatz wurde erfolgreich geladen.\n")
    else:
        print("The Netflix dataset has been loaded successfully.\n")
    
    return df, df_table


df, df_table = load_summary_data();

# --- Plots ---

def interactive_plot(f):
    interact(f, min_year = widgets.FloatSlider(value=1900,
                                               min=1850.0,
                                               max=2005.0,
                                               step=1.0))
def create_rls_av_std_plot(min_year):
    fig  = px.scatter(df, 
                      x="releaseyear", 
                      y="avg",  
                      color="std", 
                      hover_name="title",
                      labels={"avg":"durchschnittl. Rating", "std":"Standardabw. Rating","count":"Anzahl Ratings","releaseyear":"Erscheinungsjahr"},
                      color_continuous_scale=px.colors.sequential.Plasma,
                 )
    fig.update_layout( 
            font_size=22
    )
    fig.update_xaxes(range=[min_year,2005])
    return fig.show()


def create_rls_av_std_plot_en(min_year):
    fig  = px.scatter(df, 
                      x="releaseyear", 
                      y="avg",  
                      color="std", 
                      hover_name="title", 
                      labels={"avg":"Average rating", "std":"Standard deviation","count":"Number of ratings","releaseyear":"Release year"},
                      color_continuous_scale=px.colors.sequential.Plasma,
                 )
    fig.update_layout( 
            font_size=22
    )
    fig.update_xaxes(range=[min_year,2005])
    return fig.show()


def create_rls_av_std_plot_old(df,minyear):
    fig  = px.scatter(df, 
                      x="releaseyear", 
                      y="avg",  
                      color="std", 
                      #size="count", 
                      hover_name="title",
                      labels={"avg":"durchschnittl. Rating", "std":"Standardabw. Rating","count":"Anzahl Ratings","releaseyear":"Erscheinungsjahr"},
                      color_continuous_scale=px.colors.sequential.Plasma,
                 )
    fig.update_layout( # customize font and legend orientation & position
            font_size=22
    )
    fig.update_xaxes(range=[minyear,2005])

    return fig.show()

def create_rls_count_av_plot(min_year):
    fig  = px.scatter(df, x="releaseyear", 
                      y="count",  
                      color="avg", 
                      #size="count", 
                      hover_name="title",
                      labels={"avg":"durchschnittl. Rating", "std":"Standardabw. Rating","count":"Anzahl Ratings", "releaseyear":"Erscheinungsjahr"},
                      color_continuous_scale=px.colors.sequential.Plasma,
                     )
    fig.update_layout( # customize font and legend orientation & position
            font_size=22
    )
    fig.update_xaxes(range=[min_year,2005])

    return fig.show()


def create_rls_count_av_plot_en(min_year):
    fig  = px.scatter(df, x="releaseyear", 
                      y="count",  
                      color="avg", 
                      #size="count", 
                      hover_name="title",
                      labels={"avg":"Average rating", "std":"Standard deviation","count":"Number of ratings","releaseyear":"Release year"}, 
                      color_continuous_scale=px.colors.sequential.Plasma,
                     )
    fig.update_layout( # customize font and legend orientation & position
            font_size=22
    )
    fig.update_xaxes(range=[min_year,2005])

    return fig.show()

    

def create_count_av_rls_plot(df,minyear):

    dftmp = df[df["releaseyear"] >= minyear]
    fig  = px.scatter(
        dftmp,
        x="count",
        y="avg",
        color="releaseyear",
        size="count",
        hover_name="title",
        labels={
            "avg": "durchschnittl. Rating",
            "std": "Rating standard deviation",
            "count": "Anzahl Ratings",
            "releaseyear": "Erscheinungsjahr"
        },
        width=1200, 
        height=800,
        # color_continuous_scale= px.colors.sequential.Plasma_r,
        # template="simple_white"
    )
    fig.update_layout(
        # customize font and legend orientation & position
        font_size=22
    )

    return fig.show()


def create_av_std_rls_plot(df):

    fig  = px.scatter(df, x="avg", y="std",  color="releaseyear", size="count", hover_name="title",
                  labels={"avg":"Rating average", "std":"Rating standard deviation","count":"Rating count","releaseyear":"Erscheinungsjahr"},
                   width=1200, height=800,
                 #template="simple_white"
                  #color_continuous_scale = px.colors.sequential.Plasma_r,
                  #range_color=[1940,2005],
                 )
        
    fig.update_layout( # customize font and legend orientation & position
        font_size=22
    )

    return fig.show()






def create_histogram_ratings(V):

    #unique, counts = np.unique(V, return_counts=True)
    #percentage = counts/100480507 * 100
    
    fig, ax = plt.subplots(figsize=(10,8))
    ax.hist(V, bins = [0.5,1.5,2.5,3.5,4.5,5.5], color = tuple(np.array(RGBs["orange"])/255), edgecolor='black', alpha=0.95)
    ax.yaxis.get_offset_text().set_fontsize(20)
    plt.title("Histogramm der Ratings", fontsize = 20)
    plt.xlabel('Rating', fontsize=20)
    plt.ylabel('Anzahl Ratings', fontsize=20)
    plt.xticks(size = 20)
    plt.yticks(size = 20)  
    
    return  plt.show()

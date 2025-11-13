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


# Interne Imports
import template.output as output
import template.questions as questions
import template.text_order_widget as text_order_widget
import template.table_widget as table

import setup_language_checks as messages

RGBs = {
    "white": [255, 255, 255],
    "black": [0, 0, 0],
    "green": [0, 150, 130],
    "blue": [70, 100, 170],
    "maygreen": [140, 182, 60],
    "yellow": [252, 229, 0],
    "orange": [223, 155, 27],
    "brown": [167, 130, 46],
    "red": [162, 34, 35],
    "purple": [163, 16, 124],
    "cyan": [35, 161, 224]
}


def load_full_netflix_data(verbose):
    I = np.load('../data/npys/I.npy', allow_pickle=True)  # Movie ID
    J = np.load('../data/npys/J.npy', allow_pickle=True)  # User ID
    V = np.load('../data/npys/V.npy', allow_pickle=True)  # Rating of user j for movie i.

    print(messages.avg_rating_dataset)
    print(V.mean())

    if verbose:
        print(messages.matrix_propertys)
        # Show some stats
        print(messages.n0_entrys)
        print(I.size)
        print(messages.movie_count)
        print(np.unique(I).size)
        print(messages.user_count)
        print(np.unique(J).size)

    else:

        return I, J, V


def load_summary_data(include_genre = False):
    df = pd.read_csv("../data/summary.csv")
    del df['genres']
    df_ger = df.copy()
    df_ger = df_ger.rename(
        columns=messages.columns)
    del df_ger['ID']
    df_ger = df_ger.convert_dtypes()
    return df, df_ger


df, df_ger = load_summary_data()

# --- Plots ---


def interactive_plot(f):
    interact(f, start_year=widgets.FloatSlider(value=1900,
                                            min=1850.0,
                                            max=2005.0,
                                            step=1.0))


def create_rls_av_std_plot(start_year):
    fig = px.scatter(df,
                     x="releaseyear",
                     y="avg",
                     color="std",
                     hover_name="title",
                     labels=messages.lables,
                     color_continuous_scale=px.colors.sequential.Plasma,
                     )
    fig.update_layout(  # customize font and legend orientation & position
        font_size=18
    )
    fig.update_xaxes(range=[start_year, 2005])

    return fig.show()


def create_rls_count_av_plot(start_year):
    fig = px.scatter(df, x="releaseyear",
                     y="count",
                     color="avg",
                     hover_name="title",
                     labels=messages.lables,
                     color_continuous_scale=px.colors.sequential.Plasma,
                     )
    fig.update_layout(
        # customize font and legend orientation & position
        font_size=22
    )
    fig.update_xaxes(range=[start_year, 2005])

    return fig.show()


def create_count_av_rls_plot(df, minyear):

    dftmp = df[df["releaseyear"] >= minyear]
    fig = px.scatter(
        dftmp,
        x="count",
        y="avg",
        color="releaseyear",
        size="count",
        hover_name="title",
        labels=messages.lables,
        width=1200,
        height=800,
    )
    fig.update_layout(
        # customize font and legend orientation & position
        font_size=22
    )

    return fig.show()


def create_av_std_rls_plot(df):

    fig = px.scatter(df, x="avg", y="std",  color="releaseyear", size="count", hover_name="title",
                     labels=messages.lables,
                     width=1200, height=800,
                     )

    fig.update_layout(
        # customize font and legend orientation & position
        font_size=22
    )

    return fig.show()


def create_histogram_ratings(V):

    # unique, counts = np.unique(V, return_counts=True)
    # percentage = counts/100480507 * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(V, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], color=tuple(np.array(RGBs["orange"])/255), edgecolor='black', alpha=0.95)
    ax.yaxis.get_offset_text().set_fontsize(20)
    plt.title(messages.rating_histogram, fontsize=20)
    plt.xlabel('Rating', fontsize=20)
    plt.ylabel(messages.count_rating, fontsize=20)
    plt.xticks(size=20)
    plt.yticks(size=20)

    return plt.show()


def prompt_1_0_1():
    questions.prompt_answer("WS1-0a", input_prompt=messages.best_film)
    
def prompt_1_0_2():
    questions.prompt_answer("WS1-0b", input_prompt=messages.oldes_newest_film)
    
def prompt_1_0_3():
    questions.prompt_answer("WS1-0c", input_prompt=messages.rating_distribution)
    
def prompt_1_0_4():
    questions.prompt_answer("WS1-0d", input_prompt=messages.most_ratings)

def prompt_1_0_5():
    questions.prompt_answer("WS1-0e", input_prompt=messages.lowest_std)
    
def promt_1_2_a():
    questions.prompt_answer("WS1-2a", input_prompt=messages.your_ideas)
    
def promt_1_2_b():
    questions.prompt_answer("WS1-2b", input_prompt=messages.contex)
    
def promt_1_2_c():
    questions.prompt_answer("WS1-2c", input_prompt=messages.procedure_and_recomendations)
    
def promt_1_z():
    questions.prompt_answer("WS1-z", input_prompt=messages.known_percentage)
    
import numpy as np

# Interne Imports
import template.output as output
import template.questions as questions
import template.text_order_widget as text_order_widget
import template.table_widget as table

import setup_language_checks as messages

import matplotlib.pyplot as plt

# Define colors
orange1 = (255/255, 127/255, 14/255)
green1 = (44/255, 160/255, 44/255)
purple1 = (148/255, 103/255, 189/255)
red1 = (214/255, 39/255, 40/255)
brown1 = (140/255, 86/255, 75/255)

# Plotting function
def plotUserMovieVectors():
    # Save Matrices
    U = np.array([[1, 0], [0, 1], [1, 1]])
    M = np.array([[4, 1, 2], [1, 4, 2]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.quiver([0, 0, 0], [0, 0, 0], M[0, :], M[1, :], angles='xy', scale_units='xy', scale=1, linestyle='dashdot', linewidth=3)
    ax.quiver([0, 0, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1], angles='xy', scale_units='xy', scale=1, linewidth=3)

    ax.annotate("User 1", (0.85, -0.1), fontsize=14, ha='right', color='black')
    ax.annotate("User 2", (-0.1, 0.8), fontsize=14, ha='right', color='black', rotation=90)
    ax.annotate("User 3", (1, 0.85), fontsize=14, ha='right', color='black', rotation=45)

    ax.annotate("Film 1", (M[0, 0], M[1, 0] - 0.125), fontsize=14, ha='right', color='black', rotation=12)
    ax.annotate("Film 2", (M[0, 1] + 0.1, M[1, 1]), fontsize=14, ha='right', color='black', rotation=76)
    ax.annotate("Film 3", (M[0, 2], M[1, 2] - 0.15), fontsize=14, ha='right', color='black', rotation=42)

    ax.set_xlim(-0.05, 4)
    ax.set_ylim(-0.05, 4)
    ax.set_xlabel("Eigenschaft 1", fontsize=16)
    ax.set_ylabel("Eigenschaft 2", fontsize=16)
    ax.grid(False)
    plt.show()

def promt_SKP_b():
    questions.prompt_answer("WS3-2d", input_prompt = "Begründung", input_description="Deine Begründung")

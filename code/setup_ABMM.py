import numpy as np
import matplotlib.pyplot as plt

import template.output as output

u11 = 2 
u12 = 0.2
u21 = -1.2
u22 = 3.4
u32 = 2.11
u31 = 0.11
u41 = -0.6
u42 = 1000
u51 = 1000
u52 = 1000
m12 = 1000
m13 = 1000
m15 = 1000
m16 = 1000
m21 = 1000
m22 = 1000
m23 = 1000
m25 = 1000
m16 = 1000
m14 = 1.1
m24 = -0.5

u1A = 2 
u1C = 0.2
u2A = -1.2
u2C = 3.4
u3C = 2.11
u3A = np.sin(2)
u4A = -np.sin(1)
u4C = 10
u5A = 1000
u5C = 1000

mA1 = 1000
mA2 = 1000
mA3 = 1000
mA4 = 1.1
mA5 = 1000
mA6 = 1000
mC1 = 1000
mC2 = 1000
mC3 = 1000
mC4 = -0.5
mC5 = 1000
mC6 = 1000

aaa = 5


####### Ausgaben  

def check_Block1(AB_1):
    if AB_1:
        output.wrong("Das ist leider Falsch")
    else:
        output.success("Das ist richtig!")


def checkCompute_r34_AC(compute_r34):
    
    r34_sol = r34_sol = u3A * mA4 + u3C * mC4;
    if compute_r34 is None:
        output.wrong("Gib eine Formel ein")
    elif np.abs(r34_sol-compute_r34) < 0.0001:
        output.success("Deine Formel für r34 ist korrekt.")
    else:
        output.wrong("Deine Formel für r34 ist noch nicht korrekt.")    
    
    
def checkCompute_r34(compute_r34):
    
    r34_sol = u31 * m14 + u32 * m24;
    if compute_r34 is None:
        output.wrong("Gib eine Formel ein")
    elif np.abs(r34_sol-compute_r34) < 0.0001:
        output.success("Deine Formel für r34 ist korrekt.")
    else:
        output.wrong("Deine Formel für r34 ist noch nicht korrekt.")



def checkPreferences(A1,C1,A2,C2,A3,C3,A4,C4):

    if (C1 == 0) & (A1 ==  1) & (C2 == 1) &  (A2 ==  0) & (C3  == 0)   & (A3 == 1) & (C4  ==  1) & (A4 == 1):
        output.success("Du hast richtig erkannt, welche der Nutzer Action bzw. Comedy zu mögen scheinen und welche nicht.\nDie Einschätzung, ob die Nutzer die jeweiligen Kategorien mögen oder nicht, können wir ebenfalls in einer Tabelle zusammenfassen.\nNeben der Rating-Tabelle R und der Movie-Tabelle M haben wir damit noch eine dritte Tabelle: die User-Tabelle U")
    
        img = plt.imread("../figs/AB1_A2_a_feedback.png")
        plt.imshow(img)
    else:
        if (A1 == 1) & (C1 == 0):
            output.success("Die Präferenzen von User 1 sind korrekt.")
        else:
            output.wrong("Die Präferenzen von User 1 sind noch nicht korrekt.")
        
        if (A2 == 0) & (C2 == 1):
            output.success("Die Präferenzen von User 2 sind korrekt.")
        else:
            output.wrong("Die Präferenzen von User 2 sind noch nicht korrekt.")
        
        if (A3 == 1) & (C3 == 0):
            output.success("Die Präferenzen von User 3 sind korrekt.")
        else:
            output.wrong("Die Präferenzen von User 3 sind noch nicht korrekt.")
        
        if (A4 == 1) & (C4 == 1):
            output.success("Die Präferenzen von User 4 sind korrekt.")
        else:
            output.wrong("Die Präferenzen von User 4 sind noch nicht korrekt.")    
    
    return 


def checkPrefUser5(RatingUser5Movie2, RatingUser5Movie3):
    
    if (RatingUser5Movie2 == 2) & (RatingUser5Movie3 == 4):
        output.success("Deine Antwort ist korrekt!")
    elif RatingUser5Movie2 == 2:
        output.success("Deine Antwort für Film 2 ist korrekt!")
        output.wrong("Deine Antwort für Film 3 ist nicht korrekt.")
    elif RatingUser5Movie3 == 4:
        output.success("Deine Antwort für Film 3 ist korrekt!")
        output.wrong("Deine Antwort für Film 2 ist nicht korrekt.")
    else:
        output.wrong("Deine Antwort scheint noch nicht korrekt.")
        
    return

def checkRatingmatrix(R):
    U3 = np.array([[0, 1], [1, 1], [1, 0], [1, 0.5]])
    M3 = np.array([[3, 2, 1, 1], [2, 2, 4, 4]])
    
    R3_sol = np.dot(U3, M3)
    
    if None in R:
        output.wrong("Ersetze alle None")
    else:
        comparison = R == R3_sol
        if comparison.all():
            output.success("Deine Matrix ist korrekt")
        else:
            diff_indices = np.where(comparison == False)
            diff_indices_plus_one = [(i+1, j+1) for i, j in zip(*diff_indices)]
            output.wrong(f"Deine Matrix ist noch an folgenden Indizes falsch: {diff_indices_plus_one}")


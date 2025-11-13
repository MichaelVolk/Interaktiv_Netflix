import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

import template.output as output
import template.questions as questions
import template.text_order_widget as text_order_widget
import template.table_widget as table

import setup_language_checks as messages

# AB 3

DecompTrue = "Deine Zerlegung ist korrekt. Sie liefert die gewünschte Rating-Matrix:"
DecompFalse = "Deine Zerlegung liefert noch nicht die gewünschte Rating-Matrix. Das Ergebnis deiner Zerlegung lautet stattdessen:"

RechercheFeedack = "\nDu hast ein sinnvolles Fehlermaß definiert! Wir werden jedoch im Workshop mit der Summe der Fehlerquadrate weiterarbeiten.\nDabei werden die einzelnen Abweichungen erst quadriert (anstelle des Absolutbetrags) und dann aufsummiert. Ändere deine Formel entsprechend ab.  \nWir werden später diskutieren, warum dieses Fehlermaß für unsere Zwecke noch etwas besser geeignet ist."

BilanzFeedback = "Du verwendest kein sinnvolles Fehlermaß, da in deiner Summe auch negative Abweichungen einzelner Ratings auftauchen können. "

errorMeasure = "Du berechnest den Fehler mit dem Fehlermaß: "
errorNegative = "Mit deiner bisherigen Formel kann es passieren, dass der Fehler negativ ist oder null wird, obwohl die vorhergesagten Bewertungen stark von den tatsächlichen Bewertungen abweichen.\nDas ist für die Bewertung und den Vergleich von verschiedenen Zerlegungen nicht sinnvoll. Korrigiere deine Formel. "

AE = "Summe der absoluten Abweichungen."
Note = "Hinweis: "
MAE = "gemittelte Summe der absoluten Abweichungen."

SE = "Summe der Fehlerquadrate."
MSE = "gemittelte Summe der Fehlerquadrate."
RMSE = "Du berechnest die Güte der Zerlegung mit dem Fehlermaß der Wurzel des mittleren quadratischen Fehlers."
ErrorMeasureFalse = "Du verwendest noch kein sinnvolles Fehlermaß."

ExtendDecomp = "\nDeine Formel wurde für beliebig große Rating-Matrizen erweitert. Mit deiner Formel beträgt der Fehler für die Zerlegungen aus Teil a:\n"
Decomp = "Zerlegung"

# pretty tables


Rmatrix = "\nRating-Matrix R:"
Pmatrix = "\nVorhersage-Matrix P:"
Umatrix = "\nUser-Matrix U:"
Mmatrix = "\nMovie-Matrix M:"
trainmatrix = "\nTrainingsmatrix:"
testmatrix = "\nTestmatrix:"

###

R_22 = np.array([[5.0, 3.0],
             [3.0, 1.0]])

R_24 = np.array([[3.0, 2.0, 1.0, 4.0],
             [1.0, 2.0, 3.0, 4.0]])


def checkDecomposition(U1, M1):
    if np.any(U1 == None) or np.any(M1 == None):
        output.wrong("Ersetzen alle None")
        return
    P1 = np.dot(U1,M1);
    print(P1)
    error = np.sum(np.abs(R_22-np.dot(U1,M1)));
    print("Summe der absoluten Abweichungen von der Ratingmatrix: " + str(error))

def checkError_P(computeError, R1, U4, M4, U5, M5, U6, M6):

    RechercheFeedack = "\nDu hast ein sinnvolles Fehlermaß definiert! Wir werden jedoch im Workshop mit der Summe der Fehlerquadrate weiterarbeiten.\nDabei werden die einzelnen Abweichungen erst quadriert (anstelle des Absolutbetrags) und dann aufsummiert. Ändere deine Formel entsprechend ab.  \nWir werden später diskutieren, warum dieses Fehlermaß für unsere Zwecke noch etwas besser geeignet ist."

    BilanzFeedback = "Du verwendest kein sinnvolles Fehlermaß, da in deiner Summe auch negative Abweichungen einzelner Ratings auftauchen können. "

    errorMeasure = "Du berechnest den Fehler mit dem Fehlermaß: "

    r11 = R1[0, 0]
    r12 = R1[0, 1]
    r21 = R1[1, 0]
    r22 = R1[1, 1]

    p11 = 200
    p12 = 9
    p21 = -10
    p22 = 20

    errorDecomp = computeError(r11, r12, r21, r22, p11, p12, p21, p22)

    if (errorDecomp is None):
        output.wrong("Setze eine Formel ein")
        return
    # Lösung 1: Mean absolute error
    errorAE = np.abs(r11-p11) + np.abs(r12-p12) + np.abs(r21-p21) + np.abs(r22-p22)
    errorMAE = errorAE / 4

    # Lösung 2: Mean squared error
    errorSE = (r11-p11)**2 + (r12-p12)**2 + (r21-p21)**2 + (r22-p22)**2
    errorMSE = errorSE / 4

    # Lösung 3: Root squared error
    errorRMSE = np.sqrt(errorMSE)

    if errorDecomp < 0:
        output.wrong("Mit deiner bisherigen Formel kann es passieren, dass der Fehler negativ ist oder null wird, obwohl die vorhergesagten Bewertungen stark von den tatsächlichen Bewertungen abweichen.\nDas ist für die Bewertung und den Vergleich von verschiedenen Zerlegungen nicht sinnvoll. Korrigiere deine Formel. ")
        errorDecomp1n = np.sum((R1-np.dot(U4, M4)))
        errorDecomp2n = np.sum((R1-np.dot(U5, M5)))
        errorDecomp3n = np.sum((R1-np.dot(U6, M6)))

    elif np.isclose(errorDecomp, errorAE, atol= 1e-08):
        output.success(errorMeasure + "Summe der absoluten Abweichungen.")
        print("Hinweis: " + RechercheFeedack)

    elif np.isclose(errorDecomp, errorMAE, atol= 1e-08):
        print(errorMeasure + "gemittelte Summe der absoluten Abweichungen.")

    elif np.isclose(errorDecomp, errorSE, atol= 1e-08):
        output.success(errorMeasure + "Summe der Fehlerquadrate.")

    elif np.isclose(errorDecomp, errorMSE, atol= 1e-08):
        output.success(errorMeasure + "gemittelte Summe der Fehlerquadrate.")

    elif np.isclose(errorDecomp, errorRMSE, atol= 1e-08):
        output.success("Du berechnest die Güte der Zerlegung mit dem Fehlermaß der Wurzel des mittleren quadratischen Fehlers.")

    else:
        output.wrong("Du verwendest noch kein sinnvolles Fehlermaß.")


def checkError(computeError, R1, U3, M3, U4, M4, U5, M5):
    errorDecomp1 = computeError(R1[0, 0], R1[0, 1], R1[1, 0], R1[1, 1], U3[0, 0], U3[0, 1],
                                U3[1, 0], U3[1, 1], M3[0, 0], M3[0, 1], M3[1, 0], M3[1, 1])
    errorDecomp2 = computeError(R1[0, 0], R1[0, 1], R1[1, 0], R1[1, 1], U4[0, 0], U4[0, 1],
                                U4[1, 0], U4[1, 1], M4[0, 0], M4[0, 1], M4[1, 0], M4[1, 1])
    errorDecomp3 = computeError(R1[0, 0], R1[0, 1], R1[1, 0], R1[1, 1], U5[0, 0], U5[0, 1],
                                U5[1, 0], U5[1, 1], M5[0, 0], M5[0, 1], M5[1, 0], M5[1, 1])
    if (errorDecomp1 is None):
        output.wrong("Setze eine Formel ein")
        return

    r11 = R1[0, 0]
    r12 = R1[0, 1]
    r21 = R1[1, 0]
    r22 = R1[1, 1]

    u11 = U4[0, 0]
    u12 = U4[0, 1]
    u21 = U4[1, 0]
    u22 = U4[1, 1]

    m11 = M4[0, 0]
    m21 = M4[0, 1]
    m12 = M4[1, 0]
    m22 = M4[1, 1]

    errorDecomp = computeError(r11, r12, r21, r22, u11, u12, u21, u22, m11, m12, m21, m22)

    # Lösung 1: Mean absolute error
    errorAE = np.abs(r11-(u11*m11+u12*m21)) + np.abs(r12-(u11*m12+u12*m22)) + \
        np.abs(r21-(u21*m11+u22*m21)) + np.abs(r22-(u21*m12+u22*m22))
    errorMAE = errorAE / 4

    # Lösung 2: Mean squared error
    errorSE = (r11-(u11*m11+u12*m21))**2 + (r12-(u11*m12+u12*m22))**2 + (r21-(u21*m11+u22*m21))**2 + (r22-(u21*m12+u22*m22))**2
    errorMSE = errorSE / 4

    # Lösung 3: Root  squared error
    errorRMSE = np.sqrt(errorMSE)

    if errorDecomp < 0:
        output.wrong(errorNegative)

        errorDecomp1n = np.sum((R1-np.dot(U3, M3)))
        errorDecomp2n = np.sum((R1-np.dot(U4, M4)))
        errorDecomp3n = np.sum((R1-np.dot(U5, M5)))

    elif errorDecomp == errorAE:
        output.success(errorMeasure + AE)
        print(stderr, string(Note, RechercheFeedack))
        errorDecomp1n = np.sum(np.abs(R1-np.dot(U3, M3)))
        errorDecomp2n = np.sum(np.abs(R1-np.dot(U4, M4)))
        errorDecomp3n = np.sum(np.abs(R1-np.dot(U5, M5)))

    elif errorDecomp == errorMAE:
        print(errorMeasure + MAE)
        print(RechercheFeedack)
        errorDecomp1n = np.sum(np.abs(R1-np.dot(U3, M3))) / len(R1)
        errorDecomp2n = np.sum(np.abs(R1-np.dot(U4, M4))) / len(R1)
        errorDecomp3n = np.sum(np.abs(R1-np.dot(U5, M5))) / len(R1)

    elif errorDecomp == errorSE:
        output.success(errorMeasure + SE)
        errorDecomp1n = np.sum((R1-np.dot(U3, M3))**2)
        errorDecomp2n = np.sum((R1-np.dot(U4, M4))**2)
        errorDecomp3n = np.sum((R1-np.dot(U5, M5))**2)

    elif errorDecomp == errorMSE:
        output.success(errorMeasure + MSE)
        errorDecomp1n = np.sum((R1-np.dot(U3, M3))**2) / len(R1)
        errorDecomp2n = np.sum((R1-np.dot(U4, M4))**2) / len(R1)
        errorDecomp3n = np.sum((R1-np.dot(U5, M5))**2) / len(R1)

    elif errorDecomp == errorRMSE:
        output.success(RMSE)
        errorDecomp1n = sqrt(np.sum((R1-np.dot(U3, M3))**2) / len(R1))
        errorDecomp2n = sqrt(np.sum((R1-np.dot(U4, M4))**2) / len(R1))
        errorDecomp3n = sqrt(np.sum((R1-np.dot(U5, M5))**2) / len(R1))
    else:
        output.wrong(ErrorMeasureFalse)
        errorDecomp1n = NaN
        errorDecomp2n = NaN
        errorDecomp3n = NaN

    print(ExtendDecomp +
          Decomp + " 1: " + f"{errorDecomp1n}\n" +
          Decomp + " 2: " + f"{errorDecomp2n}\n" +
          Decomp + " 3: " + f"{errorDecomp3n}\n")

def calc_pred(U4, M4):
    P4 = np.dot(U4,M4);
    print("Vorhersage Matrix: \n"+ str(P4))
    print("Rating Matrix: \n" + str(R_24))

def promt_3_2_b():
    questions.prompt_answer("WS3-2b", input_prompt="Deine berechneten Abweichungen", input_description="Deine Antwort")


def promt_3_2_d():
    questions.prompt_answer("WS3-2d", input_prompt="Antwort und Begründung", input_description="Deine Antwort")

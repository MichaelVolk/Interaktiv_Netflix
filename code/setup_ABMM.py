import numpy as np

import template.output as output

####### Ausgaben  

def check_Block1(answers):
    real_answers = [True, False, True, True]
    if None in answers:
        output.wrong("Ersetze alle \"None\"")
        return
    for i in range(len(answers)):
        if answers[i] != real_answers[i]:
            output.wrong("Es stimmen noch nicht alle Ergebnisse!")
            return
    output.success("Alle Ergebnisse stimmen!")

def check_product(C):
    real_answer = np.array([[7,8],[13,14]])
    if None in C:
        output.wrong("Ersetze alle \"None\".")
        return
    if (C==real_answer).all():
        output.success("Das Matrixprodukt ist richtig!")
        print(C)
        return
    output.wrong("Das scheint was noch nicht zu stimmen...")


def check_produkt_eintragen(C):
    real_answer = np.array([[],
                            []])
    if (C==real_answer).all():
        output.success("Die Funktion ist richtig implementiert!")
    else:
        output.wrong("Das passt was noch nicht!")


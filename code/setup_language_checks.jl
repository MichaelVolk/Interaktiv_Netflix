# Define Labels for Plots 
if language == "german"

 TrueResult ="Dein Ergebnis ist korrekt."   
 FalseResult ="Dein Ergebnis ist noch nicht korrekt."   
    
 
# AB 2
 pref1True =  "Die Präferenzen von User 1 sind korrekt."
 pref1False = "Die Präferenzen von User 1 sind noch nicht korrekt."
    
 pref2True =  "Die Präferenzen von User 2 sind korrekt."
 pref2False = "Die Präferenzen von User 2 sind noch nicht korrekt."
    
 pref3True =  "Die Präferenzen von User 3 sind korrekt."
 pref3False = "Die Präferenzen von User 3 sind noch nicht korrekt."
    
 pref4True =  "Die Präferenzen von User 4 sind korrekt."
 pref4False = "Die Präferenzen von User 4 sind noch nicht korrekt."
    
 prefAllTrue = "Du hast richtig erkannt, welche der Nutzer Action bzw. Comedy zu mögen scheinen und welche nicht.\n_________________________________________________________________________\n 
Die Einschätzung, ob die Nutzer die jeweiligen Kategorien mögen oder nicht, können wir ebenfalls in einer Tabelle 
zusammenfassen. 
            
Neben der Rating-Tabelle R und der Movie-Tabelle M haben wir damit noch eine dritte Tabelle: die User-Tabelle U"      
 
 TrueAnswer = "Deine Antwort ist korrekt!"
 FalseAnswer = "Deine Antwort scheint noch nicht korrekt."
 Movie2True = "Deine Antwort für Film 2 ist korrekt!"
 Movie3True = "Deine Antwort für Film 3 ist korrekt!"
 Movie2False = "Deine Antwort für Film 2 ist nicht korrekt."
 Movie3False = "Deine Antwort für Film 3 ist nicht korrekt."

 Formula_r34_True =   "Deine Formel für r34 ist korrekt."
 Formula_r34_False =  "Deine Formel für r34 ist noch nicht korrekt."
    
  
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

# AB 4

 EnterForNaN = "Gib deine Formel für NaN ein."
 R1saved = "Die  Rating-Matrix R1 wurde gespeichert."
    
 enter_iters =  "Gib eine natürliche Zahl für die Variable iterations ein."
 enter_factors = "Gib eine natürliche Zahl für die Variable features ein."
 enter_lambda = "Gib eine Zahl größer 0 für die Variable lambda ein."      
    
  iters = " Wiederholungen: "
   MSEafterALS      =   "\n________________________________________________\nGemittelte Summe der Fehlerquadrate nach "
     
# AB 5 
    
 DatasetInfo = "Der Datensatz beinhaltet\n"
 MovieInfo = "Filme\n"
 UserInfo = "Nutzer\n"
 RatingsInfo = "Bewertungen\n"
 TrainsetInfo = "Bewertungen im Trainingsdatensatz\n"
 TestsetInfo = "Bewertungen im Testdatensatz\n"
    
 PrintFilterInfo = "Nutzer ohne irgendeine Bewertung für die ausgewählten Filme wurden entfernt.\n"

  # utils_als
 PrintRepetitions = "Wiederholungen"
 TrainMSE_After_ALS = "\nDie gemittelte Summe der Fehlerquadrate auf den Trainingsdaten nach "
 TestMSE_After_ALS = "\nDie gemittelte Summe der Fehlerquadrate auf den Testdaten nach "
    
 # utils_dict
 PrintPredForUser = "Die beste Vorhersage für  User "
 PrintIsMovie = " ist der Film: "
 PrintPredictedMovieRating = "Das vorhergesagte Rating für diesen Film lautet: " 
    
######### English ############    
    
    
elseif language == "english"
 
 TrueResult ="Your solution is correct."  
 FalseResult ="Your solution is not yet correct."
    
    # AB 2
 pref1True =  "The preferences of user 1 are correct."
 pref1False = "The preferences of user 1 are not yet correct."
    
 pref2True =  "The preferences of user 2 are correct."
 pref2False = "The preferences of user 2 are not yet correct."
    
 pref3True =  "The preferences of user 3 are correct."
 pref3False = "The preferences of user 3 are not yet correct."
    
 pref4True =  "The preferences of user 4 are correct."
 pref4False = "The preferences of user 4 are not yet correct."
    
    
 prefAllTrue = "You have correctly identified which of the users seem to like action or comedy respectively and which don't. \n_________________________________________________________________________\n 
We can also summarize the assessment of whether the users like or dislike the respective categories in a table. 
            
In addition to the rating table R and the movie table M, we thus have a third table: the user table U"
  

 TrueAnswer = "Your answer is correct!"
 FalseAnswer = "Your answer is not yet correct."
 Movie2True = "Your answer for movie 2 is correct!"
 Movie3True = "Your answer for movie 3 is correct!"
 Movie2False = "Your answer for movie 2 is not yet correct."
 Movie3False = "Your answer for movie 3 is not yet correct."
 
 Formula_r34_True =   "Your formula for r34 is correct."
 Formula_r34_False =  "Your formula for r34 is not yet correct."
    

    
 # AB 3
  
 DecompTrue = "Your decomposition is correct. It returns the desired rating matrix:"
 DecompFalse = "Your decomposition does not yet return the desired rating matrix. The result of your decomposition is instead:"
    
 ResearchFeedack = "\nYou have defined a reasonable error measure! However, we will continue in the workshop with the sum of the squared of the errors.\nIn this case, the individual deviations are first squared (instead of the absolute amount) and then summed up. Modify your formula accordingly.  \nWe'll discuss later why this error measure is slightly better for our purposes."
    
 BalanceFeedback = "You are not using a meaningful error measure, since negative deviations of individual ratings may also appear in your sum. "
    
 errorMeasure = "You calculate the error with the error measure: "
 errorNegative = "With your formula so far, it can happen that the error is negative or becomes zero, although the predicted ratings differ strongly from the actual ratings.\nThis is not useful for rating and comparing different decompositions. Correct your formula. "
    
 AE = "Sum of absolute differences."
 Note = "Note: "
 MAE = "averaged sum of absolute differences."
    
 SE = "sum of squared differences."
 MSE = "mean squared error."
 RMSE = "You calculate the quality of the decomposition with the error measure root mean squared error."
 ErrorMeasureFalse = "You are not yet using a meaningful error measure."
    
 ExtendDecomp = "\nYour formula has been extended for arbitrarily large rating matrices. With your formula, the error for the decompositions from part a is:\n"
 Decomp = "decomposition"
    
    
 # pretty tables
    
    
Rmatrix = "\nRating matrix R:"
Pmatrix = "\nPrediction matrix P:"
Umatrix = "\nUser matrix U:"
Mmatrix = "\nMovie matrix M:"
trainmatrix = "\nTraining matrix:"
testmatrix = "\nTest matrix:"

 # AB 4   
 EnterForNaN = "Enter your formula for NaN."
 R1saved = "The rating matrix R1 has been saved."
  enter_iters = "Enter a natural number for the variable iterations."
 enter_factors = "Enter a natural number for the variable features."
 enter_lambda = "Enter a number greater than 0 for the variable lambda."      
    
 iters = " iterations: "
 MSEafterALS      =   "\n________________________________________________\nThe Mean Squared Error after "
       
# AB 5 
    
 DatasetInfo = "The dataset contains:\n"
 MovieInfo = "Movies\n"
 UserInfo = "Users\n"
 RatingsInfo = "Ratings\n"
 TrainsetInfo = "Ratings in the training dataset\n"
 TestsetInfo = "Ratings in the test dataset\n"  

 PrintFilterInfo = "users were removed because they did not submit a single\nrating for the selected movies.\n"

 # utils_als
 PrintRepetitions = " repetitions: "
 TrainMSE_After_ALS = "\nThe mean squared error on the training data after "
 TestMSE_After_ALS = "\nThe  mean squared error on the test data after "

# utils_dict
 PrintPredForUser = "The best prediction for user "
 PrintIsMovie = " is the movie: "
 PrintPredictedMovieRating = "The predicted rating for this movie is: "
    
elseif language == "spanish"
    
    
    
    
    
end
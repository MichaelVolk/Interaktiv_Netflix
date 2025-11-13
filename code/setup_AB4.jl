
using Suppressor, DelimitedFiles 

@suppress begin

using LaTeXStrings
using LinearAlgebra
using Calculus
using StatsBase
using YAML
using Random
using PrettyTables

# using NPZ, SparseArrays, NMF, PrettyTables, DataFrames

include("output.jl")
include("utils.jl")
include("utils_ALS.jl")
include("utils_plots.jl")
include("utils_pretty_tables.jl");
include("utils_plotlyJS.jl");
include("utils_split_data.jl");
include("loss_functions.jl");
include("utils_lsqFIT.jl")
include("setup_language.jl")

end


# Definition Ratingmatrix Aufgabe 2
# Bei dieser Matrix steigt error wieder:
RS = [3 1 0 3 1 0;
     1 0 4 1 3 1;
     0 1 1 0 1 4;
     0 0 5 0 4 0]

# AB 5:
ROLD = [3 1 0 4 1 0 2 4 1 ;
     1 0 4 1 3 1 0 0 5;
     0 1 1 0 1 4 2 0 5;
     1 0 3 0 2 0 1 0 5;
     3 1 0 4 1 0 1 4 1;
     0 0 5 0 4 0 0 0 5 ]



# Überprüfefunktionen 

 function computeErrorSol(R,U,M)  
    
    nonZeros = findall(!isnan, R)
    sum((((R - U * M).^2))[nonZeros])/length(R)
end


function checkErrorU(errorUCal)
    
    if isnan(errorUCal(1,1))
        printError("Gib deine Formel für NaN ein.")
        
    else
    
        u1 = [2,-2,-3, 3,0,10]
        u2 = [1, 2,-3, 2, 0.1, -0.4]

        eps = 0.01;

        errorUSol(u11,u12) = (2 - (u11 + u12))^2 + (3 - (u11 + u12))^2;

        cal = errorUCal.(u1,u2)
        sol = errorUSol.(u1,u2)

        if maximum(abs.(cal - sol)) > eps 

            printError("Deine Lösung für die Fehlerfunktion ist noch nicht noch nicht korrekt.\n")
        else
            printSuccess("Deine Lösung für die Fehlerfunktion ist korrekt.\n")
            plotError(errorUCal,-1,3)
        end
    end
end


function checkGradientU(ErrorGradient1,ErrorGradient2)
    u1 = [2,-2,-3, 3,0,10]
    u2 = [1, 2,-3, 2, 0.1, -0.4]
    
    eps = 0.01;
    
    ErrorGradient1Sol(u11,u12) =  -2*(2-u11-u12)-2*(3-u11-u12);
    ErrorGradient2Sol(u11,u12) =  -2*(2-u11-u12)-2 *(3-u11-u12);
    
    
    # u11
    if isnan(ErrorGradient1(1,2))
        printError("Gib die partielle Ableitung nach u11 für das erste NaN ein.")
        
    else
        cal1 = ErrorGradient1.(u1,u2)
        sol1 = ErrorGradient1Sol.(u1,u2)
        
         if maximum(abs.(cal1 - sol1)) > eps 
        
            printError("Deine Lösung für die Ableitung nach u11 ist nicht noch nicht korrekt.")
        else
            printSuccess("Deine Lösung für die Ableitung nach u11 ist korrekt.")
        end
    end
    
    
    # u12
    if isnan(ErrorGradient2(1,2))
        printError("Gib die partielle Ableitung nach u12 für das zweite NaN ein.")
        
    else
        cal2 = ErrorGradient2.(u1,u2)
        sol2 = ErrorGradient2Sol.(u1,u2)
    
        
        if maximum(abs.(cal2 - sol2)) > eps 

            printError("Deine Lösung für die Ableitung nach u12 ist nicht noch nicht korrekt.")
        else
            printSuccess("Deine Lösung für die Ableitung nach u12 ist korrekt.")
        end
    end
    
end


function checkMinU(u11_min,u12_min,errorU)
    
    eps = 0.0001;
    
    # Ableitung bezüglich u11:
    ErrorGradient1(u11,u12) = +2*(2-u11-u12)*(-1) + 2*(3-u11-u12)* (-1); # Ableitung nach u11

    # Ableitung bezüglich u12:
    ErrorGradient2(u11,u12) =  +2*(2-u11-u12)*(-1) + 2 *(3-u11-u12)*(-1) ; # Ableitung nach u12
    
    
    if abs(ErrorGradient1(u11_min, u12_min)) < eps && abs(ErrorGradient2(u11_min, u12_min)) < eps

        printSuccess("Dein Ergebnis ist korrekt.")
        
        # Auswertung der Fehlerfunktion am Minimum
        error = errorU(u11_min, u12_min);
        println("Einsetzen der gefundenen Lösung u11 = $u11_min und u12 = $u12_min in die Fehlerfunktion liefert einen Fehler von $error."); 
        
    else
        printError("Deine gefundenen Einträge von U liefern noch nicht das Minimum der Fehlerfunktion.")
    end
    
end



function checkMinU(u11_min,u12_min,errorU, ErrorGradient1, ErrorGradient2)
    
    eps = 0.0001;

    if abs(ErrorGradient1(u11_min, u12_min)) < eps && abs(ErrorGradient2(u11_min, u12_min)) < eps

        printSuccess("Dein Ergebnis ist korrekt.")
        
        # Auswertung der Fehlerfunktion am Minimum
        error = errorU(u11_min, u12_min);
        println("Einsetzen der gefundenen Lösung u11 = $u11_min und u12 = $u12_min in die Fehlerfunktion liefert einen Fehler von $error."); 
        
    else
        printError("Deine gefundenen Einträge von U liefern noch nicht das Minimum der Fehlerfunktion.")
    end
    
end



function checkGradientM(ErrorGradientM11,ErrorGradientM12,ErrorGradientM21,ErrorGradientM22,u1min,u2min)
    
    m1 = [2,-2,-3, 3,0,10]
    m2 = [1, 2,-3, 2, 0.1, -0.4]
    m3 = [0, 2,-3, 2, 0.1, -0.4]
    m4 = [-10, 2,-3, 2, 2.1, 0.4]
    
    eps = 0.01;

    ErrorGradientM11Sol(m11,m12,m21,m22) =  2 * (2 - u1min * m11 - u2min* m21)*(-u1min);
    ErrorGradientM12Sol(m11,m12,m21,m22) =  2 * (3 - u1min * m12 - u2min * m22)*(-u1min);
    ErrorGradientM21Sol(m11,m12,m21,m22) =  2 * (2 - u1min * m11 - u2min* m21)*(-u2min);
    ErrorGradientM22Sol(m11,m12,m21,m22) =  2 * (3 - u1min * m12 - u2min * m22)*(-u2min);
    
    
    # m11
    if isnan(ErrorGradientM11(1,2,3,4))
        printError("Gib die partielle Ableitung nach m11 für das erste NaN ein.")
        
    else
        cal1 = ErrorGradientM11.(m1,m2,m3,m4)
        sol1 = ErrorGradientM11Sol.(m1,m2,m3,m4)
        
        if maximum(abs.(cal1 - sol1)) > eps 

            printError("Deine Lösung für die Ableitung nach m11 ist nicht noch nicht korrekt.")
        else
            printSuccess("Deine Lösung für die Ableitung nach m11 ist korrekt.")
        end  
    end
    
    
    # m12
    if isnan(ErrorGradientM12(1,2,3,4))
        printError("Gib die partielle Ableitung nach m12 für das zweite NaN ein.")
        
    else
        cal2 = ErrorGradientM12.(m1,m2,m3,m4)
        sol2 = ErrorGradientM12Sol.(m1,m2,m3,m4)
        
        if maximum(abs.(cal2 - sol2)) > eps 
        
            printError("Deine Lösung für die Ableitung nach m12 ist nicht noch nicht korrekt.")
        else
            printSuccess("Deine Lösung für die Ableitung nach m12 ist korrekt.")
        end 
    end
        
    
    # m21
    if isnan(ErrorGradientM21(1,2,3,4))
        printError("Gib die partielle Ableitung nach m21 für das dritte NaN ein.")
        
    else
        cal3 = ErrorGradientM21.(m1,m2,m3,m4)
        sol3 = ErrorGradientM21Sol.(m1,m2,m3,m4)
        
        if maximum(abs.(cal3 - sol3)) > eps 

            printError("Deine Lösung für die Ableitung nach m21 ist nicht noch nicht korrekt.")
        else
            printSuccess("Deine Lösung für die Ableitung nach m21 ist korrekt.")
        end
    end

        
    # m22
    if isnan(ErrorGradientM22(1,2,3,4))
        printError("Gib die partielle Ableitung nach m22 für das vierte NaN ein.")
        
    else
        cal4 = ErrorGradientM22.(m1,m2,m3,m4)
        sol4 = ErrorGradientM22Sol.(m1,m2,m3,m4)
        
        if maximum(abs.(cal4 - sol4)) > eps 

            printError("Deine Lösung für die Ableitung nach m22 ist nicht noch nicht korrekt.")
        else
            printSuccess("Deine Lösung für die Ableitung nach m22 ist korrekt.")
        end
    end
    
end


function checkMinM(m11_min,m12_min,m21_min, m22_min, errorM, ErrorGradientM11, ErrorGradientM12, ErrorGradientM21, ErrorGradientM22)
    
    eps = 0.0001;

    if abs(ErrorGradientM11(m11_min, m12_min,m21_min,m22_min)) < eps && abs(ErrorGradientM12(m11_min, m12_min,m21_min,m22_min)) < eps && abs(ErrorGradientM21(m11_min, m12_min,m21_min,m22_min)) < eps && abs(ErrorGradientM22(m11_min, m12_min,m21_min,m22_min)) < eps 

        printSuccess("Dein Ergebnis ist korrekt.")
        
        # Auswertung der Fehlerfunktion am Minimum
        error = errorM(m11_min, m12_min, m21_min, m22_min);
        println("Einsetzen der gefundenen Lösung m11 = $m11_min, m12 = $m12_min, m21 = $m21_min und m22 = $m22_min in die Fehlerfunktion liefert einen Fehler von $error."); 
        
    else
        printError("Deine gefundenen Werte für die Einträge von U liefern noch nicht das Minimum der Fehlerfunktion.")
    end
    
end 

function printSave()
         println("Die  Rating-Matrix R1 wurde gespeichert.")
end


function printUtimesM()

    U = [u11_min u12_min]
    M = [m11_min m12_min; m21_min m22_min]
    println("\nDeine Zerlegung in U und M lautet:")
    println("\nUsermatrix U:")
    printSimple(U)
    println("\nMovie-Matrix M:")
    printSimple(M)
    println("\nDas Produkt U*M und damit die Vorhersagematrix P lautet:")
    P = U*M
    printSimple(P)

end






function catchErrorALS()
    println("\n________________________________________________\n\nFalls du eine Fehlermeldung erhältst, lies die nächste Teilaufgabe auf diesem Arbeitsblatt!\n________________________________________________\n")
end



function computeALS(n_iters, n_factors, train_matrix_default, test_matrix_default, lambda,normalization,datasplit, verbose = false)
    
""" 
    n_iters   = number of iterations to train the algorithm
    n_factors = number of latent factors to use in matrix 
    average = average of the ratings in the training matrix
    lambda    = regularization term for item/user latent factors 
    normalization = binary variable. true = training data is normalized, false otherwise
    datasplit = binary variable. true = data was split into test and train data, false = factorization on the whole data 
"""
   
    test_mse_record  = [] 
    train_mse_record = [] 
    PredictionMatrix = []
    
    
    if isnan(train_matrix_default[1]) 
        printError("Wähle eine Rating-Matrix.")
        
    elseif isnan(n_iters) || (n_iters % 1) != 0 || (n_iters)<0
        printError("Gib eine natürliche Zahl für die Variable iterations ein.") 
        
    elseif isnan(n_factors)
        printError("Gib eine natürliche Zahl für die Variable features ein.")
        
    elseif isnan(lambda)
        printError("Gib eine Zahl größer 0 für die Variable lambda ein.")    
        
    else
    
    
    train_matrix_default = convert(Array{Float64,2},train_matrix_default)
    idx = findall(!iszero, train_matrix_default)
    D = zeros(size(train_matrix_default));
    D[idx] .= 1  
        
        
        if normalization == false    
            average = 0
            train_matrix = train_matrix_default[:,:]

        else 

            idx = findall(!iszero, train_matrix_default)
            average = mean(train_matrix_default[idx])
            train_matrix = train_matrix_default[:,:];
            train_matrix[idx] = train_matrix[idx] .- average
        end

        
        Random.seed!(123)
        U = rand(size(train_matrix, 1), n_factors) 
        M = rand(n_factors, size(train_matrix, 2))  
 
        for k = 1 : n_iters

            if k % 2 == 0 
                M = als_step_m_nonzeros_LSQFIT(M, U, train_matrix, lambda,D)[:,:];  
                if verbose println("\nWiederholung $k: Die User-Matrix ist fest. Die Movie-Matrix wird optimiert.") end
    
            else 
                U = als_step_u_nonzeros_LSQFIT(U, M, train_matrix, lambda,D)[:,:];
                if verbose println("\nWiederholung $k: Die Movie-Matrix ist fest. Die User-Matrix wird optimiert.") end
            end
            
            predictions = predict(U , M).+average
            train_mse = compute_mse(train_matrix, predictions); 
            append!(train_mse_record, train_mse); 
        
            if datasplit == true
                test_matrix = convert(Array{Float64,2},test_matrix_default)
                test_mse = compute_mse(test_matrix, predictions);
                append!(test_mse_record, test_mse);
            else
            
            end

        end
        
        PredictionMatrix = U * M
        PredictionMatrix = round.(PredictionMatrix, digits = 4)

        if datasplit == false
        println("\n________________________________________________\nGemittelte Summe der Fehlerquadrate nach $iterations Wiederholungen: \n$(round(train_mse_record[n_iters], digits =5))\n")


        else
            println("\n________________________________________________\nGemittelte Summe der Fehlerquadrate auf den Trainingsdaten nach $iterations Wiederholungen: \n$(round(train_mse_record[n_iters], digits =5))\n")
     #        println("\nGemittelte Summe der Fehlerquadrate auf den Testdaten nach $iterations Wiederholungen: \n$(round(test_mse_record[n_iters], digits =5))\n")  

        end

       if verbose println("Hinweis: Um den Fehler auf verschieden großen Rating-Matrizen besser vergleichen zu können, wurde die Summe der Fehlerquadrate  gemittelt. Dazu wurde der Fehler durch die Anzahl der bekannten Ratings geteilt. Wir betrachten also die gemittelte Summe der Fehlerquadrate.\n") end
    
    end
    
    return train_mse_record, PredictionMatrix, U, M
    
end



using NPZ, SparseArrays, StatsBase, Plots, NMF, LinearAlgebra, Random, Calculus, PrettyTables, DataFrames

include("utils.jl")
include("utils_ALS.jl")
include("utils_plots.jl");
include("loss_functions.jl");
include("utils_dict.jl"); 



col = Array{Int64}(undef,15) # zeros(15,1)
rating = zeros(15,1)

function load_and_split_data()
    movieIndex, userIndex, ratings = load_netflix_data(); 
    movie_subset, user_subset, ratings_subset = subset_top_movies(movieIndex, userIndex, ratings,120000);
    movie_subset, user_subset, ratings_subset = subset_random_users(movie_subset, user_subset, ratings_subset, 6500);  
    Rtrain, Rtest, ids = split_data(movie_subset, user_subset, ratings_subset, 0.1); 
    printDatasetInfo(Rtrain,Rtest)
    
    return Rtrain, Rtest, ids
end


function printDatasetInfo(movie_subset, user_subset, ratings_subset)
    println("Der Datensatz beeinhaltet\n
        $(length(unique(movie_subset))) Filme\n
        $(length(unique(user_subset))) Nutzer\n
        $(length(ratings_subset)) Bewertungen\n
        ")
end


function printDatasetInfo(Rtrain, Rtest)
    println("Der Datensatz beeinhaltet\n
        $(size(Rtrain,2)) Filme\n
        $(size(Rtrain,1))  Nutzer\n
        $(length(findall(!iszero, Rtrain)))  Bewertungen im Trainingsdatensatz\n
        $(length(findall(!iszero, Rtest)))  Bewertungen im Testdatensatz\n
        ")
end



function create_user_vec(col,rating)
    
    user_vec = zeros(size(Rtrain,2),1)
    
        for i = 1 : length(col)
            user_vec[col[i]] = rating[i]
        end    

    return user_vec
end


function create_extended_dataset(Rtrain,Rtest,col,rating)
    
    user_vec = create_user_vec(col,rating);
    Rtrain = [Rtrain;user_vec']
    Rtest = [Rtest;zeros(size(Rtest,2))']
    
    return Rtrain, Rtest
end
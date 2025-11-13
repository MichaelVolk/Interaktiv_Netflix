import YAML
using NPZ

include("setup_language.jl")

mutable struct params
    iters::Int 
    n_factors::Int 
    lambda::Float64
    train_mse::Float64
    test_mse::Float64
end

# DATA LOADING -------------------------------------------------------------------------------------------------------

function load_netflix_data()
    """Load all netflix data in human-readable format."""
    
    movieIndex = npzread("../data/npys/I.npy")
    userIndex = npzread("../data/npys/J.npy")
    ratings = npzread("../data/npys/V.npy")
    
    return convert(Array{Int64,1}, movieIndex), convert(Array{Int64,1}, userIndex), convert(Array{Int64,1}, ratings)

end
 

# SAMPLE -------------------------------------------------------------------------------------------------------

function subset_movie_genre(movieIndex, userIndex, ratings, genres=["Action"])
    """Filter movie, user and ratings by movie genre"""

    # We search through the overview file to find all movie IDs 
    # which match the genre.
    genre_to_id = YAML.load_file("../data/genre_to_id.yml"; dicttype=Dict{String,Any})
    movie_ids = zeros(0)

    for genre in genres
        if haskey(genre_to_id, genre)
            append!(movie_ids, genre_to_id[genre])
        else
            println("Could not find genre: $genre.")
        end
    end
    
    idx = findall(in(movie_ids), movieIndex)
    
    sub_movieIndex = movieIndex[idx]
    sub_userIndex = userIndex[idx]
    sub_ratings = ratings[idx]
     
    return sub_movieIndex, sub_userIndex, sub_ratings
    
end

function subset_top_movies(movieIndex, userIndex, ratings, n_ratings=150000)
    """Create a matrix with all reviews from movies with more than 'n_ratings' ratings."""    

    # Check if number of ratings is in list of pre-computed datasets
    # If not load overview file
    if isfile("../data/npys/count_$n_ratings.npy")
      #  println("Loading cached dataset.")
        idx = npzread("../data/npys/count_$n_ratings.npy")
        idx .+= 1  # Julia starts counting at 1

    else
        # We search through the overview file to find all movie IDs 
        # which have at least n_ratings.
        # Note that the order of idx in the if and else branch
        # will not match.
        overview = YAML.load_file("../data/overview.yml")
        movie_ids = zeros(0)
        for (id, value) in overview
            if value["count"] > n_ratings
                append!(movie_ids, id) 
            end
        end

        idx = findall(in(movie_ids), movieIndex)

        # Cache file to save time when rerunning
        npzwrite("../data/npys/count_$n_ratings.npy", idx)
        println("Caching dataset.")

    end

    sub_movieIndex = movieIndex[idx]
    sub_userIndex = userIndex[idx]
    sub_ratings = ratings[idx]
     
    return sub_movieIndex, sub_userIndex, sub_ratings
    
end

function create_subset_random(n_users, sparse_rating_matrix)
    """Create a sparse matrix with all reviews from n random users."""
    # Convert into sorted vectors
    movieVec, userVec, ratingVec = findnz(sparse_rating_matrix);
            
    # Sample userIDs by random numbers
    Random.seed!(123)
    selected_user_vec = sample(unique(userVec), n_users, replace=false) 

    
    return movieIndex[idx], userIndex[idx], ratings[idx]
    
end

function subset_random_users(movies, users, ratings, n_users) 
    Random.seed!(123)
    selected_user_vec = sample(unique(users), n_users, replace=false)
    idx = findall(in(selected_user_vec), users)
    
    return movies[idx], users[idx], ratings[idx]
end

function subset_random_movies(movies, users, ratings, n_movies)    
    Random.seed!(123)
    selected_movie_vec = sample(unique(movies), n_movies, replace=false) 
    idx = findall(in(selected_movie_vec), movies)
    
    return movies[idx], users[idx], ratings[idx]
end


# SPLIT DATA -------------------------------------------------------------------------------------------------------

function split_data(movies, users, ratings, test_size=0.1)
    """Split a sparse rating matrix into a separate test movie, user and rating vec.
    
    Note that this functions replaces existing ratings with zeros.
    """
    ((test_size >= 0) && (test_size <= 1)) || error("test size must be in range [0,1].")
    
    n_ratings = length(ratings)
        
    # Sample vector in the range 1 to n_rating with length: test_size * n_ratings 
    Random.seed!(111)
    test_idx = sample(1:n_ratings, Int(floor(test_size * n_ratings)), replace=false) 
    train_idx = [x for x in setdiff(Set(1:n_ratings), Set(test_idx))]    
    
    ratings_train_matrix = copy(ratings)
    ratings_test_matrix = copy(ratings)

    ratings_train_matrix[test_idx] .= 0
    ratings_test_matrix[train_idx] .= 0
    
    train_matrix, movie_to_column, user_to_row = build_rating_matrix(movies, users, ratings_train_matrix)
    # To not have any issues with orderings or randomness, we forward the used user/movie
    # to row/column dictionary
    test_matrix, _, _  = build_rating_matrix(movies, users, ratings_test_matrix, movie_to_column, user_to_row)
    
    col_to_movie = Dict(value => key for (key, value) in movie_to_column)
    row_to_user = Dict(value => key for (key, value) in user_to_row)
        
    train_matrix, test_matrix, idx = filter_users_without_ratings(train_matrix, test_matrix)
    train_matrix, test_matrix, idx = filter_movies_without_ratings(train_matrix, test_matrix)    
    
    # Remove all non idx entries from the dict and update afterwards
    idx_remove = [x for x in setdiff(Set(1:n_ratings), Set(idx))]
    for k in idx_remove
        delete!(row_to_user, k)
    end
            
    return train_matrix, test_matrix, col_to_movie, row_to_user

end

# DATA POSTPROCESSING -------------------------------------------------------------------------------------------------------

function filter_users_without_ratings(train_matrix, test_matrix)
    temp = vec(sum(>(0), train_matrix, dims = 2))
    idx = findall(!iszero, temp)
    rm = size(train_matrix, 1) - length(idx)
    
    println("$rm " * PrintFilterInfo)
    
    train_matrix = train_matrix[idx, :]
    test_matrix = test_matrix[idx, :]
    
    return train_matrix, test_matrix, idx
end


function filter_movies_without_ratings(train_matrix, test_matrix)
    temp = vec(sum(>(0), train_matrix, dims = 1))
    idx = findall(!iszero, temp)
    rm = size(train_matrix, 2) - length(idx)
    
    println("$rm " * "Filme ohne Bewertungen im Trainingsdatensatz wurden entfernt.\n")
    
    train_matrix = train_matrix[:,idx]
    test_matrix = test_matrix[:,idx]
    
    return train_matrix, test_matrix, idx
end

# UTILITY -------------------------------------------------------------------------------------------------------

function convert_idx_set_to_rating_matrix(movieIndex, userIndex, ratings)
    """Convert set of indices to a 2D matrix."""
    return sparse(UInt32.(movieIndex), UInt32.(userIndex), ratings)
    
end

function build_rating_matrix(movie_all_data, user_all_data, ratings_data, movie_to_column=nothing, user_to_row=nothing)
    """Create a rating matrix of shape users x movies."""
    
    # Create empty matrix for ratings
    unique_movies = unique(movie_all_data)
    unique_users = unique(user_all_data)
    n_movies = length(unique_movies)
    n_users = length(unique_users)
    rating_matrix = zeros(Int8, n_users, n_movies)
    
    # We do not want to assume that the movie vector is grouped/sorted
    # and therefore define a mapping between movie/user to column/row
    if isnothing(movie_to_column)
        movie_to_column = Dict(zip(unique_movies, 1:n_movies))
    end
    if isnothing(user_to_row)
        user_to_row = Dict(zip(unique_users, 1:n_users))
    end
    
    # Loop over movies and user, look up their row and column and place
    # the rating at this position
    for (idx, (m, u)) in enumerate(zip(movie_all_data, user_all_data))
        rating_matrix[user_to_row[u], movie_to_column[m]] = ratings_data[idx]
    end
    
    # We also return the mapping for later usage
    return rating_matrix, movie_to_column, user_to_row

end

function predict(A::Matrix, B::Matrix)
    """Compute the product of two matrices.
    
    We use this funciton to "predict" user ratings by
    multiplying user and movie matrix.    
    """
    return A * B
end
 
function predict_simple(user_factors::Matrix, item_factors::Matrix, u, i)
    user_factors[u, :] * item_factors[:, i]
end

function predict_with_item_mean(RatingMatrix)
    PredictedMatrix = convert(Array{Float64,2}, RatingMatrix[:,:])
    item_mean = compute_item_mean(RatingMatrix)
    
    for i = 1:size(RatingMatrix, 2)
        idx = findall(iszero, RatingMatrix[:,i])
        PredictedMatrix[idx, i] .= item_mean[i]
    end
    
    return PredictedMatrix
end 

function predict_with_user_mean(RatingMatrix)
    PredictedMatrix = convert(Array{Float64,2}, RatingMatrix[:,:])
    user_mean = compute_user_mean(RatingMatrix)
    
    for i = 1:size(RatingMatrix, 1)
        idx = findall(iszero, RatingMatrix[i,:])
        PredictedMatrix[i, idx] .= user_mean[i]
    end
    
    return PredictedMatrix
end 

function predict_with_rating_mean(RatingMatrix)
    PredictedMatrix = convert(Array{Float64,2}, RatingMatrix[:,:])
    rating_mean = compute_rating_mean(RatingMatrix)
    
    idx = findall(iszero, RatingMatrix)
    PredictedMatrix[idx] .= rating_mean
    
    return PredictedMatrix
end 

function compute_user_mean(RatingMatrix)

    nz_entries = sum(RatingMatrix .!= 0, dims=2);
    user_mean =  sum(RatingMatrix, dims=2) ./ nz_entries
    
    return user_mean
end

function compute_item_mean(RatingMatrix)
    nz_entries = sum(RatingMatrix .!= 0, dims=1);
    item_mean =  sum(RatingMatrix, dims=1) ./ nz_entries

    return item_mean
end

function compute_rating_mean(RatingMatrix)
    nz_entries = sum(RatingMatrix .!= 0);
    rating_mean = (sum(RatingMatrix) ./ nz_entries);

    return rating_mean
end

function cross_validate(iter, reg_term, latent_factors, train_matrix, test_matrix, loss, verbose, weighted)
    
    """

    
    
    """ 
    
    
    best_params = params(0, 1, 1, Inf, Inf)

    # Der Befehl `Iterators.product` berechnet alle Kombinationsmoeglichkeiten
    # zwischen den angegebenen Listen
    df = DataFrame(Iterators.product(iter, latent_factors, reg_term))
    names!(df, Symbol.(["iters", "latent_factors", "reg_term"])) 
    insert!(df, 4, Inf, :loss)

    for idx in 1:nrow(df)
        iterations = df[idx, "iters"]
        fact = df[idx, "latent_factors"]
        reg = df[idx, "reg_term"]
        test_mse_record, train_mse_record = compute_factorization_als_bias(iterations, latent_factors, train_matrix, test_matrixs,zeros(size(Rtrain)), "RMSE", reg_term,reg_term, false,false,false); 
        
        df[idx, "loss"] = round(test_mse_record[end], digits=4)

        if verbose
            println("No. of iterations: ", iterations)
            println("No. of latent factors: ", fact)
            println("Regularization term: ", reg)
            println("-------------------------------------")
        end

        if test_mse_record[end] < best_params.test_mse
            best_params.iters = iterations
            best_params.n_factors = fact
            best_params.lambda = reg
            best_params.train_mse = train_mse_record[end]
            best_params.test_mse = test_mse_record[end]
        end

    end
    
    if verbose
        pretty_table(sort(df, [:loss]), border_crayon=crayon"blue", show_row_number=true, screen_size=(1000, 1000), alignment=[:r,:r,:r,:l])
    else
    end
    
    println("Optimal Parameters:\n", best_params)
    
    return best_params
    
end

PrintMSETrain = "\n________________________________________________\nGemittelte Summe der Fehlerquadrate auf den Trainingsdaten nach "

function printTrainError()

    println( TrainMSE_After_ALS * "$iterations" *  PrintRepetitions * "\n$(round(train_loss[iterations], digits =3))\n")
    
    
end

function printTrainTestError()

    println("\n________________________________________________\nGemittelte Summe der Fehlerquadrate auf den Trainingsdaten nach $iterations Wiederholungen: \n$(round(train_loss[iterations], digits =3))\n")
    println("\n________________________________________________\nGemittelte Summe der Fehlerquadrate auf den Testdaten nach $iterations Wiederholungen: \n$(round(test_loss[iterations], digits =3))\n")
    
    
end


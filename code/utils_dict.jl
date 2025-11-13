include("setup_language_checks.jl")

function movies_to_genre(movie_ids, genre1, genre2)
    """Filter movie, user and ratings by movie genre"""
    genre = zeros(length(movie_ids))
    
    overview = YAML.load_file("../data/overview.yml")   
    
    for (index, id) in enumerate(movie_ids)
        genre_set = overview[id]["genres"]
        if isnothing(genre_set)
            genre[index] = -1
        elseif (genre1 in genre_set) & (genre2 in genre_set)
            genre[index] = 0.5  # beide Genres sind im Film
        elseif genre2 in genre_set
            genre[index] = 1  # nur genre2
        end
    end
    
    return genre
    
end

function movie_to_title(n_col, col_to_movie)
    """Return title corresponding to the movie_id at column 'n_col'."""
    
    overview = YAML.load_file("../data/overview.yml") 
    
    # return dict-value for key 'n_col' 
    movie_id = get(col_to_movie, n_col, default)
    title = overview[movie_id]["title"]
    
    return title
end


function random_movies_to_titles(Rtrain,col_to_movie,n_movies)
    """Return array of titles corresponding to the movie_ids."""
    
    overview = YAML.load_file("../data/overview.yml")
    titles = Array{String}(undef, n_movies)
    
    
    Random.seed!(123)
    selected_movies = sample(collect(1:1:size(Rtrain,2)), n_movies, replace=false) 
    
    for (index,i) in enumerate(selected_movies)
        movie_id = get(col_to_movie, i, default)
        titles[index] = overview[movie_id]["title"]
        #println(titles)
        #println(movie_id)
    end
    
    return titles
end

function movies_to_titles(movie_ids)
    """Return array of titles corresponding to the movie_ids."""
    
    overview = YAML.load_file("../data/overview.yml")   
    titles = Array{String}(undef, length(movie_ids))
    for (index, id) in enumerate(movie_ids)
        titles[index] = overview[id]["title"]
    end
    
    return titles
end

function compute_recommendation(Pred, Rtrain, col_to_movie, user_idx)
    
    if isnan(user_idx)
        printError("Ersetze NaN durch die Zeilennummer des Users für den du eine Empfehlung berechnen willst.")
        
    else
        idx_train = findall(!iszero, Rtrain)
        Pred[idx_train].= 0;


        # find max value in each row
        val, idx = findmax(Pred, dims=2)

        # return the colum of the highest predicted movie of row user_idx
        n_col = idx[user_idx][2]

        recommendation = movie_to_title(n_col,col_to_movie)

        println(PrintPredForUser * "$user_idx" *  PrintIsMovie * "\n$recommendation.\n\n" * PrintPredictedMovieRating * "\n$(round((val[user_idx]),digits =2))")
    end
    
    return  
end


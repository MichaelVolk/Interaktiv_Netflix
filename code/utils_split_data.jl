 
function split_data_ratingmatrix(ratingmatrix, percentage)
    """Split a rating matrix into a training matrix and a test matrix
    
    Note that this function replaces existing ratings in the testmatrix with zeros.
    """
    test_size = percentage / 100;
    
    ((test_size >= 0) && (test_size <= 100)) || error("Der prozentuale Anteil der Werte in der Testmatrix muss im Bereich von [0,100] liegen.")
    
    
    rating_idx = findall(ratingmatrix .!= 0)
    n_ratings = sum(ratingmatrix .!= 0);
        
    # Sample vector in the range 1 to n_rating with length: test_size * n_ratings 
    Random.seed!(111)
    test_idx = sample(1 : n_ratings, Int(floor(test_size * n_ratings)), replace = false) 
    
    ratings_train_matrix = convert(Array{Float64,2}, copy(ratingmatrix))
    ratings_test_matrix = zeros(size(ratingmatrix))

    ratings_train_matrix[rating_idx[test_idx]] .= 0
    ratings_test_matrix[rating_idx[test_idx]] = ratingmatrix[rating_idx[test_idx]]
    
    return ratings_train_matrix, ratings_test_matrix

end
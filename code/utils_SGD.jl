using Printf

function compute_mf_sgd(  
    n_iters,
    n_factors,
    train_matrix,
    test_matrix,
    learning_rate,
    loss_function="RMSE",
    lambda=0.055,
    verbose=true,
    optimize_bias_terms = false,
    weighted_sgd= false,
    lambda_user_bias = 0.0,
    lambda_item_bias = 0.0,
    init_users=nothing,
    init_movies=nothing,
)
    """Compute factorization of the train matrix.

    n_iters: number of iterations to train the algorithm
    n_factors: number of latent factors to use in matrix 
    train_matrix_default: matrix to compute factorization from
    test_matrix: matrix to track testing loss
    loss_function: loss function to evaluate training progress
    lambda: regularization term for item/user latent factors 
    verbose: print information during computation
    """ 
    
     
    if n_factors > minimum(size(train_matrix))
        @assert false "Die Anzahl der features muss kleiner als die Anzahl der Nutzer und der Filme sein."
    end
    
    train_matrix = convert(Array{Float64}, train_matrix[:,:])
    train_matrix_default = copy(train_matrix)
    test_matrix = convert(Array{Float64}, test_matrix)
    user_bias = zeros(size(train_matrix,1))
    item_bias = zeros(size(train_matrix,2))
    
    idx = findall(!iszero, train_matrix_default)
    D = zeros(size(train_matrix_default));
    D[idx] .= 1
    
    if optimize_bias_terms    
        average = sum(train_matrix) / sum(train_matrix .!= 0)
    else
        average = 0
    end
            
    test_loss_record  = zeros(n_iters) 
    train_loss_record = zeros(n_iters)
    predictions = []
    
    Random.seed!(123)
    if init_users == nothing
        U = rand(size(train_matrix, 1), n_factors)  
    else
        U = init_users
    end
    
    if init_movies == nothing
        M = rand(n_factors, size(train_matrix, 2)) 
    else
        M = init_movies
    end
    
 
        
    if weighted_sgd
        tikhonov_user = (vec(sum(>(0), train_matrix_default, dims = 2)))
        tikhonov_movies = (vec(sum(>(0), train_matrix_default, dims = 1)))
    else
        tikhonov_user = ones(size(train_matrix_default))
        tikhonov_movies = ones(size(train_matrix_default))
    end
    
    
    
    
    for k = 1 : n_iters
        if (k%10==0) println("Iteration $k") else end
        
            rng = MersenneTwister(1234);
            rand_u = shuffle(Vector(1:size(U,1)))  
            rand_m = shuffle(Vector(1:size(M,2)))
        
        for i in rand_u  
            
            for j in rand_m  
                
                # Only consider known ratings
                if train_matrix_default[i,j] > 0
                    e = train_matrix[i,j] - (U[i,:]' * M[:,j] + user_bias[i] + item_bias[j] + average)

                    # Update latent factors
                    Ui_old = U[i,:]    
                    U[i,:] += learning_rate *  D[i,j] * (e * M[:,j] - (lambda * tikhonov_user[i]) * U[i,:]) 
                    M[:,j] += learning_rate *  D[i,j] * (e * Ui_old - (lambda * tikhonov_user[j]) * M[:,j])                 

                    if optimize_bias_terms    
                    # Update user bias
                        user_bias[i] += learning_rate * (e - (lambda * tikhonov_user[i]) * user_bias[i])
                        item_bias[j] += learning_rate * (e - (lambda * tikhonov_user[j]) * item_bias[j])
                    else
                    end
        
                else
                end
            end
        end
    
        
        zeros_idx = findall(iszero, train_matrix_default)
        predictions = (U * M) + ones(size(train_matrix)) .* average + ones(size(train_matrix)) .*  user_bias + ones(size(train_matrix)) .* item_bias' 
        
        lfs = Dict{String, Function}(
            "RMSE"=>compute_rmse,
            "MSE"=>compute_mse,
            "MAE"=>compute_mae,
        )
          
        test_loss = lfs[loss_function](test_matrix, predictions);
        train_loss = lfs[loss_function](train_matrix_default, predictions);
        
         
        test_loss_record[k] = test_loss;
        train_loss_record[k] = train_loss;
        
    end
    
    

        println(
            TrainMSE_After_ALS * "$n_iters " * 
            PrintRepetitions * "\n$(train_loss_record[n_iters])\n"
        )
        if verbose
        println(
           TestMSE_After_ALS * "$n_iters " * 
            PrintRepetitions * "\n$(test_loss_record[n_iters])\n"
        )
    end
    
          
    return test_loss_record , train_loss_record, U, M, predictions
end



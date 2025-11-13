# -*- coding: utf-8 -*-
using Printf



# ALS Factorization - missings (zeros) are not fitted!
function compute_factorization_als_bias(
    n_iters,
    n_factors,
    train_matrix,
    test_matrix,
    baseline_matrix,
    loss_function="RMSE",
    lambda_movies=0.05,
    lambda_users=0.05,
    lsqfit = false,
    verbose=true,
    weighted_als = false,
    bias_fit = false,
    init_users=nothing,
    init_movies=nothing,
)
    """Compute factorization of the train matrix.

    n_iters: number of iterations to train the algorithm
    n_factors: number of latent factors to use in matrix 
    train_matrix_default: matrix to compute factorization from
    test_matrix: matrix to track testing loss
    baseline_matrix: matrix used for pre- and postprocessing of the data
    loss_function: loss function to evaluate training progress
    lambda: regularization term for item/user latent factors 
    lsqfit: binary variable. false = solve ALS via normal equations, true = solve via svd
    verbose: print information during computation
    """
    
    if n_factors > minimum(size(train_matrix))
        @assert false "Die Anzahl der features muss kleiner als die Anzahl der Nutzer und der Filme sein."
    end
    
    train_matrix = convert(Array{Float64}, train_matrix[:,:])
    train_matrix_default = copy(train_matrix)
    test_matrix = convert(Array{Float64}, test_matrix)
    
    idx = findall(!iszero, train_matrix_default)
    D = zeros(size(train_matrix_default));
    D[idx] .= 1
    
    if weighted_als
        tikhonov_user = (vec(sum(>(0), train_matrix_default, dims = 2)))
        tikhonov_movies = (vec(sum(>(0), train_matrix_default, dims = 1)))
    else
        tikhonov_user = ones(size(train_matrix_default))
        tikhonov_movies = ones(size(train_matrix_default))
    end
    
        
    if bias_fit
        user_bias = rand(size(train_matrix, 1),1)
        movie_bias = rand(size(train_matrix, 2),1)
        average = sum(train_matrix_default) / sum(train_matrix_default .!= 0)
        baseline_matrix = ones(size(train_matrix_default)) .* average
    else
        user_bias = zeros(size(train_matrix, 1),1)
        movie_bias = zeros(1,size(train_matrix, 2))
    end
    
    train_matrix -= baseline_matrix

    
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
        M = rand(n_factors, size(train_matrix, 2)) # * 0.01
        M[1,:] = (vec(sum(train_matrix, dims = 1)) ./ vec(sum(>(0), train_matrix, dims = 1)))'
        
    else
        M = init_movies
    end
 
    for k = 1 : n_iters
        
        if k % 2 == 0 
            if lsqfit
                U = als_step_u_nonzeros_LSQFIT(U, M, train_matrix, lambda_users,D)[:,:];
            elseif bias_fit
                U , user_bias = als_step_u_nonzeros_bias(U, M, train_matrix, lambda_users,D,tikhonov_user,movie_bias, n_factors)
                U = U[:,:]
            else
                U = als_step_u_nonzeros(U, M, train_matrix, lambda_users,D,tikhonov_user)[:,:];
            end   

            
        else 

            if lsqfit             
                M = als_step_m_nonzeros_LSQFIT(M, U, train_matrix, lambda_movies,D)[:,:];  
            elseif bias_fit
                M , movie_bias = als_step_m_nonzeros_bias(M, U, train_matrix, lambda_movies,D,tikhonov_movies, user_bias, n_factors)
                M = M[:,:]
            else 
                M = als_step_m_nonzeros(M, U, train_matrix, lambda_movies,D,tikhonov_movies)[:,:];  
            end
            

        end   
    
        zeros_idx = findall(iszero, train_matrix_default)
        predictions = U * M  + baseline_matrix .+ user_bias  + (ones(size(train_matrix_default))' .* movie_bias')' 
        
        

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

    if verbose
        println(
            TrainMSE_After_ALS * "$n_iters " * 
            PrintRepetitions * "\n$(train_loss_record[n_iters])\n"
        )
        println(
           TestMSE_After_ALS * "$n_iters " * 
            PrintRepetitions * "\n$(test_loss_record[n_iters])\n"
        )
    end

    return test_loss_record , train_loss_record, U, M, predictions , user_bias, movie_bias

end

 


function als_step_u_nonzeros_bias(U::Array, M::Array, rating_matrix::Array, lambda::Number, D::Array, tikhonov_user, movie_bias, n_factors)
    """Compute single Alternating Least Squares (ALS) step."""

     U_with_bias = zeros(size(U,1),(n_factors+1))
        
    for u = 1 : size(U, 1)
              
        M_red = (([M ; ones(1,size(M,2))])' .* D[u,:])'[:,:]
    
        lambdaI = I(n_factors+1) * lambda * tikhonov_user[u]
        r_i_with_movie_bias = (rating_matrix[u,:]' - movie_bias)'
        
        YYT = M_red * M_red'
        U_with_bias[u:u,:] = ((YYT + lambdaI) \ (r_i_with_movie_bias' * M_red')') # rating_matrix[u:u,:] * M_red' * (YYT + lambdaI)^(-1)
    end
    
        U = U_with_bias[:,1:n_factors]
        user_bias = U_with_bias[:,(n_factors+1)]
    
    return U , user_bias

end 


function als_step_u_nonzeros(U::Array, M::Array, rating_matrix::Array, lambda::Number, D::Array, tikhonov_user)
    """Compute single Alternating Least Squares (ALS) step."""

    for u = 1 : size(U, 1)
        M_red = (M' .* D[u,:])'
        lambdaI = I(size(M,1)) * lambda * tikhonov_user[u]
        
        # Solve system of normal equations
        YYT = M_red * M_red'
        U[u:u,:] = ((YYT + lambdaI) \ (rating_matrix[u:u,:] * M_red')') # rating_matrix[u:u,:] * M_red' * (YYT + lambdaI)^(-1)
    end

    return U

end 



function als_step_m_nonzeros_bias(M::Array, U::Array, rating_matrix::Array, lambda::Number, D::Array,tikhonov_movies, user_bias, n_factors)
    """Compute single Alternating Least Squares (ALS) step."""

        M_with_bias = zeros(n_factors+1,size(M,2))    
    
    for j = 1 : size(M, 2)
        lambdaI = I(n_factors+1) * lambda   * tikhonov_movies[j] # .* D[:,i]
        U_red = [U ones(size(U,1))] .* D[:,j]    
        r_j_with_user_bias = rating_matrix[:,j:j] .- user_bias
        
        
        # Solve system of normal equations
        XTX = U_red' * U_red
        M_with_bias[:,j:j] =  ((XTX + lambdaI)\(r_j_with_user_bias' * U_red)')' #((rating_matrix[:,i:i]' * U_red) * (XTX + lambdaI)^(-1))'
    end
        M = M_with_bias[1:n_factors,:]
        movie_bias = (M_with_bias[n_factors + 1,:])'[:,:]
    return M, movie_bias

end  


function als_step_m_nonzeros(M::Array, U::Array, rating_matrix::Array, lambda::Number, D::Array,tikhonov_movies)
    """Compute single Alternating Least Squares (ALS) step."""


    for i = 1 : size(M, 2)
        lambdaI = I(size(U,2)) * lambda   * tikhonov_movies[i] # .* D[:,i]
        U_red = U .* D[:,i]    
        
        # Solve system of normal equations
        XTX = U_red' * U_red
        M[:,i:i] =  ((XTX + lambdaI)\(rating_matrix[:,i:i]' * U_red)')'#((rating_matrix[:,i:i]' * U_red) * (XTX + lambdaI)^(-1))'
        
        
    end
    
    return M

end  


   


 

function als_step_u_nonzeros_LSQFIT(U::Array, M::Array, rating_matrix::Array, lambda::Number, D::Array)
    """Compute single Alternating Least Squares (ALS) step."""
    

    lambdaI = I(size(M,1)) * lambda

    for u = 1 : size(U, 1)
        M_red = (M' .* D[u,:])'
        U[u:u,:] = solveLSQ(M_red',rating_matrix[u:u,:]',lambda)

    end

    return U

end 


function als_step_m_nonzeros_LSQFIT(M::Array, U::Array, rating_matrix::Array, lambda::Number, D::Array)
    """Compute single Alternating Least Squares (ALS) step."""

    lambdaI = I(size(U,2)) * lambda

    for i = 1 : size(M, 2)
        U_red = U .* D[:,i]    
        M[:,i:i] = solveLSQ(U_red,rating_matrix[:,i:i],lambda)
            
    end
    
    return M

end  

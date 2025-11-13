function compute_baseline_estimates_gd(train_matrix,test_matrix,n_iters,learning_rate,lambda = 0)
 
    average = sum(train_matrix) / sum(train_matrix .!= 0)
    user_bias = vec(sum(train_matrix, dims = 2)) ./ vec(sum(>(0), train_matrix, dims = 2)) .-  average ; # zeros(size(train_matrix,1))  
    item_bias = vec(sum(train_matrix, dims = 1)) ./ vec(sum(>(0), train_matrix, dims = 1)) .-  average ;
    train_loss_record = zeros(n_iters)
    test_loss_record = zeros(n_iters)
    

    for k = 1 : n_iters
        for i = 1 : size(train_matrix,1)
            for j = 1 : size(train_matrix,2)
                if train_matrix[i,j] > 0
                    e = train_matrix[i,j] - (user_bias[i] + item_bias[j] + average)
                    user_bias[i] += learning_rate * (e - lambda * user_bias[i])
                    item_bias[j] += learning_rate * (e - lambda * item_bias[j])
                end
            end
        end
        predictions = ones(size(train_matrix)) .* average + ones(size(train_matrix)) .*  user_bias + ones(size(train_matrix)) .* item_bias' 
        train_loss = compute_rmse(train_matrix, predictions);
        test_loss = compute_rmse(test_matrix, predictions);
        train_loss_record[k] = train_loss;
        train_loss_record[k] = test_loss;
    end
    P = ones(size(Rtrain)) .* average + ones(size(Rtrain)) .* user_bias + ones(size(Rtrain)) .* item_bias'
    return test_loss_record, train_loss_record, user_bias, item_bias , P
    
end


# - ALS according to Koren 2010

function add_item_bias(train_matrix_default, overall_average, lambda_item_bias=0)   
     
        # compute item bias
        item_bias = zeros(size(train_matrix_default,2))  

        # substract item bias
        for j = 1 : size(train_matrix_default,2)
            idx = findall(!iszero, train_matrix_default[:,j])
            
            item_bias[j] =  (sum(train_matrix_default[idx,j] .- overall_average)) / (lambda_item_bias + length(idx))
        end  
    
    return item_bias
end

function add_user_bias_old(train_matrix_default, overall_average, lambda_user_bias=0, item_bias = zeros(size(train_matrix_default,2)))
        
        # compute user bias
        user_bias =  zeros(size(train_matrix_default,1))
        
        # substract user bias
        for i = 1 : size(train_matrix_default,1)
            idx = findall(!iszero, train_matrix_default[i,:])
        
            user_bias[i] = (sum(train_matrix_default[i,idx] .- overall_average - item_bias[idx])) / (lambda_user_bias + length(idx))
        end  
        
    return user_bias
end

function add_user_bias(train_matrix_default, overall_average, lambda_user_bias=0, item_bias = zeros(size(train_matrix_default,2)))
        
        # compute user bias
        user_bias =  zeros(size(train_matrix_default,1))
        
        # substract user bias
        for i = 1 : size(train_matrix_default,1)
            idx = findall(!iszero, train_matrix_default[i,:])
        
            user_bias[i] = (sum(train_matrix_default[i,idx] .- overall_average - item_bias[idx])) / (lambda_user_bias + length(idx))
        end  
        
    return user_bias
end

function compute_baseline_estimate(
    train_matrix_default, 
    normalize_user_bias = true, 
    normalize_item_bias = true, 
    lambda_item_bias = 1.0,
    lambda_user_bias = 1.0,
    
)
    
    """ Compute baseline estimate
    Implemented according to Koren, Y. (2010) paper: Factor in the Neighbours
    overall_average: average rating of training data
    normalization: binary variable. true = training data is normalized by overall_average, false otherwise
    normalize_user_bias: binary variable. true = training data is normalized by user_bias, false otherwise 
    normalize_item_bias: binary variable. true = training data is normalized by item_bias, false otherwise
    lambda_user_bias: parameter adjusting the influence of the user_bias, default: 0,
    lambda_item_bias: parameter adjusting the influence of the item_bias, default: 0 
    """

    idx = findall(!iszero, train_matrix_default)
    idx_zeros = findall(!iszero, train_matrix_default)
    baseline_matrix = zeros(size(train_matrix_default))
    av = sum(train_matrix_default) / sum(train_matrix_default .!= 0)
    
    
    if normalize_item_bias 
        item_bias = add_item_bias(train_matrix_default, av, lambda_item_bias)  
    else
        item_bias = zeros(size(train_matrix_default,2))
    end
    
    if normalize_user_bias
        user_bias = add_user_bias(train_matrix_default, av, lambda_user_bias, item_bias)
    else
        user_bias = zeros(size(train_matrix_default,1))
    end

     baseline_matrix .+= (kron( ones(size(train_matrix_default,1))',item_bias))'  .+ user_bias 
    
     # add the average to the positions with missings
     temp = zeros(size(train_matrix_default))
     temp[idx_zeros] .= av
     baseline_matrix = baseline_matrix + ones(size(train_matrix_default)) .* av 
    return baseline_matrix , item_bias, user_bias
end


 
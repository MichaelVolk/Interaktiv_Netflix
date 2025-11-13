function compute_mse(trues, preds)
    """Compute MSE."""
    trues = convert(Array{Float64}, trues[:,:])
    idx = findall(!iszero, trues)  # Get index of non zero terms
    return  (msd(trues[idx], preds[idx])) # round(msd(trues[idx], preds[idx]), digits = 5)

end

function compute_rmse(trues, preds)
    """Compute RMSE."""
    trues = convert(Array{Float64}, trues[:,:])
    idx = findall(!iszero, trues)  # Get index of non zero terms
    return rmsd(trues[idx], preds[idx])

end

function compute_mae(trues, preds)
    """Compute MAE."""
    trues = convert(Array{Float64}, trues[:,:])
    idx = findall(!iszero, trues)  # Get index of non zero terms
    return mean(abs.(trues[idx] - preds[idx]))

end
using Suppressor, NPZ, SparseArrays, StatsBase, Plots, LinearAlgebra, Random, Calculus, PrettyTables, DataFrames
# NMF

include("utils.jl")
include("utils_ALS.jl")
include("utils_plots.jl");
include("utils_pretty_tables.jl");
include("loss_functions.jl");
include("utils_dict.jl"); 
include("output.jl"); 
include("utils_lsqFIT.jl")
include("setup_language.jl")
include("setup_kit_colors.jl")
include("utils_baseline.jl")


function printDatasetInfo(movie_subset, user_subset, ratings_subset)
    println(DatasetInfo* 
        "  $(length(unique(movie_subset))) " * MovieInfo * 
        "  $(length(unique(user_subset))) " * UserInfo * 
        "  $(length(ratings_subset)) " * RatingsInfo)
end


function printDatasetInfo(Rtrain, Rtest)
    println(DatasetInfo * "$(size(Rtrain,2)) " * MovieInfo *
        "  $(size(Rtrain,1)) " * UserInfo *
        "  $(length(findall(!iszero, Rtrain))) " * TrainsetInfo *
        "  $(length(findall(!iszero, Rtest))) " * TestsetInfo)
end


function print_statistics_dataset(Rtrain, Rtest)
    idx = findall(!iszero, Rtrain);
    idx_test = findall(!iszero, Rtest);
    train_sparsity = length(idx) / (size(Rtrain,1) * size(Rtrain,2)) * 100
    test_sparsity = length(idx_test) / (size(Rtest,1) * size(Rtest,2)) * 100
    train_average = mean(Rtrain[Rtrain.!=0])
    test_average = mean(Rtest[Rtest.!=0])
    train_std = std(Rtrain[Rtrain.!=0])
    test_std = std(Rtest[Rtest.!=0])

    println("Sparsity:")
    println("% of known ratings in the training data: $(train_sparsity) ")
    println("% of known ratings in the test data: $(test_sparsity)\n")
    println("mean rating of the training data: $(train_average)")
    println("mean rating of the test data: $(test_average)\n")
    println("standard deviation training data: $(train_std)")
    println("standarad deviation test data: $(test_std)\n")
end
using PrettyTables

include("setup_language.jl")

function printPretty(R,P,U,M)
    
    println(Rmatrix)
    pretty_table(R, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(Pmatrix)
    pretty_table(P, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(Umatrix)
    pretty_table(U, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(Mmatrix)
    pretty_table(M, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3)) 
    
end

function printPrettyTrainTest(Rtrain,Rtest,P, n = size(Rtrain,1) , m = size(Rtrain,2))
    

    println(Pmatrix)
    pretty_table(P[1:n,1:m], tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(trainmatrix)
    pretty_table(Rtrain[1:n,1:m], tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(testmatrix)
    pretty_table(Rtest[1:n,1:m], tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    
end


function printPrettyTrainTest(Rtrain,Rtest,P)
    

    println(Pmatrix)
    pretty_table(P, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(trainmatrix)
    pretty_table(Rtrain, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(testmatrix)
    pretty_table(Rtest, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    
end


function printRP(R,P)
    
    println(Pmatrix)
    pretty_table(P, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(Rmatrix)
    pretty_table(R, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    
end

function printTrainTest(Rtrain,Rtest)
    
    println(trainmatrix)
    pretty_table(Rtrain, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(testmatrix)
    pretty_table(Rtest, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    
end

function printTrainTest(Rtrain,Rtest,n,m)
    
    println(trainmatrix)
    pretty_table(Rtrain[1:n,1:m], tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    println(testmatrix)
    pretty_table(Rtest[1:n,1:m], tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    
end


function printSimple(R)
    
    pretty_table(R, tf = tf_borderless, noheader = true, crop = :horizontal, formatters = ft_round(3))
    
end


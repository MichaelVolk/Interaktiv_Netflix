using Plots, Suppressor

default(
    size           = [800,700],  # width x height
    lw             = 6,          # line width 
    titlefontsize  = 16,
    guidefontsize  = 16, 
    legendfontsize = 16, 
    xtickfontsize  = 16,
    ytickfontsize  = 16,
    top_margin     = :match,
    title_location = :center,
    # legendtitlefontsize = 16,
)



function visualize_training(train_mse_record, test_mse_record)
    """Plots losses from train- und testset over iterations."""
    plotly() 
    Plots.plot(
        1:size(train_mse_record,1),
        [train_mse_record],
        label="Trainingsfehler",
        size = (550,450),
        xlabel=" Wiederholung",
        ylabel="gemittelte Summe der Fehlerquadrate",
        linewidth = 6
    )
    Plots.plot!([test_mse_record], label = "Testfehler",linewidth = 6)
end


function plotMSEError(train_error)
    
    @suppress begin
    
    end
    
    if length(train_error) == 0

    else
        plotly()
        Plots.plot(collect(1:iterations),train_error, linewidth = 3, size = (550,450), label = "", xlabel = "Wiederholung", ylabel = "gemittelte Summe der Fehlerquadrate")    
    end
end




function plotFeatureVectors(U,M)
    
    Utmp1 = (U[1:min(size(U,1),1000),1]);
    Utmp2 = (U[1:min(size(U,1),1000),2]);

    Mtmp1 = (M[1,:]);
    Mtmp2 = (M[2,:]);

    plotly()
    plot(Utmp1, Utmp2, seriestype = :scatter, xlabel = "Eigenschaft 1", ylabel = "Eigenschaft 2", label = "User-Eigenschafts-Vektor")
    plot!(Mtmp1, Mtmp2, seriestype = :scatter,label = "Movie-Eigenschaft-Vektor") 
    
end
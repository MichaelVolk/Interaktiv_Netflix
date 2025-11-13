using Plotly
 import PlotlyJS

using Suppressor


function plotError(f,xleft,xright)
       println("Wenn du mit der Maus über die Grafik fährst, werden dir die Werte für u11 (=x) und u12 (=y) sowie für den Fehler (=z) angezeigt. \nDu kannst die Grafik mit der Maus bewegen und hineinzoomen. ")
   
   
    stepsize = 0.05
    @suppress begin
 
    x = [ i + j for i=xleft:stepsize:xright, j=xleft:stepsize: xright ]
    y = [ -i + j for  i=xleft:stepsize:xright, j=xleft:stepsize: xright ]
    z = round.(f.(x,y),digits = 2)
    
        layout = Layout(;paper_bgcolor="rgb(255, 255, 255)",
        plot_bgcolor="rgb(229, 229, 229)",
        height = 500,
        width = 500,
        margin=attr(l=65, r=50, b=65, t=90), 
   
        title = "",
        scene=attr(
            xaxis=attr(
                gridcolor="rgb(255, 255, 255)",
                title = "u11",
              
                showgrid=true,
                showline=true,
                showticklabels=true,
                tickcolor="rgb(127, 127, 127)",
                ticks="outside",
                zeroline=false
            ),
                
            yaxis=attr(
                gridcolor="rgb(255, 255, 255)",
                title = "u12",
                showgrid=true,
                showline=true,
                
                showticklabels=true,
                tickcolor="rgb(127, 127, 127)",
                ticks="outside",
                zeroline=false
            ),
                
            zaxis=attr(
                gridcolor="rgb(255, 255, 255)",
                title = "Fehler",
                showgrid=true,
                showline=true,
                showticklabels=true,
                tickcolor="rgb(127, 127, 127)",
                ticks="outside",
                labelsize = 
                zeroline=false
            )
        )
    )
    PlotlyJS.plot([PlotlyJS.surface(x=x,y=y,z=z)], layout)
end
end

# AB Regularisierung

function plotErrorReg(f,xleft,xright, lambda)
    
    if lambda < 0 
        printError("Der Wert für λ sollte nicht negativ sein. ")
    
    else 
    end
    
    
       println("Wenn du mit der Maus über die Grafik fährst, werden dir die Werte für u11 (=x) und u12 (=y) sowie für den Fehler (=z) angezeigt. \nDu kannst die Grafik mit der Maus bewegen und hineinzoomen. ")
   
   
    stepsize = 0.2
    @suppress begin
 
    x = [ i + j for i=xleft:stepsize:xright, j=xleft:stepsize: xright ]
    y = [ -i + j for  i=xleft:stepsize:xright, j=xleft:stepsize: xright ]
    z = round.(f.(x,y,lambda),digits = 2)
    
        layout = Layout(;paper_bgcolor="rgb(255, 255, 255)",
        plot_bgcolor="rgb(229, 229, 229)",
        height = 800,
        width = 800,
        margin=attr(l=65, r=50, b=65, t=90), 
    #    showticklabels=true,
        title = "",
        scene=attr(
            xaxis=attr(
                gridcolor="rgb(255, 255, 255)",
                title = "u11",
              #  range=[1, 10],
                showgrid=true,
                showline=true,
                showticklabels=true,
                tickcolor="rgb(127, 127, 127)",
                ticks="outside",
                zeroline=false
            ),
            yaxis=attr(
                gridcolor="rgb(255, 255, 255)",
                title = "u12",
                showgrid=true,
                showline=true,
                
                showticklabels=true,
                tickcolor="rgb(127, 127, 127)",
                ticks="outside",
                zeroline=false
            ),
            zaxis=attr(
                gridcolor="rgb(255, 255, 255)",
                title = "Fehler",
                showgrid=true,
                showline=true,
                showticklabels=true,
                tickcolor="rgb(127, 127, 127)",
                ticks="outside",
                labelsize = 
                zeroline=false
            )
        )
    )
    PlotlyJS.plot([PlotlyJS.surface(x=x,y=y,z=z)], layout)
end
end
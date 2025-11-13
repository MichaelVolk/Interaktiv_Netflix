# https://www2.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_10_Linear_least_squares_reg.pdf

using LinearAlgebra

function solveLSQ(A,b,lambda)
u,s,vT = svd(A,full = true)

x = zeros(size(A,2),1)

minSingularValue = 1e-15
    for (i,si) in enumerate(s)
        if abs(si)>minSingularValue
            x .+= (((u[:,i]'* b)*si) / (si^2+lambda)) .* vT[:,i:i]
        end
    end

   return x 
end
    
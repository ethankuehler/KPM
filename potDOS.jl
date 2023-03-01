
using Plots
using DelimitedFiles

#for doing vectors in the ploting phase
function T_nv(x, n)
    return cos.(n .* acos.(x))
end


#jackson kernal
function Jackson(N, n)
    a1 = (N-n+1)*cos((pi*n)/(N+1))
    a2 = sin((pi*n)/(N+1))
    a3 = cot(pi/(N+1))
    return (1/(N + 1))*(a1 + a2 * a3)
end


mu = vec(readdlm("mu.csv", ','))

order = size(mu)[1]
Length_of_output = 2*order
#display(mu)
jack(n) = Jackson(order, n)

display(mu)
#plot dos
x = collect(range(-0.999,stop=0.999,length=Length_of_output))
y = ((pi*sqrt.(-x.^2 .+ 1)).\1) .* (mu[1]*jack(0) .+ sum(2*mu[i]*T_nv(x, i-1)*jack(i) for i in (2:order)))
display(plot(x, y))
println("done")


writedlm( "data.csv",  y, ',')


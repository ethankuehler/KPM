using DelimitedFiles
using Interpolations
using Plots
using QuadGK
using Printf
using LaTeXStrings

function mag(b, mu, m, g)
    t1(x) = b/2 * (sqrt(x^2 + m^2) - mu) 
    t2(x) = b/2 * (sqrt(x^2 + m^2) + mu) 
    f(x) = (1/2)*(g(x)/sqrt(x^2 + m^2)) * (tanh(t1(x)) + tanh(t2(x)))
    return quadgk(f, -1/2, 1/2, rtol=10^-4)
end 


g_data = vec(readdlm("data.csv", ',', Float64))
x = vec(collect(range(-0.999,stop=0.999,length=size(g_data)[1])))
g = linear_interpolation(x, g_data)

M = 0.00001
MU = [0 0.1 0.2 0.3 0.4]

N = 500
B = vec(collect(range(start=0.00001, stop=1, length=N)))

#=

plot()
for mu in MU
    I = zeros(N)
    for (i, B) in enumerate(B)
        I[i] = mag(1/B, mu, M, g)[1]
    end
    I = 1 ./ I
    plot!(B, I, label="\\mu = $mu")
end

xlabel!("T-temp")
ylabel!("I - minimum interaction strength")
title!("minimum Inteaction strength for ANTI-FERRO SDW")
=#


N = 1000

MU = vec(collect(range(start=0, stop=0.4, length=N)))
I2 = zeros(N)
for (i, mu) in enumerate(MU)
    I2[i] = 1/mag(1/0.0001, mu, M, g)[1]
end

plot(MU, I2)
using LinearAlgebra
using Plots
using Printf
using SparseArrays
using BenchmarkTools


function find_mu_fast(H, R, order)
    N = size(H)[1]
    NR = size(R)[1]
    mu = zeros(order) 
    
    for (i, v) in enumerate(R)
        T0 = v
        T1 = H*v
        @printf("doing the %i R ", i)
        @time for i in 1:order
            t = v'*T0
            mu[i] += real(t)
            T2 = 2*H*T1 .- T0
            T0 = T1
            T1 = T2
        end
        @printf("\n")
    end
    return mu/NR
end


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

function HG(Nx, Ny, Nz, tp, t, m)
    sx  = [0 1; 1 0]
    sy  = [0 1im; -1im 0]
    sz  = [1 0; 0 -1]
    s   = [sx, sy, sz]
    tpv = [tp/2, tp/2, 0]
    tv  = [t/2, t/2, t/2]

    T = [tv[n] * sz + tpv[n]* s[n] for n in 1:3]

    N = (2*Nx)*(2*Ny)*(2*Nz)

    H = (Complex.(spzeros(N, N)))

    diag_x = (Complex.(spzeros(2*Nx, 2*Nx)))
    diag_y = (Complex.(spzeros(2*Nx, 2*Nx)))
    diag_z = (Complex.(spzeros(2*Nx, 2*Nx)))

    #main diagonal matrix
    for i in 1:2:2*Nx
        diag_x[i:i+1, i:i+1] = -m*sz
        if i+3 <= 2*Nx
            diag_x[i:i+1, i+2:i+3] .= T[1]
            diag_x[i+2:i+3, i:i+1] .= T[1]
        end
    end

    #off diagonal block for Y jumps
    for i in 1:2:2*Nx
        diag_y[i:i+1, i:i+1] = T[2]
    end
    
    #off diagonal black for X jumps
    for i in 1:2:2*Nx
        diag_z[i:i+1, i:i+1] = T[3]
    end

    display(diag_x)
    display(diag_y)
    display(diag_z)
    #put the main diagonal
    for i in 1:2*Nx:N
        if i%(N/100) == 0 || i%(N/100) == 1
            @printf("H: %i ", (i/N)*100)    
        end
        H[i:i+2*Nx-1, i:i+2*Nx-1] = diag_x

        j = i+2*Nx
        if j < N
            if floor(i/(2*Ny))%3 != 2
                H[i:i+2*Nx-1, j:j+2*Nx-1] = diag_y
                H[j:j+2*Nx-1, i:i+2*Nx-1] = diag_y
            end
        end

        j = i+2*Nx*Ny
        if j < N
            H[i:i+2*Nx-1, j:j+2*Nx-1] = diag_z
            H[j:j+2*Nx-1, i:i+2*Nx-1] = diag_z
        end
    end

    return H
end

#define constants
m     = 0.1
t     = 1
tp    = 1
NR    = 50 #number of random vector samples
order = 500 # kpm order
Nx    = 50
Ny    = 50
Nz    = 50
L     = Nx*Ny*Nz


@time H = HG(Nx,Ny,Nz, tp, t, m)/8
display(spy(real(H)))
#make R's
R = [(2*rand(size(H)[1]).- 1) for _ in 1:NR]
for i in 1:NR
    R[i] = R[i]/norm(R[i])
end

@time mu = find_mu_fast(H, R, order) # get coeffcients 
@printf("mu time above")
display(size(H))
#display(mu)
jack(n) = Jackson(order, n)

#plot dos
x = collect(range(-1,stop=1,length=100))
y = ((pi*sqrt.(-x.^2 .+ 1)).\1) .* (mu[1]*jack(0) .+ sum(2*mu[i]*T_nv(x, i-1)*jack(i) for i in (2:order)))
display(plot(x, y))
println("done")

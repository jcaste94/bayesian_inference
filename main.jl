# -----------------------------------------------------------------------------
# INTRODUCTION TO BAYESIAN INFERENCE
# Econometrics IV, Part 1
# Prof. Frank Schorfheide
#
# Exercise 1, PS1
# -----------------------------------------------------------------------------

# ---------
# Packages
# ---------
using Distributions
using Plots, LaTeXStrings

# --------
# Modules
# --------
include("simul.jl")
using .simul

include("DS.jl")
using .DS

include("fanChart.jl")
using .TSplot

# --------------------
# 1. AR(1) simulation
# --------------------
# Initilize
y_0 = 0
ϕ = 0.95
T = 100

# Simulate
y, u = simul.AR1(ϕ, T+1, y_0)

# Graphs
pSimulationAR = plot(1:T+1,y, linecolor=:black, xlabel="Number of simulations", label=L"y_t = \phi y_{t-1} + u_t", legend=:best)
savefig(pSimulationAR, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pSimulationAR.pdf")

pNoiseAR = plot(1:T+1, u, linecolor=:black, xlabel="Number of simulations", label=L"u_t", legend=:best)
savefig(pNoiseAR, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pNoiseAR.pdf")


# -------------------
# 2. Direct sampling
# -------------------
# Initilize
N_run = 100                 # number of times the algorithm is run
N = [10,100,1000,10000]     # number of draws
τ_diffuse = 1000            # prior precision (diffuse)
τ_concentrate = 0.001       # prior precision (concentrated)
flag = "concentrate"        # "concentrate" or "diffuse"

# Analytical solution - posterior
Y = y[2:T+1]
X = y[1:T]

if flag == "concentrate"
    τ = τ_concentrate
elseif flag == "diffuse"
    τ = τ_diffuse
end

posteriorMean = inv(X'*X + τ^(-2)) * (X'*Y)
posteriorVariance = inv(X'*X + τ^(-2))


# Algorithm
vMeanMC = Float64[]
vSqMeanMC = Float64[]
vVarianceMC = Float64[]
vSqVarianceMC = Float64[]

for iDraws in 1:length(N)

    # 1. Pre-allocation
    vConditionalMean = zeros(N_run, 1)
    vConditionalSqMean = zeros(N_run, 1)


    # 2. Monte Carlo
    for i in 1:N_run

        vConditionalMean[i], vConditionalSqMean[i] = DS.posterior_approx(posteriorMean, posteriorVariance, N[iDraws])

    end

    # 2.1. Mean
    mcMean = mean(vConditionalMean)
    mcSqMean = mean(vConditionalSqMean)

    # 2.2. Variance
    mcVar = var(vConditionalMean)
    mcSqVar = var(vConditionalSqMean)


    # 3. Results
    global vMeanMC = push!(vMeanMC, mcMean)
    global vSqMeanMC = push!(vMeanMC, mcSqMean)
    global vVarianceMC = push!(vVarianceMC, mcVar)
    global vSqVarianceMC = push!(vSqVarianceMC, mcSqVar)

end

plot(N, vVarianceMC, marker=:o, markercolor=:white, linecolor=:black, label="", xlabel = "number of draws", ylabel = "sampling variance")

# ---------------
# 3. Forecasting
# ---------------


#=
# Example FanChart
nMC = 1000
yMC = zeros(T,nMC)
for iMC in 1:nMC
    yMC[:,iMC], u = simul.AR1(ϕ,T,y_0)
end

p1 = TSplot.fanChart(1:100, yMC)
plot(p1)
=#

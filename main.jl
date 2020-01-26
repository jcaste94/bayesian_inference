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
using Plots, LaTeXStrings

# --------
# Modules
# --------
include("simul.jl")
using .simul

# --------------------
# 1. AR(1) simulation
# --------------------
# Initilize
y_0 = 0
ϕ = 0.95
T = 100

# Simulate
Y, U = simul.AR1(ϕ, T, y_0)

# Graphs
pSimulationAR = plot(1:T,Y, linecolor=:black, xlabel="Number of simulations", label=L"y_t = \phi y_{t-1} + u_t", legend=:best)
savefig(pSimulationAR, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pSimulationAR.pdf")

pNoiseAR = plot(1:T, U, linecolor=:black, xlabel="Number of simulations", label=L"u_t", legend=:best)
savefig(pNoiseAR, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV/PS/PS1/LaTeX/pNoiseAR.pdf")

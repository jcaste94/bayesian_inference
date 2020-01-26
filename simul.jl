# -----------------------------------------------------------------------------
# INTRODUCTION TO BAYESIAN INFERENCE
# Econometrics IV, Part 1
# Prof. Frank Schorfheide
#
# Simulation of AR(1)
# -----------------------------------------------------------------------------

module simul

export AR1

# --------
# Packages
# --------
using Distributions, Random

# ----------
# Functions
# ----------
function AR1(ϕ::Real, T::Integer, y_0::Real)

    # pre-allocation
    y = zeros(T,1)
    y[1] = y_0

    # noise
    dist = Normal(0,1)
    u = rand(dist,T)

    # simulation
    for t in 1:T-1
        y[t+1]= ϕ*y[t] + u[t]
    end

    return y, u

end

end

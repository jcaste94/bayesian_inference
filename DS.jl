# -----------------------------------------------------------------------------
# INTRODUCTION TO BAYESIAN INFERENCE
# Econometrics IV, Part 1
# Prof. Frank Schorfheide
#
# Direct sampling algorithm
# -----------------------------------------------------------------------------

module DS

export posterior_approx

# ---------
# Packages
# ---------
using Distributions

# ----------
# Algorithm
# ----------
function posterior_approx(posteriorMean::Real, posteriorVariance::Real, N::Integer)

    # 1. draws from posterior distribution
    d = Normal(posteriorMean, posteriorVariance)
    θ = rand(d, N)

    # 2. MC approximation of E[θ|Y] and E[θ^2|Y]
    conditionalMean = mean(θ)
    conditionalSqMean = mean(θ.^2)

    return conditionalMean, conditionalSqMean
end

end

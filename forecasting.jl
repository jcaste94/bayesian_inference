# -----------------------------------------------------------------------------
# INTRODUCTION TO BAYESIAN INFERENCE
# Econometrics IV, Part 1
# Prof. Frank Schorfheide
#
# Fan chart plot
# -----------------------------------------------------------------------------

module forecast

export h_step, fanChart

# -------
# Packages
# --------
using Distributions
using Statistics
using Plots

# ---------
# Functions
# ---------
function h_step(y_hist::Array, h::Real, posteriorMean::Real, posteriorVariance::Real)

    # 1. Parameters: draws from posterior distribution
    dp = Normal(posteriorMean, sqrt(posteriorVariance))
    θ = rand(dp,1)[1]

    # 2. Innovations: draws from its distribution
    d = Normal(0,1)
    u = rand(d, h)

    # 3. Forecast
    y_f = zeros(h,1)
    y_f[1] = y_hist[end]
    for i in 1:h-1
        y_f[i+1] = θ * y_f[i] + u[i]
    end

    #y = [y_hist; y_f]

    return y_f

end

function fanChart(paths::Array, T::Integer, h::Integer; centerType = "median", percentiles = 0.05:0.05:0.95)

    # Housekeeping
    TT = T + h
    nMC = size(paths,2)
    @assert size(paths,1) == TT

    # Set the centering type
    if centerType == "median"
        y_center = median(paths,dims=2)
    elseif centerType == "mean"
        y_center = mean(paths,dims=2)
    else
        error(""" centerType can only take the values "median" or "mean""")
    end

    # Percentiles
    p = collect(0.05:0.05:0.95)
    y_p = zeros(TT,length(p))
    for t in 1:TT
        y_p[t,:] = quantile(paths[t,:],p)
    end

    # Plots
    p1 = plot(1:TT, y_p, color=:red, alpha=0.5, label="")
    plot!(1:TT, y_center, color=:black, label="", xlabel = "Time")
    vline!(p1, [T+1],linestyle=:dash,color=:black, label="")
    vspan!(p1, [T+1, T+h+1], color=:grey, alpha=0.2, label="")

    return p1
end

end

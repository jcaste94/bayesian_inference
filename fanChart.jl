# -----------------------------------------------------------------------------
# INTRODUCTION TO BAYESIAN INFERENCE
# Econometrics IV, Part 1
# Prof. Frank Schorfheide
#
# Fan chart plot
# -----------------------------------------------------------------------------

module TSplot

export fanChart

# -------
# Packages
# --------
using Statistics
using Plots

# ---------
# Function
# ---------
function fanChart(x, paths::Array; centerType = "median", percentiles = 0.05:0.05:0.95)

    # Housekeeping
    T, nMC = size(paths)
    @assert length(x) == T

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
    y_p = zeros(T,length(p))
    for t in 1:T
        y_p[t,:] = quantile(paths[t,:],p)
    end

    # Plots
    p1 = plot(x, y_center, color=:black, lw=2,label="")
    plot!(x, y_p, color=:black, alpha=0.25, label="")

    return p1
end

end

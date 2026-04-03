using Plots

# Data extraction from the text
labels = ["Original", "Transformed"]
mean_suv = [3.4913, 3.4660]
volume_cm3 = [52.55, 51.86]

# Creating grouped bar plots
# We use two separate plot panes (subplots) to handle the different y-axis scales
p1 = bar(["Original", "Transformed"], mean_suv, 
         ylabel="Mean SUV", 
         title="Mean SUV Comparison",
         legend=false, 
         color=:steelblue,
         yguidefont=font(12),
         xtickfont=font(10))

# Add data labels
for i in 1:length(mean_suv)
    annotate!(p1, [(i, mean_suv[i] / 2, text(string(mean_suv[i]), 10, :white, :center))])
end

p2 = bar(["Original", "Transformed"], volume_cm3, 
         ylabel="Volume (cm³)", 
         title="Tumor Volume Comparison",
         legend=false, 
         color=:darkorange,
         yguidefont=font(12),
         xtickfont=font(10))

for i in 1:length(volume_cm3)
    annotate!(p2, [(i, volume_cm3[i] / 2, text(string(volume_cm3[i]), 10, :white, :center))])
end

# Combine plots layout
final_plot = plot(p1, p2, layout=(1,2), size=(800, 400), margin=5Plots.mm)

# Save the plot
savefig(final_plot, "elsarticle/suv_volume_comparison.pdf")
savefig(final_plot, "elsarticle/suv_volume_comparison.png")

println("Plots generated and saved as elsarticle/suv_volume_comparison.pdf and .png")

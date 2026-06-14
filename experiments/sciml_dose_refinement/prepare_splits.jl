using Random

# The 56 cases provided by the user
raw_cases = """FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat45(16.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat46(14.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat54(08.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat51(30.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat54(06.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat61(30.10.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat47(07.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat52(13.06.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat56(20.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat54(25.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat48(08.02.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat46(25.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_5__Pat47(07.11.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat51(17.10.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat49(18.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_6__Pat49(20.02.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat60(06.03.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_6__Pat47(06.02.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat60(29.08.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat44(07.12.2023)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat61(20.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_5__Pat49(07.11.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat45(22.02.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat47(18.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat47(25.01.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_7__Pat47(10.04.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat52(24.07.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat52(30.04.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat55(19.08.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat50(07.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat46(11.12.2023)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat49(07.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_4__Pat52(13.10.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat51(04.07.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat53(25.04.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_5__Pat52(24.11.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat58(16.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat55(06.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat54(25.07.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat52(01.09.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_7__Pat49(08.05.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat53(08.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_4__Pat51(06.02.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat46(01.02.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat56(02.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat58(27.06.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat60(23.01.2025)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_4__Pat47(22.08.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat49(26.01.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat50(26.01.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat55(17.10.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_1__Pat48(21.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_0__Pat45(18.01.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_2__Pat45(28.03.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat49(30.05.2024)
FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_3__Pat47(06.06.2024)"""

# Parsing and mapping
data_dir = "data/dosimetry_data/"
valid_cases = []
for line in split(raw_cases, '\n')
    m = match(r"FDM_DPI-2024-7-KRN_Lu177_PSMA__dosemap_(\d)__Pat(\d+)", line)
    if m !== nothing
        x, y = m.captures
        # Try both Tc and Iodine prefixes
        for prefix in ["SPECT_Tc", "SPECT_Iodine"]
            dir_name = "FDM_DPI-2024-7-KRN_Lu177_PSMA__$(prefix)_$(x)__Pat$(y)"
            if isdir(joinpath(data_dir, dir_name))
                push!(valid_cases, dir_name)
                break
            end
        end
    end
end

println("Total valid cases found in processed data: ", length(valid_cases))

# Fixed seed shuffle
Random.seed!(42)
shuffle!(valid_cases)

# 15% validation
val_count = Int(floor(0.15 * length(valid_cases)))
val_cases = valid_cases[1:val_count]
train_cases = valid_cases[val_count+1:end]

println("Validation cases ($(val_count)): ", val_cases)
println("Training cases ($(length(train_cases))): ", length(train_cases))

# Save splits to file for future use
open("experiments/sciml_dose_refinement/splits.txt", "w") do f
    println(f, "VALIDATION:")
    for c in val_cases; println(f, c); end
    println(f, "\nTRAINING:")
    for c in train_cases; println(f, c); end
end
println("Splits updated in experiments/sciml_dose_refinement/splits.txt")

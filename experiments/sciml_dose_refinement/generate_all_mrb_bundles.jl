using Pkg
Pkg.activate("/home/user/MedImages.jl")

using NIfTI, Statistics

const DOSE_CONV = 8.478f-8 

function hu_to_den(hu)
    return hu <= 0 ? max(0.01f0, 1.0f0 + 0.001f0 * Float32(hu)) : 1.0f0 + 0.0007f0 * Float32(hu)
end

function generate_bundles()
    val_out_root = "val_outputs_full"
    dataset_dir = "data/dosimetry_data"
    
    # Get all patient folders in validation output
    pats = filter(d -> isdir(joinpath(val_out_root, d)) && startswith(d, "FDM"), readdir(val_out_root))
    
    for pat in pats
        println(">>> Generating MRB for: $pat")
        pat_out_dir = joinpath(val_out_root, pat)
        pat_data_dir = joinpath(dataset_dir, pat)
        
        # 1. Generate Analytical Baseline
        ct_f = niread(joinpath(pat_data_dir, "ct.nii.gz"))
        sp_f = niread(joinpath(pat_data_dir, "spect.nii.gz"))
        
        extract(f) = ndims(f) == 3 ? f : f[:,:,:,1]
        ct_i = extract(ct_f); sp_i = extract(sp_f)
        vol_p = Float32(prod(ct_f.header.pixdim[2:4]))
        
        den = hu_to_den.(ct_i)
        baseline = (Float32.(sp_i) .* DOSE_CONV) ./ (vol_p .* den .+ 1f-4)
        
        ni_b = NIVolume(baseline); ni_b.header.pixdim = ct_f.header.pixdim
        niwrite(joinpath(pat_out_dir, "baseline_analytical.nii.gz"), ni_b)
        
        # 2. Copy Monte Carlo Gold Standard
        cp(joinpath(pat_data_dir, "dosemap_mc.nii.gz"), joinpath(pat_out_dir, "monte_carlo_gold.nii.gz"), force=true)
        
        # 3. Create MRML Scene
        mrml_content = """<MRML  >
 <Volume id="vtkMRMLScalarVolumeNode1" name="Monte_Carlo_Gold" storageNodeRef="vtkMRMLVolumeArchetypeStorageNode1" isForeground="0" isBackground="1"></Volume>
 <VolumeArchetypeStorage id="vtkMRMLVolumeArchetypeStorageNode1" fileName="monte_carlo_gold.nii.gz"></VolumeArchetypeStorage>
 <Volume id="vtkMRMLScalarVolumeNode2" name="UDE_Improved" storageNodeRef="vtkMRMLVolumeArchetypeStorageNode2" isForeground="1" isBackground="0"></Volume>
 <VolumeArchetypeStorage id="vtkMRMLVolumeArchetypeStorageNode2" fileName="ude_improved_full.nii.gz"></VolumeArchetypeStorage>
 <Volume id="vtkMRMLScalarVolumeNode3" name="Baseline_Analytical" storageNodeRef="vtkMRMLVolumeArchetypeStorageNode3" isForeground="0" isBackground="0"></Volume>
 <VolumeArchetypeStorage id="vtkMRMLVolumeArchetypeStorageNode3" fileName="baseline_analytical.nii.gz"></VolumeArchetypeStorage>
</MRML>
"""
        open(joinpath(pat_out_dir, "scene.mrml"), "w") do f
            write(f, mrml_content)
        end
        
        # 4. Package MRB using Python zip (since zip utility missing)
        run(`/usr/bin/python3 -c "
import zipfile, os
pat_dir = '$pat_out_dir'
files = ['scene.mrml', 'monte_carlo_gold.nii.gz', 'ude_improved_full.nii.gz', 'baseline_analytical.nii.gz']
with zipfile.ZipFile(os.path.join(pat_dir, 'dosimetry_comparison.mrb'), 'w') as mrb:
    for f in files:
        mrb.write(os.path.join(pat_dir, f), f)
"`)
        println("  Done: $pat/dosimetry_comparison.mrb")
    end
end

generate_bundles()

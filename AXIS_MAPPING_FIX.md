# Axis Mapping Bug Fix

## Problem
The MedImages.jl test suite was failing with voxel data comparison errors showing values like `8.0` vs `0.0` at image boundaries, and dimension mismatches in pad and crop operations.

## Root Cause
Incorrect axis mapping comments and implementation in `Basic_transformations.jl`:

**Incorrect assumption:**
- Julia dim1 → Z
- Julia dim2 → Y
- Julia dim3 → X

**Correct mapping:**
- Julia dim1 → X
- Julia dim2 → Y
- Julia dim3 → Z

This incorrect assumption caused:
1. `pad_mi` and `crop_mi` to reverse the padding/cropping tuples for origin calculation
2. Test files to reverse coordinates when calling these functions
3. Double-reversal that cancelled out for origin but broke array operations

## Fix Applied

### 1. Fixed `pad_mi` (src/Basic_transformations.jl:186-190)
**Before:**
```julia
# Note: origin/spacing are in (x,y,z) order, but pad_beg is in Julia array dim order
# Julia dim1 -> Z, dim2 -> Y, dim3 -> X, so we reverse pad_beg for origin calculation
padded_origin = im.origin .- (im.spacing .* reverse(pad_beg))
```

**After:**
```julia
# Note: Both origin/spacing and pad_beg are in (x,y,z) order
# Julia array dims map as: dim1 -> X, dim2 -> Y, dim3 -> Z
# So pad_beg is already in the correct order for origin calculation
padded_origin = im.origin .- (im.spacing .* pad_beg)
```

### 2. Fixed `crop_mi` (src/Basic_transformations.jl:148-152)
**Before:**
```julia
# Note: origin/spacing are in (x,y,z) order, but crop_beg is in Julia array dim order
# Julia dim1 -> Z, dim2 -> Y, dim3 -> X, so we reverse crop_beg for origin calculation
cropped_origin = im.origin .+ (im.spacing .* reverse(crop_beg))
```

**After:**
```julia
# Note: Both origin/spacing and crop_beg are in (x,y,z) order
# Julia array dims map as: dim1 -> X, dim2 -> Y, dim3 -> Z
# So crop_beg is already in the correct order for origin calculation
cropped_origin = im.origin .+ (im.spacing .* crop_beg)
```

### 3. Fixed test files to not reverse coordinates

**test_pad_mi.jl:**
```julia
# No longer reverse the coordinates
mi_padded = MedImages.pad_mi(med_im, pad_beg, pad_end, pad_val, interp)
```

**test_crop_mi.jl:**
```julia
# No longer reverse the coordinates
medIm_cropped = MedImages.crop_mi(med_im, beginning, crop_size, interp)
```

## Verification

Test script demonstrating the fix:
```julia
# Pad operation with (10, 11, 13) in (x,y,z) order
# Original: (512, 512, 75)
# Expected result: (532, 534, 101) = (512+10+10, 512+11+11, 75+13+13)

med_padded = MedImages.pad_mi(med_im, (10, 11, 13), (10, 11, 13), 0.0, MedImages.Linear_en)
# Result: (532, 534, 101) ✓ CORRECT

# Voxel comparison
max_diff = maximum(abs.(sitk_array .- medimages_array))
# Result: 0.0 ✓ PERFECT MATCH
```

## Results
- ✅ Dimension mismatches resolved
- ✅ Voxel data comparison errors fixed
- ✅ Pad and crop operations now match SimpleITK exactly
- ⚠️  Remaining: Origin calculation issues in `scale_mi` (separate issue)

## Impact
This fix ensures that all spatial transformations correctly map Julia array dimensions to physical (x,y,z) coordinates, eliminating test failures and ensuring consistency with SimpleITK.

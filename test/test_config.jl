# test/test_config.jl
# Test configuration constants for MedImages.jl test suite

module TestConfig

using MedImages

export TEST_NIFTI_FILE, TEST_SYNTHETIC_FILE, TEST_DICOM_DIR
export TEST_HDF5_FILE, TEST_FUNC_DATA_FILE, DEBUG_DIR
export ROTATION_AXES, ROTATION_ANGLES
export INTERPOLATORS, INTERPOLATOR_NAMES
export CROP_TEST_CASES, PAD_TEST_CASES
export TRANSLATION_VALUES, TRANSLATION_AXES
export SCALE_ZOOM_VALUES
export SPACING_TEST_VALUES
export AVAILABLE_ORIENTATIONS

# Import TEST_DATA_DIR from TestHelpers
const TEST_DATA_DIR = joinpath(@__DIR__, "..", "test_data")

# Primary test files
const TEST_NIFTI_FILE = joinpath(TEST_DATA_DIR, "volume-0.nii.gz")
const TEST_SYNTHETIC_FILE = joinpath(TEST_DATA_DIR, "synthethic_small.nii.gz")
const TEST_DICOM_DIR = joinpath(TEST_DATA_DIR, "ScalarVolume_0")
const TEST_HDF5_FILE = joinpath(TEST_DATA_DIR, "debug.h5")
const TEST_FUNC_DATA_FILE = joinpath(TEST_DATA_DIR, "filtered_func_data.nii.gz")
const DEBUG_DIR = joinpath(TEST_DATA_DIR, "debug")

# Interpolation methods to test
const INTERPOLATORS = [
    MedImages.Nearest_neighbour_en,
    MedImages.Linear_en,
    MedImages.B_spline_en
]

const INTERPOLATOR_NAMES = Dict(
    MedImages.Nearest_neighbour_en => "Nearest",
    MedImages.Linear_en => "Linear",
    MedImages.B_spline_en => "BSpline"
)

# Rotation test parameters
const ROTATION_AXES = [1, 2, 3]
const ROTATION_ANGLES = [30.0, 60.0, 90.0]

# Crop test cases: [(crop_beginning, crop_size), ...]
const CROP_TEST_CASES = [
    ((0, 0, 0), (151, 156, 50)),
    ((15, 17, 7), (150, 150, 53))
]

# Pad test cases: [(pad_begin, pad_end, pad_value), ...]
const PAD_TEST_CASES = [
    ((10, 11, 13), (10, 11, 13), 0.0),
    ((10, 11, 13), (15, 17, 19), 0.0),
    ((10, 11, 13), (10, 11, 13), 111.5),
    ((10, 11, 13), (15, 17, 19), 111.5)
]

# Translation test parameters
const TRANSLATION_VALUES = [1, 10]
const TRANSLATION_AXES = [1, 2, 3]

# Scale zoom values
const SCALE_ZOOM_VALUES = [0.6, 0.9, 1.3]

# Spacing test values for resampling
const SPACING_TEST_VALUES = [
    (5.0, 0.9, 0.7),
    (1.0, 2.0, 1.1),
    (0.5, 0.5, 0.5),
    (2.0, 2.0, 2.0)
]

# Orientation test values
const AVAILABLE_ORIENTATIONS = [
    MedImages.ORIENTATION_RAS,
    MedImages.ORIENTATION_LAS,
    MedImages.ORIENTATION_RPI,
    MedImages.ORIENTATION_LPI,
    MedImages.ORIENTATION_RAI,
    MedImages.ORIENTATION_LAI,
    MedImages.ORIENTATION_RPS,
    MedImages.ORIENTATION_LPS
]

end # module TestConfig

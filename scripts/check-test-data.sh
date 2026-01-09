#!/bin/bash
# Check test data availability for MedImages.jl
# Usage: ./scripts/check-test-data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DATA_DIR="$PROJECT_DIR/test_data"

echo "MedImages.jl Test Data Check"
echo "============================"
echo ""
echo "Test data directory: $TEST_DATA_DIR"
echo ""

# Track status
all_ok=true

echo "Required test files:"
echo ""

# Check primary NIfTI file
if [ -f "$TEST_DATA_DIR/volume-0.nii.gz" ]; then
    size=$(du -h "$TEST_DATA_DIR/volume-0.nii.gz" | cut -f1)
    echo "[OK]      volume-0.nii.gz ($size)"
else
    echo "[MISSING] volume-0.nii.gz - Primary test file"
    all_ok=false
fi

# Check synthetic test file
if [ -f "$TEST_DATA_DIR/synthethic_small.nii.gz" ]; then
    size=$(du -h "$TEST_DATA_DIR/synthethic_small.nii.gz" | cut -f1)
    echo "[OK]      synthethic_small.nii.gz ($size)"
else
    echo "[MISSING] synthethic_small.nii.gz - Synthetic test file"
    all_ok=false
fi

# Check DICOM directory
if [ -d "$TEST_DATA_DIR/ScalarVolume_0" ]; then
    count=$(find "$TEST_DATA_DIR/ScalarVolume_0" -type f | wc -l)
    echo "[OK]      ScalarVolume_0/ ($count files)"
else
    echo "[MISSING] ScalarVolume_0/ - DICOM test directory"
    all_ok=false
fi

echo ""
echo "Optional files:"
echo ""

# Check functional data
if [ -f "$TEST_DATA_DIR/filtered_func_data.nii.gz" ]; then
    size=$(du -h "$TEST_DATA_DIR/filtered_func_data.nii.gz" | cut -f1)
    echo "[OK]      filtered_func_data.nii.gz ($size)"
else
    echo "[SKIP]    filtered_func_data.nii.gz - Optional functional data"
fi

echo ""
echo "============================"

if [ "$all_ok" = true ]; then
    echo "All required test files present."
    echo "Run: make test"
    exit 0
else
    echo ""
    echo "Some test files are missing."
    echo "Tests that require missing files will be skipped."
    echo ""
    echo "For benchmarks without real data, use:"
    echo "  make benchmark  (uses synthetic data)"
    exit 1
fi

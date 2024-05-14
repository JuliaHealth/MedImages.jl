#sudo apt-get install aptitude

using Cxx

# Initialize the C++ ITK environment
cxx"""
#include <itkImage.h>
#include <itkImageFileReader.h>
"""

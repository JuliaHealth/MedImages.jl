# MedImages.jl Docker Image
# Julia 1.11.6 with CUDA support for GPU benchmarks and SimpleITK for comparison
#
# Build: docker build -t medimages:latest .
# Run:   docker run --gpus all -it medimages:latest

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Build arguments
ARG JULIA_VERSION=1.11.6
ARG JULIA_MINOR=1.11

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV JULIA_PATH=/usr/local/julia
ENV PATH="${JULIA_PATH}/bin:${PATH}"
ENV JULIA_DEPOT_PATH=/root/.julia
ENV JULIA_NUM_THREADS=auto

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic utilities
    curl \
    ca-certificates \
    wget \
    git \
    nano \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # HDF5 support (required by HDF5.jl)
    libhdf5-serial-dev \
    hdf5-tools \
    # Python 3.10
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    # ITK/SimpleITK build dependencies
    ninja-build \
    # Graphics libraries (for potential visualization)
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Julia 1.11.6
RUN curl -fL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MINOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    | tar -C /usr/local -xz --strip-components=1 -f - \
    && julia --version

# Install Python packages for benchmarks
RUN python3 -m pip install --no-cache-dir \
    numpy \
    SimpleITK \
    && python3 -c "import SimpleITK; print('SimpleITK version:', SimpleITK.Version())"

# Create workspace directory
WORKDIR /workspace/MedImages.jl

# Copy package manifest files first for dependency caching
COPY Project.toml ./
COPY benchmark/Project.toml ./benchmark/

# Install dependencies only (not precompile - source files not present yet)
RUN julia --project=. -e ' \
    using Pkg; \
    Pkg.instantiate()'

# Configure PyCall to use system Python
RUN julia --project=. -e ' \
    ENV["PYTHON"] = "/usr/bin/python3"; \
    using Pkg; \
    Pkg.build("PyCall")'

# Copy full source code
COPY . /workspace/MedImages.jl/

# Now precompile with source files present
RUN julia --project=. -e ' \
    using Pkg; \
    Pkg.precompile()'

# Install benchmark dependencies
WORKDIR /workspace/MedImages.jl/benchmark
RUN julia --project=. -e ' \
    using Pkg; \
    Pkg.develop(path=".."); \
    Pkg.instantiate(); \
    Pkg.precompile()'

# Attempt CUDA precompilation (may fail if no GPU at build time)
WORKDIR /workspace/MedImages.jl
RUN julia --project=. -e ' \
    try \
        using CUDA; \
        if CUDA.functional() \
            CUDA.precompile_runtime(); \
            println("CUDA precompilation complete"); \
        else \
            println("CUDA not functional at build time - will initialize at runtime"); \
        end \
    catch e \
        println("CUDA setup deferred to runtime: ", e); \
    end'

# Create entrypoint script for thread configuration
RUN echo '#!/bin/bash\n\
num_cores=$(nproc)\n\
if [ "$num_cores" -gt 1 ]; then\n\
    export JULIA_NUM_THREADS=$((num_cores - 1)),1\n\
else\n\
    export JULIA_NUM_THREADS=1\n\
fi\n\
echo "Julia threads: $JULIA_NUM_THREADS"\n\
exec "$@"' > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# Set working directory
WORKDIR /workspace/MedImages.jl

# Default entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["julia", "--project=."]

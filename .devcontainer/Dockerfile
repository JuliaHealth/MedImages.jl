ARG JULIA_RELEASE=1.10
ARG JULIA_VERSION=1.10.10
# Using Ubuntu 22.04 to ensure broad package support.
# This contains Python 3.10 which is stable and compatible with most scientific packages.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Redeclare ARGs to make them available in this stage
ARG JULIA_RELEASE
ARG JULIA_VERSION
ARG CUDA_VERSION
ARG JULIA_CPU_TARGET=native
ARG PACKAGE_SPEC=CUDA

# Install dependencies for downloading and extracting Julia
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tar \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Julia
RUN curl -fL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_RELEASE}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | tar -C /usr/local --strip-components=1 -xz

ENV JULIA_CPU_TARGET=${JULIA_CPU_TARGET}

LABEL org.opencontainers.image.authors="Tim Besard <tim.besard@gmail.com>" \
    org.opencontainers.image.description="A CUDA.jl container with CUDA ${CUDA_VERSION} and Julia ${JULIA_VERSION}" \
    org.opencontainers.image.title="CUDA.jl" \
    org.opencontainers.image.url="https://juliagpu.org/cuda/" \
    org.opencontainers.image.source="https://github.com/JuliaGPU/CUDA.jl" \
    org.opencontainers.image.licenses="MIT"


# system-wide packages

ENV JULIA_DEPOT_PATH=/usr/local/share/julia:

# pre-install the CUDA toolkit from an artifact. we do this separately from CUDA.jl so that
# this layer can be cached independently. it also avoids double precompilation of CUDA.jl in
# order to call `CUDA.set_runtime_version!`.
RUN julia -e '#= make bundled depot non-writable (JuliaLang/Pkg.jl#4120) =# \
    for depot in DEPOT_PATH \
    compiled = joinpath(depot, "compiled"); \
    if isdir(compiled) \
    run(`find $compiled -type f -writable -exec chmod -w \{\} \;`); \
    end \
    end; \
    #= configure the preference =# \
    env = "/usr/local/share/julia/environments/v$(VERSION.major).$(VERSION.minor)"; \
    mkpath(env); \
    write("$env/LocalPreferences.toml", \
    "[CUDA_Runtime_jll]\nversion = \"'${CUDA_VERSION}'\""); \
    \
    #= install the JLL =# \
    using Pkg; \
    Pkg.add("CUDA_Runtime_jll"); \
    #= revert bundled depot changes =# \
    for depot in DEPOT_PATH \
    compiled = joinpath(depot, "compiled"); \
    if isdir(compiled) \
    run(`find $compiled -type f -writable -exec chmod +w \{\} \;`); \
    end \
    end' && \
    #= demote the JLL to an [extras] dep =# \
    find /usr/local/share/julia/environments -name Project.toml -exec sed -i 's/deps/extras/' {} + && \
    #= remove nondeterminisms =# \
    cd /usr/local/share/julia && \
    rm -rf compiled registries scratchspaces logs && \
    find -exec touch -h -d "@0" {} + && \
    touch -h -d "@0" /usr/local/share

# install CUDA.jl itself
RUN julia -e 'using Pkg; pkg"add '${PACKAGE_SPEC}'"; \
    using CUDA; CUDA.precompile_runtime()' && \
    #= remove useless stuff =# \
    cd /usr/local/share/julia && \
    rm -rf registries scratchspaces logs

# pre-install additional Julia dependencies to speed up startup
RUN julia -e 'using Pkg; Pkg.add(["NIfTI", "LuxCUDA", "Optimisers", "Plots", "Revise", "Zygote", "Lux", "GLMakie"])'


# user environment

# we hard-code the primary depot regardless of the actual user, i.e., we do not let it
# default to `$HOME/.julia`. this is for compatibility with `docker run --user`, in which
# case there might not be a (writable) home directory.

RUN mkdir -m 0777 /depot

# we add the user environment from a start-up script
# so that the user can mount `/depot` for persistency
COPY <<EOF /usr/local/share/julia/config/startup.jl
if !isdir("/depot/environments/v$(VERSION.major).$(VERSION.minor)")
    if isinteractive() && Base.JLOptions().quiet == 0
        println("""Welcome to this CUDA.jl container!

                   Since this is the first time you're running this container,
                   we'll set up a user depot for you at `/depot`. For persistency,
                   you can mount a volume at this location.

                   The CUDA.jl package is pre-installed, and ready to be imported.
                   Remember that you need to invoke Docker with e.g. `--gpus=all`
                   to access the GPU.""")
    end
    mkpath("/depot/environments")
    cp("/usr/local/share/julia/environments/v$(VERSION.major).$(VERSION.minor)",
       "/depot/environments/v$(VERSION.major).$(VERSION.minor)")
end
pushfirst!(DEPOT_PATH, "/depot")
EOF

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    nano \
    git \
    wget \
    unzip \
    zip \
    p7zip-full \
    jq \
    ninja-build \
    ffmpeg \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# SECTION 2: JULIA RUNTIME INSTALLATION
# =============================================================================
# Skip manual Julia install: base image `julia:1.10` already includes Julia

# =============================================================================
# SECTION 3: PYTHON INSTALLATION
# =============================================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # Build deps for CGAL and other scientific packages
    cmake \
    libgmp-dev \
    libmpfr-dev \
    libcgal-dev \
    swig \
    # Headless execution & Plotting (Xvfb)
    xvfb \
    xauth \
    libfontconfig1 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libxi6 \
    # Runtime deps for PyVista/VTK (headless rendering)
    libgl1-mesa-glx \
    libxrender1 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup a virtual environment to avoid PEP 668 "externally-managed-environment" errors
# This replaces the need for --break-system-packages or messing with system pip
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip packages into the virtual environment
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    monai \
    trimesh[all] \
    tqdm \
    lark \
    outlines \
    lightning \
    SimpleITK \
    morphomatics \
    pandas \
    numpy \
    matplotlib \
    pyvista \
    nibabel \
    tensorboardX \
    cgal \
    tensorboard

WORKDIR "/workspace"
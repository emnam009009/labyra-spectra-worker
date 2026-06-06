# Quantum ESPRESSO 7.4.1 runtime for Labyra Cloud Batch DFT tasks (CPU / OpenMPI).
# Used by compute backend B (cloud-batch) ONLY — backend generate-only never pulls this.
# Built from the OFFICIAL QEF/q-e source (reproducible, exact version, GPL source public).
#
#   docker build -t REGION-docker.pkg.dev/PROJECT/REPO/quantum-espresso:7.4.1 .
#   docker push  REGION-docker.pkg.dev/PROJECT/REPO/quantum-espresso:7.4.1
#
# GPU (preset high-gpu, NVIDIA L4 = Ada, compute capability 8.9) is a SEPARATE heavier
# image (NVHPC + CUDA, cmake -DQE_ENABLE_CUDA=ON ... --with-cuda-cc=89) — deferred.

# ---------- build stage ----------
FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
ARG QE_VERSION=7.4.1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gfortran gcc g++ make cmake \
        libopenmpi-dev openmpi-bin libfftw3-dev libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL "https://github.com/QEF/q-e/archive/refs/tags/qe-${QE_VERSION}.tar.gz" -o qe.tar.gz \
    && tar xzf qe.tar.gz && rm qe.tar.gz
WORKDIR /src/q-e-qe-${QE_VERSION}/build
# Recipe verified against an HPC build guide: mpicc/mpicxx/mpifort + OpenMP, Release.
RUN cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
          -DCMAKE_Fortran_COMPILER=mpifort \
          -DQE_ENABLE_OPENMP=ON -DQE_ENABLE_MPI=ON \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/qe .. \
    && make -j"$(nproc)" \
    && make install

# ---------- runtime stage (smaller) ----------
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
# QE runtime libs + gsutil/gcloud (google-cloud-cli, needs python3) for GCS staging.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
        python3 python3-crcmod \
        openmpi-bin libfftw3-double3 libfftw3-single3 libblas3 liblapack3 libgfortran5 libgomp1 \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
         > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
         | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/qe /opt/qe
ENV PATH=/opt/qe/bin:$PATH
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
WORKDIR /scratch/work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# GPU image for super_resolution and the segment_mandible pipeline it drives.
#
# Build from the REPOSITORY ROOT (the image needs segment_mandible/ too):
#
#   docker build -f super_resolution/Dockerfile -t microct-sr .
#
# BASE_IMAGE must match your GPU and driver -- see "Choosing the base image"
# in super_resolution/README.md. The default runs on RTX 20xx through 50xx
# cards. For a GTX 10xx (Pascal) card pass
#   --build-arg BASE_IMAGE=pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
#
# Only segment_mandible/ and super_resolution/ are sent to the build
# (Dockerfile.dockerignore, next to this file); scans are mounted at run time,
# never baked in.

ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SR_WORK_DIR=/work

WORKDIR /app

# Dependencies before code, so editing the code does not reinstall them.
# torch is NOT installed here: it comes from the base image, built for its CUDA.
COPY super_resolution/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt && rm /tmp/requirements-docker.txt

COPY segment_mandible/ segment_mandible/
COPY super_resolution/ super_resolution/

# Fail the build rather than the first run if anything cannot import --
# including the classical pipeline, which runs headless here (no napari).
RUN python -c "from super_resolution.classical import load_classical_pipeline; \
import super_resolution.cli, super_resolution.sr_segment; load_classical_pipeline(); print('imports ok')"

# Mount points (bind them with -v or compose.yaml). /work holds the temporary
# upscaled volume: scale^3 times the size of the scan being segmented.
RUN mkdir -p /data /results /work /models && chmod 0777 /results /work /models

ENTRYPOINT ["python", "-m", "super_resolution"]
CMD ["--help"]

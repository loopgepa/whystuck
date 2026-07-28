# Sandbox image for scoring BigCodeBench-Hard solutions (v2 code study).
# Fully pinned so a rebuild reproduces bit-for-bit: the base is digest-pinned and every
# Python dependency comes from requirements-lock.txt (a `pip freeze` of this image).
# Network is disabled at RUN time by the caller (bcb_exec: --network none).
FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt
# Determinism + headless: single-thread BLAS, no display, fixed hash seed.
ENV MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 LC_ALL=C.UTF-8 TZ=UTC

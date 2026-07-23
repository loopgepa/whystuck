# Sandbox image for scoring BigCodeBench-Hard solutions (v2 code study).
# Pinned, network-free at run time; the vendoring filter keeps only tasks whose
# reference passes under these exact versions, so scoring is reproducible.
FROM python:3.12-slim
RUN pip install --no-cache-dir \
    numpy pandas matplotlib scikit-learn scipy seaborn \
    beautifulsoup4 requests pillow lxml nltk statsmodels \
    openpyxl python-dateutil pytz flask
# Determinism + headless: single-thread BLAS, no display, fixed hash seed.
ENV MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 LC_ALL=C.UTF-8 TZ=UTC

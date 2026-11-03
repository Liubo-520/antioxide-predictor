FROM python:3.10-slim

WORKDIR /app

# System libs needed by RDKit/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libglib2.0-0 \
    libexpat1 libfontconfig1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Step 1: CPU-only torch (smaller, ~200MB vs 2.4GB for CUDA)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.0+cpu

# Step 2: PyG + everything else
RUN pip install --no-cache-dir \
    torch_geometric==2.6.1 \
    rdkit \
    matplotlib \
    gradio==5.49.1

# Non-root user required by HuggingFace
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

COPY --chown=user . /app

EXPOSE 7860

CMD ["python", "app.py"]

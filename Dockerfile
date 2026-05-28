# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    CONDA_DIR=/opt/conda

# 1. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential wget curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

# 2. Conda env `sldgen`
RUN conda create -n sldgen python=3.9.19 -y && conda clean -afy
SHELL ["conda", "run", "--no-capture-output", "-n", "sldgen", "/bin/bash", "-lc"]
ENV PATH=/opt/conda/envs/sldgen/bin:$PATH

RUN pip install --no-cache-dir "setuptools<78" wheel

# 3. diffvg
RUN git clone https://github.com/BachiLi/diffvg.git /opt/diffvg \
    && cd /opt/diffvg \
    && git submodule update --init --recursive

RUN conda install -y -c pytorch pytorch torchvision
RUN conda install -y numpy scikit-image
RUN conda install -y -c conda-forge cmake=3.28 ffmpeg
RUN conda clean -afy

RUN pip install --no-cache-dir visdom --no-build-isolation
RUN pip install --no-cache-dir svgwrite svgpathtools cssutils numba torch-tools

# Force CUDA build: diffvg gates GPU support on torch.cuda.is_available(), which
# is False during `docker build` (no GPU visible). Patch the check to True.
RUN sed -i 's/torch.cuda.is_available()/True/g' /opt/diffvg/setup.py \
    && cd /opt/diffvg && python setup.py install

# 4. fab3dwire repulsion loss
RUN git clone https://github.com/kenji-tojo/fab3dwire.git /opt/fab3dwire \
    && conda install -y -c conda-forge 'eigen=3.4.*' \
    && conda clean -afy

RUN cd /opt/fab3dwire/wiregrad \
    && pip3 install --no-cache-dir --no-build-isolation \
        git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
RUN cd /opt/fab3dwire/wiregrad \
    && grep -viE '(^|[[:space:]])clip([[:space:]@=<>]|$)' requirements.txt > requirements.noclip.txt \
    && pip3 install --no-cache-dir -r requirements.noclip.txt
RUN cd /opt/fab3dwire/wiregrad && pip3 install --no-cache-dir .

# 5. Concorde TSP solver
# Modern gcc defaults to -fno-common which breaks Concorde's legacy C; force -fcommon.
RUN cd /opt \
    && wget -q https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz \
    && gunzip co031219.tgz \
    && tar xf co031219.tar \
    && rm co031219.tar \
    && cd /opt/concorde \
    && wget -q https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.a \
    && wget -q https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.h \
    && CFLAGS="-fcommon -O2" ./configure --with-qsopt=/opt/concorde --enable-ccdefaults \
    && make
ENV CONCORDE_PATH=/opt/concorde/TSP/concorde

# 6. Project dependencies
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN chmod -R a+rwX /app

# 7. HuggingFace cache location — mount a host volume here so weights persist
# across runs and are downloaded only once. Pass the HF token at runtime via
# `-e HF_TOKEN=...` so SD3.5-medium (gated) can be fetched on first run.
ENV HF_HOME=/hf-cache
RUN mkdir -p /hf-cache && chmod 777 /hf-cache

# Point matplotlib/fontconfig caches at /tmp so they work when the container
# is run with `--user $(id -u):$(id -g)` (no writable $HOME).
ENV MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

# 8. Entrypoint — env already on PATH so plain `python` works without `conda run`.
SHELL ["/bin/bash", "-lc"]
CMD ["python", "sldgen.py", "--help"]

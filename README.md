# Single-Line Drawing Generation via Semantics-Driven Optimization 

[Tanguy Magne](https://tanguymagne.com/), [Alexandre Binninger](https://alexandrebinninger.com), [Ruben Wiersma](https://rubenwiersma.nl), [Olga Sorkine-Hornung](https://igl.ethz.ch) <br />

<a href="https://igl.ethz.ch/projects/sldgen/"><img src="https://img.shields.io/badge/🔗%20Website-Project%20page-blue" alt="website"></a>
    <a href="https://igl.ethz.ch/projects/sldgen/single-line-drawing-generation-computer-graphics-forum-2026-magne-et-al.pdf" alt ="paper"> <img src="https://img.shields.io/badge/📄%20Paper-PDF_(23.3MB)-b31b1b"/></a>
    <a href="https://doi.org/10.1111/cgf.70502" alt="doi">
    <img src="https://img.shields.io/badge/DOI-10.1111%2Fcgf.70502-red?logo=doi&color=fab608" alt="website"></a>

![Header](media/teaser.svg)

This repository contains the code for the CGF paper **"Single-Line Drawing Generation via Semantics-Driven Optimization"**.

The code is partially based on the [ControlSketch](https://github.com/swiftsketch/SwiftSketch) part of SwiftSketch. We thank the authors for sharing their work.

## 🐋 About this branch

This `docker-build` branch holds the tooling used to build the Docker image published on Docker Hub as [`tanguymagne/sldgen`](https://hub.docker.com/r/tanguymagne/sldgen). It contains the `Dockerfile` and `.dockerignore` on top of the project code.

For installation, usage, and how to run the image, refer to the [`main` branch README](https://github.com/tanguymagne/SLDgen/blob/main/README.md) and the [Docker Hub page](https://hub.docker.com/r/tanguymagne/sldgen).

## 🏗️ Building the image

The image bundles all system and Python dependencies (CUDA 12.4, diffvg with GPU support, the wiregrad repulsion loss, the Concorde TSP solver, and the project code). Model weights are **not** baked in — they are downloaded at runtime — so no Hugging Face token is needed at build time.

From the repository root:
```bash
docker build -t sldgen .
```

This takes a while (it compiles diffvg, wiregrad, and Concorde from source). The resulting image works on any host with an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

To verify the image built correctly, check that the core dependencies import and that the GPU is visible:
```bash
docker run --gpus all --rm sldgen \
  python -c "import diffvg, torch, wiregrad; print('CUDA:', torch.cuda.is_available(), ', WIREGRAD:', wiregrad.cuda.is_available())"
```
This should print `CUDA: True , WIREGRAD: True `. You can also confirm the Concorde solver is in place:
```bash
docker run --gpus all --rm sldgen \
  bash -lc 'test -x "$CONCORDE_PATH" && echo "Concorde OK: $CONCORDE_PATH"'
```

## 🪪 Citation

```
@article{Magne:SLDgen:2026,
    title   = {Single Line Drawing Generation via Semantics-Driven Optimization},
    author  = {Magne, Tanguy and Binninger, Alexandre and Wiersma, Ruben and Sorkine-Hornung, Olga},
    journal = {Computer Graphics Forum},
    volume  = {n/a},
    number  = {n/a},
    pages   = {e70502},
    year    = {2026},
    doi     = {https://doi.org/10.1111/cgf.70502},
}
```
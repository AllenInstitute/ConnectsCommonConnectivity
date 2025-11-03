# hash:sha256:c3feb56b545d0024c83bef392f8403ad9d77eaa40de1333e49bcb4692cd92560
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.4.10 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient==8.0.0 \
    deltalake==1.2.1 \
    ipywidgets==8.1.8 \
    linkml==1.9.5 \
    linkml-runtime==1.9.5 \
    matplotlib==3.10.7 \
    pandas==2.3.3 \
    polars==1.34.0 \
    pyarrow==22.0.0 \
    pydantic==2.12.3 \
    pyyaml \
    scikit-learn==1.7.2 \
    seaborn==0.13.2 \
    standard-transform==1.4.1 \
    umap-learn==0.5.9.post2

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall

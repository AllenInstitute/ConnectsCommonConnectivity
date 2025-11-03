# hash:sha256:bef56f91513e59baf20b5ef5712df484107a25a2465882a8ab05d9c47a0afeb8
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.4.10 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient \
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

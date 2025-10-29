# hash:sha256:3210c5b64032785d2e7b6a64d8ccf76b26025f393b4ab8c8c85fca37782991dd
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN conda install -y \
        jupyterlab==4.4.10 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    caveclient==7.11.1 \
    deltalake==1.2.1 \
    linkml==1.9.5 \
    linkml-runtime==1.9.5 \
    pandas==2.3.3 \
    polars==1.34.0 \
    pyarrow==22.0.0 \
    pydantic==2.12.3 \
    pyyaml

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall

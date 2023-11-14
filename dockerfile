ARG BUILD_ENV=no_proxy
ARG PYTHON=3.11.5

# BUILD_ENV={proxy,no_proxy}

# ------------------

FROM python:${PYTHON}-slim as build_proxy

ONBUILD RUN echo "Building with PROXY"

# Installing certificates
ONBUILD COPY ca-bundle.crt /usr/local/share/ca-certificates/
ONBUILD ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/ca-bundle.crt
ONBUILD RUN update-ca-certificates --fresh

# Configure pipe for cache
ONBUILD COPY pip.ini /etc/pip.ini

# ------------------

FROM python:${PYTHON}-slim as build_no_proxy

ONBUILD RUN echo "Building without PROXY"

# ------------------

FROM build_${BUILD_ENV}

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install \
    -r /tmp/pip-tmp/requirements.txt && rm -rf /tmp/pip-tmp

COPY ./src /src

ARG PORT=8000
ENV PORT=$PORT

EXPOSE ${PORT}

CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}

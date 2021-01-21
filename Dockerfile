# https://pythonspeed.com/articles/conda-docker-image-size/

FROM continuumio/miniconda3 AS build

# Install the package as normal:
COPY environment.yml .
RUN conda env create -f environment.yml

# COPY requirements.txt .
# RUN conda create --name env --file requirements.txt

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n example -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack


# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM debian:buster AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

ADD app /app
ADD app.py .
# When image is run, run the code with the environment
# activated:
SHELL ["/bin/bash", "-c"]
ENTRYPOINT source /venv/bin/activate && \
           python app.py

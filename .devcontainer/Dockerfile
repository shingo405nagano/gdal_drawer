# Python - DevContiner公式イメージ
# * GitHub | devcontainer/pyhton : https://github.com/devcontainers/images/tree/main/src/python
FROM mcr.microsoft.com/devcontainers/python:3.12

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get upgrade -y \
    && apt-get -y install --no-install-recommends \
        tzdata \
        gdal-bin \
        libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    RUN pip install --upgrade pip --break-system-packages \
    && pip install wheel --break-system-packages \
    && pip install setuptools --break-system-packages \
    && pip install requests --break-system-packages \
    && pip install numpy --break-system-packages \
    && pip install gdal[numpy]=="$(gdal-config --version).*" --no-build-isolation --break-system-packages \
    && pip install matplotlib --break-system-packages \
    && pip install pandas --break-system-packages \
    && pip install shapely --break-system-packages \
    && pip install pyproj --break-system-packages \
    && pip install geopandas --break-system-packages \
    && pip install scipy --break-system-packages \
    && pip install memory-profiler --break-system-packages \
    && pip install ruff --break-system-packages \
    && pip install black --break-system-packages \
    && pip install pytest --break-system-packages \
    && pip install pytest-cov --break-system-packages \
    && pip install pytest-xdist --break-system-packages
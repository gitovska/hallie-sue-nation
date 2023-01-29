# Use phusion/baseimage Ubuntu Jammy as base image.
FROM phusion/baseimage:jammy-1.0.1

# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# install python 
RUN apt update -y \
    && apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev  \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev -y
    && wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz
    && tar -xf Python-3.10.9.tgz \
    && cd Python-3.10.9/ \
    && ./configure --enable-optimizations \
    && make -j $(nproc) \
    && make altinstall \
    && cd \
    && rm -r Python-3.10.9 Python-3.10.9.tgz

# install git, clone repo, install requirements
RUN apt install git -y \
    && git clone https://github.com/gitovska/hallie-sue-nation.git \
    && cd hallie-sue-nation \
    && pip3.10 install -r requirements.txt \
    && python3.10 -m nltk.downloader punkt \
    && pip3.10 install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Clean up APT when done
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

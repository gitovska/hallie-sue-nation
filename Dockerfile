# Use phusion/baseimage Ubuntu Jammy as base image.
FROM phusion/baseimage:jammy-1.0.1

# Use baseimage-docker's init system.
CMD ["/sbin/my_init"]

# install python 
RUN apt update -y \
    && apt install python3 -y \
    && apt install python3-pip -y


# install git and clone repo
RUN apt install git -y \
    git clone https://github.com/gitovska/hallie-sue-nation.git \
    cd hallie-sue-nation

# Clean up APT when done
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

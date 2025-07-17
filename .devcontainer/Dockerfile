FROM ubuntu:noble-20240827.1

ARG USERNAME=ubuntu

# System-level setup.
RUN \
    # Install and enable sudo.
    apt-get update \
    && apt-get install -y sudo \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    # Configure timezone.
    && ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
    # Install necessary packages.
    && apt-get install -y wget python3 python3-pip python3-venv pipx git git-lfs \
    # Install CUDA toolkit.
    && wget \
        https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
        -O /tmp/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get -y install cuda-toolkit-12-6


# User-level setup.
USER ${USERNAME}
RUN \
    # Setup pipx for the user.
    pipx ensurepath \
    # Setup CUDA tools for the user.
    && echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> /home/ubuntu/.bashrc \
    # Install Poetry.
    && pipx install poetry \
    # Configure git.
    && git config --global init.defaultBranch main \
    && git config --global core.editor "code --wait" \
    && git lfs install \
    # Configure a longer Bash history.
    && sed -i "s/HISTSIZE=.*/HISTSIZE=1000000/" /home/ubuntu/.bashrc \
    && sed -i "s/HISTFILESIZE=.*/HISTFILESIZE=2000000/" /home/ubuntu/.bashrc

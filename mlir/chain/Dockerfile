FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq wget gnupg ca-certificates \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-22 main" > /etc/apt/sources.list.d/llvm-22.list \
    && apt-get update -qq \
    && apt-get install -y -qq mlir-22-tools llvm-22 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/lib/llvm-22/bin:${PATH}"
WORKDIR /work

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Build context: repository root (the server is a workspace member).
# Build:   docker build -f benchmarks-website/server/Dockerfile .
# Toolchain pinned to match rust-toolchain.toml.

FROM rust:1.91-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release -p vortex-bench-server --bin vortex-bench-server

FROM debian:bookworm-slim

# Keep this in lockstep with libduckdb-sys in Cargo.lock.
ARG DUCKDB_VERSION=1.5.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        unzip \
        wget \
    && wget -q "https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/duckdb_cli-linux-aarch64.zip" -O /tmp/duckdb.zip \
    && unzip -q /tmp/duckdb.zip -d /usr/local/bin/ \
    && chmod +x /usr/local/bin/duckdb \
    && rm /tmp/duckdb.zip \
    && apt-get purge -y --auto-remove unzip wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/target/release/vortex-bench-server /usr/local/bin/vortex-bench-server

WORKDIR /app/data

EXPOSE 3000

CMD ["/usr/local/bin/vortex-bench-server"]

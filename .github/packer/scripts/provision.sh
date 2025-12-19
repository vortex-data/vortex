#!/bin/bash
set -euo pipefail

# Variables (passed from Packer)
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.89}"
PROTOC_VERSION="${PROTOC_VERSION:-29.3}"
FLATC_VERSION="${FLATC_VERSION:-25.9.23}"

echo "=== Installing Vortex CI dependencies ==="

# Install build dependencies
echo "Installing system packages..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cmake \
  ninja-build \
  clang \
  lld \
  llvm \
  pkg-config \
  libssl-dev

# Install Rust
echo "Installing Rust ${RUST_TOOLCHAIN}..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"
source "$HOME/.cargo/env"
echo 'source $HOME/.cargo/env' >> "$HOME/.bashrc"

# Install Rust components
rustup component add clippy rustfmt
rustup toolchain install nightly
rustup component add --toolchain nightly rustfmt clippy rust-src miri llvm-tools-preview

echo "Rust installed:"
cargo --version
rustc --version

# Install protoc
echo "Installing protoc ${PROTOC_VERSION}..."
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  PROTOC_ARCH=linux-x86_64
else
  PROTOC_ARCH=linux-aarch_64
fi
curl -fsSL -o /tmp/protoc.zip "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-${PROTOC_ARCH}.zip"
sudo unzip -o /tmp/protoc.zip -d /usr/local bin/protoc 'include/*'
sudo chmod +x /usr/local/bin/protoc
rm /tmp/protoc.zip
protoc --version

# Install flatc
echo "Installing flatc ${FLATC_VERSION}..."
if [ "$ARCH" = "x86_64" ]; then
  curl -fsSL -o /tmp/flatc.zip "https://github.com/google/flatbuffers/releases/download/v${FLATC_VERSION}/Linux.flatc.binary.clang++-18.zip"
  sudo unzip -o /tmp/flatc.zip -d /usr/local/bin
  sudo chmod +x /usr/local/bin/flatc
  rm /tmp/flatc.zip
else
  # Build from source for ARM64
  git clone --depth 1 --branch "v${FLATC_VERSION}" https://github.com/google/flatbuffers.git /tmp/flatbuffers
  cd /tmp/flatbuffers
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .
  ninja
  sudo cp flatc /usr/local/bin/
  cd -
  rm -rf /tmp/flatbuffers
fi
flatc --version

# Install cargo tools
echo "Installing cargo tools..."
source "$HOME/.cargo/env"
cargo install cargo-nextest --locked
cargo install cargo-hack --locked
cargo install grcov --locked

# Cleanup
echo "Cleaning up..."
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
rm -rf /tmp/*

echo "=== Vortex CI dependencies installed successfully ==="

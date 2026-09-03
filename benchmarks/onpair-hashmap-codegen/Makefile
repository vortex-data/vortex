CARGO ?= cargo
GXX ?= g++
CLANGXX ?= clang++
BOOST_ROOT ?= /tmp/boost_1_89_0
TARGET_DIR ?= target/native
RUST_CPU ?= native
CORPUS ?= /tmp/onpair-corpora/fineweb-32mib.onpair
TRACE ?= /tmp/onpair-hashmap-codegen.oph
BITS ?= 16
WARMUPS ?= 3
ITERATIONS ?= 15

MANIFEST := Cargo.toml
RUST_BENCH := $(TARGET_DIR)/release/bench
TRACE_TOOL := $(TARGET_DIR)/release/trace
BOOST_GCC := $(TARGET_DIR)/boost-gcc
BOOST_CLANG := $(TARGET_DIR)/boost-clang
CXXFLAGS := -std=c++20 -O3 -DNDEBUG -march=native

.PHONY: all rust gcc clang trace run asm clean

all: rust gcc clang

rust:
	CARGO_TARGET_DIR=$(TARGET_DIR) RUSTFLAGS="-C target-cpu=$(RUST_CPU) -C linker=gcc -C link-arg=-fuse-ld=bfd" \
		$(CARGO) build --release --manifest-path $(MANIFEST)

gcc:
	mkdir -p $(TARGET_DIR)
	$(GXX) $(CXXFLAGS) -I$(BOOST_ROOT) boost.cpp -o $(BOOST_GCC)

clang:
	mkdir -p $(TARGET_DIR)
	$(CLANGXX) $(CXXFLAGS) -I$(BOOST_ROOT) boost.cpp -o $(BOOST_CLANG)

trace: rust
	ONPAIR_BITS=$(BITS) $(TRACE_TOOL) $(CORPUS) $(TRACE)

run: all trace
	HASH_PROBE_STATS=1 HASH_WARMUPS=$(WARMUPS) HASH_ITERATIONS=$(ITERATIONS) \
		$(RUST_BENCH) $(TRACE)
	HASH_WARMUPS=$(WARMUPS) HASH_ITERATIONS=$(ITERATIONS) $(BOOST_GCC) $(TRACE)
	HASH_WARMUPS=$(WARMUPS) HASH_ITERATIONS=$(ITERATIONS) $(BOOST_CLANG) $(TRACE)

asm: all
	objdump -d --no-show-raw-insn -Mintel $(RUST_BENCH) | \
		sed -n '/<asm_rust_group15_long>:/,/^$$/p'
	objdump -d --no-show-raw-insn -Mintel $(RUST_BENCH) | \
		sed -n '/<asm_rust_group15_get4_long>:/,/^$$/p'
	objdump -d --no-show-raw-insn -Mintel $(BOOST_GCC) | \
		sed -n '/<asm_boost_long>:/,/^$$/p'
	objdump -d --no-show-raw-insn -Mintel $(BOOST_CLANG) | \
		sed -n '/<asm_boost_long>:/,/^$$/p'

clean:
	$(CARGO) clean --manifest-path $(MANIFEST) --target-dir $(TARGET_DIR)

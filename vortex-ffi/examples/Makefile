# Directory containing the Rust crate
ROOT_DIR = ../..
RUST_CRATE_DIR = ..
# Directory where cargo places the built libraries
RUST_TARGET_DIR = $(ROOT_DIR)/target
# The library name (without lib prefix and .so/.dylib suffix)
LIBRARY_NAME = vortex_ffi

# Determine platform-specific library extension
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB_EXT = so
else ifeq ($(UNAME_S),Darwin)
    LIB_EXT = dylib
else
    # Default to .dll for Windows
    LIB_EXT = dll
endif

# Path to the library depending on build mode
DEBUG_LIB = $(RUST_TARGET_DIR)/debug/lib$(LIBRARY_NAME).$(LIB_EXT)
RELEASE_LIB = $(RUST_TARGET_DIR)/release/lib$(LIBRARY_NAME).$(LIB_EXT)

# Default to debug mode
LIB = $(DEBUG_LIB)

# C flags
CFLAGS = -Wall -I$(RUST_CRATE_DIR)/cinclude

# Linking flags
LDFLAGS = -L$(RUST_TARGET_DIR)/debug -l$(LIBRARY_NAME)

# Final executable name
EXECUTABLE = hello_vortex

.PHONY: all clean run release debug

# Default target
all: debug

# Debug build
debug: LDFLAGS = -L$(RUST_TARGET_DIR)/debug -l$(LIBRARY_NAME)
debug: build

# Release build
release: LDFLAGS = -L$(RUST_TARGET_DIR)/release -l$(LIBRARY_NAME)
release: LIB = $(RELEASE_LIB)
release: build

# Build rule
build: $(LIB) $(EXECUTABLE)

# Build the Rust library
$(DEBUG_LIB):
	cd $(RUST_CRATE_DIR) && cargo build

$(RELEASE_LIB):
	cd $(RUST_CRATE_DIR) && cargo build --release

# Build the C executable
$(EXECUTABLE): hello-vortex.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run the executable with the correct library path
run: build
	@echo "Running example..."
	LD_LIBRARY_PATH=$(dir $(LIB)) ./$(EXECUTABLE)

# Clean build artifacts
clean:
	rm -f $(EXECUTABLE)
	cd $(RUST_CRATE_DIR) && cargo clean

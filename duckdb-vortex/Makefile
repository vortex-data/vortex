PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=vortex_duckdb
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
EXT_FLAGS=-DDISABLE_VPTR_SANITIZER=ON -DOVERRIDE_GIT_DESCRIBE=v1.2.2
EXT_FLAGS += -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0

export MACOSX_DEPLOYMENT_TARGET=13.0
export VCPKG_OSX_DEPLOYMENT_TARGET=13.0
export VCPKG_FEATURE_FLAGS=-binarycaching

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

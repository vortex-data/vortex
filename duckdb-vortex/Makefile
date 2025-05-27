PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

EXT_NAME=vortex_duckdb
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
EXT_FLAGS=-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0

export MACOSX_DEPLOYMENT_TARGET=12.0

# The version of DuckDB and its Vortex extension is either implicitly set by Git tag, e.g. v1.2.2, or commit
# SHA if the current commit does not have a tag. The implicitly set version can be overridden by defining the
# `OVERRIDE_GIT_DESCRIBE` environment variable. In context of the DuckDB community extension build, we have to
# rely on the Git tag, as DuckDB's CI performs a checkout by Git tag. Therefore, the version can't be explicitly
# set via environment variable for the community extension build.

export OVERRIDE_GIT_DESCRIBE=v1.3.0
export VCPKG_FEATURE_FLAGS=-binarycaching
export VCPKG_OSX_DEPLOYMENT_TARGET=12.0
export VCPKG_TOOLCHAIN_PATH := ${PROJ_DIR}vcpkg/scripts/buildsystems/vcpkg.cmake

export BUILD_MAIN_DUCKDB_LIBRARY=0
export DISABLE_BUILTIN_EXTENSIONS=1

# This is not needed on macOS, we don't see a tls error on load there.
ifeq ($(shell uname), Linux)
    export CFLAGS=-ftls-model=global-dynamic
endif

include extension-ci-tools/makefiles/duckdb_extension.Makefile

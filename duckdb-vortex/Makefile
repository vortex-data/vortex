PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

EXT_NAME=vortex_duckdb
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
EXT_FLAGS=-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 -DOVERRIDE_GIT_DESCRIBE=v1.2.2

export MACOSX_DEPLOYMENT_TARGET=12.0
export VCPKG_OSX_DEPLOYMENT_TARGET=12.0
export VCPKG_FEATURE_FLAGS=-binarycaching
export VCPKG_TOOLCHAIN_PATH := ${PROJ_DIR}vcpkg/scripts/buildsystems/vcpkg.cmake

include extension-ci-tools/makefiles/duckdb_extension.Makefile

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Convenience targets for the Vortex Go bindings. cgo links against
# libvortex_ffi; `make lib` builds it.

REPO_ROOT := $(abspath ../..)
PROFILE   ?= release

.PHONY: all lib build test lint vet fmt fmt-check clean

all: build

# Build the vortex-ffi static/shared library (release by default).
lib:
	cd $(REPO_ROOT) && cargo build --profile $(PROFILE) -p vortex-ffi

build: lib
	go build ./...

test: lib
	go test ./...

lint: fmt-check vet

vet:
	go vet ./...

fmt:
	gofmt -w .

fmt-check:
	test -z "$$(gofmt -l .)"

clean:
	go clean ./...

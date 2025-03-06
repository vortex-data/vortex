#!/bin/bash

# This is an example script that can be used to install additional toolchain dependencies. Feel free to remove this script
# if no additional toolchains are required

# To enable this script, set the `custom_toolchain_script` option to true when calling the reusable workflow
# `.github/workflows/_extension_distribution.yml` from `https://github.com/duckdb/extension-ci-tools`

# note that the $DUCKDB_PLATFORM environment variable can be used to discern between the platforms
echo "This is the sample custom toolchain script running for architecture '$DUCKDB_PLATFORM' for the quack extension."


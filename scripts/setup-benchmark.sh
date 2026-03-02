#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -Eeu -o pipefail -x

if [ "$EUID" -ne 0 ]; then
    echo "Environment setup script for benchmarks should run as root."
    exit 0
fi

# Really discourage swapping to disk
sudo sysctl vm.swappiness=0

# Disable ASLR - https://docs.kernel.org/admin-guide/sysctl/kernel.html#randomize-va-space
sudo sysctl kernel.randomize_va_space=0

# Reduce kernel logging to minimum
dmesg -n 1

# Disable some unused services and features
sudo systemctl stop apparmor ModemManager
sudo systemctl disable apparmor ModemManager

# mask prevents them from being started by other services
sudo systemctl mask ModemManager

# For apparmor specifically, also teardown loaded profiles
sudo aa-teardown

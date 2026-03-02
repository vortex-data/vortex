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
sudo sysctl kernel.randomize_va_space

# Reduce kernel logging to minimum
dmesg -n 1

# Disable some unused services and features
sudo systemctl stop apparmor snapd unattended-upgrades multipathd ModemManager
sudo systemctl disable apparmor snapd unattended-upgrades multipathd ModemManager

# mask prevents them from being started by other services
sudo systemctl mask snapd unattended-upgrades multipathd ModemManager

# For apparmor specifically, also teardown loaded profiles
sudo aa-teardown

# For auditd, also disable kernel audit
sudo auditctl -e 0
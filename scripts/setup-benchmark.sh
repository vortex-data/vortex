#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -Eeu -o pipefail -x

if [ "$EUID" -ne 0 ]; then
    echo "Environment setup script for benchmarks should run as root."
    exit 0
fi

# Disable turbo boost so benchmark runs stay at a more stable clock rate.
[[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]] && echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo || true
[[ -w /sys/devices/system/cpu/cpufreq/boost ]] && echo 0 > /sys/devices/system/cpu/cpufreq/boost || true

# Really discourage swapping to disk
sysctl vm.swappiness=0
swapoff -a || true

# Might be worse if a single application uses the OS
# https://www.intel.com/content/www/us/en/developer/articles/technical/measuring-impact-of-numa-migrations-on-performance.html
sysctl -w kernel.numa_balancing=0

# Disable ASLR - https://docs.kernel.org/admin-guide/sysctl/kernel.html#randomize-va-space
sysctl kernel.randomize_va_space=0

# This is a desktop optimization, making sure its disabled
sysctl -w kernel.sched_autogroup_enabled=0

# Reduce kernel logging to minimum
dmesg -n 1

# Disable some unused services and features
systemctl stop apparmor ModemManager
systemctl disable apparmor ModemManager

# mask prevents them from being started by other services
systemctl mask ModemManager

# For apparmor specifically, also teardown loaded profiles
aa-teardown

# Reduce background activity (Ubuntu-specific)
for unit in \
  irqbalance \
  apt-daily.service \
  apt-daily-upgrade.service \
  apt-daily.timer \
  apt-daily-upgrade.timer \
  motd-news.service \
  motd-news.timer \
  apport
do
  systemctl disable --now "$unit" 2>/dev/null
done
systemctl mask irqbalance 2>/dev/null

CPU_COUNT="$(nproc)"
HOUSEKEEPING_CPUS="0-1"
BENCH_CPUS="2-$((CPU_COUNT - 1))"

# Pin all IRQs to housekeeping CPUs
for f in /proc/irq/[0-9]*/smp_affinity_list; do
  if [[ -w "$f" ]]; then
    # Some IRQs are kernel-managed and reject writes with EPERM even as root.
    echo "$HOUSEKEEPING_CPUS" > "$f" 2>/dev/null || true
  fi
done

# Persist CPU affinity ranges for non-root benchmark steps in CI.
cat > /tmp/vortex-benchmark.env <<EOF
HOUSEKEEPING_CPUS=$HOUSEKEEPING_CPUS
BENCH_CPUS=$BENCH_CPUS
EOF
chmod 0644 /tmp/vortex-benchmark.env

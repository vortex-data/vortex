#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -Eeu -o pipefail -x

if [ "$EUID" -ne 0 ]; then
    echo "Environment setup script for benchmarks should run as root."
    exit 0
fi

# Write a value to a sysfs file, skipping hosts where the file is absent or read-only
# (e.g. virtualized guests without frequency control).
sysfs_write() {
    [[ -n "$2" && -w "$1" ]] && echo "$2" > "$1" || true
}

# Expand a cpulist like "0-2,8" into one CPU index per line.
expand_cpulist() {
    [[ -n "$1" ]] || return 0
    local parts part
    IFS=, read -ra parts <<< "$1"
    for part in "${parts[@]}"; do
        seq "${part%-*}" "${part#*-}"
    done
}

# Disable turbo boost so benchmark runs stay at a more stable clock rate.
sysfs_write /sys/devices/system/cpu/intel_pstate/no_turbo 1
sysfs_write /sys/devices/system/cpu/cpufreq/boost 0

# Pin the frequency governor to performance and raise the floor to the policy max so
# clocks hold at a constant base clock (turbo is capped above) instead of ramping
# under powersave/schedutil.
for policy in /sys/devices/system/cpu/cpufreq/policy*; do
    sysfs_write "$policy/scaling_governor" performance
    sysfs_write "$policy/scaling_min_freq" "$(cat "$policy/scaling_max_freq" 2>/dev/null)"
done

# Really discourage swapping to disk
sysctl -w vm.swappiness=0 || true
swapoff -a || true

# Might be worse if a single application uses the OS
# https://www.intel.com/content/www/us/en/developer/articles/technical/measuring-impact-of-numa-migrations-on-performance.html
sysctl -w kernel.numa_balancing=0 || true

# Disable ASLR - https://docs.kernel.org/admin-guide/sysctl/kernel.html#randomize-va-space
sysctl -w kernel.randomize_va_space=0 || true

# This is a desktop optimization, making sure its disabled
sysctl -w kernel.sched_autogroup_enabled=0 || true

# Keep armed timers on the CPU that armed them instead of migrating them onto busy
# (i.e. benchmark) CPUs.
sysctl -w kernel.timer_migration=0 || true

# Reduce kernel logging to minimum
dmesg -n 1

# Stop background services that could wake up mid-benchmark (Ubuntu-specific list).
# Masking prevents them from being restarted by other services.
for unit in \
  apparmor \
  ModemManager \
  irqbalance \
  apt-daily.service \
  apt-daily-upgrade.service \
  apt-daily.timer \
  apt-daily-upgrade.timer \
  motd-news.service \
  motd-news.timer \
  apport
do
  systemctl disable --now "$unit" 2>/dev/null || true
  systemctl mask "$unit" 2>/dev/null || true
done

# For apparmor, also teardown already-loaded profiles
aa-teardown || true

# Split CPUs between the benchmark and everything else ("housekeeping"). On machines
# with multiple NUMA nodes the benchmark owns all of node 0 (bench-taskset.sh binds
# its memory to node 0) and housekeeping moves to the other nodes entirely. On
# single-node machines housekeeping gets cores 0-1 plus their SMT siblings, so
# background work never shares a physical core with the benchmark.
node_dirs=(/sys/devices/system/node/node[0-9]*)
if [[ ${#node_dirs[@]} -gt 1 && -r "${node_dirs[1]:-}/cpulist" ]]; then
    BENCH_CPUS="$(cat /sys/devices/system/node/node0/cpulist)"
    HOUSEKEEPING_CPUS="$(cat /sys/devices/system/node/node[1-9]*/cpulist | paste -sd, -)"
else
    HOUSEKEEPING_CPUS="$(expand_cpulist "$(cat /sys/devices/system/cpu/cpu{0,1}/topology/thread_siblings_list 2>/dev/null | paste -sd, -)" | sort -un | paste -sd, -)"
    bench=()
    for cpu in $(expand_cpulist "$(cat /sys/devices/system/cpu/online)"); do
        [[ ",$HOUSEKEEPING_CPUS," == *",$cpu,"* ]] || bench+=("$cpu")
    done
    BENCH_CPUS="$(IFS=,; echo "${bench[*]}")"
fi

# Pin all IRQs to housekeeping CPUs. Some IRQs are kernel-managed and reject writes
# with EPERM even as root.
for f in /proc/irq/[0-9]*/smp_affinity_list; do
  echo "$HOUSEKEEPING_CPUS" > "$f" 2>/dev/null || true
done

# Steer unbound kernel workqueues (writeback etc.) to the housekeeping CPUs. The
# sysfs file takes a hex cpumask rather than a cpulist.
if command -v python3 >/dev/null 2>&1; then
    wq_mask="$(expand_cpulist "$HOUSEKEEPING_CPUS" \
        | python3 -c 'import sys; print(f"{sum(1 << int(l) for l in sys.stdin):x}")' || true)"
    sysfs_write /sys/devices/virtual/workqueue/cpumask "$wq_mask"
fi

# Keep everything that is not explicitly re-pinned on the housekeeping CPUs. Two
# layers: a systemd manager default so services (re)started from now on land there,
# and a one-shot re-pin of everything already running (daemons, the CI runner agent,
# movable kernel threads). Plain affinity is a default rather than a ceiling (unlike
# cpusets), so bench-taskset.sh can still move the benchmark onto BENCH_CPUS.
mkdir -p /etc/systemd/system.conf.d
cat > /etc/systemd/system.conf.d/99-vortex-benchmark.conf <<EOF
[Manager]
CPUAffinity=$HOUSEKEEPING_CPUS
EOF
systemctl daemon-reexec || true

{ set +x; } 2>/dev/null  # the per-PID loop is far too noisy for trace output
for pid_dir in /proc/[0-9]*; do
    taskset -a -pc "$HOUSEKEEPING_CPUS" "${pid_dir##*/}" >/dev/null 2>&1 || true
done
set -x

# Persist CPU affinity ranges for non-root benchmark steps in CI.
cat > /tmp/vortex-benchmark.env <<EOF
HOUSEKEEPING_CPUS=$HOUSEKEEPING_CPUS
BENCH_CPUS=$BENCH_CPUS
EOF
chmod 0644 /tmp/vortex-benchmark.env

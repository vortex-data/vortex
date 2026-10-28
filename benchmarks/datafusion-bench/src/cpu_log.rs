// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `--opt cpu-log=<path>`: the CPU each run takes, one JSON line a run. The process's CPU time is
//! exact (`CLOCK_PROCESS_CPUTIME_ID`, every thread); the kernel's ticks split it into user and
//! system time (`/proc/self/stat`). The whole machine's busy ticks (`/proc/stat`) also count
//! interrupts and kernel threads, which no process is charged for: a drive's completions land
//! there. Ticks are the kernel's USER_HZ, 10 ms on Linux. And the bytes the md RAID devices read
//! in the run (`/proc/diskstats`), the drives' reads the run itself made. Linux only.

use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;

/// Where [`Self::log_run`] appends each run's line.
pub struct CpuLog {
    path: PathBuf,
}

/// The CPU counters at one instant.
pub struct CpuSample {
    at: Instant,
    process_ns: u64,
    process: ProcessTicks,
    machine: MachineTicks,
    raid_sectors: u64,
}

/// The process's user and system ticks, from `/proc/self/stat`.
#[derive(Clone, Copy)]
struct ProcessTicks {
    user: u64,
    system: u64,
}

/// The machine's ticks over all CPUs, from `/proc/stat`'s first line.
#[derive(Clone, Copy)]
struct MachineTicks {
    /// user and nice.
    user: u64,
    system: u64,
    /// irq and softirq.
    interrupts: u64,
    /// idle and iowait.
    idle: u64,
}

impl CpuLog {
    pub fn logging_to(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The counters now: taken before a run's timer starts.
    pub fn sample(&self) -> anyhow::Result<CpuSample> {
        Ok(CpuSample {
            at: Instant::now(),
            process_ns: CpuSample::process_cpu_ns()?,
            process: ProcessTicks::read()?,
            machine: MachineTicks::read()?,
            raid_sectors: CpuSample::raid_read_sectors()?,
        })
    }

    /// Appends what the CPU did since `start`, after a run's timer stops.
    pub fn log_run(&self, start: &CpuSample) -> anyhow::Result<()> {
        let end = self.sample()?;
        let line = format!(
            "{{\"wall_ms\":{:.3},\"process_cpu_ms\":{:.3},\"process_user_ticks\":{},\"process_system_ticks\":{},\"machine_user_ticks\":{},\"machine_system_ticks\":{},\"machine_interrupt_ticks\":{},\"machine_idle_ticks\":{},\"raid_read_mb\":{:.3}}}\n",
            end.at.duration_since(start.at).as_secs_f64() * 1e3,
            end.process_ns.saturating_sub(start.process_ns) as f64 / 1e6,
            end.process.user - start.process.user,
            end.process.system - start.process.system,
            end.machine.user - start.machine.user,
            end.machine.system - start.machine.system,
            end.machine.interrupts - start.machine.interrupts,
            end.machine.idle - start.machine.idle,
            (end.raid_sectors - start.raid_sectors) as f64 * 512.0 / 1e6,
        );
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?
            .write_all(line.as_bytes())?;
        Ok(())
    }
}

impl CpuSample {
    #[cfg(target_os = "linux")]
    fn process_cpu_ns() -> anyhow::Result<u64> {
        let now = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
        Ok(u64::try_from(now.tv_sec)? * 1_000_000_000 + u64::try_from(now.tv_nsec)?)
    }

    #[cfg(not(target_os = "linux"))]
    fn process_cpu_ns() -> anyhow::Result<u64> {
        anyhow::bail!("cpu-log reads Linux's process CPU clock")
    }

    /// The 512-byte sectors every md device has read, from `/proc/diskstats`: the drives under
    /// a RAID, counted once. 0 on a machine without one.
    fn raid_read_sectors() -> anyhow::Result<u64> {
        let stats =
            std::fs::read_to_string("/proc/diskstats").context("reading /proc/diskstats")?;
        let mut sectors = 0;
        for line in stats.lines() {
            // major minor name reads merged sectors-read ...
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.get(2).is_some_and(|name| name.starts_with("md")) {
                sectors += fields
                    .get(5)
                    .context("/proc/diskstats' line is short")?
                    .parse::<u64>()?;
            }
        }
        Ok(sectors)
    }
}

impl ProcessTicks {
    fn read() -> anyhow::Result<Self> {
        let stat = std::fs::read_to_string("/proc/self/stat").context("reading /proc/self/stat")?;
        // The fields after the command's closing parenthesis start at the third, the state, so
        // utime (the 14th) is the 12th of them and stime the 13th.
        let after_command = stat
            .rsplit_once(')')
            .context("no command in /proc/self/stat")?
            .1;
        let fields: Vec<&str> = after_command.split_whitespace().collect();
        let field = |i: usize| -> anyhow::Result<u64> {
            Ok(fields
                .get(i)
                .context("/proc/self/stat ended early")?
                .parse()?)
        };
        Ok(Self {
            user: field(11)?,
            system: field(12)?,
        })
    }
}

impl MachineTicks {
    fn read() -> anyhow::Result<Self> {
        let stat = std::fs::read_to_string("/proc/stat").context("reading /proc/stat")?;
        let line = stat.lines().next().context("/proc/stat is empty")?;
        // cpu user nice system idle iowait irq softirq ...
        let ticks: Vec<u64> = line
            .split_whitespace()
            .skip(1)
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        anyhow::ensure!(ticks.len() >= 7, "/proc/stat's cpu line is short: {line}");
        Ok(Self {
            user: ticks[0] + ticks[1],
            system: ticks[2],
            interrupts: ticks[5] + ticks[6],
            idle: ticks[3] + ticks[4],
        })
    }
}

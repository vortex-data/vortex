//! Measurement utilities: per-chunk latency quantiles, CPU utilization, and
//! multi-iteration run reporting.

use std::time::Duration;
use std::time::Instant;

use sysinfo::Pid;
use sysinfo::ProcessRefreshKind;
use sysinfo::ProcessesToUpdate;
use sysinfo::System;

/// One pass through the corpus.
#[derive(Clone, Debug)]
pub struct IterationResult {
    pub elapsed: Duration,
    pub bytes: u64,
    pub vectors: u64,
    pub chunk_latencies_us: Vec<u64>,
    pub cpu_percent: f32,
    pub sink_sum: f64,
    pub sink_max: f32,
}

impl IterationResult {
    pub fn gib_per_sec(&self) -> f64 {
        (self.bytes as f64) / self.elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn gb_per_sec(&self) -> f64 {
        (self.bytes as f64) / self.elapsed.as_secs_f64() / 1e9
    }

    pub fn vecs_per_sec(&self) -> f64 {
        (self.vectors as f64) / self.elapsed.as_secs_f64()
    }

    /// Returns (p50, p99) latency in microseconds across chunks.
    pub fn latency_p50_p99_us(&self) -> (u64, u64) {
        if self.chunk_latencies_us.is_empty() {
            return (0, 0);
        }
        let mut v = self.chunk_latencies_us.clone();
        v.sort_unstable();
        let p50 = v[v.len() / 2];
        let p99 = v[(v.len() * 99 / 100).min(v.len() - 1)];
        (p50, p99)
    }
}

/// Aggregated stats across multiple iterations.
pub struct RunSummary {
    pub iters: Vec<IterationResult>,
}

impl RunSummary {
    pub fn report(&self, label: &str) {
        if self.iters.is_empty() {
            println!("[{label}] no iterations completed");
            return;
        }
        let mut gbs: Vec<f64> = self.iters.iter().map(|i| i.gb_per_sec()).collect();
        gbs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = gbs[gbs.len() / 2];
        let min = gbs[0];
        let max = gbs[gbs.len() - 1];
        let last = self.iters.last().unwrap();
        let (p50, p99) = last.latency_p50_p99_us();

        println!(
            "[{label}] median {:.2} GB/s (min {:.2} / max {:.2}) | {:.1} Mvec/s | chunk latency p50 {}us p99 {}us | CPU {:.0}%",
            median,
            min,
            max,
            last.vecs_per_sec() / 1e6,
            p50,
            p99,
            last.cpu_percent,
        );
        println!(
            "[{label}] sink: sum={:.6} max={:.6} vectors={} bytes={} elapsed={:.3}s",
            last.sink_sum,
            last.sink_max,
            last.vectors,
            last.bytes,
            last.elapsed.as_secs_f64(),
        );
    }
}

/// Samples this process's CPU time against wall-clock time.
pub struct CpuSampler {
    sys: System,
    pid: Pid,
    start: Instant,
    start_cpu: Duration,
}

impl Default for CpuSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuSampler {
    pub fn new() -> Self {
        let pid = Pid::from_u32(std::process::id());
        let mut sys = System::new();
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::new().with_cpu(),
        );
        let start_cpu = process_cpu_time();
        Self {
            sys,
            pid,
            start: Instant::now(),
            start_cpu,
        }
    }

    /// Finish sampling and return CPU utilization as a percentage of a single
    /// core. Values above 100% indicate cross-core parallelism.
    pub fn finish(mut self) -> f32 {
        self.sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            true,
            ProcessRefreshKind::new().with_cpu(),
        );
        let cpu = process_cpu_time().saturating_sub(self.start_cpu);
        let wall = self.start.elapsed();
        if wall.is_zero() {
            return 0.0;
        }
        (cpu.as_secs_f64() / wall.as_secs_f64() * 100.0) as f32
    }
}

/// Returns process CPU time (user + system) on Unix; falls back to zero
/// elsewhere. Uses getrusage for stable cross-distro semantics.
#[cfg(unix)]
fn process_cpu_time() -> Duration {
    // SAFETY: getrusage(RUSAGE_SELF, &out) only writes to the provided struct.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if rc != 0 {
        return Duration::ZERO;
    }
    let user = Duration::new(
        usage.ru_utime.tv_sec as u64,
        (usage.ru_utime.tv_usec * 1000) as u32,
    );
    let sys = Duration::new(
        usage.ru_stime.tv_sec as u64,
        (usage.ru_stime.tv_usec * 1000) as u32,
    );
    user + sys
}

#[cfg(not(unix))]
fn process_cpu_time() -> Duration {
    Duration::ZERO
}

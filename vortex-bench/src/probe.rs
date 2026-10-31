// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Engine-independent measurements of a query, read from `/proc`.
//!
//! Process figures are this process's own. Device figures cover the whole disk, so another
//! process on the host shows up in them.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use vortex::scan::first_output;

const SAMPLE_INTERVAL: Duration = Duration::from_millis(5);
const SECTOR_BYTES: u64 = 512;
/// Linux reports process CPU time in units of `USER_HZ`, which is 100 on every supported target.
const MS_PER_TICK: u64 = 10;

/// The disk to watch, from `VORTEX_BENCH_DISK`.
fn disk() -> String {
    std::env::var("VORTEX_BENCH_DISK").unwrap_or_else(|_| "nvme0n1".to_owned())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counters {
    proc_read_bytes: u64,
    proc_read_syscalls: u64,
    cpu_ms: u64,
    dev_reads: u64,
    dev_read_bytes: u64,
    dev_read_ms: u64,
}

fn field(text: &str, key: &str) -> u64 {
    text.lines()
        .find_map(|line| line.strip_prefix(key)?.trim().parse().ok())
        .unwrap_or(0)
}

/// Reads in flight in `/sys/block/<disk>/inflight` text, which holds reads then writes.
///
/// `/proc/diskstats` is no use for this: it mixes in writes and reports zero for a whole NVMe disk.
fn parse_inflight_reads(text: &str) -> Option<u64> {
    text.split_whitespace().next()?.parse().ok()
}

/// `(reads, read sectors, read ms)` of `disk` in `/proc/diskstats` text.
fn parse_diskstats(text: &str, disk: &str) -> Option<(u64, u64, u64)> {
    let fields: Vec<&str> = text
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .find(|fields| fields.get(2) == Some(&disk))?;
    let number = |index: usize| fields.get(index)?.parse().ok();
    Some((number(3)?, number(5)?, number(6)?))
}

/// User plus system ticks in `/proc/self/stat` text; `comm` may hold spaces, so count from `)`.
fn parse_cpu_ticks(text: &str) -> Option<u64> {
    let (_, rest) = text.rsplit_once(')')?;
    let mut fields = rest.split_whitespace().skip(11);
    Some(fields.next()?.parse::<u64>().ok()? + fields.next()?.parse::<u64>().ok()?)
}

fn counters(disk: &str) -> Counters {
    let io = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let disk_stats = std::fs::read_to_string("/proc/diskstats").unwrap_or_default();
    let (dev_reads, sectors, dev_read_ms) = parse_diskstats(&disk_stats, disk).unwrap_or_default();
    Counters {
        proc_read_bytes: field(&io, "read_bytes:"),
        proc_read_syscalls: field(&io, "syscr:"),
        cpu_ms: parse_cpu_ticks(&stat).unwrap_or(0) * MS_PER_TICK,
        dev_reads,
        dev_read_bytes: sectors * SECTOR_BYTES,
        dev_read_ms,
    }
}

/// What a [`QueryProbe`] saw over a query's iterations.
#[derive(Debug)]
pub struct ProbeReport {
    /// Delay from each iteration's start to its first scan batch, where there was one.
    pub first_row: Vec<Duration>,
    /// Wall time of the iterations.
    pub wall: Duration,
    /// Bytes this process caused to be read from storage, which page cache hits do not count in.
    pub proc_read_bytes: u64,
    /// Read system calls of this process.
    pub proc_read_syscalls: u64,
    /// CPU time of this process, over all its threads.
    pub cpu: Duration,
    /// Reads the disk completed.
    pub dev_reads: u64,
    /// Bytes the disk read.
    pub dev_read_bytes: u64,
    /// Mean milliseconds per read the disk completed.
    pub dev_read_ms_mean: f64,
    /// Mean reads in flight on the disk, over the 5ms samples.
    pub dev_queue_mean: f64,
    /// Most reads in flight on the disk in a sample.
    pub dev_queue_max: u64,
}

/// Measures one query across its iterations.
pub struct QueryProbe {
    disk: String,
    start: Instant,
    before: Counters,
    iteration: Instant,
    first_row: Vec<Duration>,
    stop: Arc<AtomicBool>,
    samples: Arc<AtomicU64>,
    in_flight_sum: Arc<AtomicU64>,
    in_flight_max: Arc<AtomicU64>,
}

impl QueryProbe {
    /// Starts measuring, with a thread that samples the disk's queue until the probe finishes.
    pub fn start() -> Self {
        let disk = disk();
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(AtomicU64::new(0));
        let in_flight_sum = Arc::new(AtomicU64::new(0));
        let in_flight_max = Arc::new(AtomicU64::new(0));
        {
            let inflight = format!("/sys/block/{disk}/inflight");
            let stop = Arc::clone(&stop);
            let (samples, sum, max) = (
                Arc::clone(&samples),
                Arc::clone(&in_flight_sum),
                Arc::clone(&in_flight_max),
            );
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let text = std::fs::read_to_string(&inflight).unwrap_or_default();
                    if let Some(in_flight) = parse_inflight_reads(&text) {
                        samples.fetch_add(1, Ordering::Relaxed);
                        sum.fetch_add(in_flight, Ordering::Relaxed);
                        max.fetch_max(in_flight, Ordering::Relaxed);
                    }
                    std::thread::sleep(SAMPLE_INTERVAL);
                }
            });
        }
        let before = counters(&disk);
        let now = Instant::now();
        Self {
            disk,
            start: now,
            before,
            iteration: now,
            first_row: Vec::new(),
            stop,
            samples,
            in_flight_sum,
            in_flight_max,
        }
    }

    /// Call as an iteration starts.
    pub fn start_iteration(&mut self) {
        first_output::reset();
        self.iteration = Instant::now();
    }

    /// Call as an iteration ends.
    pub fn end_iteration(&mut self) {
        if let Some(delay) = first_output::since(self.iteration) {
            self.first_row.push(delay);
        }
    }

    /// Stops measuring and reports.
    pub fn finish(self) -> ProbeReport {
        let wall = self.start.elapsed();
        let after = counters(&self.disk);
        self.stop.store(true, Ordering::Relaxed);
        let samples = self.samples.load(Ordering::Relaxed).max(1);
        let dev_reads = after.dev_reads - self.before.dev_reads;
        ProbeReport {
            first_row: self.first_row,
            wall,
            proc_read_bytes: after.proc_read_bytes - self.before.proc_read_bytes,
            proc_read_syscalls: after.proc_read_syscalls - self.before.proc_read_syscalls,
            cpu: Duration::from_millis(after.cpu_ms - self.before.cpu_ms),
            dev_reads,
            dev_read_bytes: after.dev_read_bytes - self.before.dev_read_bytes,
            dev_read_ms_mean: (after.dev_read_ms - self.before.dev_read_ms) as f64
                / dev_reads.max(1) as f64,
            dev_queue_mean: self.in_flight_sum.load(Ordering::Relaxed) as f64 / samples as f64,
            dev_queue_max: self.in_flight_max.load(Ordering::Relaxed),
        }
    }
}

impl ProbeReport {
    /// One `key=value` line that benchmark logs can be summarized from.
    pub fn line(&self, query_idx: usize, format: impl std::fmt::Display) -> String {
        let first_row = self
            .first_row
            .iter()
            .map(|delay| format!("{:.2}", delay.as_secs_f64() * 1e3))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "probe query={query_idx} format={format} first_row_ms={first_row} wall_ms={} cpu_ms={} \
             proc_read_bytes={} proc_read_syscalls={} dev_reads={} dev_read_bytes={} \
             dev_read_ms_mean={:.3} dev_queue_mean={:.2} dev_queue_max={}",
            self.wall.as_millis(),
            self.cpu.as_millis(),
            self.proc_read_bytes,
            self.proc_read_syscalls,
            self.dev_reads,
            self.dev_read_bytes,
            self.dev_read_ms_mean,
            self.dev_queue_mean,
            self.dev_queue_max,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_diskstats() {
        let text =
            " 259 0 nvme0n1 10 1 200 30 5 0 8 2 7 40 50\n 259 1 nvme0n1p1 1 0 2 3 0 0 0 0 0 1 1\n";
        assert_eq!(parse_diskstats(text, "nvme0n1"), Some((10, 200, 30)));
        assert_eq!(parse_diskstats(text, "sda"), None);
    }

    #[test]
    fn inflight_reads_come_first() {
        assert_eq!(parse_inflight_reads("       2       12\n"), Some(2));
    }

    #[test]
    fn parses_cpu_ticks_past_spaces_in_the_name() {
        let text = "1 (a b) S 0 0 0 0 0 0 0 0 0 0 12 30 0 0";
        assert_eq!(parse_cpu_ticks(text), Some(42));
    }
}

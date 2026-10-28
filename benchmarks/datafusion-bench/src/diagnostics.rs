// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Write;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::sync::LazyLock;
use std::time::Duration;
#[cfg(target_os = "linux")]
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ipc::writer::FileWriter;
use parking_lot::Mutex;
use serde_json::json;
use url::Url;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_bench::Format;

pub struct Diagnostics {
    output: Mutex<Option<File>>,
    results: Option<PathBuf>,
    files: Vec<PathBuf>,
    cold_delay: Duration,
    iterations: Mutex<HashMap<String, usize>>,
}

/// Work performed outside the query timer to establish verified cold file pages.
#[derive(Clone, Copy)]
pub struct ColdPreparation {
    pages: usize,
    pacing: Duration,
    elapsed: Duration,
    mapping_release: Duration,
    mappings_released: usize,
}

impl Diagnostics {
    pub fn new(output: Option<&Path>, results: Option<PathBuf>, cold: bool,
        cold_delay: Duration, base: &Url, formats: &[Format]) -> anyhow::Result<Self> {
        let mut files = Vec::new();
        if cold {
            anyhow::ensure!(cfg!(target_os = "linux"), "Cold-cache eviction requires Linux");
            let base = base.to_file_path().map_err(|_| anyhow::anyhow!("Cold runs require local data"))?;
            for format in formats {
                collect_files(&base.join(format.name()), format.ext(), &mut files)?;
            }
            anyhow::ensure!(!files.is_empty(), "No data files found to evict");
        }
        if let Some(path) = &results { std::fs::create_dir_all(path)?; }
        Ok(Self { output: Mutex::new(output.map(File::create).transpose()?), results, files, cold_delay,
            iterations: Mutex::default() })
    }

    pub fn prepare(&self) -> anyhow::Result<Option<ColdPreparation>> {
        if self.files.is_empty() { return Ok(None); }
        #[cfg(target_os = "linux")]
        {
            let pacing = if self.cold_delay.is_zero() {
                Duration::ZERO
            } else {
                let started = Instant::now();
                std::thread::sleep(self.cold_delay);
                started.elapsed()
            };
            let started = Instant::now();
            #[cfg(feature = "io-uring")]
            let mappings_released = datafusion_bench::uring::release_mappings();
            #[cfg(not(feature = "io-uring"))]
            let mappings_released = 0;
            let mapping_release = started.elapsed();
            let mut pages = 0;
            for path in &self.files { pages += evict_file(path)?; }
            Ok(Some(ColdPreparation {
                pages, pacing, elapsed: started.elapsed(), mapping_release, mappings_released,
            }))
        }
        #[cfg(not(target_os = "linux"))]
        anyhow::bail!("Cold-cache eviction requires Linux")
    }

    #[expect(clippy::too_many_arguments)]
    pub fn record(&self, query: usize, format: Format, time: Duration, before: Snapshot,
        after: Snapshot, cold: Option<ColdPreparation>, batches: &[RecordBatch], schema: &SchemaRef)
        -> anyhow::Result<()> {
        let key = format!("{}-q{query}", format.name());
        let iteration = {
            let mut iterations = self.iterations.lock();
            let count = iterations.entry(key.clone()).or_default();
            let result = *count;
            *count += 1;
            result
        };
        if let Some(output) = self.output.lock().as_mut() {
            let row = json!({
                "query": query, "format": format.name(), "iteration": iteration,
                "seconds": time.as_secs_f64(),
                "snapshot_before_unix_ns": before.unix_ns,
                "snapshot_after_unix_ns": after.unix_ns,
                "process_stats": after.process.zip(before.process).map(|(after, before)| after.delta(before)),
                "rows": batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                "proc_read_bytes": after.read_bytes.zip(before.read_bytes).map(|(a,b)| a.saturating_sub(b)),
                "logical_reads": after.logical_reads.saturating_sub(before.logical_reads),
                "logical_bytes": after.logical_bytes.saturating_sub(before.logical_bytes),
                "physical_reads": after.physical_reads.saturating_sub(before.physical_reads),
                "physical_bytes": after.physical_bytes.saturating_sub(before.physical_bytes),
                "resident_reads": after.resident_reads.saturating_sub(before.resident_reads),
                "resident_bytes": after.resident_bytes.saturating_sub(before.resident_bytes),
                "verified_cold_pages": cold.map(|preparation| preparation.pages),
                "cold_pacing_seconds": cold.map(|preparation| preparation.pacing.as_secs_f64()),
                "cold_prepare_seconds": cold.map(|preparation| preparation.elapsed.as_secs_f64()),
                "cold_mapping_release_seconds": cold.map(|preparation| preparation.mapping_release.as_secs_f64()),
                "cold_mappings_released": cold.map(|preparation| preparation.mappings_released),
                "cold_cycle_seconds": cold.map(|preparation| (preparation.elapsed + time).as_secs_f64()),
                "cold_iteration_seconds": cold.map(|preparation| (preparation.pacing + preparation.elapsed + time).as_secs_f64()),
            });
            serde_json::to_writer(&mut *output, &row)?;
            writeln!(output)?;
            output.flush()?;
        }
        if iteration == 0 && let Some(directory) = &self.results {
            let mut writer = FileWriter::try_new(File::create(directory.join(format!("{key}.arrow")))?, schema)?;
            for batch in batches { writer.write(batch)?; }
            writer.finish()?;
        }
        Ok(())
    }
}

fn collect_files(directory: &Path, extension: &str, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            collect_files(&entry.path(), extension, files)?;
        } else if entry.path().extension().is_some_and(|value| value == extension) {
            files.push(entry.path());
        }
    }
    Ok(())
}

#[derive(Default)]
pub struct Snapshot {
    unix_ns: Option<u64>,
    process: Option<ProcessStats>,
    read_bytes: Option<u64>,
    logical_reads: u64,
    logical_bytes: u64,
    physical_reads: u64,
    physical_bytes: u64,
    resident_reads: u64,
    resident_bytes: u64,
}

// RUSAGE_SELF sums threads in this process, rather than measuring the whole machine. The two
// snapshots also bracket their small bookkeeping overhead, so they supplement the query timer.
#[derive(Clone, Copy)]
struct ProcessStats {
    user_us: u64,
    system_us: u64,
    minor_faults: u64,
    major_faults: u64,
    voluntary_switches: u64,
    involuntary_switches: u64,
}

impl ProcessStats {
    #[cfg(target_os = "linux")]
    fn take() -> Option<Self> {
        static ENABLED: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("VORTEX_BENCH_PROCESS_STATS").as_deref() == Ok("1")
        });
        if !*ENABLED {
            return None;
        }
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage writes a complete rusage to this valid output pointer on success.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: the successful call above initialized the entire structure.
        let usage = unsafe { usage.assume_init() };
        let micros = |time: libc::timeval| -> Option<u64> {
            u64::try_from(time.tv_sec).ok()?.checked_mul(1_000_000)?
                .checked_add(u64::try_from(time.tv_usec).ok()?)
        };
        Some(Self {
            user_us: micros(usage.ru_utime)?,
            system_us: micros(usage.ru_stime)?,
            minor_faults: u64::try_from(usage.ru_minflt).ok()?,
            major_faults: u64::try_from(usage.ru_majflt).ok()?,
            voluntary_switches: u64::try_from(usage.ru_nvcsw).ok()?,
            involuntary_switches: u64::try_from(usage.ru_nivcsw).ok()?,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn take() -> Option<Self> {
        None
    }

    fn delta(self, before: Self) -> serde_json::Value {
        json!({
            "user_us": self.user_us.saturating_sub(before.user_us),
            "system_us": self.system_us.saturating_sub(before.system_us),
            "minor_faults": self.minor_faults.saturating_sub(before.minor_faults),
            "major_faults": self.major_faults.saturating_sub(before.major_faults),
            "voluntary_switches": self.voluntary_switches.saturating_sub(before.voluntary_switches),
            "involuntary_switches": self.involuntary_switches.saturating_sub(before.involuntary_switches),
        })
    }
}

impl Snapshot {
    pub fn take() -> Self {
        let unix_ns = SystemTime::now().duration_since(UNIX_EPOCH).ok()
            .and_then(|duration| u64::try_from(duration.as_nanos()).ok());
        let process = ProcessStats::take();
        let read_bytes = std::fs::read_to_string("/proc/self/io").ok().and_then(|value| {
            value.lines().find_map(|line| line.strip_prefix("read_bytes: ")?.trim().parse().ok())
        });
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        {
            let stats = datafusion_bench::uring::snapshot();
            Self { unix_ns, process, read_bytes, logical_reads: stats.logical_reads, logical_bytes: stats.logical_bytes,
                physical_reads: stats.physical_reads, physical_bytes: stats.physical_bytes,
                resident_reads: stats.resident_reads, resident_bytes: stats.resident_bytes }
        }
        #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
        Self { unix_ns, process, read_bytes, ..Self::default() }
    }
}

#[cfg(target_os = "linux")]
fn evict_file(path: &Path) -> anyhow::Result<usize> {
    let file = File::open(path)?;
    let length = usize::try_from(file.metadata()?.len())?;
    if length == 0 { return Ok(0); }
    // SAFETY: sysconf has no pointer arguments.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    anyhow::ensure!(page_size > 0, "Could not determine page size");
    let pages = length.div_ceil(page_size as usize);
    // SAFETY: mapping a valid file with PROT_NONE does not access or fault in data. We never
    // dereference the mapping; it is used only to query residency and is then unmapped.
    let mapping = unsafe { libc::mmap(std::ptr::null_mut(), length, libc::PROT_NONE,
        libc::MAP_SHARED, file.as_raw_fd(), 0) };
    if mapping == libc::MAP_FAILED { return Err(std::io::Error::last_os_error().into()); }
    let mut residency = vec![0u8; pages];
    // DONTNEED is advisory and does not guarantee immediate eviction. Retry outside the
    // timed region, requiring the same zero-residency condition on every run.
    let evicted = (|| -> anyhow::Result<()> {
        let mut resident = pages;
        for _ in 0..100 {
            // SAFETY: the descriptor remains open; this only advises on its cache pages.
            let error = unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED)
            };
            if error != 0 { return Err(std::io::Error::from_raw_os_error(error).into()); }
            // SAFETY: the mapping is page-aligned and alive, and residency has one byte per page.
            let result = unsafe { libc::mincore(mapping, length, residency.as_mut_ptr()) };
            if result != 0 { return Err(std::io::Error::last_os_error().into()); }
            resident = residency.iter().filter(|value| **value & 1 != 0).count();
            if resident == 0 { return Ok(()); }
            std::thread::sleep(Duration::from_millis(1));
        }
        anyhow::bail!("{} still has {resident}/{pages} resident pages after eviction", path.display())
    })();
    // SAFETY: this is exactly the mapping and length returned above, with no remaining users.
    let unmap = unsafe { libc::munmap(mapping, length) };
    evicted?;
    if unmap != 0 { return Err(std::io::Error::last_os_error().into()); }
    Ok(pages)
}

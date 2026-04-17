//! Local-file scan using a blocking threadpool and (on Linux) O_DIRECT.
//!
//! The file is split into fixed-size chunks. Each worker thread holds a single
//! aligned buffer (reused across chunks), pulls the next unread chunk offset
//! from an atomic counter, issues a blocking `pread`, runs the dot-product
//! kernel over the freshly read bytes, and absorbs per-vector scores into its
//! local sink. Sinks merge at the end.
//!
//! On macOS we fall back to a regular buffered open and use `F_NOCACHE` as a
//! best-effort page-cache bypass. The resulting numbers there will be cache-
//! warmed after the first pass - documented in the README.

use std::fs::File;
use std::fs::OpenOptions;
use std::os::unix::fs::FileExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Instant;

use aligned_vec::AVec;
use aligned_vec::ConstAlign;
use anyhow::Context;
use anyhow::Result;

use crate::generate::ALIGN;
use crate::kernel::DotKernel;
use crate::kernel::ScanSink;
use crate::kernel::scan_block;
use crate::metrics::CpuSampler;
use crate::metrics::IterationResult;

/// Configuration for a single scan-local run.
pub struct LocalScanConfig {
    pub path: std::path::PathBuf,
    pub dim: usize,
    pub query: Vec<f32>,
    pub threads: usize,
    pub chunk_bytes: usize,
    pub kernel: DotKernel,
    pub direct: bool,
}

/// Open a file handle suitable for O_DIRECT-style access. On Linux we set
/// O_DIRECT; on macOS we hint F_NOCACHE; elsewhere we fall back to buffered.
fn open_direct(path: &Path, direct: bool) -> Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true);

    #[cfg(target_os = "linux")]
    {
        if direct {
            // 0o40000 is O_DIRECT. Not all filesystems support it (tmpfs will
            // reject the open), so we fall back if the open fails.
            opts.custom_flags(libc::O_DIRECT);
            match opts.open(path) {
                Ok(f) => return Ok(f),
                Err(e) => {
                    eprintln!(
                        "warning: O_DIRECT open failed ({}); falling back to buffered",
                        e
                    );
                }
            }
            // Reset flags and retry without O_DIRECT.
            opts = OpenOptions::new();
            opts.read(true);
        }
    }

    let file = opts
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;

    #[cfg(target_os = "macos")]
    if direct {
        use std::os::unix::io::AsRawFd;
        // F_NOCACHE=48 hints the kernel to bypass the unified buffer cache.
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        if rc != 0 {
            eprintln!("warning: F_NOCACHE failed on macOS, results will be cache-warmed");
        }
    }
    let _ = direct;
    Ok(file)
}

/// Run a single pass over the corpus and return measurement data.
pub fn run_once(cfg: &LocalScanConfig) -> Result<IterationResult> {
    anyhow::ensure!(cfg.dim > 0, "dim must be positive");
    anyhow::ensure!(cfg.threads > 0, "threads must be positive");
    anyhow::ensure!(cfg.chunk_bytes >= ALIGN, "chunk_bytes must be >= {ALIGN}");
    anyhow::ensure!(
        cfg.chunk_bytes.is_multiple_of(ALIGN),
        "chunk_bytes must be {ALIGN}-aligned"
    );
    let bytes_per_vec = cfg.dim * std::mem::size_of::<f32>();
    anyhow::ensure!(
        cfg.chunk_bytes.is_multiple_of(bytes_per_vec),
        "chunk_bytes ({}) must be a multiple of vector size ({})",
        cfg.chunk_bytes,
        bytes_per_vec
    );
    anyhow::ensure!(cfg.query.len() == cfg.dim, "query dim mismatch");

    let file = Arc::new(open_direct(&cfg.path, cfg.direct)?);
    let total_bytes = file.metadata().context("stat corpus file")?.len();
    anyhow::ensure!(
        total_bytes.is_multiple_of(bytes_per_vec as u64),
        "corpus file length {} is not a multiple of vector size {}",
        total_bytes,
        bytes_per_vec
    );

    let cursor = Arc::new(AtomicU64::new(0));
    let kernel = cfg.kernel;
    let dim = cfg.dim;
    let chunk_bytes = cfg.chunk_bytes;
    let query = Arc::new(cfg.query.clone());

    let cpu = CpuSampler::new();
    let t0 = Instant::now();
    let handles: Vec<_> = (0..cfg.threads)
        .map(|_| {
            let file = Arc::clone(&file);
            let cursor = Arc::clone(&cursor);
            let query = Arc::clone(&query);
            thread::spawn(move || -> Result<(ScanSink, Vec<u64>)> {
                // Aligned, reused, per-thread buffer.
                let mut buf: AVec<u8, ConstAlign<ALIGN>> = AVec::with_capacity(ALIGN, chunk_bytes);
                buf.resize(chunk_bytes, 0);

                let mut sink = ScanSink::new();
                let mut latencies = Vec::<u64>::new();

                loop {
                    let offset = cursor.fetch_add(chunk_bytes as u64, Ordering::Relaxed);
                    if offset >= total_bytes {
                        break;
                    }
                    let want = ((total_bytes - offset) as usize).min(chunk_bytes);
                    let start = Instant::now();
                    // For O_DIRECT we must only read in multiples of ALIGN. If
                    // the final chunk is smaller than chunk_bytes but still >=
                    // ALIGN and the corpus is vector-aligned (enforced above),
                    // we still need to round the read size UP to ALIGN and
                    // then trim. We over-allocated buf[chunk_bytes] so there's
                    // room. If `want` is not a multiple of ALIGN we read the
                    // rounded-up size and only consider the first `want`
                    // bytes.
                    let read_size = want.div_ceil(ALIGN) * ALIGN;
                    let read_size = read_size.min(chunk_bytes);
                    let slice = &mut buf[..read_size];
                    file.read_at(slice, offset)
                        .with_context(|| format!("pread at offset {} len {}", offset, read_size))?;
                    latencies.push(start.elapsed().as_micros() as u64);

                    // Reinterpret the read region as f32. The buffer is 4 KB
                    // aligned and dim * 4 divides chunk_bytes, so every vector
                    // is intact and f32-aligned.
                    let bytes = &buf[..want];
                    // SAFETY: buf is 4 KB aligned -> 4-byte aligned, `want` is
                    // a multiple of 4 (it is a multiple of dim * 4).
                    let floats: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            bytes.as_ptr() as *const f32,
                            bytes.len() / std::mem::size_of::<f32>(),
                        )
                    };
                    scan_block(kernel, &query, floats, dim, &mut sink);
                }
                Ok((sink, latencies))
            })
        })
        .collect();

    let mut combined = ScanSink::new();
    let mut all_latencies = Vec::<u64>::new();
    for h in handles {
        let (sink, lat) = h.join().expect("worker panicked")?;
        combined.merge(&sink);
        all_latencies.extend_from_slice(&lat);
    }
    let elapsed = t0.elapsed();
    let cpu_percent = cpu.finish();

    // Defeat DCE: the sink values propagate up into IterationResult which the
    // caller prints, but add a black-box for good measure.
    std::hint::black_box(combined.sum);
    std::hint::black_box(combined.max);

    Ok(IterationResult {
        elapsed,
        bytes: total_bytes,
        vectors: total_bytes / bytes_per_vec as u64,
        chunk_latencies_us: all_latencies,
        cpu_percent,
        sink_sum: combined.sum,
        sink_max: combined.max,
    })
}

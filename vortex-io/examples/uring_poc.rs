// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures random reads across two files through a plan's `SourcesIoService`, served three ways:
//!
//! - `serial`: develop's `ReadAtIoSource`, one blocking read per `wait()`.
//! - `pool`: `ReadAtIoService` over `FileReadAt`, each read on a blocking-pool thread.
//! - `uring`: files opened with `IoTarget::Open(UringOpenable)` on one shared ring, served by the
//!   default `ReadAtIoService`.
//!
//! Each round submits one batch of reads, half to each file, and waits for all of them, the way a
//! planner's `NeedsIO` round does. Cold runs drop both files from the page cache first.
//!
//! ```text
//! cargo run --release -p vortex-io --features io-uring --example uring_poc -- [dir] [MiB per file]
//! ```

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::VortexReadAt;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_io::request::OpenTarget;
use vortex_io::request::ReadAtIoService;
use vortex_io::request::ReadAtIoSource;
use vortex_io::request::SourceId;
use vortex_io::request::SourcesIoService;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::std_file::FileReadAt;
use vortex_io::uring::UringOpenable;
use vortex_io::uring::UringRing;

const READ: usize = 4096;
const READS: usize = 4096;

/// Develop's blocking source as a service, one fresh source per session.
struct SerialService {
    read: Arc<dyn VortexReadAt>,
    runtime: Arc<CurrentThreadRuntime>,
}

impl IoService for SerialService {
    fn session(&self) -> Arc<dyn IoSource> {
        Arc::new(ReadAtIoSource::new(
            Arc::clone(&self.read),
            Arc::clone(&self.runtime),
        ))
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Serial,
    Pool,
    Uring,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Serial => "serial (ReadAtIoSource)",
            Mode::Pool => "pool (ReadAtIoService + FileReadAt)",
            Mode::Uring => "uring (UringOpenable, one ring)",
        }
    }
}

fn fetch(source: SourceId, id: u32, target: IoTarget) -> IoRequest {
    IoRequest {
        source,
        intent: IoIntent::Fetch,
        request: IoRequestId(id),
        target,
    }
}

fn write_file(path: &Path, mib: usize, seed: u8) -> VortexResult<()> {
    if path.metadata().is_ok_and(|m| m.len() == (mib << 20) as u64) {
        return Ok(());
    }
    let mut file = File::create(path)?;
    let chunk: Vec<u8> = (0..1usize << 20)
        .map(|i| (i as u8).wrapping_mul(31) ^ seed)
        .collect();
    for _ in 0..mib {
        file.write_all(&chunk)?;
    }
    file.sync_all()?;
    Ok(())
}

fn drop_from_page_cache(path: &Path) -> VortexResult<()> {
    let file = File::open(path)?;
    // SAFETY: plain syscall on an open descriptor.
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if rc != 0 {
        vortex_bail!("posix_fadvise: {}", std::io::Error::from_raw_os_error(rc));
    }
    Ok(())
}

/// Deterministic pseudo-random 4 KiB-aligned offsets.
fn offsets(file_len: u64) -> Vec<u64> {
    let blocks = file_len / READ as u64;
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..READS)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % blocks) * READ as u64
        })
        .collect()
}

struct Run {
    elapsed: Duration,
    open: Duration,
    submits: Option<u64>,
    /// Completions the polling thread reaped itself, of all completions.
    inline: Option<f64>,
}

fn run(mode: Mode, paths: &[PathBuf; 2], batch: usize, cold: bool) -> VortexResult<Run> {
    if cold {
        paths.iter().try_for_each(|p| drop_from_page_cache(p))?;
    }
    let runtime = Arc::new(CurrentThreadRuntime::new());
    let sources = SourcesIoService::new();
    let session = sources.session();
    let owner = IoOwnerId(1);
    let ring = UringRing::new(256)?;

    let started = Instant::now();
    let ids: Vec<SourceId> = match mode {
        Mode::Serial | Mode::Pool => paths
            .iter()
            .map(|path| -> VortexResult<SourceId> {
                let read: Arc<dyn VortexReadAt> =
                    Arc::new(FileReadAt::open(path, runtime.handle())?);
                let service: Arc<dyn IoService> = if mode == Mode::Serial {
                    Arc::new(SerialService {
                        read,
                        runtime: Arc::clone(&runtime),
                    })
                } else {
                    Arc::new(ReadAtIoService::new(read))
                };
                Ok(sources.register(service))
            })
            .collect::<VortexResult<_>>()?,
        Mode::Uring => {
            let opens = paths
                .iter()
                .enumerate()
                .map(|(i, path)| {
                    let path = path.to_str().ok_or_else(|| vortex_err!("non-UTF-8 path"))?;
                    let open = OpenTarget::new(UringOpenable::new(ring.clone(), path));
                    Ok(fetch(sources.id(), i as u32, IoTarget::Open(open)))
                })
                .collect::<VortexResult<Vec<_>>>()?;
            session.submit(owner, opens)?;
            let mut ids = [None, None];
            for _ in 0..2 {
                let completion = session.wait()?;
                let IoResult::Opened(id) = completion.result? else {
                    vortex_bail!("open answered without an id");
                };
                ids[completion.request.0 as usize] = Some(id);
            }
            ids.into_iter().flatten().collect()
        }
    };
    let open = started.elapsed();

    let file_len = paths[0].metadata()?.len();
    let offsets = offsets(file_len);
    let before = ring.stats();
    let started = Instant::now();
    for (round, chunk) in offsets.chunks(batch).enumerate() {
        let requests = chunk
            .iter()
            .enumerate()
            .map(|(i, offset)| {
                let id = (round * batch + i) as u32;
                fetch(
                    ids[i % 2],
                    id,
                    IoTarget::Range {
                        offset: *offset,
                        len: READ,
                        alignment: Alignment::new(READ),
                    },
                )
            })
            .collect();
        session.submit(owner, requests)?;
        for _ in 0..chunk.len() {
            let completion = session.wait()?;
            let IoResult::Bytes(bytes) = completion.result? else {
                vortex_bail!("read answered without bytes");
            };
            if bytes.len() != READ {
                vortex_bail!("short read");
            }
        }
    }
    let elapsed = started.elapsed();
    let after = ring.stats();
    Ok(Run {
        elapsed,
        open,
        submits: (mode == Mode::Uring).then(|| after.submits - before.submits),
        inline: (mode == Mode::Uring).then(|| {
            let inline = after.reaped_inline - before.reaped_inline;
            let total = inline + after.reaped_by_reaper - before.reaped_by_reaper;
            inline as f64 / total.max(1) as f64
        }),
    })
}

fn main() -> VortexResult<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().unwrap_or_else(|| "target/uring-poc".into()));
    let mib: usize = args
        .next()
        .map(|a| a.parse())
        .transpose()
        .map_err(|e| vortex_err!("MiB per file: {e}"))?
        .unwrap_or(1024);
    std::fs::create_dir_all(&dir)?;
    let paths = [dir.join("a.bin"), dir.join("b.bin")];
    for (seed, path) in paths.iter().enumerate() {
        write_file(path, mib, seed as u8)?;
    }

    println!(
        "{READS} random {READ}-byte reads across two {mib} MiB files, batches split between them\n"
    );
    println!("| cache | batch | mode | reads/s | total ms | open µs | io_uring submits | reaped inline |");
    println!("|---|---:|---|---:|---:|---:|---:|---:|");
    for cold in [true, false] {
        for batch in [1, 16, 64, 256] {
            for mode in [Mode::Serial, Mode::Pool, Mode::Uring] {
                // A warm-up pass, so warm runs read from the page cache.
                if !cold {
                    run(mode, &paths, batch, false)?;
                }
                let r = run(mode, &paths, batch, cold)?;
                println!(
                    "| {} | {batch} | {} | {:.0} | {:.1} | {} | {} | {} |",
                    if cold { "cold" } else { "warm" },
                    mode.name(),
                    READS as f64 / r.elapsed.as_secs_f64(),
                    r.elapsed.as_secs_f64() * 1e3,
                    r.open.as_micros(),
                    r.submits.map_or("".into(), |s| s.to_string()),
                    r.inline.map_or("".into(), |i| format!("{:.0}%", i * 100.0)),
                );
            }
        }
    }
    Ok(())
}

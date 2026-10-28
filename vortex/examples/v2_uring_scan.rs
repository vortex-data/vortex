// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare full V1 and V2 scans with the existing reader, its completion adapter, and worker-local
//! io_uring. Usage: v2_uring_scan MODE ITERATIONS WORKERS DIRECTORY [QUEUE_DEPTH].
//! Modes: v1, legacy (V2), adapter, uring, uring-unmerged, direct, direct-unmerged.
//! Every column is converted to Arrow, forcing decompression. Opening/footer reads are excluded.

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("v2_uring_scan requires Linux");
}

#[cfg(target_os = "linux")]
mod linux {
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use anyhow::Context;
    use anyhow::ensure;
    use arrow_array::RecordBatch;
    use arrow_array::cast::AsArray;
    use futures::SinkExt;
    use futures::StreamExt;
    use futures::TryStreamExt;
    use futures::channel::mpsc;
    use futures::stream::BoxStream;
    use serde_json::json;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::VortexSessionExecute;
    use vortex::arrow::ArrowSessionExt;
    use vortex::error::VortexResult;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::io::VortexReadAt;
    use vortex::io::read_at2::FutureReadAt2;
    use vortex::io::read_at2::ReadAt2IoService;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::Handle;
    use vortex::io::runtime::tokio::TokioRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::io::std_file::FileReadAt;
    use vortex::io::uring::FileMode;
    use vortex::io::uring::UringFile;
    use vortex::io::uring::UringOptions;
    use vortex::io::uring::UringStats;
    use vortex::io::uring::UringWorkerPool;
    use vortex::layout::scan::v2::ChunkPlanner;
    use vortex::layout::scan::v2::PreparedScan;
    use vortex::layout::scan::v2::ScanBuilder;
    use vortex::scan::planning::driver::Progress;
    use vortex::scan::planning::driver::Run;
    use vortex::scan::planning::planner::WorkScope;
    use vortex::session::VortexSession;
    use vortex::utils::parallelism::get_available_parallelism;

    #[global_allocator]
    static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

    #[tokio::main]
    pub async fn run() -> anyhow::Result<()> {
        let args: Vec<_> = env::args().collect();
        ensure!(
            args.len() >= 5,
            "usage: v2_uring_scan MODE ITERATIONS WORKERS DIRECTORY [QUEUE_DEPTH]"
        );
        let mode = &args[1];
        ensure!(
            [
                "v1",
                "legacy",
                "adapter",
                "uring",
                "uring-unmerged",
                "direct",
                "direct-unmerged"
            ]
            .contains(&mode.as_str()),
            "Unknown mode {mode}"
        );
        let iterations: usize = args[2].parse()?;
        let workers: usize = args[3].parse()?;
        ensure!(iterations > 0, "Iterations must be positive");
        ensure!(
            Some(workers) == get_available_parallelism(),
            "Set process affinity with taskset so all backends have the requested CPU count"
        );
        let directory = PathBuf::from(&args[4]);
        let queue_depth = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(64);
        let runtime = TokioRuntime::new(tokio::runtime::Handle::current());
        let session = VortexSession::default().with_handle(runtime.handle());
        let pool = if mode != "legacy" && mode != "v1" {
            Some(UringWorkerPool::new(
                workers,
                UringOptions {
                    queue_depth,
                    coalesce: !mode.ends_with("unmerged"),
                    ..Default::default()
                },
            )?)
        } else {
            None
        };
        let mut paths = fs::read_dir(&directory)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        paths.retain(|p| p.extension().is_some_and(|extension| extension == "vortex"));
        paths.sort();
        ensure!(
            !paths.is_empty(),
            "No Vortex files in {}",
            directory.display()
        );
        let file_bytes: u64 = paths
            .iter()
            .map(|p| fs::metadata(p).map(|m| m.len()))
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .sum();
        let mut files = Vec::new();
        for path in &paths {
            let file = session
                .open_options()
                .open_path(path)
                .await
                .with_context(|| format!("opening {}", path.display()))?;
            let native = if mode.starts_with("uring") || mode.starts_with("direct") {
                Some(UringFile::open(
                    path,
                    if mode.starts_with("direct") {
                        FileMode::Direct
                    } else {
                        FileMode::Buffered
                    },
                )?)
            } else {
                None
            };
            let old: Arc<dyn VortexReadAt> = Arc::new(FileReadAt::open(path, session.handle())?);
            files.push((path, file, native, old));
        }
        for iteration in 0..iterations {
            let read_bytes_before = process_read_bytes()?;
            let start = Instant::now();
            let mut rows = 0u64;
            let mut batches = 0u64;
            let mut decoded_bytes = 0usize;
            let mut io = UringStats::default();
            for (path, file, native, old) in &files {
                let before = native.as_ref().map(UringFile::stats).unwrap_or_default();
                let schema = Arc::new(session.arrow().to_arrow_schema(file.dtype())?);
                let stream = if mode == "v1" {
                    file.scan()?
                        .with_concurrency(1)
                        .into_record_batch_stream(schema)?
                        .boxed()
                } else {
                    let mut scan = ScanBuilder::new(session.clone(), file.scan_file());
                    if let Some(native) = native {
                        scan =
                            scan.with_io_service(Arc::new(ReadAt2IoService::new(native.factory())));
                    } else if mode == "adapter" {
                        let old = Arc::clone(old);
                        scan = scan.with_io_service(Arc::new(ReadAt2IoService::new(Arc::new(
                            move || Ok(Arc::new(FutureReadAt2::new(Arc::clone(&old)))),
                        ))));
                    }
                    let scan = Arc::new(scan.prepare()?);
                    let batches = drive(scan, session.handle(), pool.clone())
                        .map_err(|error| arrow_schema::ArrowError::ExternalError(Box::new(error)));
                    let arrow_session = session.clone();
                    let field = arrow_schema::Field::new_struct("", schema.fields().clone(), false);
                    batches
                        .and_then(move |array| {
                            let mut ctx = arrow_session.create_execution_ctx();
                            let result = arrow_session
                                .arrow()
                                .execute_arrow(array, Some(&field), &mut ctx)
                                .map(|arrow| RecordBatch::from(arrow.as_struct().clone()))
                                .map_err(|error| {
                                    arrow_schema::ArrowError::ExternalError(Box::new(error))
                                });
                            std::future::ready(result)
                        })
                        .boxed()
                };
                futures::pin_mut!(stream);
                let mut file_rows = 0u64;
                while let Some(batch) = stream.next().await {
                    let batch: RecordBatch =
                        batch.with_context(|| format!("scanning {}", path.display()))?;
                    file_rows += batch.num_rows() as u64;
                    decoded_bytes += batch.get_array_memory_size();
                    batches += 1;
                }
                ensure!(
                    file_rows == file.row_count(),
                    "Row count mismatch in {}: {file_rows} != {}",
                    path.display(),
                    file.row_count()
                );
                rows += file_rows;
                let after = native.as_ref().map(UringFile::stats).unwrap_or_default();
                io.logical_reads += after.logical_reads - before.logical_reads;
                io.logical_bytes += after.logical_bytes - before.logical_bytes;
                io.physical_reads += after.physical_reads - before.physical_reads;
                io.physical_bytes += after.physical_bytes - before.physical_bytes;
            }
            let seconds = start.elapsed().as_secs_f64();
            let process_read_bytes = process_read_bytes()?.saturating_sub(read_bytes_before);
            println!(
                "{}",
                json!({"mode": mode, "iteration": iteration, "workers": workers,
                "queue_depth": queue_depth, "directory": directory, "files": files.len(),
                "file_bytes": file_bytes, "rows": rows, "batches": batches,
                "arrow_buffer_bytes": decoded_bytes, "seconds": seconds,
                "proc_read_bytes": process_read_bytes,
                "logical_reads": io.logical_reads, "logical_bytes": io.logical_bytes,
                "physical_reads": io.physical_reads, "physical_bytes": io.physical_bytes})
            );
        }
        Ok(())
    }

    /// The example drives each file's prepared plan on one task; DataFusion owns its own driver.
    fn drive(
        scan: Arc<PreparedScan>,
        handle: Handle,
        workers: Option<UringWorkerPool>,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let (mut send, recv) = mpsc::channel(1);
        let work = move || async move {
            let io = scan.io();
            let mut run = Run::new();
            run.admit(
                Box::new(ChunkPlanner::new(
                    Arc::clone(&scan),
                    scan.rows().clone(),
                    scan.selection().clone(),
                    None,
                )),
                WorkScope {
                    file_ordinal: 0,
                    rows: scan.rows().clone(),
                },
                io.session(),
            );
            loop {
                match run.advance()? {
                    Progress::Batch(batch) => {
                        if send.send(Ok(batch.array)).await.is_err() {
                            return Ok(());
                        }
                    }
                    Progress::Spawn(scope, child) => {
                        run.admit(child, scope, io.session());
                    }
                    Progress::RootDone(_) if run.live_roots() > 0 => {}
                    Progress::RootDone(_) | Progress::Idle => return Ok(()),
                    Progress::Waiting => std::future::poll_fn(|cx| run.poll_completion(cx)).await?,
                }
            }
        };
        let task: futures::future::BoxFuture<'static, VortexResult<()>> = match &workers {
            Some(workers) => workers.spawn_async(work),
            None => Box::pin(handle.spawn(work())),
        };
        let errors = futures::stream::once(task)
            .filter_map(|result| std::future::ready(result.err().map(Err)));
        futures::stream::select(recv, errors)
            .chain(futures::stream::poll_fn(move |_| {
                let _ = &workers;
                std::task::Poll::Ready(None)
            }))
            .boxed()
    }

    fn process_read_bytes() -> anyhow::Result<u64> {
        let io = fs::read_to_string("/proc/self/io")?;
        let value = io
            .lines()
            .find_map(|line| line.strip_prefix("read_bytes:"))
            .context("Missing read_bytes in /proc/self/io")?;
        Ok(value.trim().parse()?)
    }
}

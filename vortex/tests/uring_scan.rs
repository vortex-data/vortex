// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(target_os = "linux")]

#[cfg(test)]
mod tests {
    use std::future::poll_fn;
    use std::sync::Arc;

    use tempfile::tempdir;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::Buffer;
    use vortex::dtype::DType;
    use vortex::error::VortexResult;
    use vortex::expr::and;
    use vortex::expr::eq;
    use vortex::expr::get_item;
    use vortex::expr::gt;
    use vortex::expr::gt_eq;
    use vortex::expr::lit;
    use vortex::expr::lt;
    use vortex::expr::lt_eq;
    use vortex::expr::root;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::VortexReadAt;
    use vortex::io::read_at2::FutureReadAt2;
    use vortex::io::read_at2::ReadAt2IoService;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::tokio::TokioRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::io::std_file::FileReadAt;
    use vortex::io::uring::FileMode;
    use vortex::io::uring::SharedUringPool;
    use vortex::io::uring::UringFile;
    use vortex::io::uring::UringOptions;
    use vortex::io::uring::UringWorkerPool;
    use vortex::layout::scan::v2::ChunkPlanner;
    use vortex::layout::scan::v2::PreparedScan;
    use vortex::layout::scan::v2::PrunePlanner;
    use vortex::layout::scan::v2::ScanBuilder;
    use vortex::scan::planning::driver::Progress;
    use vortex::scan::planning::driver::Run;
    use vortex::scan::planning::planner::Planner;
    use vortex::scan::planning::planner::WorkScope;
    use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
    use vortex::session::VortexSession;

    #[tokio::test]
    async fn v2_uring_preserves_values_nulls_and_sparse_rows() -> VortexResult<()> {
        let runtime = TokioRuntime::new(tokio::runtime::Handle::current());
        let session = VortexSession::default().with_handle(runtime.handle());
        let directory = tempdir()?;
        let path = directory.path().join("scan.vortex");
        let make = |rows: Vec<u64>| -> VortexResult<_> {
            Ok(StructArray::from_fields(&[
                ("id", Buffer::from(rows.clone()).into_array()),
                (
                    "nullable_id",
                    PrimitiveArray::from_option_iter(
                        rows.iter().map(|&row| (row % 3 != 0).then_some(row)),
                    )
                    .into_array(),
                ),
                (
                    "text",
                    VarBinArray::from_iter(
                        rows.iter()
                            .map(|row| (row % 3 != 0).then(|| format!("row {row}"))),
                        DType::Utf8(true.into()),
                    )
                    .into_array(),
                ),
            ])?
            .into_array())
        };
        let all = make((0..70003).collect())?;
        let mut output = tokio::fs::File::create(&path).await?;
        session
            .write_options()
            .write(&mut output, all.to_array_stream())
            .await?;
        drop(output);
        let file = session.open_options().open_path(&path).await?;
        let old: Arc<dyn VortexReadAt> = Arc::new(FileReadAt::open(&path, session.handle())?);
        let mut factories = Vec::new();
        for max_active_jobs in [1, 4] {
            let pool = UringWorkerPool::new(
                1,
                UringOptions {
                    queue_depth: 2,
                    max_active_jobs,
                    ..Default::default()
                },
            )?;
            for mode in [FileMode::Buffered, FileMode::Direct] {
                factories.push((UringFile::open(&path, mode)?.factory(), Some(pool.clone())));
            }
            // SAFETY: the fixture is immutable after writing, through all readers and buffers.
            let mapped = unsafe { UringFile::open_mapped(&path) }?;
            factories.push((mapped.factory(), Some(pool.clone())));
            let old = Arc::clone(&old);
            factories.push((
                Arc::new(move || Ok(Arc::new(FutureReadAt2::new(Arc::clone(&old))) as _)),
                Some(pool),
            ));
        }
        for rings in [1, 2] {
            let reactors = SharedUringPool::new(
                rings,
                UringOptions {
                    reuse_inflight: true,
                    ..Default::default()
                },
            )?;
            for mode in [FileMode::Buffered, FileMode::Direct] {
                factories.push((reactors.factory(UringFile::open(&path, mode)?), None));
            }
            // SAFETY: the fixture is immutable after writing, through all readers and buffers.
            let mapped = unsafe { UringFile::open_mapped(&path) }?;
            factories.push((reactors.factory(mapped), None));
        }
        for max_active_jobs in [1, 4] {
            let pool = UringWorkerPool::new(
                2,
                UringOptions {
                    queue_depth: 2,
                    max_active_jobs,
                    reuse_live_buffers: true,
                    ..Default::default()
                },
            )?;
            // SAFETY: the fixture is immutable through all readers and returned buffers.
            let mapped = unsafe { UringFile::open_mapped(&path) }?;
            factories.push((mapped.factory(), Some(pool)));
        }
        for rings in [1, 2] {
            let reactors = SharedUringPool::new(
                rings,
                UringOptions {
                    queue_depth: 2,
                    reuse_inflight: true,
                    reuse_live_buffers: true,
                    ..Default::default()
                },
            )?;
            // SAFETY: the fixture is immutable through all readers and returned buffers.
            let mapped = unsafe { UringFile::open_mapped(&path) }?;
            factories.push((reactors.factory(mapped), None));
        }
        for (factory, workers) in factories {
            for chunk_rows in [8192, 65536, 262144] {
                for cache_bytes in [0, 8 << 20] {
                    for rows in [
                        (0..70003).collect::<Vec<_>>(),
                        vec![0, 1, 999, 1001, 3000, 65535, 65536, 70002],
                    ] {
                        for predicate in 0..5 {
                            let (filter, kept) = match predicate {
                                0 => (None, rows.clone()),
                                1 => (
                                    Some(and(
                                        gt(get_item("id", root()), lit(999_u64)),
                                        lt(get_item("id", root()), lit(66000_u64)),
                                    )),
                                    rows.iter()
                                        .copied()
                                        .filter(|row| *row > 999 && *row < 66000)
                                        .collect(),
                                ),
                                2 => (
                                    Some(eq(get_item("text", root()), lit("row 1001"))),
                                    rows.iter().copied().filter(|row| *row == 1001).collect(),
                                ),
                                3 => (
                                    Some(and(
                                        gt_eq(get_item("nullable_id", root()), lit(1001_u64)),
                                        lt_eq(get_item("nullable_id", root()), lit(66001_u64)),
                                    )),
                                    rows.iter()
                                        .copied()
                                        .filter(|row| {
                                            *row >= 1001 && *row <= 66001 && *row % 3 != 0
                                        })
                                        .collect(),
                                ),
                                _ => (Some(lt(get_item("id", root()), lit(0_u64))), Vec::new()),
                            };
                            let expected = make(kept)?;
                            let scan = ScanBuilder::new(session.clone(), file.scan_file())
                                .with_io_cache_bytes(cache_bytes)
                                .with_chunk_rows(chunk_rows)
                                .with_io_service(Arc::new(ReadAt2IoService::new(Arc::clone(
                                    &factory,
                                ))))
                                .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(
                                    rows.clone(),
                                ))?)
                                .with_some_filter(
                                    filter.map(|filter| filter.bind(file.dtype())).transpose()?,
                                );
                            let scan = Arc::new(scan.prepare()?);
                            let chunks = if let Some(workers) = &workers {
                                workers.spawn_async(move || collect(scan)).await?
                            } else {
                                collect(scan).await?
                            };
                            let actual = ChunkedArray::try_new(chunks, expected.dtype().clone())?
                                .into_array();
                            assert_arrays_eq!(
                                actual,
                                expected,
                                &mut session.create_execution_ctx()
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn collect(scan: Arc<PreparedScan>) -> VortexResult<Vec<ArrayRef>> {
        let chunks = {
            let scan = Arc::clone(&scan);
            move |kept| -> Box<dyn Planner> {
                Box::new(ChunkPlanner::new(
                    Arc::clone(&scan),
                    scan.rows().clone(),
                    scan.selection().clone(),
                    kept,
                ))
            }
        };
        let root: Box<dyn Planner> = if scan.filter().is_some() {
            Box::new(PrunePlanner::new(
                Arc::clone(&scan),
                scan.rows().clone(),
                Box::new(chunks),
            ))
        } else {
            chunks(None)
        };
        let io = scan.io();
        let mut run = Run::new();
        run.admit(
            root,
            WorkScope {
                file_ordinal: 0,
                rows: scan.rows().clone(),
            },
            io.session(),
        );
        let mut batches = Vec::new();
        loop {
            match run.advance()? {
                Progress::Batch(batch) => batches.push((batch.scope.rows.start, batch.array)),
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, io.session());
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => break,
                Progress::Waiting => poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
        batches.sort_by_key(|(start, _)| *start);
        Ok(batches.into_iter().map(|(_, array)| array).collect())
    }
}

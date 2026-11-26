// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A table-like plan over in-memory files: one planner spawns an [`OpenFilePlanner`] per file, and
//! one driver opens, reads and scans them all through one sources service.

use std::fmt;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_io::VortexReadAt;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_io::request::OpenTarget;
use vortex_io::request::Openable;
use vortex_io::request::SourceId;
use vortex_io::request::SourcesIoService;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_scan::planning::drive::ParallelDriver;
use vortex_scan::planning::drive::PlanDriver;
use vortex_scan::planning::drive::PlanRoot;
use vortex_scan::planning::drive::ScopedBatch;
use vortex_scan::planning::drive::SequentialDriver;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

use super::OpenFilePlanner;
use crate::Footer;
use crate::OpenOptionsSessionExt;
use crate::WriteOptionsSessionExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    crate::register_default_encodings(&session);
    crate::enable_all_registered_array_encodings(&session);
    session
});

const ROWS: i32 = 3_000;

/// The values of file `i`, which it holds in chunks of 1,000 rows.
fn values_of(i: i32) -> Vec<i32> {
    (i * ROWS..(i + 1) * ROWS).collect()
}

async fn write(i: i32) -> VortexResult<ByteBuffer> {
    let chunks = ChunkedArray::from_iter((0..3).map(|chunk| {
        let start = i * ROWS + chunk * 1_000;
        Buffer::from_iter(start..start + 1_000).into_array()
    }));
    let mut buf = ByteBufferMut::empty();
    SESSION
        .write_options()
        .write(&mut buf, chunks.into_array().to_array_stream())
        .await?;
    Ok(buf.freeze())
}

/// A file in memory that counts its size requests and the reads that reach its last byte.
struct File {
    key: Arc<str>,
    bytes: ByteBuffer,
    sizes: Arc<AtomicUsize>,
    tail_reads: Arc<AtomicUsize>,
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("File").field(&self.key).finish()
    }
}

struct CountingRead {
    bytes: ByteBuffer,
    sizes: Arc<AtomicUsize>,
    tail_reads: Arc<AtomicUsize>,
}

impl VortexReadAt for CountingRead {
    fn concurrency(&self) -> usize {
        4
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.sizes.fetch_add(1, Ordering::Relaxed);
        self.bytes.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        if offset + length as u64 == self.bytes.len() as u64 {
            self.tail_reads.fetch_add(1, Ordering::Relaxed);
        }
        self.bytes.read_at(offset, length, alignment)
    }
}

impl Openable for File {
    fn key(&self) -> &Arc<str> {
        &self.key
    }

    fn open(&self) -> BoxFuture<'static, VortexResult<Arc<dyn VortexReadAt>>> {
        let read: Arc<dyn VortexReadAt> = Arc::new(CountingRead {
            bytes: self.bytes.clone(),
            sizes: Arc::clone(&self.sizes),
            tail_reads: Arc::clone(&self.tail_reads),
        });
        async move { Ok(read) }.boxed()
    }
}

/// What a table's manifest would know about a file.
#[derive(Clone, Copy, Debug)]
enum Known {
    Nothing,
    Size,
    Footer,
}

/// Spawns one [`OpenFilePlanner`] per file, a scan of the rows above `above`.
struct TablePlanner {
    sources: SourceId,
    files: Vec<(OpenTarget, u64, Footer)>,
    known: Known,
    above: Option<i32>,
    next: usize,
}

impl IoConsumer for TablePlanner {
    fn set_io_result(&mut self, _: IoRequestId, _: IoResult) {}
}

impl Planner for TablePlanner {
    fn state(&self) -> State {
        if self.next < self.files.len() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let i = self.next;
        self.next += 1;
        let (target, size, footer) = self.files[i].clone();
        let scope = WorkScope {
            file_ordinal: i as u64,
            rows: 0..ROWS as u64,
        };
        let above = self.above;
        let planner = OpenFilePlanner::scan(
            self.sources,
            target,
            SESSION.clone(),
            scope.clone(),
            move |builder| {
                let filter = above.map(|above| gt(root(), lit(above)));
                let dtype = builder.dtype().clone();
                builder.with_some_filter(filter.map(|filter| {
                    filter
                        .bind(&dtype)
                        .unwrap_or_else(|err| panic!("bind failed: {err}"))
                }))
            },
        );
        let planner = match self.known {
            Known::Nothing => planner,
            Known::Size => planner.with_file_size(size),
            Known::Footer => planner.with_file_size(size).with_footer(footer),
        };
        Ok(PlannerOutput::Spawn(scope, Box::new(planner)))
    }
}

struct Table {
    files: Vec<File>,
    footers: Vec<Footer>,
}

impl Table {
    async fn new(files: i32) -> VortexResult<Self> {
        let mut table = Table {
            files: vec![],
            footers: vec![],
        };
        for i in 0..files {
            let bytes = write(i).await?;
            table.footers.push(
                SESSION
                    .open_options()
                    .open_buffer(bytes.clone())?
                    .footer()
                    .clone(),
            );
            table.files.push(File {
                key: format!("mem://table/{i}.vortex").into(),
                bytes,
                sizes: Arc::default(),
                tail_reads: Arc::default(),
            });
        }
        Ok(table)
    }

    fn root(&self, sources: &SourcesIoService, known: Known, above: Option<i32>) -> PlanRoot {
        let files = self
            .files
            .iter()
            .zip(&self.footers)
            .map(|(file, footer)| {
                let target = OpenTarget::new(File {
                    key: Arc::clone(&file.key),
                    bytes: file.bytes.clone(),
                    sizes: Arc::clone(&file.sizes),
                    tail_reads: Arc::clone(&file.tail_reads),
                });
                (target, file.bytes.len() as u64, footer.clone())
            })
            .collect();
        PlanRoot::new(
            WorkScope {
                file_ordinal: 0,
                rows: 0..ROWS as u64 * self.files.len() as u64,
            },
            Box::new(TablePlanner {
                sources: sources.id(),
                files,
                known,
                above,
                next: 0,
            }),
        )
    }

    fn count(&self, counter: impl Fn(&File) -> usize) -> usize {
        self.files.iter().map(counter).sum()
    }
}

fn sorted_values(batches: Vec<ScopedBatch>) -> VortexResult<Vec<i32>> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut values = vec![];
    for batch in batches {
        let array = batch.batch.execute::<PrimitiveArray>(&mut ctx)?;
        values.extend_from_slice(array.as_slice::<i32>());
    }
    values.sort_unstable();
    Ok(values)
}

#[derive(Clone, Copy, Debug)]
enum Driver {
    Sequential,
    Parallel,
}

async fn scan(
    table: &Table,
    driver: Driver,
    known: Known,
    above: Option<i32>,
) -> VortexResult<(Vec<i32>, usize)> {
    let sources = SourcesIoService::new();
    let io: Arc<dyn IoService> = Arc::new(sources.clone());
    let driver: Box<dyn PlanDriver> = match driver {
        Driver::Sequential => Box::new(SequentialDriver::new(io)),
        Driver::Parallel => {
            Box::new(ParallelDriver::new(TokioRuntime::current(), io).with_workers(3))
        }
    };
    let batches: Vec<ScopedBatch> = driver
        .execute(table.root(&sources, known, above))
        .try_collect()
        .await?;
    Ok((sorted_values(batches)?, sources.len()))
}

#[rstest]
#[case(Driver::Sequential, Known::Nothing)]
#[case(Driver::Parallel, Known::Nothing)]
#[case(Driver::Sequential, Known::Size)]
#[case(Driver::Parallel, Known::Footer)]
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn one_driver_opens_and_scans_every_file(
    #[case] driver: Driver,
    #[case] known: Known,
) -> VortexResult<()> {
    let table = Table::new(4).await?;
    let (values, sources) = scan(&table, driver, known, None).await?;
    assert_eq!(values, (0..4).flat_map(values_of).collect::<Vec<_>>());
    assert_eq!(sources, 4, "each file opened to a source of its own");
    let sizes = table.count(|file| file.sizes.load(Ordering::Relaxed));
    let tails = table.count(|file| file.tail_reads.load(Ordering::Relaxed));
    match known {
        Known::Nothing => assert_eq!(sizes, 4),
        Known::Size => assert_eq!(sizes, 0),
        Known::Footer => {
            assert_eq!(sizes, 0);
            assert_eq!(tails, 0, "a known footer is never read");
        }
    }
    Ok(())
}

#[rstest]
#[case(Driver::Sequential)]
#[case(Driver::Parallel)]
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn filters_run_in_every_file(#[case] driver: Driver) -> VortexResult<()> {
    let table = Table::new(3).await?;
    let above = ROWS + 500;
    let (values, _) = scan(&table, driver, Known::Size, Some(above)).await?;
    let expected: Vec<i32> = (0..3).flat_map(values_of).filter(|v| *v > above).collect();
    assert_eq!(values, expected);
    Ok(())
}

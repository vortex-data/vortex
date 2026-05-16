//! `VortexScanSource`: read a column projection from a Vortex file
//! and emit each backing array as a Batch.
//!
//! Async path (preferred): when a [`DriverIo`] is attached to the
//! `SpawnRuntime`, `init_local` spawns a forwarding task onto
//! DriverIo's smol executor. That task opens the file, builds
//! Vortex's async `ArrayStream`, and pushes each batch into a
//! bounded mpsc channel. The lane's `poll_next` just polls the
//! receiver — `Pending` if the next batch hasn't landed yet,
//! `Ready(Some(batch))` when it has. This is what lets I/O overlap
//! with compute on the lane.
//!
//! Blocking fallback: when no `DriverIo` is present (e.g. tests
//! using `LocalInitRuntime::default()`), we fall back to the old
//! `into_array_iter(BlockingRuntime)` path. Same behaviour, but no
//! overlap.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::task::Poll;

use vortex_array::dtype::DType;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_array::iter::ArrayIterator;
use vortex_array::stream::SendableArrayStream;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::Cardinality;
use crate::Domain;
use crate::DomainId;
use crate::DomainSpan;
use crate::EngineError;
use crate::EngineResult;
use crate::OutputContract;
use crate::physical_plan::DriverIo;
use crate::physical_plan::abi::{
    Batch, LocalInitRuntime, OperatorPoll, Parallelism, SourceCtx, SourceNode,
};
use crate::physical_plan::error::BuildResult;
use crate::physical_plan::lowering::{LoweringCtx, LoweringCtxExt, PipelineTail};
use crate::physical_plan::plan::Operator;

/// A source that reads one column of a Vortex file.
pub struct VortexScanSource {
    label: String,
    path: PathBuf,
    column: String,
    output_domain: Domain,
    output_contract: OutputContract,
}

impl VortexScanSource {
    /// Open the file once to discover its row count and the column's
    /// dtype, then return a source over that column.
    pub fn open(
        label: impl Into<String>,
        path: impl AsRef<Path>,
        column: impl Into<String>,
    ) -> EngineResult<Self> {
        let label = label.into();
        let path = path.as_ref().to_path_buf();
        let column = column.into();
        let (row_count, dtype) = probe_file(&path, &column)?;
        let output_domain = Domain::new(
            DomainId::new(format!("{}:{column}", path.display())),
            Cardinality::Exact(row_count),
        );
        let output_contract = OutputContract::new(dtype);
        Ok(Self {
            label,
            path,
            column,
            output_domain,
            output_contract,
        })
    }

    pub fn output_domain(&self) -> &Domain {
        &self.output_domain
    }
    pub fn output_contract(&self) -> &OutputContract {
        &self.output_contract
    }
}

fn default_session() -> VortexSession {
    use vortex::VortexSessionDefault;
    let session = VortexSession::default();
    crate::kernels::install(&session);
    session
}

fn probe_file(path: &Path, column: &str) -> EngineResult<(u64, DType)> {
    let runtime = CurrentThreadRuntime::new();
    let session = default_session().with_handle(runtime.handle());
    let path_buf = path.to_path_buf();
    let session_for_open = session.clone();
    let file: VortexFile = runtime
        .block_on(async move { session_for_open.open_options().open_path(path_buf).await })
        .map_err(|e| EngineError::message(format!("open vortex file: {e}")))?;

    let row_count = file.row_count();
    let scan = file
        .scan()
        .map_err(|e| EngineError::message(format!("scan builder: {e}")))?
        .with_projection(get_item(column, root()));
    let dtype = scan
        .dtype()
        .map_err(|e| EngineError::message(format!("scan dtype: {e}")))?;
    Ok((row_count, dtype))
}

/// Per-lane state. The async variant holds a `SendableArrayStream`
/// whose I/O tasks run on `DriverIo`'s executor; the lane polls this
/// stream directly with the lane's own waker, so wake-ups originate
/// from the I/O thread without an mpsc hop. The blocking variant is
/// the legacy fallback used when no `DriverIo` is attached.
pub enum VortexScanLocal {
    Async {
        // We keep the file alive for the duration of the scan even
        // though `into_array_stream` already references it: file
        // metadata (segment cache) lives on the file, not the stream.
        _file: VortexFile,
        stream: SendableArrayStream,
        rows_read: u64,
    },
    Blocking {
        runtime: Arc<CurrentThreadRuntime>,
        iter: Box<dyn ArrayIterator + Send>,
        rows_read: u64,
    },
}

impl SourceNode for VortexScanSource {
    type LocalState = VortexScanLocal;

    fn label(&self) -> &str {
        &self.label
    }

    fn parallelism(&self) -> Parallelism {
        Parallelism::serial()
    }

    fn init_local(&self, runtime: &mut LocalInitRuntime<'_>) -> EngineResult<Self::LocalState> {
        // Prefer the async path when a SpawnRuntime is attached.
        if let Some(io) = runtime.spawn().map(|s| Arc::clone(s.io())) {
            return self.init_local_async(io);
        }
        self.init_local_blocking()
    }

    fn poll_next(
        &self,
        local: &mut Self::LocalState,
        ctx: &mut SourceCtx<'_, '_>,
    ) -> OperatorPoll<Option<Batch>> {
        match local {
            VortexScanLocal::Async {
                stream, rows_read, ..
            } => match stream.as_mut().poll_next(ctx.cx()) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(None) => Poll::Ready(Ok(None)),
                Poll::Ready(Some(Err(e))) => Poll::Ready(Err(EngineError::message(format!(
                    "vortex scan stream: {e}"
                )))),
                Poll::Ready(Some(Ok(array))) => {
                    let rows = array.len() as u64;
                    let span = DomainSpan::new(*rows_read, rows);
                    *rows_read += rows;
                    Poll::Ready(Ok(Some(Batch::new(array, span))))
                }
            },
            VortexScanLocal::Blocking {
                iter, rows_read, ..
            } => match iter.next() {
                None => Poll::Ready(Ok(None)),
                Some(Err(e)) => Poll::Ready(Err(EngineError::message(format!(
                    "vortex scan iter: {e}"
                )))),
                Some(Ok(array)) => {
                    let rows = array.len() as u64;
                    let span = DomainSpan::new(*rows_read, rows);
                    *rows_read += rows;
                    Poll::Ready(Ok(Some(Batch::new(array, span))))
                }
            },
        }
    }
}

impl VortexScanSource {
    /// Async path: open the file synchronously (one block_on on the
    /// io executor), build Vortex's `ArrayStream`, and store it on
    /// the lane. The lane polls the stream directly — Vortex's
    /// internal I/O tasks are spawned on DriverIo's executor via the
    /// session's `Handle`, and the resulting wake-ups land on the
    /// lane's waker. No mpsc indirection.
    fn init_local_async(&self, io: Arc<DriverIo>) -> EngineResult<VortexScanLocal> {
        let session = default_session().with_handle(io.vortex_handle());
        let path = self.path.clone();
        let session_for_open = session.clone();
        // Run the file open on the io executor; `executor.run(future)`
        // processes other tasks while we wait, but in practice this
        // is the only outstanding work and finishes quickly.
        let file: VortexFile = smol::block_on(io.executor().run(async move {
            session_for_open.open_options().open_path(path).await
        }))
        .map_err(|e| EngineError::message(format!("open vortex file: {e}")))?;
        let scan = file
            .scan()
            .map_err(|e| EngineError::message(format!("scan builder: {e}")))?
            .with_projection(get_item(self.column.as_str(), root()));
        let stream = scan
            .into_array_stream()
            .map_err(|e| EngineError::message(format!("into_array_stream: {e}")))?;
        let stream: SendableArrayStream = Box::pin(stream);

        Ok(VortexScanLocal::Async {
            _file: file,
            stream,
            rows_read: 0,
        })
    }

    /// Blocking fallback: same v0 behaviour as before — open the file
    /// synchronously via a per-lane `CurrentThreadRuntime` and walk
    /// the iterator in `poll_next`.
    fn init_local_blocking(&self) -> EngineResult<VortexScanLocal> {
        let runtime = Arc::new(CurrentThreadRuntime::new());
        let session = default_session().with_handle(runtime.handle());
        let path = self.path.clone();
        let session_for_open = session.clone();
        let file: VortexFile = runtime
            .block_on(async move { session_for_open.open_options().open_path(path).await })
            .map_err(|e| EngineError::message(format!("open vortex file: {e}")))?;

        let scan = file
            .scan()
            .map_err(|e| EngineError::message(format!("scan builder: {e}")))?
            .with_projection(get_item(self.column.as_str(), root()));
        let iter = scan
            .into_array_iter(runtime.as_ref())
            .map_err(|e| EngineError::message(format!("into_array_iter: {e}")))?;

        Ok(VortexScanLocal::Blocking {
            runtime,
            iter: Box::new(iter),
            rows_read: 0,
        })
    }
}

impl Operator for VortexScanSource {
    fn lower(&self, ctx: &mut dyn LoweringCtx, tail: PipelineTail) -> BuildResult<()> {
        ctx.register_domain(self.output_domain.clone())?;
        let source = VortexScanSource {
            label: self.label.clone(),
            path: self.path.clone(),
            column: self.column.clone(),
            output_domain: self.output_domain.clone(),
            output_contract: self.output_contract.clone(),
        };
        ctx.emit_pipeline(
            tail,
            self.output_domain.clone(),
            self.output_contract.clone(),
            source,
        )?;
        Ok(())
    }
}

// Force-assert Send: VortexScanLocal must be Send so the lane state
// can move across the executor's local task boundary. Both variants
// are Send.
fn _assert_send() {
    fn assert_send<T: Send>() {}
    assert_send::<VortexScanLocal>();
}

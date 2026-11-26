// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading an index's content, wherever it is stored.

use std::any::Any;
use std::any::TypeId;
use std::sync::Arc;

use futures::TryStreamExt;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_expr::BoundExpression;
use vortex_expr::Expression;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::scan::v2::ScanFile;
use vortex_scan::execution::driver::ScanDriver;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

/// Run a scan v2 `builder` to its end, collecting its batches in row order.
///
/// Index content and catalog reads are small, so they run on the calling task rather than paying
/// for a hop onto the runtime.
pub async fn collect(builder: ScanBuilder) -> VortexResult<Vec<ArrayRef>> {
    let handle = builder.session().handle();
    ScanDriver::new(handle, 0, true)
        .run(builder)
        .try_collect()
        .await
}

/// One step of an index search: a scan of index rows to read next, or the answer.
///
/// A search is a chain of steps, so it can run anywhere reads can be done: awaited here through
/// [`IndexStep::run`], or as planners whose reads the plan's driver performs, with each `Read`
/// becoming a gathering planner over the index file.
pub enum IndexStep<T> {
    /// Scan these index rows, then continue with their batches, in row order.
    Read(Box<ScanBuilder>, StepFn<T>),
    /// The search's answer.
    Done(T),
}

/// What a lookup found, and how exactly: `None` when the index makes no claim on the predicate.
pub type Located<T> = Option<(IndexExactness, T)>;

/// What a search does with the batches of one [`IndexStep::Read`].
pub type StepFn<T> = Box<dyn FnOnce(Vec<ArrayRef>) -> VortexResult<IndexStep<T>> + Send>;

impl<T: Send + 'static> IndexStep<T> {
    /// Run every step to the answer, reading on the calling task.
    pub async fn run(self) -> VortexResult<T> {
        let mut step = self;
        loop {
            match step {
                Self::Done(answer) => return Ok(answer),
                Self::Read(scan, then) => step = then(collect(*scan).await?)?,
            }
        }
    }

    /// Continue with `f` once this search has its answer.
    pub fn and_then<U: Send + 'static>(
        self,
        f: impl FnOnce(T) -> VortexResult<IndexStep<U>> + Send + 'static,
    ) -> VortexResult<IndexStep<U>> {
        Ok(match self {
            Self::Done(answer) => f(answer)?,
            Self::Read(scan, then) => {
                IndexStep::Read(scan, Box::new(move |batches| then(batches)?.and_then(f)))
            }
        })
    }

    /// Transform this search's answer with `f`.
    pub fn map<U: Send + 'static>(
        self,
        f: impl FnOnce(T) -> VortexResult<U> + Send + 'static,
    ) -> VortexResult<IndexStep<U>> {
        self.and_then(move |answer| Ok(IndexStep::Done(f(answer)?)))
    }
}

/// The sorted keys a query reads, by key type.
///
/// A keyed kind's query is exactly the sorted keys its filter matches, so its rows can be found
/// through a cached copy of the index's key column instead of filtering the whole index.
pub enum QueryKeys {
    /// Keys of a `Utf8` key column.
    Strings(Vec<String>),
    /// Keys of a `u32` key column.
    U32s(Vec<u32>),
}

/// An index kind that [`IndexRows`] can answer predicates for.
///
/// Kinds whose index rows are sorted by a key column report a query's keys, so a lookup reads
/// only their rows. Others keep the default and are read through their plan's filter.
pub trait IndexKind: IndexVTable {
    /// The keys `query` reads, if this kind's rows are sorted by them.
    fn query_keys(&self, query: &Self::Query) -> Option<QueryKeys> {
        let _ = query;
        None
    }
}

/// A scored row returned by a ranked search, in the index's row space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// The row, counted from the start of the indexed data.
    pub row: u64,
    /// Higher is better.
    pub score: f32,
}

/// An opened index: its kind and options, plus a reader over its content.
///
/// The content is the same rows whether the index is a child of a `vortex.indexed` layout inside
/// a data file, or the root of a standalone index file, so filtering and ranked search are
/// implemented once here and shared by both.
pub struct IndexRows<V: IndexVTable> {
    pub(crate) vtable: Arc<V>,
    pub(crate) options: V::Options,
    pub(crate) source_dtype: DType,
    pub(crate) index_dtype: DType,
    /// Rows of indexed data this index covers.
    pub(crate) data_row_count: u64,
    /// The index content's layout, and where its segments are, for scans of it.
    pub(crate) file: ScanFile,
    pub(crate) session: VortexSession,
    /// State an index kind derives once from its content and reuses across searches, such as
    /// the IVF centroids, one value per type.
    pub(crate) memo: Mutex<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

impl<V: IndexVTable> IndexRows<V> {
    /// The index kind's parsed options.
    pub fn options(&self) -> &V::Options {
        &self.options
    }

    /// The dtype of the column the index was built over.
    pub fn source_dtype(&self) -> &DType {
        &self.source_dtype
    }

    /// Rows of indexed data the index covers.
    pub fn data_row_count(&self) -> u64 {
        self.data_row_count
    }

    /// The index kind.
    pub fn vtable(&self) -> &Arc<V> {
        &self.vtable
    }

    /// The dtype of the index rows, as the kind declared it.
    pub fn index_dtype(&self) -> &DType {
        &self.index_dtype
    }

    /// Rows of index content.
    pub fn index_row_count(&self) -> u64 {
        self.file.layout.row_count()
    }

    /// The session the index is read with.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// The value of type `T` kept by [`Self::memo_set`], if any.
    pub fn memo_get<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        self.memo
            .lock()
            .get(&TypeId::of::<T>())
            .and_then(|value| Arc::clone(value).downcast::<T>().ok())
    }

    /// Keep `value` for later searches. A concurrent first search may have kept one already;
    /// both derived it from the same content.
    pub fn memo_set<T: Any + Send + Sync>(&self, value: Arc<T>) {
        self.memo.lock().insert(TypeId::of::<T>(), value);
    }

    /// A scan of the index rows.
    pub fn scan(&self) -> ScanBuilder {
        ScanBuilder::new(self.session.clone(), self.file.clone())
    }

    /// Read and decode the index rows matching `filter`, or every row without one.
    ///
    /// Index content is an ordinary layout, so the filter is pruned by its zone maps before any
    /// segment is fetched. The index kinds in this crate keep each index row self-contained, so
    /// decoding any subset of rows is valid.
    pub async fn chunks(
        self: &Arc<Self>,
        filter: Option<Expression>,
    ) -> VortexResult<Vec<Arc<V::Chunk>>> {
        self.chunks_step(filter)?.run().await
    }

    /// [`chunks`](Self::chunks) as a step.
    pub fn chunks_step(
        self: &Arc<Self>,
        filter: Option<Expression>,
    ) -> VortexResult<IndexStep<Vec<Arc<V::Chunk>>>> {
        let filter = filter.map(|f| f.bind(&self.index_dtype)).transpose()?;
        Ok(self.bound_chunks_step(filter))
    }

    /// Read the index rows at `rows`, strictly ascending, without evaluating a filter.
    ///
    /// For kinds whose key is the row position, so a lookup needs no key column read.
    pub fn chunks_at_step(
        self: &Arc<Self>,
        rows: Vec<u64>,
    ) -> VortexResult<IndexStep<Vec<Arc<V::Chunk>>>> {
        let indices = StrictSortedBuffer::try_new(Buffer::from(rows))?;
        Ok(self.decode_step(self.scan().with_row_indices(indices)))
    }

    fn bound_chunks_step(
        self: &Arc<Self>,
        filter: Option<BoundExpression>,
    ) -> IndexStep<Vec<Arc<V::Chunk>>> {
        self.decode_step(self.scan().with_some_filter(filter))
    }

    /// Read `scan` and decode its rows.
    fn decode_step(self: &Arc<Self>, scan: ScanBuilder) -> IndexStep<Vec<Arc<V::Chunk>>> {
        let this = Arc::clone(self);
        IndexStep::Read(
            Box::new(scan),
            Box::new(move |arrays| Ok(IndexStep::Done(this.decode_all(arrays)?))),
        )
    }

    fn decode_all(&self, arrays: Vec<ArrayRef>) -> VortexResult<Vec<Arc<V::Chunk>>> {
        let mut ctx = self.session.create_execution_ctx();
        arrays
            .into_iter()
            .filter(|array| !array.is_empty())
            .map(|array| {
                self.vtable
                    .decode(array, &self.options, &mut ctx)
                    .map(Arc::new)
            })
            .collect()
    }
}

impl<V: IndexKind> IndexRows<V> {
    /// Answer a predicate over the indexed column, if this index can.
    ///
    /// `expr` is scoped to the column, so `root()` is the indexed value. Returns `None` when the
    /// index makes no claim on it, in which case the caller scans the data instead.
    pub async fn locate(self: &Arc<Self>, expr: &Expression) -> VortexResult<Located<RowLocator>> {
        self.locate_steps(expr)?.run().await
    }

    /// [`locate`](Self::locate) as steps.
    pub fn locate_steps(
        self: &Arc<Self>,
        expr: &Expression,
    ) -> VortexResult<IndexStep<Located<RowLocator>>> {
        let bound = expr.bind(&self.source_dtype)?;
        let Some(plan) =
            self.vtable
                .plan(&bound, &self.source_dtype, &self.index_dtype, &self.options)?
        else {
            return Ok(IndexStep::Done(None));
        };
        let chunks = match self.vtable.query_keys(&plan.query) {
            Some(QueryKeys::Strings(keys)) => self.chunks_for_keys_step(keys)?,
            Some(QueryKeys::U32s(keys)) => self.chunks_for_keys_step(keys)?,
            None => self.bound_chunks_step(Some(plan.filter)),
        };
        let this = Arc::clone(self);
        let exactness = plan.exactness;
        let query = plan.query;
        chunks.map(move |chunks| {
            let mut ctx = this.session.create_execution_ctx();
            let locator = this.vtable.resolve(
                &query,
                &chunks,
                this.data_row_count,
                &this.options,
                &mut ctx,
            )?;
            Ok(Some((exactness, locator)))
        })
    }
}

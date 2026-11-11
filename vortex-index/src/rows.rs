// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading an index's content, wherever it is stored.

use std::any::Any;
use std::sync::Arc;
use std::sync::OnceLock;

use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::scan::v2::ScanFile;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use crate::InvertedIndex;
use crate::TrigramIndex;

/// Run a scan v2 `builder` to its end, collecting its batches in row order.
///
/// Index content and catalog reads are small, so they run on the calling task rather than paying
/// for a hop onto the runtime.
pub async fn collect(builder: ScanBuilder) -> VortexResult<Vec<ArrayRef>> {
    vortex_file::v2::collect(builder).await
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

/// The keys a plan reads, by key type.
enum PlanKeys {
    Strings(Vec<String>),
    U32s(Vec<u32>),
}

/// A scored row returned by a ranked search, in the index's row space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// The row, counted from the start of the indexed data.
    pub row: u64,
    /// Higher is better.
    pub score: f32,
}

/// Keep the `k` best hits, best first. Ties keep the lower row first.
pub(crate) fn top_k(mut hits: Vec<Hit>, k: usize) -> Vec<Hit> {
    let order = |a: &Hit, b: &Hit| b.score.total_cmp(&a.score).then(a.row.cmp(&b.row));
    if hits.len() > k {
        hits.select_nth_unstable_by(k, order);
        hits.truncate(k);
    }
    hits.sort_by(order);
    hits
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
    /// the IVF centroids.
    pub(crate) memo: OnceLock<Arc<dyn Any + Send + Sync>>,
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

    pub(crate) fn scan(&self) -> ScanBuilder {
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
    pub(crate) fn chunks_at_step(
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

    /// The keys a plan of this crate's keyed kinds reads: its query is exactly the sorted keys
    /// its filter matches, so the rows can be found through the cached key column instead.
    fn plan_keys(&self, query: &V::Query) -> Option<PlanKeys> {
        let kind: &dyn Any = self.vtable.as_ref();
        let query: &dyn Any = query;
        if kind.is::<InvertedIndex>() {
            return query
                .downcast_ref::<Vec<String>>()
                .map(|k| PlanKeys::Strings(k.clone()));
        }
        if kind.is::<TrigramIndex>() {
            return query
                .downcast_ref::<Vec<u32>>()
                .map(|k| PlanKeys::U32s(k.clone()));
        }
        None
    }

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
        let chunks = match self.plan_keys(&plan.query) {
            Some(PlanKeys::Strings(keys)) => self.chunks_for_keys_step(keys)?,
            Some(PlanKeys::U32s(keys)) => self.chunks_for_keys_step(keys)?,
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

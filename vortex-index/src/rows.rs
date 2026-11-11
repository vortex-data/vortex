// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading an index's content, wherever it is stored.

use std::any::Any;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::TryStreamExt;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_file::v2::ScanDriver;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::layouts::indexed::RowLocator;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_layout::scan::v2::ScanFile;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

/// Run a scan v2 `builder` to its end, collecting its batches in row order.
///
/// Index content and catalog reads are small, so one chunk runs at a time per worker.
pub async fn collect(builder: ScanBuilder) -> VortexResult<Vec<ArrayRef>> {
    ScanDriver::new(builder.session().handle(), 1, true)
        .run(builder)
        .try_collect()
        .await
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

    /// Read and decode the index rows matching `filter`, or every row without one.
    ///
    /// Index content is an ordinary layout, so the filter is pruned by its zone maps before any
    /// segment is fetched. The index kinds in this crate keep each index row self-contained, so
    /// decoding any subset of rows is valid.
    pub async fn chunks(&self, filter: Option<Expression>) -> VortexResult<Vec<Arc<V::Chunk>>> {
        let filter = filter.map(|f| f.bind(&self.index_dtype)).transpose()?;
        self.bound_chunks(filter).await
    }

    /// Read the index rows at `rows`, strictly ascending, without evaluating a filter.
    ///
    /// For kinds whose key is the row position, so a lookup needs no key column read.
    pub(crate) async fn chunks_at(&self, rows: Vec<u64>) -> VortexResult<Vec<Arc<V::Chunk>>> {
        let indices = StrictSortedBuffer::try_new(Buffer::from(rows))?;
        let arrays = collect(
            ScanBuilder::new(self.session.clone(), self.file.clone()).with_row_indices(indices),
        )
        .await?;
        self.decode_all(arrays)
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

    async fn bound_chunks(
        &self,
        filter: Option<BoundExpression>,
    ) -> VortexResult<Vec<Arc<V::Chunk>>> {
        let arrays = collect(
            ScanBuilder::new(self.session.clone(), self.file.clone()).with_some_filter(filter),
        )
        .await?;
        self.decode_all(arrays)
    }

    /// Answer a predicate over the indexed column, if this index can.
    ///
    /// `expr` is scoped to the column, so `root()` is the indexed value. Returns `None` when the
    /// index makes no claim on it, in which case the caller scans the data instead.
    pub async fn locate(
        &self,
        expr: &Expression,
    ) -> VortexResult<Option<(IndexExactness, RowLocator)>> {
        let bound = expr.bind(&self.source_dtype)?;
        let Some(plan) =
            self.vtable
                .plan(&bound, &self.source_dtype, &self.index_dtype, &self.options)?
        else {
            return Ok(None);
        };
        let chunks = self.bound_chunks(Some(plan.filter)).await?;
        let mut ctx = self.session.create_execution_ctx();
        let locator = self.vtable.resolve(
            &plan.query,
            &chunks,
            self.data_row_count,
            &self.options,
            &mut ctx,
        )?;
        Ok(Some((plan.exactness, locator)))
    }
}

//! Physical operators needed to implement scanning of Vortex arrays with pushdown.

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion_common::{DataFusionError, Result as DFResult};
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan, PlanProperties,
};
use futures::{ready, Stream};
use pin_project::pin_project;
use vortex_array::array::ChunkedArray;
use vortex_array::arrow::FromArrowArray;
use vortex_array::compute::take;
use vortex_array::{Array, IntoArrayVariant, IntoCanonical};
use vortex_dtype::field::Field;
use vortex_error::{vortex_err, vortex_panic, VortexError};
use vortex_expr::ExprRef;

/// Physical plan operator that applies a set of [filters][Expr] against the input, producing a
/// row mask that can be used downstream to force a take against the corresponding struct array
/// chunks but for different columns.
pub(crate) struct RowSelectorExec {
    filter_expr: ExprRef,
    /// cached PlanProperties object. We do not make use of this.
    cached_plan_props: PlanProperties,
    /// Full array. We only access partitions of this data.
    chunked_array: ChunkedArray,
}

static ROW_SELECTOR_SCHEMA_REF: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![arrow_schema::Field::new(
        "row_idx",
        DataType::UInt64,
        false,
    )]))
});

impl RowSelectorExec {
    pub(crate) fn try_new(filter_expr: ExprRef, chunked_array: &ChunkedArray) -> DFResult<Self> {
        let cached_plan_props = PlanProperties::new(
            EquivalenceProperties::new(ROW_SELECTOR_SCHEMA_REF.clone()),
            Partitioning::UnknownPartitioning(1),
            ExecutionMode::Bounded,
        );

        Ok(Self {
            filter_expr,
            chunked_array: chunked_array.clone(),
            cached_plan_props,
        })
    }
}

impl Debug for RowSelectorExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowSelectorExec")
            .field("filter_expr", &self.filter_expr)
            .finish()
    }
}

impl DisplayAs for RowSelectorExec {
    fn fmt_as(
        &self,
        _display_format_type: DisplayFormatType,
        f: &mut Formatter,
    ) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ExecutionPlan for RowSelectorExec {
    fn name(&self) -> &str {
        RowSelectorExec::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cached_plan_props
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // No children
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(vortex_err!(
                "Single partitioning only supported by RowSelectorExec, got partition {}",
                partition
            )
            .into());
        }

        let filter_projection = self.filter_expr.references().into_iter().cloned().collect();
        Ok(Box::pin(RowIndicesStream {
            chunked_array: self.chunked_array.clone(),
            chunk_idx: 0,
            filter_projection,
            conjunction_expr: self.filter_expr.clone(),
        }))
    }
}

/// [RecordBatchStream] of row indices, emitted by the [RowSelectorExec] physical plan node.
pub(crate) struct RowIndicesStream {
    chunked_array: ChunkedArray,
    chunk_idx: usize,
    conjunction_expr: ExprRef,
    filter_projection: Vec<Field>,
}

impl Stream for RowIndicesStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.chunk_idx >= this.chunked_array.nchunks() {
            return Poll::Ready(None);
        }

        let next_chunk = this.chunked_array.chunk(this.chunk_idx)?;
        this.chunk_idx += 1;

        // Get the unfiltered record batch.
        // Since this is a one-shot, we only want to poll the inner future once, to create the
        // initial batch for us to process.
        let vortex_struct = next_chunk.with_dyn(|a| {
            a.as_struct_array()
                .ok_or_else(|| vortex_err!("Not a struct array"))?
                .project(&this.filter_projection)
        })?;

        let selection = this
            .conjunction_expr
            .evaluate(vortex_struct.as_ref())
            .map_err(|e| DataFusionError::External(e.into()))?
            .into_canonical()?
            .into_arrow()?;

        // Convert the `selection` BooleanArray into a UInt64Array of indices.
        let selection_indices = selection
            .as_boolean()
            .values()
            .set_indices()
            .map(|idx| idx as u64);

        let indices = Arc::new(UInt64Array::from_iter_values(selection_indices)) as ArrayRef;
        let indices_batch = RecordBatch::try_new(ROW_SELECTOR_SCHEMA_REF.clone(), vec![indices])?;

        Poll::Ready(Some(Ok(indices_batch)))
    }
}

impl RecordBatchStream for RowIndicesStream {
    fn schema(&self) -> SchemaRef {
        ROW_SELECTOR_SCHEMA_REF.clone()
    }
}

/// Physical that receives a stream of row indices from a child operator, and uses that to perform
/// a `take` operation on tha backing Vortex array.
pub(crate) struct TakeRowsExec {
    plan_properties: PlanProperties,

    // Array storing the indices used to take the plan nodes.
    projection: Vec<Field>,

    // Input plan, a stream of indices on which we perform a take against the original dataset.
    input: Arc<dyn ExecutionPlan>,

    output_schema: SchemaRef,

    // The original Vortex array holding the fields we have not decoded yet.
    table: ChunkedArray,
}

impl TakeRowsExec {
    pub(crate) fn new(
        schema_ref: SchemaRef,
        projection: &[usize],
        row_indices: Arc<dyn ExecutionPlan>,
        table: &ChunkedArray,
    ) -> Self {
        let output_schema = Arc::new(schema_ref.project(projection).unwrap_or_else(|err| {
            vortex_panic!("Failed to project schema: {}", VortexError::from(err))
        }));
        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::UnknownPartitioning(1),
            ExecutionMode::Bounded,
        );

        Self {
            plan_properties,
            projection: projection.iter().copied().map(Field::from).collect(),
            input: row_indices,
            output_schema: output_schema.clone(),
            table: table.clone(),
        }
    }
}

impl Debug for TakeRowsExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TakeRowsExec")
            .field("projection", &self.projection)
            .field("output_schema", &self.output_schema)
            .finish()
    }
}

impl DisplayAs for TakeRowsExec {
    fn fmt_as(&self, _display_type: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl ExecutionPlan for TakeRowsExec {
    fn name(&self) -> &str {
        TakeRowsExec::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Get the row indices for the given chunk.
        let row_indices_stream = self.input.execute(partition, context)?;

        Ok(Box::pin(TakeRowsStream {
            row_indices_stream,
            chunk_idx: 0,
            output_projection: self.projection.clone(),
            output_schema: self.output_schema.clone(),
            vortex_array: self.table.clone(),
        }))
    }
}

/// Stream of outputs emitted by the [TakeRowsExec] physical operator.
#[pin_project]
pub(crate) struct TakeRowsStream<F> {
    // Stream of row indices arriving from upstream operator.
    #[pin]
    row_indices_stream: F,

    // The current chunk. Every time we receive a new RecordBatch from the upstream operator
    // we treat it as a set of row-indices that are zero-indexed relative to this chunk number
    // in the `vortex_array`.
    chunk_idx: usize,

    // Projection based on the schema here
    output_projection: Vec<Field>,
    output_schema: SchemaRef,

    // The original Vortex array we're taking from
    vortex_array: ChunkedArray,
}

impl<F> Stream for TakeRowsStream<F>
where
    F: Stream<Item = DFResult<RecordBatch>>,
{
    type Item = DFResult<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        // Get the indices provided by the upstream operator.
        let record_batch = match ready!(this.row_indices_stream.poll_next(cx)) {
            None => {
                // Row indices stream is complete, we are also complete.
                return Poll::Ready(None);
            }
            Some(result) => result?,
        };

        assert!(
            *this.chunk_idx <= this.vortex_array.nchunks(),
            "input yielded too many RecordBatches"
        );

        let row_indices =
            Array::from_arrow(record_batch.column(0).as_primitive::<UInt64Type>(), false);

        // If no columns in the output projection, we send back a RecordBatch with empty schema.
        // This is common for COUNT queries.
        if this.output_projection.is_empty() {
            let opts = RecordBatchOptions::new().with_row_count(Some(row_indices.len()));
            return Poll::Ready(Some(Ok(RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &opts,
            )
            .map_err(DataFusionError::from)?)));
        }

        let chunk = this.vortex_array.chunk(*this.chunk_idx)?.into_struct()?;

        *this.chunk_idx += 1;

        // TODO(aduffy): this re-decodes the fields from the filter schema, which is wasteful.
        //  We should find a way to avoid decoding the filter columns and only decode the other
        //  columns, then stitch the StructArray back together from those.
        let projected_for_output = chunk.project(this.output_projection)?;
        let decoded = take(projected_for_output, &row_indices)?
            .into_canonical()?
            .into_arrow()?;

        // Send back a single record batch of the decoded data.
        let output_batch = RecordBatch::from(decoded.as_struct());

        Poll::Ready(Some(Ok(output_batch)))
    }
}

impl<F> RecordBatchStream for TakeRowsStream<F>
where
    F: Stream<Item = DFResult<RecordBatch>>,
{
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::{RecordBatch, UInt64Array};
    use datafusion_common::ToDFSchema;
    use datafusion_expr::execution_props::ExecutionProps;
    use datafusion_expr::{and, col, lit};
    use datafusion_physical_expr::create_physical_expr;
    use itertools::Itertools;
    use vortex_array::array::{BoolArray, ChunkedArray, PrimitiveArray, StructArray};
    use vortex_array::arrow::infer_schema;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayDType, IntoArray};
    use vortex_dtype::field::Field;
    use vortex_dtype::FieldName;
    use vortex_expr::datafusion::convert_expr_to_vortex;

    use crate::memory::plans::{RowIndicesStream, ROW_SELECTOR_SCHEMA_REF};

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_filtering_stream() {
        let chunk = StructArray::try_new(
            Arc::new([FieldName::from("a"), FieldName::from("b")]),
            vec![
                PrimitiveArray::from(vec![0u64, 1, 2]).into_array(),
                BoolArray::from(vec![false, false, true]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        let dtype = chunk.dtype().clone();
        let chunked_array =
            ChunkedArray::try_new(vec![chunk.clone(), chunk.clone()], dtype).unwrap();

        let schema = infer_schema(chunk.dtype()).unwrap();
        let logical_expr = and((col("a")).eq(lit(2u64)), col("b").eq(lit(true)));
        let df_expr = create_physical_expr(
            &logical_expr,
            &schema.clone().to_dfschema().unwrap(),
            &ExecutionProps::new(),
        )
        .unwrap();

        let filtering_stream = RowIndicesStream {
            chunked_array: chunked_array.clone(),
            chunk_idx: 0,
            conjunction_expr: convert_expr_to_vortex(df_expr).unwrap(),
            filter_projection: vec![Field::from(0), Field::from(1)],
        };

        let rows: Vec<RecordBatch> = futures::executor::block_on_stream(filtering_stream)
            .try_collect()
            .unwrap();

        assert_eq!(rows.len(), 2);

        // The output of row selection is a RecordBatch of indices that can be used as selectors
        // against the original RecordBatch.
        assert_eq!(
            rows[0],
            RecordBatch::try_new(
                ROW_SELECTOR_SCHEMA_REF.clone(),
                vec![Arc::new(UInt64Array::from(vec![2u64])),]
            )
            .unwrap()
        );

        assert_eq!(
            rows[1],
            RecordBatch::try_new(
                ROW_SELECTOR_SCHEMA_REF.clone(),
                vec![Arc::new(UInt64Array::from(vec![2u64])),]
            )
            .unwrap()
        );
    }
}

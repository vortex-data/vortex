//! Connectors to enable DataFusion to read Vortex data.

#![allow(clippy::nonminimal_bool)]

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion_common::{exec_datafusion_err, DataFusionError, Result as DFResult, Statistics};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::{Expr, Operator};
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::Stream;
use memory::{VortexMemTable, VortexMemTableOptions};
use persistent::config::VortexTableOptions;
use persistent::provider::VortexFileTableProvider;
use vortex_array::array::ChunkedArray;
use vortex_array::{Array, ArrayDType, IntoArrayVariant};
use vortex_dtype::field::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::statistics::chunked_array_df_stats;

pub mod memory;
pub mod persistent;

mod plans;
mod statistics;

const SUPPORTED_BINARY_OPS: &[Operator] = &[
    Operator::Eq,
    Operator::NotEq,
    Operator::Gt,
    Operator::GtEq,
    Operator::Lt,
    Operator::LtEq,
];

fn supported_data_types(dt: DataType) -> bool {
    dt.is_integer()
        || dt.is_floating()
        || dt.is_null()
        || dt == DataType::Boolean
        || dt == DataType::Binary
        || dt == DataType::Utf8
        || dt == DataType::Binary
        || dt == DataType::BinaryView
        || dt == DataType::Utf8View
        || dt == DataType::Date32
        || dt == DataType::Date64
        || matches!(
            dt,
            DataType::Timestamp(_, _) | DataType::Time32(_) | DataType::Time64(_)
        )
}

pub trait SessionContextExt {
    fn register_mem_vortex<S: AsRef<str>>(&self, name: S, array: Array) -> DFResult<()> {
        self.register_mem_vortex_opts(name, array, VortexMemTableOptions::default())
    }

    fn register_mem_vortex_opts<S: AsRef<str>>(
        &self,
        name: S,
        array: Array,
        options: VortexMemTableOptions,
    ) -> DFResult<()>;

    fn read_mem_vortex(&self, array: Array) -> DFResult<DataFrame> {
        self.read_mem_vortex_opts(array, VortexMemTableOptions::default())
    }

    fn read_mem_vortex_opts(
        &self,
        array: Array,
        options: VortexMemTableOptions,
    ) -> DFResult<DataFrame>;

    fn register_disk_vortex_opts<S: AsRef<str>>(
        &self,
        name: S,
        url: ObjectStoreUrl,
        options: VortexTableOptions,
    ) -> DFResult<()>;

    fn read_disk_vortex_opts(
        &self,
        url: ObjectStoreUrl,
        options: VortexTableOptions,
    ) -> DFResult<DataFrame>;
}

impl SessionContextExt for SessionContext {
    fn register_mem_vortex_opts<S: AsRef<str>>(
        &self,
        name: S,
        array: Array,
        options: VortexMemTableOptions,
    ) -> DFResult<()> {
        if !array.dtype().is_struct() {
            return Err(vortex_err!(
                "Vortex arrays must have struct type, found {}",
                array.dtype()
            )
            .into());
        }

        let vortex_table = VortexMemTable::new(array, options);
        self.register_table(name.as_ref(), Arc::new(vortex_table))
            .map(|_| ())
    }

    fn read_mem_vortex_opts(
        &self,
        array: Array,
        options: VortexMemTableOptions,
    ) -> DFResult<DataFrame> {
        if !array.dtype().is_struct() {
            return Err(vortex_err!(
                "Vortex arrays must have struct type, found {}",
                array.dtype()
            )
            .into());
        }

        let vortex_table = VortexMemTable::new(array, options);

        self.read_table(Arc::new(vortex_table))
    }

    fn register_disk_vortex_opts<S: AsRef<str>>(
        &self,
        name: S,
        url: ObjectStoreUrl,
        options: VortexTableOptions,
    ) -> DFResult<()> {
        let provider = Arc::new(VortexFileTableProvider::try_new(url, options)?);
        self.register_table(name.as_ref(), provider as _)?;

        Ok(())
    }

    fn read_disk_vortex_opts(
        &self,
        url: ObjectStoreUrl,
        options: VortexTableOptions,
    ) -> DFResult<DataFrame> {
        let provider = Arc::new(VortexFileTableProvider::try_new(url, options)?);
        self.read_table(provider)
    }
}

fn can_be_pushed_down(expr: &Expr, schema: &Schema) -> bool {
    match expr {
        Expr::BinaryExpr(expr)
            if expr.op.is_logic_operator() || SUPPORTED_BINARY_OPS.contains(&expr.op) =>
        {
            can_be_pushed_down(expr.left.as_ref(), schema)
                & can_be_pushed_down(expr.right.as_ref(), schema)
        }
        Expr::Column(col) => match schema.column_with_name(col.name()) {
            Some((_, field)) => supported_data_types(field.data_type().clone()),
            _ => false,
        },
        Expr::Literal(lit) => supported_data_types(lit.data_type()),
        _ => false,
    }
}

/// Physical plan node for scans against an in-memory, possibly chunked Vortex Array.
#[derive(Clone)]
struct VortexScanExec {
    array: ChunkedArray,
    scan_projection: Vec<usize>,
    plan_properties: PlanProperties,
    statistics: Statistics,
}

impl VortexScanExec {
    pub fn try_new(
        array: ChunkedArray,
        scan_projection: Vec<usize>,
        plan_properties: PlanProperties,
    ) -> VortexResult<Self> {
        let statistics = chunked_array_df_stats(&array, &scan_projection)?;
        Ok(Self {
            array,
            scan_projection,
            plan_properties,
            statistics,
        })
    }
}

impl Debug for VortexScanExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexScanExec")
            .field("array_length", &self.array.len())
            .field("array_dtype", &self.array.dtype())
            .field("scan_projection", &self.scan_projection)
            .field("plan_properties", &self.plan_properties)
            .finish_non_exhaustive()
    }
}

impl DisplayAs for VortexScanExec {
    fn fmt_as(&self, _display_type: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

pub(crate) struct VortexRecordBatchStream {
    schema_ref: SchemaRef,

    idx: usize,
    num_chunks: usize,
    chunks: ChunkedArray,

    projection: Vec<Field>,
}

impl Stream for VortexRecordBatchStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.idx >= self.num_chunks {
            return Poll::Ready(None);
        }

        // Grab next chunk, project and convert to Arrow.
        let chunk = self.chunks.chunk(self.idx)?;
        self.idx += 1;

        let struct_array = chunk
            .into_struct()
            .map_err(|vortex_error| DataFusionError::Execution(format!("{}", vortex_error)))?;

        let projected_struct = struct_array
            .project(&self.projection)
            .map_err(|vortex_err| {
                exec_datafusion_err!("projection pushdown to Vortex failed: {vortex_err}")
            })?;

        Poll::Ready(Some(Ok(projected_struct.try_into()?)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.num_chunks, Some(self.num_chunks))
    }
}

impl RecordBatchStream for VortexRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema_ref)
    }
}

impl ExecutionPlan for VortexScanExec {
    fn name(&self) -> &str {
        VortexScanExec::static_name()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.plan_properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // Leaf node
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Send back a stream of RecordBatch that returns the next element of the chunk each time.
        Ok(Box::pin(VortexRecordBatchStream {
            schema_ref: self.schema().clone(),
            idx: 0,
            num_chunks: self.array.nchunks(),
            chunks: self.array.clone(),
            projection: self
                .scan_projection
                .iter()
                .copied()
                .map(Field::from)
                .collect(),
        }))
    }

    fn statistics(&self) -> DFResult<Statistics> {
        Ok(self.statistics.clone())
    }
}

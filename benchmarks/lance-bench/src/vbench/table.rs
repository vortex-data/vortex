// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::FixedSizeListArray;
use datafusion::arrow::array::Float32Array;
use datafusion::arrow::array::Int64Array;
use datafusion::arrow::array::UInt64Array;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Field;
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::Session;
use datafusion::catalog::TableProvider;
use datafusion::common::Result;
use datafusion::common::ScalarValue;
use datafusion::common::exec_err;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Expr;
use datafusion::logical_expr::Operator;
use datafusion::logical_expr::TableProviderFilterPushDown;
use datafusion::logical_expr::TableType;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::DisplayAs;
use datafusion::physical_plan::DisplayFormatType;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::Partitioning;
use datafusion::physical_plan::PlanProperties;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::Boundedness;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use lance::Dataset;
use lance::deps::arrow_array as la;
use lance::deps::arrow_array::Array as _;
use lance::deps::arrow_array::cast::AsArray as _;
use lance::deps::datafusion::physical_plan::ExecutionPlan as LancePlan;
use lance::deps::datafusion::physical_plan::displayable as display_lance_plan;
use lance_datafusion::exec::execute_plan;
use lance_linalg::distance::MetricType;
use roaring::RoaringBitmap;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::dtype::Nullability::NonNullable;
use vortex::arrow::ArrowScanExt;
use vortex::buffer::Buffer;
use vortex::expr::col;
use vortex::expr::gt_eq;
use vortex::expr::lit;
use vortex::expr::pack;
use vortex::file::VortexFile;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::scan::execution::driver::ScanDriver;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_index::IndexRows;
use vortex_index_ivf::IvfIndex;
use vortex_index_ivf::VectorSearch;

#[derive(Clone)]
pub(super) enum Backend {
    Vortex {
        file: Arc<VortexFile>,
        index: Arc<IndexRows<IvfIndex>>,
    },
    Lance(Arc<Dataset>),
}

#[derive(Clone, Debug)]
pub(super) struct Search {
    pub vector: Vec<f32>,
    pub k: usize,
    pub min_price: Option<i64>,
}

#[derive(Clone)]
pub(super) struct RecipeTable {
    backend: Backend,
    schema: SchemaRef,
    pub search: Option<Search>,
    pub push_price_filter: bool,
    probes: usize,
}

impl RecipeTable {
    pub fn new(backend: Backend, dim: i32, probes: usize) -> Self {
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let schema = Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("emb", DataType::FixedSizeList(item, dim), false),
            Field::new("price", DataType::Int64, false),
        ]);
        Self {
            backend,
            schema: Arc::new(schema),
            search: None,
            push_price_filter: false,
            probes,
        }
    }

    fn plan_name(&self) -> &'static str {
        match (&self.backend, &self.search) {
            (Backend::Vortex { .. }, Some(_)) => "VortexIvfExec",
            (Backend::Lance(_), Some(_)) => "LanceNearestExec",
            (Backend::Vortex { .. }, None) => "VortexScanExec",
            (Backend::Lance(_), None) => "LanceScanExec",
        }
    }
}

#[derive(Clone)]
enum NativeScan {
    Vortex {
        file: Arc<VortexFile>,
        index: Arc<IndexRows<IvfIndex>>,
        search: Option<Search>,
        min_price: Option<i64>,
        probes: usize,
    },
    Lance(Arc<dyn LancePlan>),
}

impl NativeScan {
    async fn batches(
        &self,
        schema: SchemaRef,
        target_partitions: usize,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<RecordBatch>>> {
        match self {
            Self::Vortex {
                file,
                index,
                search,
                min_price,
                probes,
            } => {
                let session = file.session();
                let scan = || ScanBuilder::new(session.clone(), file.scan_file());
                let columns = pack(
                    ["id", "emb", "price"].map(|name| (name, col(name))),
                    NonNullable,
                )
                .bind(file.dtype())?;
                // An identity scan sizes work chunks using the file's bytes per row.
                let mut read = if search.is_none() && columns.dtype() == file.dtype() {
                    scan()
                } else {
                    scan().with_projection(columns)
                };
                if let Some(search) = search {
                    let hits = if let Some(floor) = *min_price {
                        let dtype = file.dtype();
                        let mut eligible = ScanDriver::new(session.handle(), 0, true).run(
                            scan()
                                .with_projection(row_idx().bind(dtype)?)
                                .with_some_filter(Some(
                                    gt_eq(col("price"), lit(floor)).bind(dtype)?,
                                )),
                        );
                        let mut rows = RoaringBitmap::new();
                        let mut ctx = session.create_execution_ctx();
                        while let Some(chunk) = eligible.try_next().await? {
                            let indices = chunk.execute::<PrimitiveArray>(&mut ctx)?;
                            for &row in indices.as_slice::<u64>() {
                                rows.insert(u32::try_from(row)?);
                            }
                        }
                        index
                            .search_vectors_filtered(&search.vector, search.k, *probes, rows)
                            .await?
                    } else {
                        index
                            .search_vectors(&search.vector, search.k, *probes)
                            .await?
                    };
                    if hits.is_empty() {
                        return Ok(stream::empty().boxed());
                    }
                    let mut rows: Vec<_> = hits.into_iter().map(|hit| hit.row).collect();
                    rows.sort_unstable();
                    read = read.with_row_indices(StrictSortedBuffer::try_new(Buffer::from(rows))?);
                } else if let Some(floor) = *min_price {
                    read = read.with_some_filter(Some(
                        gt_eq(col("price"), lit(floor)).bind(file.dtype())?,
                    ));
                }
                // Full scans can stream completed chunks in any order.
                // An indexed take fetches only the candidate rows in a single small read.
                let driver = if search.is_some() {
                    ScanDriver::new(session.handle(), 0, true)
                } else {
                    ScanDriver::new(session.handle(), target_partitions, false)
                };
                Ok(driver
                    .run_record_batches(read, Some(schema))
                    .map_err(anyhow::Error::from)
                    .boxed())
            }
            Self::Lance(plan) => Ok(execute_plan(Arc::clone(plan), Default::default())?
                .map(move |batch| import_lance(&schema, &batch?))
                .boxed()),
        }
    }
}

// Lance 14 returns Arrow 58. Bridge only the three non-null benchmark columns into Arrow 60.
fn import_lance(schema: &SchemaRef, batch: &la::RecordBatch) -> anyhow::Result<RecordBatch> {
    let column = |name: &str| {
        batch
            .column_by_name(name)
            .ok_or_else(|| anyhow::anyhow!("missing Lance column {name}"))
    };
    let ids = column("id")?.as_primitive::<la::types::UInt64Type>();
    let lists = column("emb")?.as_fixed_size_list();
    let prices = column("price")?.as_primitive::<la::types::Int64Type>();
    let values = lists.values().as_primitive::<la::types::Float32Type>();
    anyhow::ensure!(
        values.null_count() == 0,
        "recipe embedding contains a null coordinate"
    );
    let DataType::FixedSizeList(item, dim) = schema.field(1).data_type() else {
        anyhow::bail!("expected fixed-size embeddings");
    };
    let vectors = FixedSizeListArray::try_new(
        Arc::clone(item),
        *dim,
        Arc::new(Float32Array::from(values.values().to_vec())),
        None,
    )?;
    Ok(RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(UInt64Array::from(ids.values().to_vec())),
            Arc::new(vectors),
            Arc::new(Int64Array::from(prices.values().to_vec())),
        ],
    )?)
}

impl fmt::Debug for RecipeTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.plan_name())
    }
}

fn price_floor(filter: &Expr) -> Option<i64> {
    let Expr::BinaryExpr(binary) = filter else {
        return None;
    };
    match (binary.left.as_ref(), binary.op, binary.right.as_ref()) {
        (
            Expr::Column(column),
            Operator::GtEq,
            Expr::Literal(ScalarValue::Int64(Some(value)), _),
        )
        | (
            Expr::Literal(ScalarValue::Int64(Some(value)), _),
            Operator::LtEq,
            Expr::Column(column),
        ) if column.name == "price" => Some(*value),
        _ => None,
    }
}

#[async_trait]
impl TableProvider for RecipeTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        let can_prefilter = self.push_price_filter
            || (self.search.is_none() && matches!(self.backend, Backend::Vortex { .. }));
        Ok(filters
            .iter()
            .map(|filter| {
                if can_prefilter && price_floor(filter).is_some() {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        _: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let search = self.search.clone();
        let min_price = filters
            .iter()
            .filter_map(price_floor)
            .chain(search.as_ref().and_then(|s| s.min_price))
            .max();
        let source = match &self.backend {
            Backend::Vortex { file, index } => NativeScan::Vortex {
                file: Arc::clone(file),
                index: Arc::clone(index),
                search,
                min_price,
                probes: self.probes,
            },
            Backend::Lance(dataset) => {
                let plan = async {
                    let mut scan = dataset.scan();
                    scan.project(&["id", "emb", "price"])?;
                    if let Some(search) = search {
                        scan.nearest(
                            "emb",
                            &la::Float32Array::from(search.vector.clone()),
                            search.k,
                        )?
                        .distance_metric(MetricType::Cosine)
                        .nprobes(self.probes);
                        if let Some(floor) = min_price {
                            scan.filter(&format!("price >= {floor}"))?.prefilter(true);
                        }
                    }
                    scan.create_plan().await
                }
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
                NativeScan::Lance(plan)
            }
        };
        let schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => Arc::clone(&self.schema),
        };
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Arc::new(RecipeExec {
            name: self.plan_name(),
            min_price,
            source,
            properties: Arc::new(properties),
            full_schema: Arc::clone(&self.schema),
            projection: projection.map(<[_]>::to_vec),
        }))
    }
}

struct RecipeExec {
    min_price: Option<i64>,
    name: &'static str,
    source: NativeScan,
    properties: Arc<PlanProperties>,
    full_schema: SchemaRef,
    projection: Option<Vec<usize>>,
}

impl fmt::Debug for RecipeExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

impl DisplayAs for RecipeExec {
    fn fmt_as(&self, format: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name)?;
        if let Some(floor) = self.min_price {
            write!(f, " prefilter=[price >= {floor}]")?;
        }
        if let (DisplayFormatType::Verbose, NativeScan::Lance(plan)) = (format, &self.source) {
            write!(f, "\n{}", display_lance_plan(plan.as_ref()).indent(false))?;
        }
        Ok(())
    }
}

impl ExecutionPlan for RecipeExec {
    fn name(&self) -> &'static str {
        self.name
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn apply_expressions(
        &self,
        _visitor: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return exec_err!("recipe scan is a leaf plan");
        }
        Ok(self)
    }
    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return exec_err!("recipe scan has one partition");
        }
        let source = self.source.clone();
        let schema = Arc::clone(&self.full_schema);
        let projection = self.projection.clone();
        let batches = stream::once(async move {
            let stream = source
                .batches(schema, ctx.session_config().target_partitions())
                .await
                .map_err(|e| DataFusionError::External(e.into()))?;
            Ok::<_, DataFusionError>(stream.map(move |batch| {
                let batch = batch.map_err(|e| DataFusionError::External(e.into()))?;
                match &projection {
                    Some(indices) => Ok(batch.project(indices)?),
                    None => Ok(batch),
                }
            }))
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            batches,
        )))
    }
}

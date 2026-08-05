// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(missing_docs)]

//! Morsel-driven I/O support for Vortex files.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::Statistics;
use datafusion_common::arrow::array::AsArray;
use datafusion_common::exec_datafusion_err;
use datafusion_common::format::MetricCategory;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::TableSchema;
use datafusion_datasource::morsel::Morsel;
use datafusion_datasource::morsel::MorselPlan;
use datafusion_datasource::morsel::MorselPlanner;
use datafusion_datasource::morsel::Morselizer;
use datafusion_execution::cache::cache_manager::CachedFileMetadataEntry;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::projection::Projector;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_adapter::replace_columns_with_literals;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::metrics::MetricBuilder;
use datafusion_pruning::FilePruner;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream::BoxStream;
use object_store::path::Path;
use tokio::sync::OnceCell;
use tracing::Instrument;
use vortex::array::VortexSessionExecute;
use vortex::dtype::FieldMask;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::file::Footer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::InstrumentedReadAt;
use vortex::io::VortexReadAt;
use vortex::layout::LayoutReader;
use vortex::layout::scan::repeated_scan::RepeatedScan;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::metrics::Label;
use vortex::metrics::MetricsRegistry;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use crate::VortexAccessPlan;
use crate::convert::ExpressionConvertor;
use crate::convert::exprs::ProcessedProjection;
use crate::convert::exprs::make_vortex_predicate;
use crate::convert::schema::calculate_physical_schema;
use crate::metrics::PARTITION_LABEL;
use crate::metrics::PATH_LABEL;
use crate::persistent::cache::CachedVortexMetadata;
use crate::persistent::stream::PrunableStream;
use crate::reader::VortexReaderFactory;

pub(crate) type NaturalSplitCache = DashMap<Path, Arc<OnceCell<Arc<NaturalSplits>>>>;

/// A file's natural split boundaries, with each split's byte assignment precomputed.
///
/// `row_boundaries` is the sorted, deduplicated split point list covering the whole file.
/// `assignment_bytes` holds the byte that owns each split (see [`split_assignment_byte`]) and is
/// sorted, so translating a DataFusion byte range into rows is two binary searches rather than a
/// scan over every split.
#[derive(Debug)]
pub(crate) struct NaturalSplits {
    row_boundaries: Arc<[u64]>,
    assignment_bytes: Box<[u64]>,
}

impl NaturalSplits {
    fn new(row_boundaries: Arc<[u64]>, total_size: u64) -> Self {
        let row_count = row_boundaries.last().copied().unwrap_or_default();
        let assignment_bytes = if row_count == 0 {
            Box::default()
        } else {
            row_boundaries
                .windows(2)
                .enumerate()
                .map(|(idx, boundaries)| {
                    split_assignment_byte(
                        idx,
                        &(boundaries[0]..boundaries[1]),
                        row_count,
                        total_size,
                    )
                })
                .collect()
        };

        debug_assert!(assignment_bytes.is_sorted());
        debug_assert_eq!(
            assignment_bytes.len() + usize::from(!row_boundaries.is_empty()),
            row_boundaries.len()
        );

        Self {
            row_boundaries,
            assignment_bytes,
        }
    }
}

/// Creates morsel planners for Vortex files.
pub struct VortexMorselizer {
    pub partition: usize,
    pub session: VortexSession,
    pub vortex_reader_factory: Arc<dyn VortexReaderFactory>,
    /// Optional table schema projection. The indices are w.r.t. the `table_schema`, which is
    /// all fields in the final scan result not including the partition columns.
    pub projection: ProjectionExprs,
    /// Filter expression optimized for pushdown into Vortex scan operations.
    /// This may be a subset of file_pruning_predicate containing only expressions
    /// that Vortex can efficiently evaluate.
    pub filter: Option<PhysicalExprRef>,
    /// Filter expression used by DataFusion's FilePruner to eliminate files based on
    /// statistics and partition values without opening them.
    pub file_pruning_predicate: Option<PhysicalExprRef>,
    pub expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    /// This is the table's schema without partition columns. It may contain fields which do
    /// not exist in the file, and are supplied by the `schema_adapter_factory`.
    pub table_schema: TableSchema,
    /// Desired row count for record batches returned from the scan.
    /// If provided, the scan will not return more than this many rows.
    pub limit: Option<u64>,
    /// A metrics object for tracking performance of the scan.
    pub metrics_registry: Arc<dyn MetricsRegistry>,
    /// DataFusion-native metrics exposed through `DataSourceExec`.
    pub df_metrics: ExecutionPlanMetricsSet,
    /// A shared cache of file readers.
    ///
    /// To save on the overhead of reparsing FlatBuffers and rebuilding the layout tree, we cache
    /// a file reader the first time we read a file.
    pub layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    /// Shared full-file natural split ranges keyed by file path.
    pub(crate) natural_splits: Arc<NaturalSplitCache>,
    /// Whether the query has output ordering specified
    pub has_output_ordering: bool,

    pub expression_convertor: Arc<dyn ExpressionConvertor>,
    pub file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    /// Whether to enable expression pushdown into the underlying Vortex scan.
    pub projection_pushdown: bool,
    pub scan_concurrency: Option<usize>,
}

impl std::fmt::Debug for VortexMorselizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexMorselizer")
            .field("partition", &self.partition)
            .field("session", &self.session)
            .field("vortex_reader_factory", &self.vortex_reader_factory)
            .field("projection", &self.projection)
            .field("filter", &self.filter)
            .field("file_pruning_predicate", &self.file_pruning_predicate)
            .field("expr_adapter_factory", &self.expr_adapter_factory)
            .field("table_schema", &self.table_schema)
            .field("limit", &self.limit)
            .field("df_metrics", &self.df_metrics)
            .field("layout_readers", &self.layout_readers)
            .field("natural_splits", &self.natural_splits)
            .field("has_output_ordering", &self.has_output_ordering)
            .field("file_metadata_cache", &self.file_metadata_cache)
            .field("projection_pushdown", &self.projection_pushdown)
            .field("scan_concurrency", &self.scan_concurrency)
            .finish_non_exhaustive()
    }
}

impl Morselizer for VortexMorselizer {
    fn plan_file(&self, file: PartitionedFile) -> DFResult<Box<dyn MorselPlanner>> {
        Ok(Box::new(VortexMorselPlanner::try_new(self, file)?))
    }
}

/// Plans morsels for a Vortex file.
#[derive(Debug)]
pub struct VortexMorselPlanner {
    state: State,
}

impl VortexMorselPlanner {
    /// Create new [`VortexMorselPlanner`]
    pub fn try_new(morselizer: &VortexMorselizer, file: PartitionedFile) -> DFResult<Self> {
        // Calculate the output schema before replacing partition columns with literals so it
        // retains the table and partition-field metadata declared by the plan.
        let output_schema = Arc::new(
            morselizer
                .projection
                .project_schema(morselizer.table_schema.table_schema())?,
        );
        let session = morselizer.session.clone();
        let metrics_registry = Arc::clone(&morselizer.metrics_registry);
        let labels = vec![
            Label::new(PATH_LABEL, file.path().to_string()),
            Label::new(PARTITION_LABEL, morselizer.partition.to_string()),
        ];

        let mut projection = morselizer.projection.clone();
        let mut filter = morselizer.filter.clone();

        let reader = morselizer
            .vortex_reader_factory
            .create_reader(&file, &session)?;

        let reader =
            InstrumentedReadAt::new_with_labels(reader, metrics_registry.as_ref(), labels.clone());

        let mut file_pruning_predicate = morselizer.file_pruning_predicate.clone();
        let expr_adapter_factory = Arc::clone(&morselizer.expr_adapter_factory);
        let file_metadata_cache = morselizer.file_metadata_cache.clone();

        let unified_file_schema = Arc::clone(morselizer.table_schema.file_schema());
        let limit = morselizer.limit;
        let layout_readers = Arc::clone(&morselizer.layout_readers);
        let natural_splits = Arc::clone(&morselizer.natural_splits);
        let has_output_ordering = morselizer.has_output_ordering;
        let scan_concurrency = morselizer.scan_concurrency;

        let expr_convertor = Arc::clone(&morselizer.expression_convertor);
        let projection_pushdown = morselizer.projection_pushdown;

        let predicate_creation_errors = MetricBuilder::new(&morselizer.df_metrics)
            .with_category(MetricCategory::Rows)
            .global_counter("num_predicate_creation_errors");

        // Replace column access for partition columns with literals
        #[expect(clippy::disallowed_types)]
        let literal_value_cols = morselizer
            .table_schema
            .table_partition_cols()
            .iter()
            .map(|f| f.name())
            .cloned()
            .zip(file.partition_values.clone())
            .collect::<std::collections::HashMap<String, ScalarValue>>();

        let predicate_uses_partition_columns =
            file_pruning_predicate.as_ref().is_some_and(|predicate| {
                collect_columns(predicate)
                    .iter()
                    .any(|column| literal_value_cols.contains_key(column.name()))
            });

        if !literal_value_cols.is_empty() {
            projection = projection.try_map_exprs(|expr| {
                replace_columns_with_literals(Arc::clone(&expr), &literal_value_cols)
            })?;
            filter = filter
                .map(|p| replace_columns_with_literals(p, &literal_value_cols))
                .transpose()?;
            file_pruning_predicate = file_pruning_predicate
                .map(|p| replace_columns_with_literals(p, &literal_value_cols))
                .transpose()?;
        }

        // FilePruner requires a statistics object even when the rewritten predicate
        // only contains partition literals. Supply unknown file-column statistics in
        // that case so static and dynamic partition predicates can still prune.
        let synthetic_statistics = (!file.has_statistics() && predicate_uses_partition_columns)
            .then(|| {
                file.clone()
                    .with_statistics(Arc::new(Statistics::new_unknown(&unified_file_schema)))
            });
        let pruning_file = synthetic_statistics.as_ref().unwrap_or(&file);

        let file_pruner = file_pruning_predicate
            .filter(|_| file.has_statistics() || predicate_uses_partition_columns)
            .and_then(|predicate| {
                FilePruner::try_new(
                    Arc::clone(&predicate),
                    &unified_file_schema,
                    pruning_file,
                    predicate_creation_errors,
                )
            });

        Ok(Self {
            state: State::Start {
                state: FileOpenState {
                    file,
                    output_schema,
                    session,
                    metrics_registry,
                    labels,
                    projection,
                    filter,
                    reader,
                    file_pruner,
                    expr_adapter_factory,
                    file_metadata_cache,
                    unified_file_schema,
                    limit,
                    layout_readers,
                    natural_splits,
                    has_output_ordering,
                    expr_convertor,
                    projection_pushdown,
                    scan_concurrency,
                },
            },
        })
    }
}

struct FileOpenState {
    file: PartitionedFile,
    output_schema: SchemaRef,
    session: VortexSession,
    metrics_registry: Arc<dyn MetricsRegistry>,
    labels: Vec<Label>,
    projection: ProjectionExprs,
    filter: Option<PhysicalExprRef>,
    reader: InstrumentedReadAt<Arc<dyn VortexReadAt>>,
    file_pruner: Option<FilePruner>,
    expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    unified_file_schema: SchemaRef,
    limit: Option<u64>,
    layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    natural_splits: Arc<NaturalSplitCache>,
    has_output_ordering: bool,
    expr_convertor: Arc<dyn ExpressionConvertor>,
    projection_pushdown: bool,
    scan_concurrency: Option<usize>,
}

enum State {
    Start {
        state: FileOpenState,
    },
    OpenFooter {
        state: FileOpenState,
    },
    OpenFile {
        state: FileOpenState,
        footer: Option<Footer>,
    },
    BuildScan {
        state: FileOpenState,
        vxf: VortexFile,
    },
    CalculateLayoutSplits {
        state: FileOpenState,
        vxf: VortexFile,
        layout_reader: Arc<dyn LayoutReader>,
        byte_range: Range<u64>,
        split_ranges: Arc<OnceCell<Arc<NaturalSplits>>>,
    },
    PrepareScan {
        state: FileOpenState,
        vxf: VortexFile,
        layout_reader: Arc<dyn LayoutReader>,
        row_range: Option<Range<u64>>,
    },
    PreparedScan {
        scan: RepeatedScan,
        file_pruner: Option<FilePruner>,
        output_schema: SchemaRef,
        session: VortexSession,
        stream_target_field: Field,
        file_location: Path,
        projector: Projector,
    },
    Done,
}

impl State {
    fn transition(self) -> DFResult<Self> {
        match self {
            State::Start { mut state } => {
                if let Some(pruner) = state.file_pruner.as_mut()
                    && pruner.should_prune()?
                {
                    Ok(Self::Done)
                } else {
                    Ok(Self::OpenFooter { state })
                }
            }
            State::OpenFooter { state } => {
                let footer = state
                    .file_metadata_cache
                    .as_ref()
                    .and_then(|cache| cache.get(state.file.path()))
                    .filter(|entry| entry.is_valid_for(&state.file.object_meta))
                    .and_then(|entry| {
                        entry
                            .file_metadata
                            .as_any()
                            .downcast_ref::<CachedVortexMetadata>()
                            .map(|vortex_metadata| vortex_metadata.footer().clone())
                    });

                Ok(State::OpenFile { state, footer })
            }
            state => Ok(state),
        }
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Start { state } => f.debug_struct("Start").field("state", state).finish(),
            Self::OpenFooter { state } => {
                f.debug_struct("OpenFooter").field("state", state).finish()
            }
            Self::OpenFile { state, footer } => f
                .debug_struct("OpenFile")
                .field("state", state)
                .field("footer", &footer.as_ref().map(|_| "<footer>"))
                .finish(),
            Self::BuildScan { state, .. } => f
                .debug_struct("BuildScan")
                .field("state", state)
                .field("vxf", &"<vortex_file>")
                .finish(),
            Self::CalculateLayoutSplits {
                state, byte_range, ..
            } => f
                .debug_struct("CalculateLayoutSplits")
                .field("state", state)
                .field("vxf", &"<vortex_file>")
                .field("layout_reader", &"<layout_reader>")
                .field("byte_range", byte_range)
                .field("split_ranges", &"<once_cell>")
                .finish(),
            Self::PrepareScan {
                state, row_range, ..
            } => f
                .debug_struct("PrepareScan")
                .field("state", state)
                .field("vxf", &"<vortex_file>")
                .field("layout_reader", &"<layout_reader>")
                .field("row_range", row_range)
                .finish(),
            Self::PreparedScan {
                file_pruner,
                output_schema,
                stream_target_field,
                file_location,
                projector,
                ..
            } => f
                .debug_struct("PreparedScan")
                .field("scan", &"<repeated_scan>")
                .field(
                    "file_pruner",
                    &file_pruner.as_ref().map(|_| "<file_pruner>"),
                )
                .field("output_schema", output_schema)
                .field("stream_target_field", stream_target_field)
                .field("file_location", file_location)
                .field("projector", projector)
                .finish(),
            Self::Done => f.debug_struct("Done").finish(),
        }
    }
}

impl std::fmt::Debug for FileOpenState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileOpenState")
            .field("file", &self.file)
            .field("output_schema", &self.output_schema)
            .field("session", &self.session)
            .field("labels", &self.labels)
            .field("projection", &self.projection)
            .field("filter", &self.filter)
            .field(
                "file_pruner",
                &self.file_pruner.as_ref().map(|_| "<file_pruner>"),
            )
            .field("expr_adapter_factory", &self.expr_adapter_factory)
            .field("file_metadata_cache", &self.file_metadata_cache)
            .field("unified_file_schema", &self.unified_file_schema)
            .field("limit", &self.limit)
            .field("layout_readers", &self.layout_readers)
            .field("natural_splits", &self.natural_splits)
            .field("has_output_ordering", &self.has_output_ordering)
            .field("expr_convertor", &"<expr_convertor>")
            .field("projection_pushdown", &self.projection_pushdown)
            .field("scan_concurrency", &self.scan_concurrency)
            .finish_non_exhaustive()
    }
}

struct VortexStreamMorsel {
    inner: BoxStream<'static, DFResult<RecordBatch>>,
}

impl std::fmt::Debug for VortexStreamMorsel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VortexStreamMorsel").finish_non_exhaustive()
    }
}

impl Morsel for VortexStreamMorsel {
    fn into_stream(self: Box<Self>) -> BoxStream<'static, DFResult<RecordBatch>> {
        self.inner
    }
}

impl MorselPlanner for VortexMorselPlanner {
    #[allow(clippy::cognitive_complexity)]
    fn plan(self: Box<Self>) -> DFResult<Option<MorselPlan>> {
        let VortexMorselPlanner { state } = *self;
        let state = match state {
            State::Done => return Ok(None),
            state => state.transition()?,
        };

        match state {
            State::OpenFile { state, footer } => {
                let footer_cache_hit = footer.is_some();
                let mut open_opts = state
                    .session
                    .open_options()
                    .with_file_size(state.file.object_meta.size)
                    .with_metrics_registry(Arc::clone(&state.metrics_registry))
                    .with_labels(state.labels.clone());

                if let Some(footer) = footer {
                    open_opts = open_opts.with_footer(footer);
                }

                Ok(Some(
                    MorselPlan::new().with_pending_planner(
                        async move {
                            let vxf =
                                open_opts
                                    .open_read(state.reader.clone())
                                    .await
                                    .map_err(|e| {
                                        exec_datafusion_err!("Failed to open Vortex file {e}")
                                    })?;

                            // On a miss, cache the parsed footer so other partitions and later executions
                            // skip the footer fetch and parse. `infer_schema`/`infer_stats` also populate
                            // this cache, but only when planning goes through `VortexFormat`.
                            if !footer_cache_hit && let Some(cache) = &state.file_metadata_cache {
                                cache.put(
                                    state.file.path(),
                                    CachedFileMetadataEntry::new(
                                        state.file.object_meta.clone(),
                                        Arc::new(CachedVortexMetadata::new(&vxf)),
                                    ),
                                );
                            }

                            let new_state = if vxf.row_count() == 0 {
                                State::Done
                            } else {
                                State::BuildScan { state, vxf }
                            };

                            Ok(Box::new(VortexMorselPlanner { state: new_state })
                                as Box<dyn MorselPlanner>)
                        }
                        .in_current_span(),
                    ),
                ))
            }
            State::BuildScan { state, vxf } => {
                let layout_reader = layout_reader_for_file(
                    state.layout_readers.as_ref(),
                    &state.file.object_meta.location,
                    &vxf,
                )?;

                let Some(byte_range) = partial_file_byte_range(&state.file)? else {
                    return Ok(Some(MorselPlan::new().with_planners(vec![Box::new(
                        Self {
                            state: State::PrepareScan {
                                state,
                                vxf,
                                layout_reader,
                                row_range: None,
                            },
                        },
                    )])));
                };

                let split_ranges = natural_split_cell_for_file(
                    state.natural_splits.as_ref(),
                    &state.file.object_meta.location,
                );

                Ok(Some(MorselPlan::new().with_planners(vec![Box::new(
                    Self {
                        state: State::CalculateLayoutSplits {
                            state,
                            vxf,
                            layout_reader,
                            byte_range,
                            split_ranges,
                        },
                    },
                )])))
            }
            State::CalculateLayoutSplits {
                state,
                vxf,
                layout_reader,
                byte_range,
                split_ranges,
            } => {
                let total_size = state.file.object_meta.size;

                Ok(Some(
                    MorselPlan::new().with_pending_planner(
                        async move {
                            let natural_splits = Arc::clone(
                                split_ranges
                                    .get_or_try_init(|| async {
                                        compute_natural_splits(layout_reader.as_ref(), total_size)
                                    })
                                    .await?,
                            );

                            let new_state = match split_aligned_row_range(
                                byte_range,
                                natural_splits.as_ref(),
                            ) {
                                Some(row_range) => State::PrepareScan {
                                    state,
                                    vxf,
                                    layout_reader,
                                    row_range: Some(row_range),
                                },
                                None => State::Done,
                            };

                            Ok(Box::new(VortexMorselPlanner { state: new_state })
                                as Box<dyn MorselPlanner>)
                        }
                        .in_current_span(),
                    ),
                ))
            }
            State::PrepareScan {
                state,
                vxf,
                layout_reader,
                row_range,
            } => {
                let FileOpenState {
                    file,
                    output_schema,
                    session,
                    metrics_registry,
                    projection,
                    filter,
                    file_pruner,
                    expr_adapter_factory,
                    unified_file_schema,
                    limit,
                    has_output_ordering,
                    expr_convertor,
                    projection_pushdown,
                    scan_concurrency,
                    ..
                } = state;

                // This is the expected arrow types of the actual columns in the file, which might have different types
                // from the unified logical schema or miss
                let this_file_schema = Arc::new(calculate_physical_schema(
                    vxf.dtype(),
                    &unified_file_schema,
                    &session.arrow(),
                )?);

                let expr_adapter = expr_adapter_factory.create(
                    Arc::clone(&unified_file_schema),
                    Arc::clone(&this_file_schema),
                )?;

                let simplifier = PhysicalExprSimplifier::new(&this_file_schema);

                // The adapter rewrites the expressions to the local file schema, allowing
                // for schema evolution and divergence between the table's schema and individual files.
                let filter = filter
                    .map(|filter| {
                        // Expression might now reference columns that don't exist in the file, so we can give it
                        // another simplification pass.
                        simplifier.simplify(expr_adapter.rewrite(filter)?)
                    })
                    .transpose()?;
                let projection =
                    projection.try_map_exprs(|p| simplifier.simplify(expr_adapter.rewrite(p)?))?;

                let ProcessedProjection {
                    scan_projection,
                    leftover_projection,
                } = if projection_pushdown {
                    expr_convertor.split_projection(
                        projection.clone(),
                        &this_file_schema,
                        output_schema.as_ref(),
                    )?
                } else {
                    // When projection pushdown is disabled, read only the required columns
                    // and apply the full projection after the scan.
                    expr_convertor.no_pushdown_projection(projection.clone(), &this_file_schema)?
                };

                // The schema of the stream returned from the vortex scan.
                // We use a reference schema for types that don't roundtrip (Dictionary, Utf8, etc.).
                let scan_dtype = scan_projection.return_dtype(vxf.dtype()).map_err(|_e| {
                    exec_datafusion_err!("Couldn't get the dtype for the underlying Vortex scan")
                })?;

                // When projection pushdown is enabled, the scan outputs the projected columns.
                // When disabled, the scan outputs raw columns and the projection is applied after.
                let scan_reference_schema = if projection_pushdown {
                    (*output_schema).clone()
                } else {
                    // Build schema from the raw columns being read
                    let column_indices = projection.column_indices();
                    let fields: Vec<_> = column_indices
                        .into_iter()
                        .map(|idx| this_file_schema.field(idx).clone())
                        .collect();
                    Schema::new_with_metadata(fields, this_file_schema.metadata().clone())
                };
                let stream_schema = calculate_physical_schema(
                    &scan_dtype,
                    &scan_reference_schema,
                    &session.arrow(),
                )?;

                let leftover_projection = leftover_projection
                    .try_map_exprs(|expr| reassign_expr_columns(expr, &stream_schema))?;
                let projector = leftover_projection.make_projector(&stream_schema)?;

                let mut scan_builder =
                    ScanBuilder::new(session.clone(), Arc::clone(&layout_reader));

                if let Some(vortex_plan) = file.extensions.get::<VortexAccessPlan>() {
                    scan_builder = vortex_plan.apply_to_builder(scan_builder);
                }

                if let Some(row_range) = row_range {
                    scan_builder = scan_builder.with_row_range(row_range);
                }

                let filter = filter
                    .and_then(|f| {
                        // Verify that all filters we've accepted from DataFusion get pushed down.
                        // This will only fail if the user has not configured a suitable
                        // PhysicalExprAdapterFactory on the file source to handle rewriting the
                        // expression to handle missing/reordered columns in the Vortex file.
                        let (pushed, unpushed): (Vec<PhysicalExprRef>, Vec<PhysicalExprRef>) =
                            split_conjunction(&f)
                                .into_iter()
                                .cloned()
                                .partition(|expr| {
                                    expr_convertor.can_be_pushed_down(expr, &this_file_schema)
                                });

                        if !unpushed.is_empty() {
                            return Some(Err(exec_datafusion_err!(
                                r#"VortexSource accepted but failed to push {} filters.
                            This should never happen if you have a properly configured
                            PhysicalExprAdapterFactory configured on the source.

                            Failed filters:

                            {unpushed:#?}
                            "#,
                                unpushed.len()
                            )));
                        }

                        make_vortex_predicate(expr_convertor.as_ref(), &pushed).transpose()
                    })
                    .transpose()?;

                if let Some(limit) = limit {
                    scan_builder = scan_builder.with_limit(limit);
                }

                if let Some(concurrency) = scan_concurrency {
                    scan_builder = scan_builder.with_concurrency(concurrency);
                }

                let scan = scan_builder
                    .with_metrics_registry(metrics_registry)
                    .with_projection(scan_projection)
                    .with_some_filter(filter)
                    .with_ordered(has_output_ordering)
                    .prepare()
                    .map_err(|e| exec_datafusion_err!("Failed to prepare Vortex scan: {e}"))?;

                let stream_target_field =
                    Field::new_struct("", stream_schema.fields().clone(), false);
                let file_location = file.object_meta.location;

                Ok(Some(MorselPlan::new().with_planners(vec![Box::new(
                    Self {
                        state: State::PreparedScan {
                            scan,
                            file_pruner,
                            output_schema,
                            session,
                            stream_target_field,
                            file_location,
                            projector,
                        },
                    },
                )])))
            }
            State::PreparedScan {
                scan,
                file_pruner,
                output_schema,
                session,
                stream_target_field,
                file_location,
                projector,
            } => {
                let stream = scan
                    .execute_array_stream(None)
                    .map_err(|e| exec_datafusion_err!("Failed to create Vortex stream: {e}"))?
                    // Convert to Arrow inline on the polling thread: DataFusion sources are expected
                    // to do their CPU work inside `poll_next`, and spawning this onto the blocking
                    // pool oversubscribes the CPU.
                    .map(move |chunk| {
                        let mut ctx = session.create_execution_ctx();
                        chunk.and_then(|chunk| {
                            let arrow_session = ctx.session().clone();
                            let arrow = arrow_session.arrow().execute_arrow(
                                chunk,
                                Some(&stream_target_field),
                                &mut ctx,
                            )?;
                            Ok(RecordBatch::from(arrow.as_struct().clone()))
                        })
                    })
                    .map_err(move |e: VortexError| vortex_file_read_error(&file_location, e))
                    .map(move |batch| -> DFResult<RecordBatch> {
                        let batch = if projector.projection().as_ref().is_empty() {
                            batch
                        } else {
                            batch.and_then(|b| projector.project_batch(&b))
                        }?;

                        let (_, columns, row_count) = batch.into_parts();
                        RecordBatch::try_new_with_options(
                            Arc::clone(&output_schema),
                            columns,
                            &RecordBatchOptions::new().with_row_count(Some(row_count)),
                        )
                        .map_err(Into::into)
                    })
                    .boxed();

                let stream = if let Some(file_pruner) = file_pruner {
                    PrunableStream::new(file_pruner, stream).boxed()
                } else {
                    stream
                };

                Ok(Some(MorselPlan::new().with_morsels(vec![
                    Box::new(VortexStreamMorsel { inner: stream }) as Box<dyn Morsel>,
                ])))
            }
            State::Done => Ok(None),
            new_state => Ok(Some(
                MorselPlan::new().with_planners(vec![Box::new(Self { state: new_state })]),
            )),
        }
    }
}

fn layout_reader_for_file(
    layout_readers: &DashMap<Path, Weak<dyn LayoutReader>>,
    path: &Path,
    vxf: &VortexFile,
) -> DFResult<Arc<dyn LayoutReader>> {
    match layout_readers.entry(path.clone()) {
        Entry::Occupied(mut entry) => {
            if let Some(reader) = entry.get().upgrade() {
                tracing::trace!("reusing layout reader for {}", entry.key());
                Ok(reader)
            } else {
                tracing::trace!("creating layout reader for {}", entry.key());
                let reader = vxf.layout_reader().map_err(|error| {
                    DataFusionError::Execution(format!("Failed to create layout reader: {error}"))
                })?;
                entry.insert(Arc::downgrade(&reader));
                Ok(reader)
            }
        }
        Entry::Vacant(entry) => {
            tracing::trace!("creating layout reader for {}", entry.key());
            let reader = vxf.layout_reader().map_err(|error| {
                DataFusionError::Execution(format!("Failed to create layout reader: {error}"))
            })?;
            entry.insert(Arc::downgrade(&reader));
            Ok(reader)
        }
    }
}

fn partial_file_byte_range(file: &PartitionedFile) -> DFResult<Option<Range<u64>>> {
    let Some(file_range) = file.range.as_ref() else {
        return Ok(None);
    };

    let byte_range = Range {
        start: u64::try_from(file_range.start)
            .map_err(|_| exec_datafusion_err!("Vortex file range start is negative"))?,
        end: u64::try_from(file_range.end)
            .map_err(|_| exec_datafusion_err!("Vortex file range end is negative"))?,
    };

    Ok((byte_range.start != 0 || byte_range.end != file.object_meta.size).then_some(byte_range))
}

fn natural_split_cell_for_file(
    natural_splits: &NaturalSplitCache,
    path: &Path,
) -> Arc<OnceCell<Arc<NaturalSplits>>> {
    match natural_splits.entry(path.clone()) {
        Entry::Occupied(entry) => Arc::clone(entry.get()),
        Entry::Vacant(entry) => {
            let split_ranges = Arc::new(OnceCell::new());
            entry.insert(Arc::clone(&split_ranges));
            split_ranges
        }
    }
}

fn compute_natural_splits(
    layout_reader: &dyn LayoutReader,
    total_size: u64,
) -> DFResult<Arc<NaturalSplits>> {
    let row_count = layout_reader.row_count();
    let row_range = 0..row_count;
    let row_boundaries = SplitBy::Layout
        .splits(layout_reader, &row_range, &[FieldMask::All])
        .map_err(|e| exec_datafusion_err!("Failed to compute Vortex natural splits: {e}"))?;

    Ok(Arc::new(NaturalSplits::new(
        row_boundaries.into(),
        total_size,
    )))
}

/// Translate a DataFusion byte range to the contiguous natural split ranges it owns.
/// Most splits are assigned by midpoint, but the leading split stays with the range that owns
/// byte 0 so a tiny first byte range still claims the first rows.
fn split_aligned_row_range(
    byte_range: Range<u64>,
    natural_splits: &NaturalSplits,
) -> Option<Range<u64>> {
    if byte_range.start >= byte_range.end {
        return None;
    }

    let first_split = natural_splits
        .assignment_bytes
        .partition_point(|&assignment_byte| assignment_byte < byte_range.start);
    let after_last_split = natural_splits
        .assignment_bytes
        .partition_point(|&assignment_byte| assignment_byte < byte_range.end);
    if first_split == after_last_split {
        return None;
    }

    Some(
        natural_splits.row_boundaries[first_split]..natural_splits.row_boundaries[after_last_split],
    )
}

fn split_assignment_byte(
    idx: usize,
    split_range: &Range<u64>,
    row_count: u64,
    total_size: u64,
) -> u64 {
    if idx == 0 && split_range.start == 0 {
        // Byte 0 is the only stable representative for the leading split. A midpoint can fall
        // into the next DataFusion byte range and leave the first range with no rows to read.
        0
    } else {
        split_midpoint_to_byte(split_range, row_count, total_size)
    }
}

fn split_midpoint_to_byte(split_range: &Range<u64>, row_count: u64, total_size: u64) -> u64 {
    let midpoint_row = split_range.start + (split_range.end - split_range.start) / 2;
    let midpoint_byte = (u128::from(midpoint_row) * u128::from(total_size)) / u128::from(row_count);

    u64::try_from(midpoint_byte).vortex_expect("midpoint byte projection should fit into u64")
}

fn vortex_file_read_error(path: &Path, error: VortexError) -> DataFusionError {
    DataFusionError::External(Box::new(
        error.with_context(format!("Failed to read Vortex file: {path}")),
    ))
}

#[cfg(test)]
mod tests;

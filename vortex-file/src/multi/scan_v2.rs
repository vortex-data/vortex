// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ScanPlan-backed multi-file data source.

use std::any::Any;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use tracing::Instrument;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::forms::conjuncts;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::root::Root;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::filesystem::FileListing;
use vortex_io::filesystem::FileSystemRef;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layout_v2::LayoutScanPlanCtx;
use vortex_layout::scan::v2::validate_temporal_comparisons;
use vortex_layout::scan::v2::with_row_idx;
use vortex_mask::Mask;
use vortex_metrics::MetricsRegistry;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::PlannedMorselScan;
use vortex_scan::PlannedMorselScanRef;
use vortex_scan::ScanMeta;
use vortex_scan::ScanRequest as DataSourceScanRequest;
use vortex_scan::ScanScheduler;
use vortex_scan::ScanSchedulerSessionExt;
use vortex_scan::plan::EvidenceScope;
use vortex_scan::plan::OwnedRowScope;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedAggregateRef;
use vortex_scan::plan::PreparedEvidenceRef;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PreparedStats;
use vortex_scan::plan::PreparedStatsRef;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::ReadContext;
use vortex_scan::plan::ReadStep;
use vortex_scan::plan::ReadTask;
use vortex_scan::plan::ReadTaskOutput;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::ScanState;
use vortex_scan::plan::ScanStateRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::downcast_state;
use vortex_scan::plan::evidence::EvidenceFragment;
use vortex_scan::plan::evidence::PredicateEvidence;
use vortex_scan::plan::evidence::PredicateEvidenceKind;
use vortex_scan::plan::evidence::PredicateId;
use vortex_scan::plan::evidence::PredicateVersion;
use vortex_scan::plan::request::EvidenceMode;
use vortex_scan::plan::request::OwnedEvidenceRequest;
use vortex_scan::plan::request::ScanRequest;
use vortex_scan::read::ReadResults;
use vortex_scan::read::ReadStore;
use vortex_scan::read::ReadStoreRef;
use vortex_scan::read::ScanIoPhase;
use vortex_scan::read::ScanRead;
use vortex_scan::selection::Selection;
use vortex_scan::task::ScanStep;
use vortex_scan::task::ScanStepResult;
use vortex_scan::task::ScanTask;
use vortex_scan::task::ScanTaskBox;
use vortex_scan::task::ScanTaskLane;
use vortex_scan::task::ScanTaskQueue;
use vortex_scan::task::ScanTaskRead;
use vortex_scan::task::scan_task_read_bytes;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use super::MultiFileDataSource;
use super::create_local_filesystem;
use super::open_file;
use crate::FileStatistics;
use crate::VortexFile;
use crate::VortexOpenOptions;

const DEFAULT_CONCURRENCY: usize = 8;
const IDEAL_SPLIT_SIZE: u64 = 100_000;
const MAX_SELECTION_RANGE_SIZE: u64 = IDEAL_SPLIT_SIZE / 25;
const MIN_SELECTION_GAP_BETWEEN_RANGES: u64 = IDEAL_SPLIT_SIZE / 2;
/// Below this demanded-row density, evaluate a residual predicate over only the demanded rows
/// (filter-first) rather than the whole morsel.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;
const INLINE_ZERO_READ_EVIDENCE_MAX_PRIORITY: u64 = 100_150;
const SCAN_SCOPE_MIN_PREDICATE_COST: u64 = 100;

struct FileStatsScanPlan {
    data: ScanPlanRef,
    stats: Arc<FileStatistics>,
    fields: StructFields,
    row_count: u64,
}

struct FileStatsExprScanPlan {
    data: ScanPlanRef,
    stats: Arc<FileStatistics>,
    field_idx: usize,
    field_dtype: DType,
    row_count: u64,
}

struct FilePreparedStats {
    stats: StatsSet,
    field_dtype: DType,
    row_count: u64,
    funcs: Vec<AggregateFnRef>,
}

impl FileStatsScanPlan {
    fn try_new(
        data: ScanPlanRef,
        stats: Arc<FileStatistics>,
        dtype: &DType,
        row_count: u64,
    ) -> Option<Self> {
        let fields = dtype.as_struct_fields_opt()?.clone();
        Some(Self {
            data,
            stats,
            fields,
            row_count,
        })
    }

    fn pushed_field(&self, expr: &Expression) -> Option<(usize, FieldName, DType)> {
        let name = root_field(expr)?;
        let field_idx = self.fields.find(name)?;
        let field_dtype = self.fields.field_by_index(field_idx)?;
        Some((field_idx, name.clone(), field_dtype))
    }
}

impl ScanPlan for FileStatsScanPlan {
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        cx.init_plan(&self.data)
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        let Some(data) = Arc::clone(&self.data).try_push_expr(expr, cx)? else {
            return Ok(None);
        };
        let Some((field_idx, _name, field_dtype)) = self.pushed_field(expr) else {
            return Ok(Some(data));
        };
        Ok(Some(Arc::new(FileStatsExprScanPlan {
            data,
            stats: Arc::clone(&self.stats),
            field_idx,
            field_dtype,
            row_count: self.row_count,
        })))
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        Arc::clone(&self.data).prepare_read(cx)
    }

    fn prepare_evidence(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>> {
        Arc::clone(&self.data).prepare_evidence(cx)
    }

    fn prepare_field_stats(
        self: Arc<Self>,
        field_path: &FieldPath,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedStatsRef>> {
        if field_path.parts().len() != 1 {
            return Arc::clone(&self.data).prepare_field_stats(field_path, funcs, cx);
        }
        let Some(name) = field_path.parts()[0].as_name() else {
            return Arc::clone(&self.data).prepare_field_stats(field_path, funcs, cx);
        };
        let Some(field_idx) = self.fields.find(name) else {
            return Ok(None);
        };
        let Some(field_dtype) = self.fields.field_by_index(field_idx) else {
            return Ok(None);
        };
        let stats = self.stats.stats_sets()[field_idx].clone();
        Ok(Some(Arc::new(FilePreparedStats {
            stats,
            field_dtype,
            row_count: self.row_count,
            funcs: funcs.to_vec(),
        })))
    }

    fn prepare_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedAggregateRef>> {
        Arc::clone(&self.data).prepare_aggregate_partial(funcs, cx)
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.data.split_hints()
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<ScanStateRef>(state)?;
        self.data.release(frontier, state.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "file_stats:")?;
        self.data.fmt_chain(f)
    }
}

impl ScanPlan for FileStatsExprScanPlan {
    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        cx.init_plan(&self.data)
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        Arc::clone(&self.data).try_push_expr(expr, cx)
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        Arc::clone(&self.data).prepare_read(cx)
    }

    fn prepare_evidence(
        self: Arc<Self>,
        cx: &mut PrepareCtx,
    ) -> VortexResult<Vec<PreparedEvidenceRef>> {
        Arc::clone(&self.data).prepare_evidence(cx)
    }

    fn prepare_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedAggregateRef>> {
        Arc::clone(&self.data).prepare_aggregate_partial(funcs, cx)
    }

    fn prepare_field_stats(
        self: Arc<Self>,
        field_path: &FieldPath,
        funcs: &[AggregateFnRef],
        cx: &mut PrepareCtx,
    ) -> VortexResult<Option<PreparedStatsRef>> {
        if !field_path.is_root() {
            return Arc::clone(&self.data).prepare_field_stats(field_path, funcs, cx);
        }
        let stats = self.stats.stats_sets()[self.field_idx].clone();
        Ok(Some(Arc::new(FilePreparedStats {
            stats,
            field_dtype: self.field_dtype.clone(),
            row_count: self.row_count,
            funcs: funcs.to_vec(),
        })))
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.data.split_hints()
    }

    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<ScanStateRef>(state)?;
        self.data.release(frontier, state.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "file_stats_expr:")?;
        self.data.fmt_chain(f)
    }
}

impl PreparedStats for FilePreparedStats {
    fn init_state(&self, _ctx: &VortexSession) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(()))
    }

    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        _io: &'a ReadContext,
        _state: &'a ScanState,
    ) -> BoxFuture<'a, VortexResult<Vec<Precision<Scalar>>>> {
        Box::pin(async move {
            if range != (0..self.row_count) {
                return Ok(absent_statistics(&self.funcs));
            }
            self.funcs
                .iter()
                .map(|func| self.stat_for_func(func))
                .collect()
        })
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "file_stats")
    }
}

impl FilePreparedStats {
    fn stat_for_func(&self, func: &AggregateFnRef) -> VortexResult<Precision<Scalar>> {
        let Some(stat) = Stat::from_aggregate_fn(func) else {
            return Ok(Precision::Absent);
        };
        let Some(dtype) = func.return_dtype(&self.field_dtype) else {
            return Ok(Precision::Absent);
        };
        self.stats
            .get(stat)
            .map(|value| Scalar::try_new(dtype, Some(value)))
            .transpose()
    }
}

fn root_field(expr: &Expression) -> Option<&FieldName> {
    let name = expr.as_opt::<GetItem>()?;
    expr.child(0).is::<Root>().then_some(name)
}

fn root_field_path(expr: &Expression) -> Option<FieldPath> {
    if expr.is::<Root>() {
        return Some(FieldPath::root());
    }
    root_field(expr).cloned().map(FieldPath::from_name)
}

/// Static cost estimate for a filter conjunct, used to order predicate evaluation cheapest-first.
///
/// We sum a per-node cost over the whole expression tree. Primitive comparisons, null checks and
/// data access (`vortex.binary`, `vortex.between`, `vortex.is_null`, `vortex.get_item`, ...) are
/// cheap; per-row string/byte work (`vortex.like`, `vortex.byte_length`, `vortex.list.contains`)
/// and opaque/dynamic functions are expensive. Unrecognized functions get a moderate cost so they
/// sort after primitives but ahead of known-expensive matchers.
fn predicate_cost(expr: &Expression) -> u64 {
    fn node_cost(expr: &Expression) -> u64 {
        match expr.id().as_str() {
            // Free or near-free structural / access nodes.
            "vortex.root" | "vortex.literal" | "vortex.get_item" => 0,
            // Cheap primitive predicates.
            "vortex.binary" | "vortex.between" | "vortex.is_null" | "vortex.is_not_null"
            | "vortex.not" | "vortex.fill_null" | "vortex.cast" => 1,
            // Expensive per-row string / byte / matching work, and fallible UDFs.
            "vortex.like" | "vortex.byte_length" | "vortex.list.contains" => 100,
            "vortex.dynamic" | "vortex.variant_get" | "vortex.parquet.variant" => 100,
            // Unknown functions: more expensive than primitives, cheaper than known matchers.
            _ => 10,
        }
    }

    let mut cost = node_cost(expr);
    for child in expr.children().iter() {
        cost = cost.saturating_add(predicate_cost(child));
    }
    cost
}

fn absent_statistics(funcs: &[AggregateFnRef]) -> Vec<Precision<Scalar>> {
    funcs.iter().map(|_| Precision::Absent).collect()
}

fn scalar_precision_to_value(precision: Precision<Scalar>) -> Precision<ScalarValue> {
    match precision {
        Precision::Exact(scalar) => scalar
            .into_value()
            .map(Precision::Exact)
            .unwrap_or(Precision::Absent),
        Precision::Inexact(scalar) => scalar
            .into_value()
            .map(Precision::Inexact)
            .unwrap_or(Precision::Absent),
        Precision::Absent => Precision::Absent,
    }
}

/// Build a scan2 [`DataSource`] from a multi-file builder.
pub(crate) async fn build_scan_plan_data_source(
    builder: MultiFileDataSource,
) -> VortexResult<ScanPlanDataSource> {
    if builder.glob_sources.is_empty() {
        vortex_bail!("MultiFileDataSource requires at least one glob pattern");
    }

    let local_fs: Option<FileSystemRef> = builder
        .glob_sources
        .iter()
        .any(|(_, fs)| fs.is_none())
        .then(|| create_local_filesystem(&builder.session))
        .transpose()?;

    let mut all_files: Vec<(FileListing, FileSystemRef)> = Vec::new();
    for (glob, maybe_fs) in &builder.glob_sources {
        let fs = maybe_fs
            .as_ref()
            .or(local_fs.as_ref())
            .map(Arc::clone)
            .unwrap_or_else(|| unreachable!("local_fs is set when any glob lacks a filesystem"));
        let files: Vec<FileListing> = fs.glob(glob)?.try_collect().await?;
        for file in files {
            all_files.push((file, Arc::clone(&fs)));
        }
    }

    if all_files.is_empty() {
        let globs: Vec<_> = builder
            .glob_sources
            .iter()
            .map(|(glob, _)| glob.as_str())
            .collect();
        vortex_bail!("No files matched the glob pattern(s): {:?}", globs);
    }

    let (first_file_listing, first_fs) = &all_files[0];
    let first_file = open_file(
        first_fs,
        first_file_listing,
        &builder.session,
        builder.metrics_registry.as_ref(),
        builder.open_options_fn.as_ref(),
    )
    .await?;

    let factories: Vec<Arc<dyn VortexFileFactory>> = all_files[1..]
        .iter()
        .map(|(file, fs)| {
            Arc::new(ScanPlanFileFactory {
                fs: Arc::clone(fs),
                file: file.clone(),
                session: builder.session.clone(),
                open_options_fn: Arc::clone(&builder.open_options_fn),
                metrics_registry: builder.metrics_registry.clone(),
            }) as Arc<dyn VortexFileFactory>
        })
        .collect();

    Ok(ScanPlanDataSource::new_with_first(
        first_file,
        factories,
        &builder.session,
    ))
}

#[async_trait]
trait VortexFileFactory: 'static + Send + Sync {
    async fn open(&self) -> VortexResult<Option<VortexFile>>;
}

struct ScanPlanFileFactory {
    fs: FileSystemRef,
    file: FileListing,
    session: VortexSession,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

#[async_trait]
impl VortexFileFactory for ScanPlanFileFactory {
    async fn open(&self) -> VortexResult<Option<VortexFile>> {
        let file = open_file(
            &self.fs,
            &self.file,
            &self.session,
            self.metrics_registry.as_ref(),
            self.open_options_fn.as_ref(),
        )
        .await?;
        Ok(Some(file))
    }
}

enum ScanPlanChild {
    Opened(VortexFile),
    Deferred(Arc<dyn VortexFileFactory>),
}

/// Multi-file data source backed by scan2 ScanPlan plans.
pub struct ScanPlanDataSource {
    dtype: DType,
    session: VortexSession,
    children: Vec<ScanPlanChild>,
    concurrency: usize,
}

impl ScanPlanDataSource {
    fn new_with_first(
        first: VortexFile,
        remaining: Vec<Arc<dyn VortexFileFactory>>,
        session: &VortexSession,
    ) -> Self {
        let dtype = first.dtype().clone();
        let concurrency = get_available_parallelism().unwrap_or(DEFAULT_CONCURRENCY);

        let mut children = Vec::with_capacity(1 + remaining.len());
        children.push(ScanPlanChild::Opened(first));
        children.extend(remaining.into_iter().map(ScanPlanChild::Deferred));

        Self {
            dtype,
            session: session.clone(),
            children,
            concurrency,
        }
    }

    async fn open_files(&self, ordered: bool) -> VortexResult<Vec<(usize, VortexFile)>> {
        let jobs = self
            .children
            .iter()
            .enumerate()
            .map(|(idx, child)| match child {
                ScanPlanChild::Opened(file) => {
                    let file = file.clone();
                    async move { Ok(Some((idx, file))) }.boxed()
                }
                ScanPlanChild::Deferred(factory) => {
                    let factory = Arc::clone(factory);
                    async move {
                        factory
                            .open()
                            .instrument(tracing::info_span!("VortexFileFactory::open"))
                            .await
                            .map(|file| file.map(|file| (idx, file)))
                    }
                    .boxed()
                }
            })
            .collect::<Vec<BoxFuture<'static, VortexResult<Option<(usize, VortexFile)>>>>>();

        let files = if ordered {
            stream::iter(jobs)
                .buffered(self.concurrency)
                .try_filter_map(|file| async move { Ok(file) })
                .try_collect::<Vec<_>>()
                .await?
        } else {
            stream::iter(jobs)
                .buffer_unordered(self.concurrency)
                .try_filter_map(|file| async move { Ok(file) })
                .try_collect::<Vec<_>>()
                .await?
        };

        let mut files = files;
        files.sort_unstable_by_key(|(idx, _)| *idx);
        Ok(files)
    }
}

#[async_trait]
impl DataSource for ScanPlanDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        let mut sum: u64 = 0;
        let mut opened_count: u64 = 0;
        let mut deferred_count: u64 = 0;

        for child in &self.children {
            match child {
                ScanPlanChild::Opened(file) => {
                    opened_count += 1;
                    sum = sum.saturating_add(file.row_count());
                }
                ScanPlanChild::Deferred(_) => {
                    deferred_count += 1;
                }
            }
        }

        let total_count = opened_count + deferred_count;
        if total_count == 0 {
            return Precision::exact(0u64);
        }

        if deferred_count == 0 {
            Precision::exact(sum)
        } else if opened_count > 0 {
            let avg = sum / opened_count;
            Precision::inexact(avg.saturating_mul(total_count))
        } else {
            Precision::Absent
        }
    }

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("ScanPlanDataSource partitions are not yet serializable")
    }

    async fn plan_morsel_partitions(
        &self,
        scan_request: DataSourceScanRequest,
        target_partitions: usize,
    ) -> VortexResult<Option<PlannedMorselScanRef>> {
        if scan_request.ordered || scan_request.limit.is_some() {
            return Ok(None);
        }

        let target_partitions = target_partitions.max(1);
        let dtype = scan_request.projection.return_dtype(&self.dtype)?;

        let meta = ScanMeta {
            label: Some("scan2".to_string()),
        };
        let provider = self.session.scan_scheduler_provider();
        let scheduler = provider.scheduler_for_scan(&meta);

        let mut planned_files = Vec::new();
        let mut total_morsels = 0usize;
        for (partition_idx, file) in self.open_files(false).await? {
            let Some(request) = file_scan_request(partition_idx, &file, scan_request.clone())?
            else {
                continue;
            };
            let row_range = request
                .row_range
                .clone()
                .ok_or_else(|| vortex_err!("scan2 partition row range missing"))?;
            let prepared = Arc::new(PreparedScanPlan::try_new(&file, &request)?);
            let execution = Arc::new(ScanExecution::try_new(file, prepared, None)?);
            let ranges = execution.splits(&row_range)?;
            if ranges.is_empty() {
                continue;
            }
            total_morsels = total_morsels.saturating_add(ranges.len());
            planned_files.push((execution, ranges));
        }

        // The physical plan may expose more engine partitions than we can fill with morsels.
        // Keep only non-empty planned partitions; engine adapters can return empty streams for
        // any surplus advertised partitions.
        let partition_count = total_morsels.min(target_partitions);
        let mut partitions = vec![Vec::new(); partition_count];
        let mut morsel_idx = 0usize;
        for (execution, ranges) in planned_files {
            for range in ranges {
                let partition = morsel_idx % partition_count;
                partitions[partition].push(PlannedScanPlanMorsel {
                    execution: Arc::clone(&execution),
                    range,
                });
                morsel_idx = morsel_idx.saturating_add(1);
            }
        }

        let read_byte_budget = read_byte_budget(&scheduler);

        Ok(Some(Arc::new(PlannedScanPlanScan {
            dtype,
            partitions,
            handle: self.session.handle(),
            read_byte_budget,
        })))
    }

    async fn scan(&self, scan_request: DataSourceScanRequest) -> VortexResult<DataSourceScanRef> {
        let meta = ScanMeta {
            label: Some("scan2".to_string()),
        };
        let provider = self.session.scan_scheduler_provider();
        let scheduler = provider.scheduler_for_scan(&meta);

        let mut ready = VecDeque::new();
        let mut deferred = VecDeque::new();

        for (index, child) in self.children.iter().enumerate() {
            match child {
                ScanPlanChild::Opened(file) => ready.push_back((index, file.clone())),
                ScanPlanChild::Deferred(factory) => {
                    deferred.push_back((index, Arc::clone(factory)));
                }
            }
        }

        let dtype = scan_request.projection.return_dtype(&self.dtype)?;
        let limit_remaining = scan_request.limit.map(AtomicU64::new).map(Arc::new);

        Ok(Box::new(ScanPlanDataSourceScan {
            dtype,
            request: scan_request,
            ready,
            deferred,
            handle: self.session.handle(),
            concurrency: self.concurrency,
            scheduler,
            limit_remaining,
        }))
    }

    async fn statistics(
        &self,
        expr: &Expression,
        funcs: &[AggregateFnRef],
    ) -> VortexResult<Vec<Precision<Scalar>>> {
        if self.children.len() != 1 {
            return Ok(absent_statistics(funcs));
        }
        let ScanPlanChild::Opened(file) = &self.children[0] else {
            return Ok(absent_statistics(funcs));
        };
        scan_plan_file_statistics(file.clone(), expr, funcs).await
    }

    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet> {
        if field_path.parts().len() != 1 {
            return Ok(StatsSet::default());
        }
        let Some(field_name) = field_path.parts()[0].as_name() else {
            return Ok(StatsSet::default());
        };
        let funcs = Stat::all()
            .filter_map(|stat| stat.aggregate_fn().map(|func| (stat, func)))
            .collect::<Vec<_>>();
        let aggregate_funcs = funcs
            .iter()
            .map(|(_, func)| func.clone())
            .collect::<Vec<_>>();
        let stats = self
            .statistics(
                &vortex_array::expr::get_item(field_name, vortex_array::expr::root()),
                &aggregate_funcs,
            )
            .await?;
        let mut stats_set = StatsSet::default();
        for ((stat, _), value) in funcs.into_iter().zip(stats) {
            stats_set.set(stat, scalar_precision_to_value(value));
        }
        Ok(stats_set)
    }

    fn supports_morsel_partitioning(&self) -> bool {
        true
    }
}

struct ScanPlanDataSourceScan {
    dtype: DType,
    request: DataSourceScanRequest,
    ready: VecDeque<(usize, VortexFile)>,
    deferred: VecDeque<(usize, Arc<dyn VortexFileFactory>)>,
    handle: Handle,
    concurrency: usize,
    scheduler: Arc<ScanScheduler>,
    limit_remaining: Option<Arc<AtomicU64>>,
}

impl DataSourceScan for ScanPlanDataSourceScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        let count = self.ready.len() + self.deferred.len();
        if self.deferred.is_empty() {
            Precision::exact(count)
        } else {
            Precision::inexact(count)
        }
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        let Self {
            dtype: _,
            request,
            ready,
            deferred,
            handle,
            concurrency,
            scheduler,
            limit_remaining,
        } = *self;

        let ordered = request.ordered;
        let ready_stream = stream::iter(ready).map(Ok);
        let spawned = stream::iter(deferred).map(move |(index, factory)| {
            handle.spawn(async move {
                factory
                    .open()
                    .instrument(tracing::info_span!("VortexFileFactory::open"))
                    .await
                    .map(|file| file.map(|file| (index, file)))
            })
        });

        let deferred_stream = if ordered {
            spawned
                .buffered(concurrency)
                .filter_map(|result| async move {
                    match result {
                        Ok(Some(file)) => Some(Ok(file)),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    }
                })
                .boxed()
        } else {
            spawned
                .buffer_unordered(concurrency)
                .filter_map(|result| async move {
                    match result {
                        Ok(Some(file)) => Some(Ok(file)),
                        Ok(None) => None,
                        Err(error) => Some(Err(error)),
                    }
                })
                .boxed()
        };

        ready_stream
            .chain(deferred_stream)
            .filter_map(move |file_result| {
                let request = request.clone();
                let scheduler = Arc::clone(&scheduler);
                let limit_remaining = limit_remaining.clone();
                async move {
                    match file_result {
                        Ok((index, file)) => {
                            file_partition(index, file, request, scheduler, limit_remaining)
                                .transpose()
                        }
                        Err(error) => Some(Err(error)),
                    }
                }
            })
            .boxed()
    }
}

fn file_partition(
    partition_idx: usize,
    file: VortexFile,
    request: DataSourceScanRequest,
    scheduler: Arc<ScanScheduler>,
    limit_remaining: Option<Arc<AtomicU64>>,
) -> VortexResult<Option<PartitionRef>> {
    let Some(request) = file_scan_request(partition_idx, &file, request)? else {
        return Ok(None);
    };
    let row_range = request
        .row_range
        .clone()
        .ok_or_else(|| vortex_err!("scan2 partition row range missing"))?;
    let prepared = Arc::new(PreparedScanPlan::try_new(&file, &request)?);

    Ok(Some(Box::new(ScanPlanPartition {
        file,
        prepared,
        row_range,
        index: partition_idx,
        scheduler,
        limit_remaining,
    })))
}

pub(crate) fn scan_plan_file_stream(
    file: VortexFile,
    request: DataSourceScanRequest,
) -> VortexResult<SendableArrayStream> {
    let dtype = request.projection.return_dtype(file.dtype())?;
    let meta = ScanMeta {
        label: Some("scan2".to_string()),
    };
    let provider = file.session().scan_scheduler_provider();
    let scheduler = provider.scheduler_for_scan(&meta);

    let limit_remaining = request.limit.map(AtomicU64::new).map(Arc::new);
    let Some(partition) = file_partition(0, file, request, scheduler, limit_remaining)? else {
        return Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            stream::empty(),
        )));
    };
    partition.execute()
}

pub(crate) async fn scan_plan_file_statistics(
    file: VortexFile,
    expr: &Expression,
    funcs: &[AggregateFnRef],
) -> VortexResult<Vec<Precision<Scalar>>> {
    let mut stats = scan_plan_file_statistics_many(file, std::slice::from_ref(expr), funcs).await?;
    Ok(stats.pop().unwrap_or_else(|| absent_statistics(funcs)))
}

pub(crate) async fn scan_plan_file_statistics_many(
    file: VortexFile,
    exprs: &[Expression],
    funcs: &[AggregateFnRef],
) -> VortexResult<Vec<Vec<Precision<Scalar>>>> {
    let session = file.session().clone();
    let root = file.scan_plan_root()?;
    let reader = ReadContext::new(session);
    let mut result = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let plan = if let Some(field_path) = root_field_path(expr) {
            Arc::clone(&root).prepare_field_stats(
                &field_path,
                funcs,
                &mut PrepareCtx::new(reader.session().clone()),
            )?
        } else {
            let pushed = push_expr(&root, expr, file.dtype(), reader.session())?;
            pushed.prepare_field_stats(
                &FieldPath::root(),
                funcs,
                &mut PrepareCtx::new(reader.session().clone()),
            )?
        };
        let Some(plan) = plan else {
            result.push(absent_statistics(funcs));
            continue;
        };
        let state = plan.init_state(reader.session())?;
        result.push(
            plan.stats(0..file.row_count(), &reader, state.as_ref())
                .await?,
        );
    }
    Ok(result)
}

pub(crate) fn scan_plan_file_splits(file: &VortexFile) -> VortexResult<Vec<Range<u64>>> {
    let root = file.scan_plan_root()?;
    Ok(split_ranges_from_node(&root, file.row_count()))
}

pub(crate) async fn scan_plan_file_plan_splits(
    file: VortexFile,
    projection: &Expression,
) -> VortexResult<Vec<Range<u64>>> {
    let session = file.session().clone();
    let root = file.scan_plan_root()?;
    let pushed = push_expr(&root, projection, file.dtype(), &session)?;
    let Some(plan) = pushed.prepare_splits(&mut PrepareCtx::new(session.clone()))? else {
        return Ok(std::iter::once(0..file.row_count()).collect());
    };
    let reader = ReadContext::new(session.clone());
    let state = plan.init_state(&session)?;
    plan.splits(0..file.row_count(), &reader, state.as_ref())
        .await
}

fn split_ranges_from_node(node: &ScanPlanRef, row_count: u64) -> Vec<Range<u64>> {
    let mut points = Vec::new();
    if let Some(hints) = node.split_hints() {
        points.extend_from_slice(hints);
    }
    let points = normalize_split_points(row_count, points);
    natural_split_ranges(&points, None)
}

pub(crate) fn build_file_scan_plan_root(file: &VortexFile) -> VortexResult<ScanPlanRef> {
    let mut plan_request = ScanRequest::empty();
    let layout = file
        .footer()
        .layout2()
        .ok_or_else(|| vortex_err!("scan2 requires a v2 footer layout"))?;
    let ctx = LayoutScanPlanCtx::new(
        file.session().clone(),
        file.segment_source(),
        file.scan_plan_segment_future_cache(),
    );
    let root = layout.new_scan_plan(&mut plan_request, &ctx)?;
    let root = with_row_idx(root, file.dtype().clone(), 0);
    Ok(match file.footer().statistics().cloned() {
        Some(stats) => FileStatsScanPlan::try_new(
            Arc::clone(&root),
            Arc::new(stats),
            file.dtype(),
            file.row_count(),
        )
        .map(|node| Arc::new(node) as ScanPlanRef)
        .unwrap_or(root),
        None => root,
    })
}

fn file_scan_request(
    partition_idx: usize,
    file: &VortexFile,
    request: DataSourceScanRequest,
) -> VortexResult<Option<DataSourceScanRequest>> {
    let partition_idx_u64 = partition_idx as u64;
    if let Some(range) = &request.partition_range
        && !range.contains(&partition_idx_u64)
    {
        return Ok(None);
    }
    match &request.partition_selection {
        Selection::IncludeByIndex(buffer) => {
            if buffer.as_slice().binary_search(&partition_idx_u64).is_err() {
                return Ok(None);
            }
        }
        Selection::ExcludeByIndex(buffer) => {
            if buffer.as_slice().binary_search(&partition_idx_u64).is_ok() {
                return Ok(None);
            }
        }
        _ => {}
    };

    let row_count = file.row_count();
    let row_range = request.row_range.clone().unwrap_or(0..row_count);
    check_range(&row_range, row_count)?;

    if let Some(filter) = &request.filter
        && file.can_prune(filter)?
    {
        return Ok(None);
    }

    Ok(Some(DataSourceScanRequest {
        row_range: Some(row_range),
        ..request
    }))
}

type QueuedWork = ScanTaskBox<WorkOutput>;

struct LaunchedWorkOutput {
    lane: ScanTaskLane,
    reads: Vec<ScanTaskRead>,
    output: VortexResult<WorkPoll>,
}

struct EvidenceWorkOutput {
    morsel_id: usize,
    predicate_idx: usize,
    version: PredicateVersion,
    source: EvidenceWorkSource,
    fragments: Vec<EvidenceFragment>,
}

struct ScanEvidenceWorkOutput {
    execution: Arc<ScanExecution>,
    morsel_id: usize,
    predicate_idx: usize,
    evidence_idx: usize,
    version: PredicateVersion,
    fragments: Option<Vec<EvidenceFragment>>,
}

enum EvidenceWorkSource {
    Provider,
    Predicate { input_rows: usize, pass_rows: usize },
}

struct ProjectionWorkOutput {
    morsel_id: usize,
    array: ArrayRef,
}

enum WorkOutput {
    Evidence(EvidenceWorkOutput),
    ScanEvidence(ScanEvidenceWorkOutput),
    Projection(ProjectionWorkOutput),
}

enum WorkPoll {
    Ready(WorkOutput),
    Pending(QueuedWork),
}

struct ScanEvidenceWaitTask {
    execution: Arc<ScanExecution>,
    morsel_id: usize,
    predicate_idx: usize,
    evidence_idx: usize,
    version: PredicateVersion,
    lane: ScanTaskLane,
    priority: u64,
}

impl ScanTask<WorkOutput> for ScanEvidenceWaitTask {
    fn morsel_id(&self) -> usize {
        self.morsel_id
    }

    fn phase(&self) -> ScanIoPhase {
        ScanIoPhase::EvidenceProbe
    }

    fn lane(&self) -> ScanTaskLane {
        self.lane
    }

    fn reads(&self) -> &[ScanTaskRead] {
        &[]
    }

    fn priority(&self) -> u64 {
        self.priority
    }

    fn into_step(self: Box<Self>) -> VortexResult<ScanStep<WorkOutput>> {
        let task = *self;
        let morsel_id = task.morsel_id;
        let lane = task.lane;
        let priority = task.priority;
        Ok(ScanStep::new(
            morsel_id,
            ScanIoPhase::EvidenceProbe,
            lane,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            move |_| {
                if !task.execution.scan_evidence_provider_ready(
                    task.predicate_idx,
                    task.evidence_idx,
                    task.version,
                ) && task.execution.predicates[task.predicate_idx].version() == task.version
                {
                    return Ok(ScanStepResult::Continue(Box::new(task)));
                }

                Ok(ScanStepResult::Ready(WorkOutput::ScanEvidence(
                    ScanEvidenceWorkOutput {
                        execution: Arc::clone(&task.execution),
                        morsel_id: task.morsel_id,
                        predicate_idx: task.predicate_idx,
                        evidence_idx: task.evidence_idx,
                        version: task.version,
                        fragments: None,
                    },
                )))
            },
        )
        .with_priority(priority))
    }
}

struct PredicateReadWorkState {
    execution: Arc<ScanExecution>,
    morsel_id: usize,
    predicate_idx: usize,
    version: PredicateVersion,
    range: Range<u64>,
    need: Mask,
    compact: bool,
    len: usize,
    priority: u64,
    lane: ScanTaskLane,
}

struct PredicateReadWorkTask {
    state: PredicateReadWorkState,
    step: ReadStep,
    reads: Vec<ScanTaskRead>,
}

impl PredicateReadWorkTask {
    fn try_new(state: PredicateReadWorkState, task: Box<dyn ReadTask>) -> VortexResult<Self> {
        let step = task.into_step()?;
        let reads = ScanTaskRead::from_scan_reads(&step.required_reads);
        Ok(Self { state, step, reads })
    }
}

impl ScanTask<WorkOutput> for PredicateReadWorkTask {
    fn morsel_id(&self) -> usize {
        self.state.morsel_id
    }

    fn phase(&self) -> ScanIoPhase {
        ScanIoPhase::PredicateRead
    }

    fn lane(&self) -> ScanTaskLane {
        self.state.lane
    }

    fn reads(&self) -> &[ScanTaskRead] {
        &self.reads
    }

    fn priority(&self) -> u64 {
        self.state.priority
    }

    fn into_step(self: Box<Self>) -> VortexResult<ScanStep<WorkOutput>> {
        let task = *self;
        let state = task.state;
        let morsel_id = state.morsel_id;
        let lane = state.lane;
        let reads = task.reads.clone();
        let priority = state.priority;
        let read_step = task.step;
        Ok(ScanStep::new(
            morsel_id,
            ScanIoPhase::PredicateRead,
            lane,
            reads,
            read_step.required_reads,
            read_step.prefetch_reads,
            move |results| {
                let reader = state.execution.read_context();
                let mut ctx = state.execution.session.create_execution_ctx();
                let array = match read_step.continuation.run(&reader, &mut ctx, results)? {
                    ReadTaskOutput::Ready(array) => array,
                    ReadTaskOutput::Continue(read_task) => {
                        return Ok(ScanStepResult::Continue(Box::new(
                            PredicateReadWorkTask::try_new(state, read_task)?,
                        )));
                    }
                };
                let result = if state.compact {
                    let compact = array.null_as_false().execute(&mut ctx)?;
                    if compact.len() != state.need.true_count() {
                        vortex_bail!(
                            "compacted residual result length {} does not match demanded row count {}",
                            compact.len(),
                            state.need.true_count()
                        );
                    }
                    state.need.intersect_by_rank(&compact)
                } else {
                    array.null_as_false().execute(&mut ctx)?
                };
                if result.len() != state.len {
                    vortex_bail!(
                        "residual result length {} does not match morsel length {}",
                        result.len(),
                        state.len
                    );
                }
                let pass = &result & &state.need;
                let input_rows = state.need.true_count();
                let pass_rows = pass.true_count();
                let exact = !&state.need | &pass;
                Ok(ScanStepResult::Ready(WorkOutput::Evidence(
                    EvidenceWorkOutput {
                        morsel_id: state.morsel_id,
                        predicate_idx: state.predicate_idx,
                        version: state.version,
                        source: EvidenceWorkSource::Predicate {
                            input_rows,
                            pass_rows,
                        },
                        fragments: vec![EvidenceFragment::new(
                            state.range.clone(),
                            PredicateEvidenceKind::ExactMask(exact),
                        )],
                    },
                )))
            },
        )
        .with_priority(priority))
    }
}

struct ProjectionReadWorkTask {
    execution: Arc<ScanExecution>,
    step: ReadStep,
    reads: Vec<ScanTaskRead>,
    morsel_id: usize,
}

impl ProjectionReadWorkTask {
    fn try_new(
        execution: Arc<ScanExecution>,
        task: Box<dyn ReadTask>,
        morsel_id: usize,
    ) -> VortexResult<Self> {
        let step = task.into_step()?;
        let reads = ScanTaskRead::from_scan_reads(&step.required_reads);
        Ok(Self {
            execution,
            step,
            reads,
            morsel_id,
        })
    }
}

impl ScanTask<WorkOutput> for ProjectionReadWorkTask {
    fn morsel_id(&self) -> usize {
        self.morsel_id
    }

    fn phase(&self) -> ScanIoPhase {
        ScanIoPhase::ProjectionRead
    }

    fn lane(&self) -> ScanTaskLane {
        ScanTaskLane::Projection
    }

    fn reads(&self) -> &[ScanTaskRead] {
        &self.reads
    }

    fn priority(&self) -> u64 {
        ScanStep::<WorkOutput>::DEFAULT_PRIORITY
    }

    fn into_step(self: Box<Self>) -> VortexResult<ScanStep<WorkOutput>> {
        let task = *self;
        let reads = task.reads.clone();
        let read_step = task.step;
        Ok(ScanStep::new(
            task.morsel_id,
            ScanIoPhase::ProjectionRead,
            ScanTaskLane::Projection,
            reads,
            read_step.required_reads,
            read_step.prefetch_reads,
            move |results| {
                let reader = task.execution.read_context();
                let mut ctx = task.execution.session.create_execution_ctx();
                match read_step.continuation.run(&reader, &mut ctx, results)? {
                    ReadTaskOutput::Ready(array) => Ok(ScanStepResult::Ready(
                        WorkOutput::Projection(ProjectionWorkOutput {
                            morsel_id: task.morsel_id,
                            array,
                        }),
                    )),
                    ReadTaskOutput::Continue(read_task) => Ok(ScanStepResult::Continue(Box::new(
                        ProjectionReadWorkTask::try_new(task.execution, read_task, task.morsel_id)?,
                    ))),
                }
            },
        ))
    }
}

async fn resolve_step_reads(read_store: ReadStoreRef, reads: Vec<ScanRead>) -> VortexResult<()> {
    let mut pending_reads = FuturesUnordered::new();
    for read in reads {
        let key = read.request.key;
        if read_store.get(key).is_none() {
            pending_reads.push(async move { read.future.await.map(|buffer| (key, buffer)) });
        }
    }
    while let Some(result) = pending_reads.next().await {
        let (key, buffer) = result?;
        read_store.insert(key, buffer);
    }
    Ok(())
}

fn prefetch_step_reads(handle: &Handle, read_store: ReadStoreRef, reads: Vec<ScanRead>) {
    if reads.is_empty() {
        return;
    }
    handle
        .spawn(async move {
            if let Err(error) = resolve_step_reads(read_store, reads).await {
                tracing::debug!(
                    target: "vortex_file::scan_v2",
                    ?error,
                    "scan2 prefetch read failed"
                );
            }
        })
        .detach();
}

async fn run_scan_task_step(
    work: QueuedWork,
    read_store: ReadStoreRef,
    handle: Handle,
) -> VortexResult<WorkPoll> {
    let mut step = work.into_step()?;
    let (required_reads, prefetch_reads) = step.take_reads();
    prefetch_step_reads(&handle, Arc::clone(&read_store), prefetch_reads);
    resolve_step_reads(Arc::clone(&read_store), required_reads).await?;
    match step.continue_with(ReadResults::new(Arc::clone(&read_store)))? {
        ScanStepResult::Ready(output) => Ok(WorkPoll::Ready(output)),
        ScanStepResult::Continue(work) => Ok(WorkPoll::Pending(work)),
    }
}

enum CompletedMorsel {
    Empty,
    Output(ArrayRef),
}

struct PlannedMorselWork {
    state: MorselState,
    evidence: Vec<QueuedWork>,
}

struct MorselState {
    execution: Arc<ScanExecution>,
    range: Range<u64>,
    selected: Mask,
    evidence: Vec<Option<PredicateEvidence>>,
    pending_evidence: Vec<usize>,
    pending_scan_evidence: Vec<usize>,
    scan_evidence_generation: Vec<u64>,
    predicate_queued: Vec<bool>,
    predicate_done: Vec<bool>,
    next_recheck_predicate: usize,
    projection_queued: bool,
}

#[derive(Default)]
struct ScanEvidenceStore {
    predicates: Vec<PredicateScanEvidenceStore>,
}

#[derive(Default)]
struct PredicateScanEvidenceStore {
    generation: u64,
    providers: Vec<ScanEvidenceSlot>,
}

#[derive(Default)]
struct ScanEvidenceSlot {
    version: Option<PredicateVersion>,
    pending: Option<PredicateVersion>,
    fragments: Vec<EvidenceFragment>,
}

enum ScanEvidenceAction {
    Ready,
    Pending,
    Prepare,
    Wait,
}

#[derive(Default)]
struct PredicateRuntimeStats {
    input_rows: u64,
    rejected_rows: u64,
}

struct PartitionWorkSchedulerState {
    pending: VecDeque<PlannedScanPlanMorsel>,
    morsels: Vec<Option<MorselState>>,
    active_morsels: usize,
    has_dynamic_predicates: bool,
    in_flight_projection_tasks: usize,
    next_morsel_id: usize,
    next_emit_morsel_id: usize,
    task_queue: ScanTaskQueue<WorkOutput>,
    in_flight: FuturesUnordered<BoxFuture<'static, LaunchedWorkOutput>>,
    read_store: ReadStoreRef,
    completed_morsels: BTreeMap<usize, CompletedMorsel>,
    handle: Handle,
    ordered: bool,
    plan_window: usize,
}

fn plan_window_for_limit(limited: bool) -> usize {
    if limited { 1 } else { usize::MAX }
}

fn read_byte_budget(scheduler: &ScanScheduler) -> u64 {
    scheduler.config().read_byte_budget().unwrap_or(u64::MAX)
}

fn partition_work_stream(
    morsels: Vec<PlannedScanPlanMorsel>,
    handle: Handle,
    ordered: bool,
    plan_window: usize,
    read_byte_budget: u64,
) -> impl futures::Stream<Item = VortexResult<ArrayRef>> + Send + 'static {
    let has_dynamic_predicates = morsels
        .iter()
        .any(|morsel| morsel.execution.has_dynamic_predicates());
    tracing::debug!(
        target: "vortex_file::scan_v2",
        morsel_count = morsels.len(),
        ordered,
        plan_window,
        read_byte_budget,
        has_dynamic_predicates,
        "created scan2 task stream"
    );
    let state = PartitionWorkSchedulerState {
        pending: VecDeque::from(morsels),
        morsels: Vec::new(),
        active_morsels: 0,
        has_dynamic_predicates,
        in_flight_projection_tasks: 0,
        next_morsel_id: 0,
        next_emit_morsel_id: 0,
        task_queue: ScanTaskQueue::new(read_byte_budget),
        in_flight: FuturesUnordered::new(),
        read_store: Arc::new(ReadStore::new()),
        completed_morsels: BTreeMap::new(),
        handle,
        ordered,
        plan_window,
    };

    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(array) = state.pop_ready_output() {
                return Some((Ok(array), state));
            }

            while state.active_morsels < state.plan_window && !state.pending.is_empty() {
                if let Err(error) = state.plan_next_morsel() {
                    state.clear();
                    return Some((Err(error), state));
                }
            }

            while state.launch_next_admissible_work() {}

            if state.in_flight.is_empty() {
                if state.is_done() {
                    return None;
                }
                let error = vortex_err!(
                    "scan2 work scheduler stalled: {} active morsels, {} pending morsels, {} evidence work items, {} predicate work items, {} projection work items, {} active read bytes",
                    state.active_morsels,
                    state.pending.len(),
                    state.task_queue.evidence_len(),
                    state.task_queue.predicate_len(),
                    state.task_queue.projection_len(),
                    state.task_queue.active_read_bytes()
                );
                state.clear();
                return Some((Err(error), state));
            }

            match state.in_flight.next().await {
                Some(output) => {
                    state.release_reads(output.lane, &output.reads);
                    match output.output {
                        Ok(WorkPoll::Ready(output)) => match state.complete_work(output) {
                            Ok(Some(array)) => return Some((Ok(array), state)),
                            Ok(None) => continue,
                            Err(error) => return Some((Err(error), state)),
                        },
                        Ok(WorkPoll::Pending(work)) => {
                            state.task_queue.push(work);
                            continue;
                        }
                        Err(error) => return Some((Err(error), state)),
                    }
                }
                None if state.is_done() => return None,
                None => continue,
            }
        }
    })
}

impl PartitionWorkSchedulerState {
    fn clear(&mut self) {
        self.pending.clear();
        self.morsels.clear();
        self.active_morsels = 0;
        self.in_flight_projection_tasks = 0;
        self.next_emit_morsel_id = 0;
        self.task_queue.clear();
        self.in_flight = FuturesUnordered::new();
        self.read_store = Arc::new(ReadStore::new());
        self.completed_morsels.clear();
    }

    fn is_done(&self) -> bool {
        self.pending.is_empty()
            && self.active_morsels == 0
            && self.task_queue.is_empty()
            && self.in_flight.is_empty()
            && self.completed_morsels.is_empty()
    }

    fn plan_next_morsel(&mut self) -> VortexResult<()> {
        let Some(morsel) = self.pending.pop_front() else {
            return Ok(());
        };
        let morsel_id = self.next_morsel_id;
        let range = morsel.range.clone();
        let Some(planned) = morsel.execution.plan_morsel(morsel_id, morsel.range)? else {
            tracing::trace!(
                target: "vortex_file::scan_v2",
                morsel_id,
                range_start = range.start,
                range_end = range.end,
                pending_morsels = self.pending.len(),
                active_morsels = self.active_morsels,
                "scan2 skipped empty morsel"
            );
            return Ok(());
        };
        self.next_morsel_id = self.next_morsel_id.saturating_add(1);
        self.active_morsels = self.active_morsels.saturating_add(1);
        if self.morsels.len() <= morsel_id {
            self.morsels.resize_with(morsel_id + 1, || None);
        }
        self.morsels[morsel_id] = Some(planned.state);
        let evidence_len = planned.evidence.len();
        self.task_queue.extend(planned.evidence);
        self.enqueue_ready_work(morsel_id)?;
        tracing::trace!(
            target: "vortex_file::scan_v2",
            morsel_id,
            range_start = range.start,
            range_end = range.end,
            pending_morsels = self.pending.len(),
            active_morsels = self.active_morsels,
            queued_evidence = evidence_len,
            evidence_queue_len = self.task_queue.evidence_len(),
            predicate_queue_len = self.task_queue.predicate_len(),
            projection_queue_len = self.task_queue.projection_len(),
            "scan2 planned morsel"
        );
        Ok(())
    }

    fn launch_next_admissible_work(&mut self) -> bool {
        let in_flight_empty = self.in_flight.is_empty();
        // Backlogged output should stop speculative projection for dynamic scans, but not the
        // single projection needed to unblock an otherwise idle ordered stream.
        let projection_admissible = !self.has_dynamic_predicates
            || (self.in_flight_projection_tasks == 0 && !self.has_completed_output_backlog())
            || in_flight_empty;
        let morsels = &self.morsels;
        let Some(task) = self.task_queue.pop_next_admissible_with_projection_gate(
            in_flight_empty,
            projection_admissible,
            |morsel_id| morsels.get(morsel_id).and_then(Option::as_ref).is_some(),
        ) else {
            return false;
        };
        let (task, lane, reads) = task.into_parts();
        self.launch_admitted(task, lane, reads);
        true
    }

    fn launch_admitted(&mut self, work: QueuedWork, lane: ScanTaskLane, reads: Vec<ScanTaskRead>) {
        let morsel_id = work.morsel_id();
        let phase = work.phase();
        let priority = work.priority();
        let bytes = scan_task_read_bytes(&reads);
        let read_count = reads.len();
        tracing::trace!(
            target: "vortex_file::scan_v2",
            morsel_id,
            ?phase,
            ?lane,
            read_count,
            read_bytes = bytes,
            priority,
            in_flight = self.in_flight.len(),
            in_flight_projection_tasks = self.in_flight_projection_tasks,
            active_morsels = self.active_morsels,
            pending_morsels = self.pending.len(),
            evidence_queue_len = self.task_queue.evidence_len(),
            predicate_queue_len = self.task_queue.predicate_len(),
            projection_queue_len = self.task_queue.projection_len(),
            active_read_count = self.task_queue.active_read_count(),
            active_read_bytes = self.task_queue.active_read_bytes(),
            active_evidence_read_bytes = self.task_queue.active_evidence_read_bytes(),
            active_predicate_read_bytes = self.task_queue.active_predicate_read_bytes(),
            active_projection_read_bytes = self.task_queue.active_projection_read_bytes(),
            "scan2 launching work"
        );
        let read_store = Arc::clone(&self.read_store);
        let handle = self.handle.clone();
        let future = async move {
            let output = run_scan_task_step(work, read_store, handle).await;
            LaunchedWorkOutput {
                lane,
                reads,
                output,
            }
        }
        .instrument(tracing::trace_span!(
            "scan2_work",
            morsel_id,
            phase = ?phase,
            lane = ?lane,
            read_count,
            read_bytes = bytes,
        ));
        let inline_zero_read = bytes == 0
            && match phase {
                ScanIoPhase::EvidenceProbe | ScanIoPhase::EvidenceSetup => {
                    priority <= INLINE_ZERO_READ_EVIDENCE_MAX_PRIORITY
                }
                ScanIoPhase::PredicateRead
                | ScanIoPhase::ProjectionRead
                | ScanIoPhase::AggregateRead => false,
            };
        if inline_zero_read {
            self.in_flight.push(future.boxed());
        } else {
            self.in_flight.push(self.handle.spawn(future).boxed());
        }
        if matches!(lane, ScanTaskLane::Projection) {
            self.in_flight_projection_tasks = self.in_flight_projection_tasks.saturating_add(1);
        }
    }

    fn release_reads(&mut self, lane: ScanTaskLane, reads: &[ScanTaskRead]) {
        self.task_queue.release_reads(lane, reads);
        if matches!(lane, ScanTaskLane::Projection) {
            self.in_flight_projection_tasks = self.in_flight_projection_tasks.saturating_sub(1);
        }
    }

    fn complete_work(&mut self, output: WorkOutput) -> VortexResult<Option<ArrayRef>> {
        match output {
            WorkOutput::Evidence(output) => self.complete_evidence(output),
            WorkOutput::ScanEvidence(output) => self.complete_scan_evidence(output),
            WorkOutput::Projection(output) => {
                Ok(self.finish_output_morsel(output.morsel_id, output.array))
            }
        }
    }

    fn complete_scan_evidence(
        &mut self,
        output: ScanEvidenceWorkOutput,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(morsel) = self
            .morsels
            .get_mut(output.morsel_id)
            .and_then(Option::as_mut)
            && let Some(pending) = morsel.pending_scan_evidence.get_mut(output.predicate_idx)
        {
            *pending = pending.saturating_sub(1);
        }

        if let Some(fragments) = output.fragments {
            output.execution.record_scan_evidence(
                output.predicate_idx,
                output.evidence_idx,
                output.version,
                fragments,
            )?;
        }

        let affected = self
            .morsels
            .iter()
            .enumerate()
            .filter_map(|(morsel_id, morsel)| {
                morsel
                    .as_ref()
                    .filter(|morsel| Arc::ptr_eq(&morsel.execution, &output.execution))
                    .map(|_| morsel_id)
            })
            .collect::<Vec<_>>();

        for morsel_id in affected {
            if self
                .morsels
                .get(morsel_id)
                .and_then(Option::as_ref)
                .is_none()
            {
                continue;
            }
            if self.refresh_morsel_scan_evidence(morsel_id, output.predicate_idx)? {
                if let Some(array) = self.finish_empty_morsel(morsel_id) {
                    return Ok(Some(array));
                }
            } else {
                self.enqueue_ready_work(morsel_id)?;
            }
        }
        Ok(None)
    }

    fn refresh_all_scan_evidence(&mut self, morsel_id: usize) -> VortexResult<bool> {
        let Some(predicate_count) = self
            .morsels
            .get(morsel_id)
            .and_then(Option::as_ref)
            .map(|morsel| morsel.execution.predicates.len())
        else {
            return Ok(false);
        };

        for predicate_idx in 0..predicate_count {
            if self.refresh_morsel_scan_evidence(morsel_id, predicate_idx)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn refresh_morsel_scan_evidence(
        &mut self,
        morsel_id: usize,
        predicate_idx: usize,
    ) -> VortexResult<bool> {
        let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
            return Ok(false);
        };
        let predicate = &morsel.execution.predicates[predicate_idx];
        let version = predicate.version();
        let (generation, fragments) =
            morsel
                .execution
                .scan_evidence_fragments(predicate_idx, version, &morsel.range)?;
        let Some(seen_generation) = morsel.scan_evidence_generation.get_mut(predicate_idx) else {
            vortex_bail!("missing scan evidence generation slot {predicate_idx}");
        };
        if generation <= *seen_generation {
            return Ok(false);
        }
        *seen_generation = generation;

        let Some(slot) = morsel.evidence.get_mut(predicate_idx) else {
            vortex_bail!("missing predicate evidence slot {predicate_idx}");
        };
        if slot
            .as_ref()
            .is_none_or(|evidence| evidence.version() != version)
        {
            *slot = Some(PredicateEvidence::new(
                predicate.id,
                version,
                morsel.range.clone(),
            )?);
        }
        let evidence = slot
            .as_mut()
            .ok_or_else(|| vortex_err!("missing predicate evidence after initialization"))?;
        for fragment in fragments {
            evidence.absorb(fragment)?;
        }
        let maybe = evidence.maybe().clone();
        let all_false = evidence.all_false();
        morsel.selected = &morsel.selected & &maybe;
        Ok(morsel.selected.all_false() || all_false)
    }

    fn complete_evidence(&mut self, output: EvidenceWorkOutput) -> VortexResult<Option<ArrayRef>> {
        let mut record_predicate = None;
        let finish_empty = {
            let Some(morsel) = self
                .morsels
                .get_mut(output.morsel_id)
                .and_then(Option::as_mut)
            else {
                return Ok(None);
            };
            match output.source {
                EvidenceWorkSource::Provider => {
                    let Some(pending) = morsel.pending_evidence.get_mut(output.predicate_idx)
                    else {
                        vortex_bail!("missing predicate evidence count {}", output.predicate_idx);
                    };
                    *pending = pending.saturating_sub(1);
                }
                EvidenceWorkSource::Predicate {
                    input_rows,
                    pass_rows,
                } => {
                    let Some(queued) = morsel.predicate_queued.get_mut(output.predicate_idx) else {
                        vortex_bail!("missing predicate queued slot {}", output.predicate_idx);
                    };
                    *queued = false;
                    let Some(done) = morsel.predicate_done.get_mut(output.predicate_idx) else {
                        vortex_bail!("missing predicate done slot {}", output.predicate_idx);
                    };
                    *done = true;
                    record_predicate = Some((
                        Arc::clone(&morsel.execution),
                        output.predicate_idx,
                        input_rows,
                        pass_rows,
                    ));
                }
            }
            let predicate = &morsel.execution.predicates[output.predicate_idx];
            let Some(slot) = morsel.evidence.get_mut(output.predicate_idx) else {
                vortex_bail!("missing predicate evidence slot {}", output.predicate_idx);
            };
            if slot
                .as_ref()
                .is_none_or(|evidence| evidence.version() != output.version)
            {
                *slot = Some(PredicateEvidence::new(
                    predicate.id,
                    output.version,
                    morsel.range.clone(),
                )?);
            }
            let evidence = slot
                .as_mut()
                .ok_or_else(|| vortex_err!("missing predicate evidence after initialization"))?;
            for fragment in output.fragments {
                evidence.absorb(fragment)?;
            }
            let maybe = evidence.maybe().clone();
            let all_false = evidence.all_false();
            morsel.selected = &morsel.selected & &maybe;
            morsel.selected.all_false() || all_false
        };

        if let Some((execution, predicate_idx, input_rows, pass_rows)) = record_predicate
            && !execution.has_dynamic_predicates()
        {
            execution.record_predicate_result(predicate_idx, input_rows, pass_rows);
        }

        if finish_empty {
            return Ok(self.finish_empty_morsel(output.morsel_id));
        }

        self.enqueue_ready_work(output.morsel_id)?;
        Ok(None)
    }

    fn enqueue_ready_work(&mut self, morsel_id: usize) -> VortexResult<()> {
        if self.refresh_all_scan_evidence(morsel_id)? {
            self.finish_empty_morsel(morsel_id);
            return Ok(());
        }

        if let Some((predicate_idx, need, priority)) = self.choose_ready_predicate(morsel_id)? {
            let Some(morsel) = self.morsels.get(morsel_id).and_then(Option::as_ref) else {
                return Ok(());
            };
            let work = morsel.execution.plan_predicate_work(
                morsel_id,
                predicate_idx,
                morsel.range.clone(),
                need,
                morsel.execution.predicates[predicate_idx].version(),
                priority,
            )?;
            let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
                return Ok(());
            };
            morsel.predicate_queued[predicate_idx] = true;
            self.task_queue.push(work);
            return Ok(());
        }

        let ready_to_project = self
            .morsels
            .get(morsel_id)
            .and_then(Option::as_ref)
            .is_some_and(|morsel| {
                !morsel.projection_queued
                    && morsel.pending_evidence.iter().all(|pending| *pending == 0)
                    && morsel
                        .pending_scan_evidence
                        .iter()
                        .all(|pending| *pending == 0)
                    && morsel.predicate_queued.iter().all(|queued| !*queued)
                    && morsel.predicate_done.iter().all(|done| *done)
            });
        if !ready_to_project {
            return Ok(());
        }

        if self.enqueue_recheck_evidence(morsel_id)? {
            return Ok(());
        }

        let Some(morsel) = self.morsels.get(morsel_id).and_then(Option::as_ref) else {
            return Ok(());
        };
        let projection = morsel.execution.plan_projection_work(
            morsel_id,
            morsel.range.clone(),
            morsel.selected.clone(),
        )?;
        let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
            return Ok(());
        };
        morsel.projection_queued = true;
        match projection {
            Some(work) => self.task_queue.push(work),
            None => {
                self.finish_empty_morsel(morsel_id);
            }
        }
        Ok(())
    }

    fn choose_ready_predicate(
        &mut self,
        morsel_id: usize,
    ) -> VortexResult<Option<(usize, Mask, u64)>> {
        loop {
            let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
                return Ok(None);
            };
            if morsel.predicate_queued.iter().any(|queued| *queued) {
                return Ok(None);
            }
            let dynamic_scan = morsel.execution.has_dynamic_predicates();
            if dynamic_scan
                && (morsel.pending_evidence.iter().any(|pending| *pending != 0)
                    || morsel
                        .pending_scan_evidence
                        .iter()
                        .any(|pending| *pending != 0))
            {
                return Ok(None);
            }

            let mut best: Option<(u64, usize, Mask)> = None;
            let mut advanced = false;
            for predicate_idx in 0..morsel.execution.predicates.len() {
                if morsel.predicate_done[predicate_idx]
                    || morsel.predicate_queued[predicate_idx]
                    || morsel.pending_evidence[predicate_idx] != 0
                    || morsel.pending_scan_evidence[predicate_idx] != 0
                {
                    continue;
                }
                if morsel.evidence[predicate_idx].is_none() {
                    let predicate = &morsel.execution.predicates[predicate_idx];
                    morsel.evidence[predicate_idx] = Some(PredicateEvidence::new(
                        predicate.id,
                        predicate.version(),
                        morsel.range.clone(),
                    )?);
                }
                let evidence = morsel.evidence[predicate_idx].as_ref().ok_or_else(|| {
                    vortex_err!(
                        "missing evidence for predicate {predicate_idx} before residual read"
                    )
                })?;
                let need = &morsel.selected & &evidence.unproven();
                if need.all_false() {
                    morsel.predicate_done[predicate_idx] = true;
                    advanced = true;
                    continue;
                }
                let priority = if dynamic_scan {
                    u64::try_from(predicate_idx).unwrap_or(u64::MAX)
                } else {
                    morsel
                        .execution
                        .predicate_priority(predicate_idx, need.true_count())
                };
                if best.as_ref().is_none_or(|(best_priority, best_idx, _)| {
                    (priority, predicate_idx) < (*best_priority, *best_idx)
                }) {
                    best = Some((priority, predicate_idx, need));
                }
            }
            if advanced {
                continue;
            }
            return Ok(best.map(|(priority, predicate_idx, need)| (predicate_idx, need, priority)));
        }
    }

    fn enqueue_recheck_evidence(&mut self, morsel_id: usize) -> VortexResult<bool> {
        loop {
            let Some(morsel) = self.morsels.get(morsel_id).and_then(Option::as_ref) else {
                return Ok(false);
            };
            if morsel.next_recheck_predicate >= morsel.execution.predicates.len() {
                return Ok(false);
            }

            let predicate_idx = morsel.next_recheck_predicate;
            let predicate = &morsel.execution.predicates[predicate_idx];
            let current_version = predicate.version();
            let evidence_version = morsel.evidence[predicate_idx]
                .as_ref()
                .map(PredicateEvidence::version)
                .unwrap_or(PredicateVersion::STATIC);
            let has_dynamic = predicate.dynamic_updates.is_some();
            let has_scan_recheck_evidence = predicate.has_scan_recheck_evidence();
            let has_morsel_recheck_evidence = predicate.has_morsel_recheck_evidence();

            if has_dynamic && has_scan_recheck_evidence && current_version != evidence_version {
                let work = morsel.execution.plan_scan_evidence_work(
                    morsel_id,
                    predicate_idx,
                    current_version,
                    EvidenceMode::RecheckBeforeProjection,
                )?;
                if !work.is_empty() {
                    let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut)
                    else {
                        return Ok(false);
                    };
                    morsel.pending_scan_evidence[predicate_idx] =
                        morsel.pending_scan_evidence[predicate_idx].saturating_add(work.len());
                    self.task_queue.extend(work);
                    return Ok(true);
                }
                if self.refresh_morsel_scan_evidence(morsel_id, predicate_idx)? {
                    self.finish_empty_morsel(morsel_id);
                    return Ok(true);
                }
            }

            let Some(morsel) = self.morsels.get(morsel_id).and_then(Option::as_ref) else {
                return Ok(false);
            };
            let evidence_version = morsel.evidence[predicate_idx]
                .as_ref()
                .map(PredicateEvidence::version)
                .unwrap_or(PredicateVersion::STATIC);

            if has_dynamic && has_morsel_recheck_evidence && current_version != evidence_version {
                let work = morsel.execution.plan_evidence_work(
                    morsel_id,
                    predicate_idx,
                    morsel.range.clone(),
                    current_version,
                    EvidenceMode::RecheckBeforeProjection,
                )?;
                if work.is_empty() {
                    let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut)
                    else {
                        return Ok(false);
                    };
                    morsel.next_recheck_predicate = morsel.next_recheck_predicate.saturating_add(1);
                    continue;
                }
                let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
                    return Ok(false);
                };
                morsel.pending_evidence[predicate_idx] =
                    morsel.pending_evidence[predicate_idx].saturating_add(work.len());
                self.task_queue.extend(work);
                return Ok(true);
            }

            let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
                return Ok(false);
            };
            morsel.next_recheck_predicate = morsel.next_recheck_predicate.saturating_add(1);
        }
    }

    fn finish_empty_morsel(&mut self, morsel_id: usize) -> Option<ArrayRef> {
        if self.finish_morsel(morsel_id) && self.ordered {
            self.completed_morsels
                .insert(morsel_id, CompletedMorsel::Empty);
            return self.pop_ready_output();
        }
        None
    }

    fn finish_output_morsel(&mut self, morsel_id: usize, array: ArrayRef) -> Option<ArrayRef> {
        if !self.finish_morsel(morsel_id) {
            return None;
        }
        if self.ordered {
            self.completed_morsels
                .insert(morsel_id, CompletedMorsel::Output(array));
            self.pop_ready_output()
        } else {
            Some(array)
        }
    }

    fn finish_morsel(&mut self, morsel_id: usize) -> bool {
        if let Some(slot) = self.morsels.get_mut(morsel_id)
            && slot.take().is_some()
        {
            self.active_morsels = self.active_morsels.saturating_sub(1);
            return true;
        }
        false
    }

    fn pop_ready_output(&mut self) -> Option<ArrayRef> {
        if !self.ordered {
            return None;
        }
        loop {
            match self.completed_morsels.remove(&self.next_emit_morsel_id) {
                Some(CompletedMorsel::Empty) => {
                    self.next_emit_morsel_id = self.next_emit_morsel_id.saturating_add(1);
                }
                Some(CompletedMorsel::Output(array)) => {
                    self.next_emit_morsel_id = self.next_emit_morsel_id.saturating_add(1);
                    return Some(array);
                }
                None => return None,
            }
        }
    }

    fn has_completed_output_backlog(&self) -> bool {
        self.completed_morsels
            .values()
            .any(|morsel| matches!(morsel, CompletedMorsel::Output(_)))
    }
}

struct ScanPlanPartition {
    file: VortexFile,
    prepared: Arc<PreparedScanPlan>,
    row_range: Range<u64>,
    index: usize,
    scheduler: Arc<ScanScheduler>,
    limit_remaining: Option<Arc<AtomicU64>>,
}

impl Partition for ScanPlanPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        let row_count = self.row_range.end - self.row_range.start;
        let row_count = self.prepared.selection().row_count(row_count);
        let row_count = self
            .prepared
            .limit()
            .map_or(row_count, |limit| row_count.min(limit));

        if self.prepared.has_filter() {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let ScanPlanPartition {
            file,
            prepared,
            row_range,
            index: _,
            scheduler,
            limit_remaining,
        } = *self;

        let execution = Arc::new(ScanExecution::try_new(file, prepared, limit_remaining)?);
        let handle = execution.session.handle();
        let dtype = execution.plan.dtype().clone();
        let ranges = execution.splits(&row_range)?;
        let ordered = execution.plan.ordered();
        let plan_window = plan_window_for_limit(execution.limit_remaining.is_some());
        let read_byte_budget = read_byte_budget(&scheduler);
        let morsels = ranges
            .into_iter()
            .map(|range| PlannedScanPlanMorsel {
                execution: Arc::clone(&execution),
                range,
            })
            .collect::<Vec<_>>();

        let stream = partition_work_stream(morsels, handle, ordered, plan_window, read_byte_budget);

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

struct PlannedScanPlanScan {
    dtype: DType,
    partitions: Vec<Vec<PlannedScanPlanMorsel>>,
    handle: Handle,
    read_byte_budget: u64,
}

#[derive(Clone)]
struct PlannedScanPlanMorsel {
    execution: Arc<ScanExecution>,
    range: Range<u64>,
}

impl PlannedMorselScan for PlannedScanPlanScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    fn partition(self: Arc<Self>, partition: usize) -> VortexResult<PartitionRef> {
        if partition >= self.partitions.len() {
            vortex_bail!(
                "planned scan partition {partition} is outside 0..{}",
                self.partitions.len()
            );
        }

        Ok(Box::new(PlannedScanPlanPartition {
            planned: self,
            index: partition,
        }))
    }
}

struct PlannedScanPlanPartition {
    planned: Arc<PlannedScanPlanScan>,
    index: usize,
}

impl Partition for PlannedScanPlanPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        let mut row_count = 0u64;
        let mut has_filter = false;

        for morsel in &self.planned.partitions[self.index] {
            let range_len = morsel.range.end - morsel.range.start;
            row_count =
                row_count.saturating_add(morsel.execution.plan.selection().row_count(range_len));
            has_filter |= morsel.execution.plan.has_filter();
        }

        if has_filter {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let PlannedScanPlanPartition { planned, index } = *self;
        let morsels = planned.partitions[index].clone();
        let dtype = planned.dtype.clone();
        let handle = planned.handle.clone();
        let stream =
            partition_work_stream(morsels, handle, false, usize::MAX, planned.read_byte_budget);

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

struct PreparedScanPlan {
    // Request-level physical plan after pushdown. This must stay free of per-scan IO state.
    dtype: DType,
    selection: Selection,
    ordered: bool,
    limit: Option<u64>,
    row_count: u64,
    split_hints: Vec<u64>,
    projection: ScanPlanRef,
    predicates: Vec<PreparedPredicatePlan>,
}

struct PreparedPredicatePlan {
    id: PredicateId,
    expr: Expression,
    plan: ScanPlanRef,
}

struct ScanExecution {
    // Runtime instantiation of a prepared plan: source binding, prepared handles, and scan state.
    session: VortexSession,
    plan: Arc<PreparedScanPlan>,
    limit_remaining: Option<Arc<AtomicU64>>,
    projection: PreparedReadRef,
    predicates: Vec<ExecutionPredicate>,
    predicate_stats: Mutex<Vec<PredicateRuntimeStats>>,
    scan_evidence: Mutex<ScanEvidenceStore>,
}

struct ExecutionPredicate {
    id: PredicateId,
    expr: Expression,
    static_cost: u64,
    dynamic_updates: Option<DynamicExprUpdates>,
    read: PreparedReadRef,
    evidence: Vec<PreparedEvidenceRef>,
}

impl ExecutionPredicate {
    fn version(&self) -> PredicateVersion {
        self.dynamic_updates
            .as_ref()
            .map(|updates| PredicateVersion::new(updates.version()))
            .unwrap_or(PredicateVersion::STATIC)
    }

    fn has_morsel_recheck_evidence(&self) -> bool {
        self.evidence
            .iter()
            .any(|plan| plan.scope() == EvidenceScope::Morsel && plan.recheck_before_projection())
    }

    fn has_scan_recheck_evidence(&self) -> bool {
        self.evidence
            .iter()
            .any(|plan| plan.scope() == EvidenceScope::Scan && plan.recheck_before_projection())
    }
}

impl PreparedScanPlan {
    fn try_new(file: &VortexFile, request: &DataSourceScanRequest) -> VortexResult<Self> {
        let session = file.session().clone();
        let dtype = request.projection.return_dtype(file.dtype())?;
        let projection = request.projection.optimize_recursive(file.dtype())?;
        let filter = request
            .filter
            .clone()
            .map(|filter| filter.optimize_recursive(file.dtype()))
            .transpose()?;

        let root = file.scan_plan_root()?;
        let projection_pushed = push_expr(&root, &projection, file.dtype(), &session)?;
        let mut split_hints = Vec::new();
        extend_split_hints(&projection_pushed, &mut split_hints);

        // Run cheap, likely-selective conjuncts first so an expensive residual (e.g. an FSST `LIKE`)
        // only evaluates over the rows that survive the cheaper predicates. AND is commutative, so
        // reordering is semantically safe; `PredicateId`s are assigned by final slot below (after the
        // sort) so each predicate's evidence/read stay self-consistent with its id.
        let mut ordered_conjuncts = filter.as_ref().map(conjuncts).unwrap_or_default();
        ordered_conjuncts.sort_by_cached_key(predicate_cost);
        let predicates = ordered_conjuncts
            .into_iter()
            .enumerate()
            .map(|(idx, expr)| {
                let id = PredicateId::new(
                    u32::try_from(idx).map_err(|_| vortex_err!("too many predicates"))?,
                );
                let pushed = push_expr(&root, &expr, file.dtype(), &session)?;
                extend_split_hints(&pushed, &mut split_hints);
                Ok(PreparedPredicatePlan {
                    id,
                    expr,
                    plan: pushed,
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Self {
            dtype,
            selection: request.selection.clone(),
            ordered: request.ordered,
            limit: request.limit,
            row_count: file.row_count(),
            split_hints,
            projection: projection_pushed,
            predicates,
        })
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn selection(&self) -> &Selection {
        &self.selection
    }

    fn ordered(&self) -> bool {
        self.ordered
    }

    fn limit(&self) -> Option<u64> {
        self.limit
    }

    fn predicates(&self) -> &[PreparedPredicatePlan] {
        &self.predicates
    }

    fn has_filter(&self) -> bool {
        !self.predicates.is_empty()
    }

    fn projection(&self) -> &ScanPlanRef {
        &self.projection
    }

    fn splits(&self, row_range: &Range<u64>) -> VortexResult<Vec<Range<u64>>> {
        check_range(row_range, self.row_count)?;
        let (splits, split_kind) = prepare_split_ranges(
            self.row_count,
            row_range,
            &self.selection,
            self.split_hints.clone(),
        );
        trace_prepared_splits(row_range, &splits, split_kind, self.has_filter());
        Ok(splits)
    }
}

impl ScanExecution {
    fn try_new(
        file: VortexFile,
        plan: Arc<PreparedScanPlan>,
        limit_remaining: Option<Arc<AtomicU64>>,
    ) -> VortexResult<Self> {
        let session = file.session().clone();
        let mut prepare_ctx =
            PrepareCtx::with_state_cache(session.clone(), file.scan_plan_state_cache());
        let projection = Arc::clone(plan.projection())
            .prepare_read(&mut prepare_ctx)?
            .ok_or_else(|| vortex_err!("scan2 could not plan read for pushed projection"))?;
        let predicates = plan
            .predicates()
            .iter()
            .map(|predicate| {
                let read = Arc::clone(&predicate.plan)
                    .prepare_read(&mut prepare_ctx)?
                    .ok_or_else(|| {
                        vortex_err!("scan2 could not plan predicate read {}", predicate.expr)
                    })?;
                let evidence = Arc::clone(&predicate.plan).prepare_evidence(&mut prepare_ctx)?;
                let dynamic_updates = DynamicExprUpdates::new(&predicate.expr);
                Ok(ExecutionPredicate {
                    id: predicate.id,
                    expr: predicate.expr.clone(),
                    static_cost: predicate_cost(&predicate.expr),
                    dynamic_updates,
                    read,
                    evidence,
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let predicate_stats = (0..predicates.len())
            .map(|_| PredicateRuntimeStats::default())
            .collect();
        let scan_evidence = ScanEvidenceStore {
            predicates: predicates
                .iter()
                .map(|predicate| PredicateScanEvidenceStore {
                    generation: 0,
                    providers: predicate
                        .evidence
                        .iter()
                        .map(|_| ScanEvidenceSlot::default())
                        .collect(),
                })
                .collect(),
        };

        Ok(Self {
            session,
            plan,
            limit_remaining,
            projection,
            predicates,
            predicate_stats: Mutex::new(predicate_stats),
            scan_evidence: Mutex::new(scan_evidence),
        })
    }

    fn read_context(&self) -> ReadContext {
        ReadContext::new(self.session.clone())
    }

    fn predicate_priority(&self, predicate_idx: usize, demand_rows: usize) -> u64 {
        let predicate = &self.predicates[predicate_idx];
        let static_cost = predicate.static_cost.max(1);
        let demand_rows = u64::try_from(demand_rows).unwrap_or(u64::MAX).max(1);
        let stats = self.predicate_stats.lock();
        let stats = &stats[predicate_idx];
        let rejection_per_mille = if stats.input_rows >= 1024 {
            stats.rejected_rows.saturating_mul(1000) / stats.input_rows.max(1)
        } else {
            // Before feedback exists, preserve the existing static cheap-first ordering while still
            // giving every predicate a nonzero expected benefit.
            500
        }
        .max(1);
        let expected_rejected = demand_rows.saturating_mul(rejection_per_mille) / 1000;
        static_cost.saturating_mul(1_000_000) / expected_rejected.max(1)
    }

    fn has_dynamic_predicates(&self) -> bool {
        self.predicates
            .iter()
            .any(|predicate| predicate.dynamic_updates.is_some())
    }

    fn record_predicate_result(&self, predicate_idx: usize, input_rows: usize, pass_rows: usize) {
        let input_rows = u64::try_from(input_rows).unwrap_or(u64::MAX);
        let pass_rows = u64::try_from(pass_rows).unwrap_or(u64::MAX);
        let rejected_rows = input_rows.saturating_sub(pass_rows);
        let mut stats = self.predicate_stats.lock();
        let stats = &mut stats[predicate_idx];
        stats.input_rows = stats.input_rows.saturating_add(input_rows);
        stats.rejected_rows = stats.rejected_rows.saturating_add(rejected_rows);
    }

    fn use_scan_scope_evidence(&self, predicate_idx: usize, mode: EvidenceMode) -> bool {
        mode == EvidenceMode::RecheckBeforeProjection
            || self.predicates[predicate_idx].static_cost >= SCAN_SCOPE_MIN_PREDICATE_COST
    }

    fn plan_morsel(
        self: &Arc<Self>,
        morsel_id: usize,
        range: Range<u64>,
    ) -> VortexResult<Option<PlannedMorselWork>> {
        let selected = self.plan.selection().row_mask(&range).mask().clone();
        if selected.all_false() {
            return Ok(None);
        }

        let mut evidence = Vec::new();
        let mut pending_evidence = Vec::with_capacity(self.predicates.len());
        let mut pending_scan_evidence = Vec::with_capacity(self.predicates.len());
        for predicate_idx in 0..self.predicates.len() {
            let version = self.predicates[predicate_idx].version();
            let scan_work = self.plan_scan_evidence_work(
                morsel_id,
                predicate_idx,
                version,
                EvidenceMode::Normal,
            )?;
            pending_scan_evidence.push(scan_work.len());
            evidence.extend(scan_work);

            let morsel_work = self.plan_evidence_work(
                morsel_id,
                predicate_idx,
                range.clone(),
                version,
                EvidenceMode::Normal,
            )?;
            pending_evidence.push(morsel_work.len());
            evidence.extend(morsel_work);
        }

        let state = MorselState {
            execution: Arc::clone(self),
            range,
            selected,
            evidence: (0..self.predicates.len()).map(|_| None).collect(),
            pending_evidence,
            pending_scan_evidence,
            scan_evidence_generation: vec![0; self.predicates.len()],
            predicate_queued: vec![false; self.predicates.len()],
            predicate_done: vec![false; self.predicates.len()],
            next_recheck_predicate: 0,
            projection_queued: false,
        };

        Ok(Some(PlannedMorselWork { state, evidence }))
    }

    fn reserve_scan_evidence(
        &self,
        predicate_idx: usize,
        evidence_idx: usize,
        version: PredicateVersion,
        create_waiter: bool,
    ) -> VortexResult<ScanEvidenceAction> {
        let mut store = self.scan_evidence.lock();
        let slot = store
            .predicates
            .get_mut(predicate_idx)
            .and_then(|predicate| predicate.providers.get_mut(evidence_idx))
            .ok_or_else(|| {
                vortex_err!(
                    "missing scan evidence slot for predicate {predicate_idx} provider {evidence_idx}"
                )
            })?;
        if slot.version == Some(version) {
            return Ok(ScanEvidenceAction::Ready);
        }
        if slot.pending == Some(version) {
            if !create_waiter {
                return Ok(ScanEvidenceAction::Pending);
            }
            return Ok(ScanEvidenceAction::Wait);
        }

        // Any older version is superseded. Polling waiters observe the version change and
        // re-enter planning for the current dynamic boundary.
        slot.pending = Some(version);
        Ok(ScanEvidenceAction::Prepare)
    }

    fn clear_scan_evidence_pending(
        &self,
        predicate_idx: usize,
        evidence_idx: usize,
        version: PredicateVersion,
    ) {
        let mut store = self.scan_evidence.lock();
        let Some(slot) = store
            .predicates
            .get_mut(predicate_idx)
            .and_then(|predicate| predicate.providers.get_mut(evidence_idx))
        else {
            return;
        };
        if slot.pending == Some(version) {
            slot.pending = None;
        }
    }

    fn scan_evidence_provider_ready(
        &self,
        predicate_idx: usize,
        evidence_idx: usize,
        version: PredicateVersion,
    ) -> bool {
        self.scan_evidence
            .lock()
            .predicates
            .get(predicate_idx)
            .and_then(|predicate| predicate.providers.get(evidence_idx))
            .is_some_and(|slot| slot.version == Some(version))
    }

    fn record_scan_evidence(
        &self,
        predicate_idx: usize,
        evidence_idx: usize,
        version: PredicateVersion,
        mut fragments: Vec<EvidenceFragment>,
    ) -> VortexResult<bool> {
        fragments.sort_by_key(|fragment| (fragment.rows.start, fragment.rows.end));
        let mut store = self.scan_evidence.lock();
        let predicate = store
            .predicates
            .get_mut(predicate_idx)
            .ok_or_else(|| vortex_err!("missing scan evidence predicate slot {predicate_idx}"))?;
        let slot = predicate.providers.get_mut(evidence_idx).ok_or_else(|| {
            vortex_err!(
                "missing scan evidence provider slot {evidence_idx} for predicate {predicate_idx}"
            )
        })?;

        if slot.pending != Some(version) && slot.version != Some(version) {
            return Ok(false);
        }

        slot.version = Some(version);
        slot.pending = None;
        slot.fragments = fragments;
        predicate.generation = predicate.generation.saturating_add(1);
        Ok(true)
    }

    fn scan_evidence_fragments(
        &self,
        predicate_idx: usize,
        version: PredicateVersion,
        range: &Range<u64>,
    ) -> VortexResult<(u64, Vec<EvidenceFragment>)> {
        let store = self.scan_evidence.lock();
        let Some(predicate) = store.predicates.get(predicate_idx) else {
            vortex_bail!("missing scan evidence predicate slot {predicate_idx}");
        };
        let generation = predicate.generation;
        let mut fragments = Vec::new();
        for slot in &predicate.providers {
            if slot.version == Some(version) {
                push_overlapping_fragments(&slot.fragments, range, &mut fragments)?;
            }
        }
        Ok((generation, fragments))
    }

    fn plan_scan_evidence_work(
        self: &Arc<Self>,
        morsel_id: usize,
        predicate_idx: usize,
        version: PredicateVersion,
        mode: EvidenceMode,
    ) -> VortexResult<Vec<QueuedWork>> {
        if !self.use_scan_scope_evidence(predicate_idx, mode) {
            return Ok(Vec::new());
        }

        let predicate = &self.predicates[predicate_idx];
        let predicate_idx_u32 =
            u32::try_from(predicate_idx).map_err(|_| vortex_err!("too many predicates"))?;
        let mut work = Vec::new();
        for (evidence_idx, plan) in predicate.evidence.iter().enumerate() {
            if plan.scope() != EvidenceScope::Scan {
                continue;
            }
            if mode == EvidenceMode::RecheckBeforeProjection && !plan.recheck_before_projection() {
                continue;
            }

            let evidence_idx_u32 =
                u32::try_from(evidence_idx).map_err(|_| vortex_err!("too many evidence plans"))?;
            let priority = plan
                .cost(
                    &OwnedEvidenceRequest {
                        id: predicate.id,
                        version,
                        predicate: predicate.expr.clone(),
                        range: 0..self.plan.row_count,
                        mode,
                    }
                    .as_request(),
                )
                .priority(0, mode == EvidenceMode::RecheckBeforeProjection)
                .saturating_add(predicate.static_cost);

            let create_waiter = mode == EvidenceMode::RecheckBeforeProjection;
            match self.reserve_scan_evidence(predicate_idx, evidence_idx, version, create_waiter)? {
                ScanEvidenceAction::Ready => {}
                ScanEvidenceAction::Pending => {}
                ScanEvidenceAction::Wait => {
                    work.push(Box::new(ScanEvidenceWaitTask {
                        execution: Arc::clone(self),
                        morsel_id,
                        predicate_idx,
                        evidence_idx,
                        version,
                        lane: ScanTaskLane::ScanEvidence {
                            predicate_idx: predicate_idx_u32,
                            evidence_idx: evidence_idx_u32,
                        },
                        priority,
                    }) as QueuedWork);
                }
                ScanEvidenceAction::Prepare => {
                    let req = OwnedEvidenceRequest {
                        id: predicate.id,
                        version,
                        predicate: predicate.expr.clone(),
                        range: 0..self.plan.row_count,
                        mode,
                    };
                    let result = (|| {
                        let task = Arc::clone(plan)
                            .create_task(req.clone(), ScanIoPhase::EvidenceProbe)?;
                        let step = task.into_step()?;
                        let work_reads = ScanTaskRead::from_scan_reads(&step.required_reads);
                        let priority = plan
                            .cost(&req.as_request())
                            .priority(
                                scan_task_read_bytes(&work_reads),
                                mode == EvidenceMode::RecheckBeforeProjection,
                            )
                            .saturating_add(predicate.static_cost);
                        let execution = Arc::clone(self);
                        Ok(ScanStep::new(
                            morsel_id,
                            ScanIoPhase::EvidenceProbe,
                            ScanTaskLane::ScanEvidence {
                                predicate_idx: predicate_idx_u32,
                                evidence_idx: evidence_idx_u32,
                            },
                            work_reads,
                            step.required_reads,
                            step.prefetch_reads,
                            move |results| {
                                let reader = execution.read_context();
                                let fragments = step.continuation.run(&reader, results)?;
                                Ok(ScanStepResult::Ready(WorkOutput::ScanEvidence(
                                    ScanEvidenceWorkOutput {
                                        execution,
                                        morsel_id,
                                        predicate_idx,
                                        evidence_idx,
                                        version,
                                        fragments: Some(fragments),
                                    },
                                )))
                            },
                        )
                        .with_priority(priority)
                        .boxed())
                    })();
                    match result {
                        Ok(task) => work.push(task),
                        Err(error) => {
                            self.clear_scan_evidence_pending(predicate_idx, evidence_idx, version);
                            return Err(error);
                        }
                    }
                }
            }
        }
        Ok(work)
    }

    fn plan_evidence_work(
        self: &Arc<Self>,
        morsel_id: usize,
        predicate_idx: usize,
        range: Range<u64>,
        version: PredicateVersion,
        mode: EvidenceMode,
    ) -> VortexResult<Vec<QueuedWork>> {
        let predicate = &self.predicates[predicate_idx];
        let req = OwnedEvidenceRequest {
            id: predicate.id,
            version,
            predicate: predicate.expr.clone(),
            range,
            mode,
        };
        let predicate_idx_u32 =
            u32::try_from(predicate_idx).map_err(|_| vortex_err!("too many predicates"))?;
        let mut work = Vec::with_capacity(predicate.evidence.len());
        for (evidence_idx, plan) in predicate.evidence.iter().enumerate() {
            if plan.scope() == EvidenceScope::Scan
                && self.use_scan_scope_evidence(predicate_idx, mode)
            {
                continue;
            }
            if mode == EvidenceMode::RecheckBeforeProjection && !plan.recheck_before_projection() {
                continue;
            }
            let evidence_idx_u32 =
                u32::try_from(evidence_idx).map_err(|_| vortex_err!("too many evidence plans"))?;
            let task = Arc::clone(plan).create_task(req.clone(), ScanIoPhase::EvidenceProbe)?;
            let step = task.into_step()?;
            let work_reads = ScanTaskRead::from_scan_reads(&step.required_reads);
            let priority = plan
                .cost(&req.as_request())
                .priority(
                    scan_task_read_bytes(&work_reads),
                    mode == EvidenceMode::RecheckBeforeProjection,
                )
                .saturating_add(predicate.static_cost);
            let execution = Arc::clone(self);
            work.push(
                ScanStep::new(
                    morsel_id,
                    ScanIoPhase::EvidenceProbe,
                    ScanTaskLane::Evidence {
                        predicate_idx: predicate_idx_u32,
                        evidence_idx: evidence_idx_u32,
                    },
                    work_reads,
                    step.required_reads,
                    step.prefetch_reads,
                    move |results| {
                        let reader = execution.read_context();
                        let fragments = step.continuation.run(&reader, results)?;
                        Ok(ScanStepResult::Ready(WorkOutput::Evidence(
                            EvidenceWorkOutput {
                                morsel_id,
                                predicate_idx,
                                version,
                                source: EvidenceWorkSource::Provider,
                                fragments,
                            },
                        )))
                    },
                )
                .with_priority(priority)
                .boxed(),
            );
        }
        Ok(work)
    }

    fn plan_predicate_work(
        self: &Arc<Self>,
        morsel_id: usize,
        predicate_idx: usize,
        range: Range<u64>,
        need: Mask,
        version: PredicateVersion,
        priority: u64,
    ) -> VortexResult<QueuedWork> {
        let len = range_len(&range)?;
        let predicate = &self.predicates[predicate_idx];
        let compact = need.density() < EXPR_EVAL_THRESHOLD;
        let rows = if compact {
            OwnedRowScope::selected(need.clone())
        } else {
            OwnedRowScope::try_new(Mask::new_true(len), need.clone())?
        };
        let phase = ScanIoPhase::PredicateRead;
        let task = Arc::clone(&predicate.read).create_task(range.clone(), rows, phase)?;

        let predicate_idx_u32 =
            u32::try_from(predicate_idx).map_err(|_| vortex_err!("too many predicates"))?;
        let state = PredicateReadWorkState {
            execution: Arc::clone(self),
            morsel_id,
            predicate_idx,
            version,
            range,
            need,
            compact,
            len,
            priority,
            lane: ScanTaskLane::Predicate {
                predicate_idx: predicate_idx_u32,
            },
        };
        Ok(Box::new(PredicateReadWorkTask::try_new(state, task)?))
    }

    fn plan_projection_work(
        self: &Arc<Self>,
        morsel_id: usize,
        range: Range<u64>,
        selected: Mask,
    ) -> VortexResult<Option<QueuedWork>> {
        // Projection consumes the final selected rows after every predicate plan has contributed
        // metadata evidence and, if needed, exact residual evidence. There is no separate
        // predicate-demand mask at this point.
        let len = range_len(&range)?;
        let selected = if let Some(limit_remaining) = &self.limit_remaining {
            limit_mask(selected, limit_remaining)?
        } else {
            selected
        };
        if selected.all_false() {
            return Ok(None);
        }
        if selected.len() != len {
            vortex_bail!(
                "scan2 projection selection length {} does not match range length {len}",
                selected.len()
            );
        }

        let rows = OwnedRowScope::selected(selected);
        let phase = ScanIoPhase::ProjectionRead;
        let task = Arc::clone(&self.projection).create_task(range, rows, phase)?;

        let execution = Arc::clone(self);
        Ok(Some(Box::new(ProjectionReadWorkTask::try_new(
            execution, task, morsel_id,
        )?)))
    }

    fn splits(&self, row_range: &Range<u64>) -> VortexResult<Vec<Range<u64>>> {
        self.plan.splits(row_range)
    }
}

fn push_overlapping_fragments(
    fragments: &[EvidenceFragment],
    range: &Range<u64>,
    output: &mut Vec<EvidenceFragment>,
) -> VortexResult<()> {
    let start = fragments
        .partition_point(|fragment| fragment.rows.start < range.start)
        .saturating_sub(1);
    for fragment in &fragments[start..] {
        if fragment.rows.start >= range.end {
            break;
        }
        if let Some(fragment) = slice_evidence_fragment(fragment, range)? {
            output.push(fragment);
        }
    }
    Ok(())
}

fn slice_evidence_fragment(
    fragment: &EvidenceFragment,
    range: &Range<u64>,
) -> VortexResult<Option<EvidenceFragment>> {
    let rows = fragment.rows.start.max(range.start)..fragment.rows.end.min(range.end);
    if rows.start >= rows.end {
        return Ok(None);
    }
    if rows == fragment.rows {
        return Ok(Some(fragment.clone()));
    }

    let local = usize::try_from(rows.start - fragment.rows.start)
        .map_err(|_| vortex_err!("evidence fragment exceeds usize"))?
        ..usize::try_from(rows.end - fragment.rows.start)
            .map_err(|_| vortex_err!("evidence fragment exceeds usize"))?;
    let kind = match &fragment.kind {
        PredicateEvidenceKind::AllFalse => PredicateEvidenceKind::AllFalse,
        PredicateEvidenceKind::AllTrue => PredicateEvidenceKind::AllTrue,
        PredicateEvidenceKind::Unknown => PredicateEvidenceKind::Unknown,
        PredicateEvidenceKind::ExactMask(mask) => {
            PredicateEvidenceKind::ExactMask(mask.slice(local))
        }
        PredicateEvidenceKind::CandidateMask(mask) => {
            PredicateEvidenceKind::CandidateMask(mask.slice(local))
        }
    };
    Ok(Some(EvidenceFragment::new(rows, kind)))
}

fn push_expr(
    root: &ScanPlanRef,
    expr: &Expression,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<ScanPlanRef> {
    validate_temporal_comparisons(expr, dtype)?;
    Arc::clone(root)
        .try_push_expr(expr, &mut PushCtx::new(session.clone()))?
        .ok_or_else(|| vortex_err!("scan2 could not push expression {expr}"))
}

fn extend_split_hints(plan: &ScanPlanRef, points: &mut Vec<u64>) {
    if let Some(hints) = plan.split_hints() {
        points.extend_from_slice(hints);
    }
}

#[derive(Clone, Copy, Debug)]
enum PreparedSplitKind {
    SelectionRanges,
    Natural,
}

fn prepare_split_ranges(
    row_count: u64,
    row_range: &Range<u64>,
    selection: &Selection,
    split_hints: Vec<u64>,
) -> (Vec<Range<u64>>, PreparedSplitKind) {
    let explicit_row_range = explicit_row_range(row_count, row_range);
    if let Some(ranges) = selection_split_ranges(selection, explicit_row_range) {
        return (ranges, PreparedSplitKind::SelectionRanges);
    }

    let file_range = 0..row_count;
    let selection_range = intersect_ranges(Some(&file_range), selection_bounding_range(selection));
    let bounded_range = intersect_ranges(explicit_row_range, selection_range);
    let points = normalize_split_points(row_count, split_hints);
    (
        natural_split_ranges(&points, bounded_range.as_ref()),
        PreparedSplitKind::Natural,
    )
}

fn explicit_row_range(row_count: u64, row_range: &Range<u64>) -> Option<&Range<u64>> {
    (row_range.start != 0 || row_range.end != row_count).then_some(row_range)
}

fn selection_split_ranges(
    selection: &Selection,
    row_range: Option<&Range<u64>>,
) -> Option<Vec<Range<u64>>> {
    let Selection::IncludeByIndex(buffer) = selection else {
        return None;
    };
    if row_range.is_some() {
        return None;
    }

    let indices = buffer.as_slice();
    if indices.is_empty() {
        return Some(Vec::new());
    }
    debug_assert!(indices.is_sorted());

    let mut ranges = Vec::with_capacity((indices.len() as u64 / MAX_SELECTION_RANGE_SIZE) as usize);
    let mut curr_start = indices[0];
    let mut curr_end = indices[0].saturating_add(1);
    for &idx in &indices[1..] {
        let idx_end = idx.saturating_add(1);
        let new_range_size = idx_end.saturating_sub(curr_start);
        let gap = idx_end.saturating_sub(curr_end);
        if new_range_size >= MAX_SELECTION_RANGE_SIZE {
            if gap >= MIN_SELECTION_GAP_BETWEEN_RANGES {
                ranges.push(curr_start..curr_end);
                curr_start = idx;
                curr_end = idx_end;
            } else {
                return None;
            }
        } else {
            curr_end = idx_end;
        }
    }
    ranges.push(curr_start..curr_end);
    Some(ranges)
}

fn selection_bounding_range(selection: &Selection) -> Option<Range<u64>> {
    match selection {
        Selection::IncludeByIndex(buffer) => {
            let indices = buffer.as_slice();
            indices
                .first()
                .zip(indices.last())
                .map(|(&first, &last)| first..last.saturating_add(1))
        }
        Selection::IncludeRoaring(roaring) if !roaring.is_empty() => {
            Some(roaring.min()?..roaring.max()?.saturating_add(1))
        }
        _ => None,
    }
}

fn intersect_ranges(left: Option<&Range<u64>>, right: Option<Range<u64>>) -> Option<Range<u64>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.start.max(right.start)..left.end.min(right.end)),
        (Some(left), None) => Some(left.clone()),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn normalize_split_points(row_count: u64, mut hints: Vec<u64>) -> Vec<u64> {
    hints.push(0);
    hints.push(row_count);
    hints.retain(|&hint| hint <= row_count);
    hints.sort_unstable();
    hints.dedup();
    hints
}

fn natural_split_ranges(split_points: &[u64], row_range: Option<&Range<u64>>) -> Vec<Range<u64>> {
    let points = if let Some(row_range) = row_range {
        if row_range.start >= row_range.end {
            return Vec::new();
        }
        let mut points = Vec::new();
        points.push(row_range.start);
        points.extend(
            split_points
                .iter()
                .copied()
                .filter(|&point| row_range.start < point && point < row_range.end),
        );
        points.push(row_range.end);
        points.sort_unstable();
        points.dedup();
        points
    } else {
        split_points.to_vec()
    };

    points
        .windows(2)
        .filter_map(|window| {
            let range = window[0]..window[1];
            (range.start < range.end).then_some(range)
        })
        .collect()
}

fn trace_prepared_splits(
    row_range: &Range<u64>,
    splits: &[Range<u64>],
    split_kind: PreparedSplitKind,
    has_filter: bool,
) {
    tracing::debug!(
        target: "vortex_file::scan_v2",
        ?split_kind,
        split_count = splits.len(),
        row_start = row_range.start,
        row_end = row_range.end,
        first_split = ?splits.first(),
        last_split = ?splits.last(),
        has_filter,
        "prepared scan2 splits"
    );
    tracing::trace!(
        target: "vortex_file::scan_v2",
        ?splits,
        "prepared scan2 split ranges"
    );
}

fn check_range(range: &Range<u64>, row_count: u64) -> VortexResult<()> {
    if range.start > range.end || range.end > row_count {
        vortex_bail!(
            "scan2 row range {:?} is out of bounds for row count {}",
            range,
            row_count
        );
    }
    range_len(range).map(|_| ())
}

fn range_len(range: &Range<u64>) -> VortexResult<usize> {
    let len = range
        .end
        .checked_sub(range.start)
        .ok_or_else(|| vortex_err!("scan2 row range end is before start: {range:?}"))?;
    usize::try_from(len).map_err(|_| vortex_err!("scan2 row range exceeds usize"))
}

fn limit_mask(mask: Mask, remaining: &AtomicU64) -> VortexResult<Mask> {
    let true_count = mask.true_count();
    let true_count =
        u64::try_from(true_count).map_err(|_| vortex_err!("mask count exceeds u64"))?;

    loop {
        let available = remaining.load(Ordering::Acquire);
        if available == 0 {
            return Ok(Mask::new_false(mask.len()));
        }

        let take = true_count.min(available);
        if remaining
            .compare_exchange_weak(
                available,
                available - take,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            continue;
        }

        if take == true_count {
            return Ok(mask);
        }

        let take = usize::try_from(take).unwrap_or(usize::MAX);
        return Ok(Mask::from_indices(
            mask.len(),
            (0..mask.len()).filter(|idx| mask.value(*idx)).take(take),
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use vortex_array::expr::get_item;
    use vortex_array::expr::like;
    use vortex_array::expr::lit;
    use vortex_array::expr::not_eq;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use super::limit_mask;
    use super::predicate_cost;

    #[test]
    fn predicate_cost_orders_cheap_before_expensive() {
        let cheap = not_eq(get_item("search", root()), lit(""));
        let expensive = like(get_item("url", root()), lit("%google%"));
        assert!(
            predicate_cost(&cheap) < predicate_cost(&expensive),
            "primitive comparison must be cheaper than LIKE: cheap={}, expensive={}",
            predicate_cost(&cheap),
            predicate_cost(&expensive),
        );
    }

    #[test]
    fn limit_mask_consumes_full_mask_when_limit_allows() -> VortexResult<()> {
        let remaining = AtomicU64::new(4);

        let selected = limit_mask(Mask::from_indices(6, [1, 2, 4]), &remaining)?;

        assert_eq!(selected.true_count(), 3);
        assert!(selected.value(1));
        assert!(selected.value(2));
        assert!(selected.value(4));
        assert_eq!(remaining.load(Ordering::Acquire), 1);
        Ok(())
    }

    #[test]
    fn limit_mask_trims_mask_to_remaining_rows() -> VortexResult<()> {
        let remaining = AtomicU64::new(2);

        let selected = limit_mask(Mask::from_indices(6, [1, 2, 4]), &remaining)?;

        assert_eq!(selected.true_count(), 2);
        assert!(selected.value(1));
        assert!(selected.value(2));
        assert!(!selected.value(4));
        assert_eq!(remaining.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn limit_mask_shared_counter_never_overselects() -> VortexResult<()> {
        let remaining = Arc::new(AtomicU64::new(10));

        let handles = (0..16)
            .map(|_| {
                let remaining = Arc::clone(&remaining);
                std::thread::spawn(move || limit_mask(Mask::new_true(8), &remaining))
            })
            .collect::<Vec<_>>();

        let mut selected_rows = 0;
        for handle in handles {
            let selected = handle
                .join()
                .map_err(|_| vortex_err!("limit mask worker thread panicked"))??;
            selected_rows += selected.true_count();
        }

        assert_eq!(selected_rows, 10);
        assert_eq!(remaining.load(Ordering::Acquire), 0);
        Ok(())
    }
}

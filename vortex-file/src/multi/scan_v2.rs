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
use vortex_scan::ScanTicket;
use vortex_scan::SegmentSourceId;
use vortex_scan::SegmentSourceMeta;
use vortex_scan::WorkRequest;
use vortex_scan::plan::FileReader;
use vortex_scan::plan::OwnedRowScope;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedAggregateRef;
use vortex_scan::plan::PreparedEvidenceRef;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PreparedStats;
use vortex_scan::plan::PreparedStatsRef;
use vortex_scan::plan::PushCtx;
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
use vortex_scan::segments::ScanIoPhase;
use vortex_scan::segments::ScheduledSegmentSource;
use vortex_scan::segments::ScheduledSegmentSourceReader;
use vortex_scan::segments::SegmentFutureCache;
use vortex_scan::segments::SegmentPlanCtx;
use vortex_scan::segments::SegmentRequests;
use vortex_scan::segments::SubmittedSegmentRequests;
use vortex_scan::segments::submit_segment_requests_cached;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use super::MultiFileDataSource;
use super::create_local_filesystem;
use super::open_file;
use crate::FileStatistics;
use crate::VortexFile;
use crate::VortexOpenOptions;

const DEFAULT_CONCURRENCY: usize = 8;
const FALLBACK_SPLIT_SIZE: u64 = 100_000;
const DEFAULT_EVIDENCE_MORSEL_WINDOW: usize = 8;

/// Below this demanded-row density, evaluate a residual predicate over only the demanded rows
/// (filter-first) rather than the whole morsel. Mirrors the V1 flat-reader threshold.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

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
        _io: &'a FileReader,
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
pub(super) async fn build_scan_plan_data_source(
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
        let provider = scan_request
            .scheduler_provider
            .clone()
            .unwrap_or_else(|| self.session.scan_scheduler_provider());
        let scheduler = provider.scheduler_for_scan(&meta);
        let ticket = scheduler.register_scan(meta);

        let mut planned_files = Vec::new();
        let mut total_morsels = 0usize;
        let mut has_runtime_evidence = false;
        for (partition_idx, file) in self.open_files(false).await? {
            let Some(request) = file_scan_request(partition_idx, &file, scan_request.clone())?
            else {
                continue;
            };
            let prepared = Arc::new(PreparedScanPlanFile::try_new(file, request, &ticket)?);
            let ranges = prepared.splits()?;
            if ranges.is_empty() {
                continue;
            }
            has_runtime_evidence |= prepared.has_runtime_evidence();
            total_morsels = total_morsels.saturating_add(ranges.len());
            planned_files.push((prepared, ranges));
        }

        // The physical plan may expose more engine partitions than we can fill with morsels.
        // Keep only non-empty planned partitions; engine adapters can return empty streams for
        // any surplus advertised partitions.
        let partition_count = total_morsels.min(target_partitions);
        let mut partitions = vec![Vec::new(); partition_count];
        let mut morsel_idx = 0usize;
        for (prepared, ranges) in planned_files {
            for range in ranges {
                let partition = morsel_idx % partition_count;
                partitions[partition].push(PlannedScanPlanMorsel {
                    prepared: Arc::clone(&prepared),
                    range,
                });
                morsel_idx = morsel_idx.saturating_add(1);
            }
        }

        let default_window = get_available_parallelism().unwrap_or(1).saturating_mul(4);
        let (morsel_plan_window, morsel_launch_window) =
            morsel_windows(&scheduler, false, has_runtime_evidence, default_window);

        Ok(Some(Arc::new(PlannedScanPlanScan {
            dtype,
            partitions,
            scheduler,
            ticket,
            morsel_plan_window,
            morsel_launch_window,
        })))
    }

    async fn scan(&self, scan_request: DataSourceScanRequest) -> VortexResult<DataSourceScanRef> {
        let meta = ScanMeta {
            label: Some("scan2".to_string()),
        };
        let provider = scan_request
            .scheduler_provider
            .clone()
            .unwrap_or_else(|| self.session.scan_scheduler_provider());
        let scheduler = provider.scheduler_for_scan(&meta);
        let ticket = scheduler.register_scan(meta);

        let mut ready = VecDeque::new();
        let mut deferred = VecDeque::new();

        for child in &self.children {
            match child {
                ScanPlanChild::Opened(file) => ready.push_back(file.clone()),
                ScanPlanChild::Deferred(factory) => deferred.push_back(Arc::clone(factory)),
            }
        }

        let dtype = scan_request.projection.return_dtype(&self.dtype)?;

        Ok(Box::new(ScanPlanDataSourceScan {
            dtype,
            request: scan_request,
            ready,
            deferred,
            handle: self.session.handle(),
            concurrency: self.concurrency,
            scheduler,
            ticket,
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
    ready: VecDeque<VortexFile>,
    deferred: VecDeque<Arc<dyn VortexFileFactory>>,
    handle: Handle,
    concurrency: usize,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
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
            ticket,
        } = *self;

        let ordered = request.ordered;
        let ready_stream = stream::iter(ready).map(Ok);
        let spawned = stream::iter(deferred).map(move |factory| {
            handle.spawn(async move {
                factory
                    .open()
                    .instrument(tracing::info_span!("VortexFileFactory::open"))
                    .await
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
            .enumerate()
            .filter_map(move |(index, file_result)| {
                let request = request.clone();
                let scheduler = Arc::clone(&scheduler);
                let ticket = ticket.clone();
                async move {
                    match file_result {
                        Ok(file) => {
                            file_partition(index, file, request, scheduler, ticket).transpose()
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
    ticket: ScanTicket,
) -> VortexResult<Option<PartitionRef>> {
    let Some(request) = file_scan_request(partition_idx, &file, request)? else {
        return Ok(None);
    };

    Ok(Some(Box::new(ScanPlanPartition {
        file,
        request,
        index: partition_idx,
        scheduler,
        ticket,
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
    let provider = request
        .scheduler_provider
        .clone()
        .unwrap_or_else(|| file.session().scan_scheduler_provider());
    let scheduler = provider.scheduler_for_scan(&meta);
    let ticket = scheduler.register_scan(meta);

    let Some(partition) = file_partition(0, file, request, scheduler, ticket)? else {
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
    let reader = FileReader::new(file.segment_source(), session);
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
    split_ranges_from_node(&root, file.row_count())
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
    let reader = FileReader::new(file.segment_source(), session.clone());
    let state = plan.init_state(&session)?;
    plan.splits(0..file.row_count(), &reader, state.as_ref())
        .await
}

fn split_ranges_from_node(node: &ScanPlanRef, row_count: u64) -> VortexResult<Vec<Range<u64>>> {
    let mut points = vec![0, row_count];
    if let Some(hints) = node.split_hints() {
        points.extend(
            hints
                .iter()
                .copied()
                .filter(|&hint| 0 < hint && hint < row_count),
        );
    }
    points.sort_unstable();
    points.dedup();
    Ok(points
        .windows(2)
        .filter_map(|window| {
            let range = window[0]..window[1];
            (range.start < range.end).then_some(range)
        })
        .collect())
}

pub(crate) fn build_file_scan_plan_root(file: &VortexFile) -> VortexResult<ScanPlanRef> {
    let mut plan_request = ScanRequest::empty();
    let layout = file
        .footer()
        .layout2()
        .ok_or_else(|| vortex_err!("scan2 requires a v2 footer layout"))?;
    let root = layout.new_scan_plan(&mut plan_request, file.session())?;
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

struct Work<T> {
    phase: ScanIoPhase,
    known_bytes: u64,
    handle: Handle,
    future: BoxFuture<'static, VortexResult<T>>,
}

impl<T: Send + 'static> Work<T> {
    fn new(
        phase: ScanIoPhase,
        handle: Handle,
        registered: SubmittedSegmentRequests,
        future: BoxFuture<'static, VortexResult<T>>,
    ) -> Self {
        let known_bytes = registered.bytes();
        let future = async move {
            let result = future.await;
            drop(registered);
            result
        }
        .boxed();
        Self {
            phase,
            known_bytes,
            handle,
            future,
        }
    }

    fn into_queued(
        self,
        morsel_id: usize,
        map: impl FnOnce(T) -> WorkOutput + Send + 'static,
    ) -> QueuedWork {
        QueuedWork {
            morsel_id,
            phase: self.phase,
            known_bytes: self.known_bytes,
            handle: self.handle,
            future: async move { self.future.await.map(map) }.boxed(),
        }
    }
}

struct QueuedWork {
    morsel_id: usize,
    phase: ScanIoPhase,
    known_bytes: u64,
    handle: Handle,
    future: BoxFuture<'static, VortexResult<WorkOutput>>,
}

struct EvidenceWorkOutput {
    morsel_id: usize,
    predicate_idx: usize,
    evidence: PredicateEvidence,
}

struct ProjectionWorkOutput {
    morsel_id: usize,
    array: ArrayRef,
}

enum WorkOutput {
    Evidence(EvidenceWorkOutput),
    Projection(ProjectionWorkOutput),
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
    prepared: Arc<PreparedScanPlanFile>,
    range: Range<u64>,
    selected: Mask,
    evidence: Vec<Option<PredicateEvidence>>,
    pending_evidence: usize,
    next_predicate: usize,
    next_recheck_predicate: usize,
}

struct PartitionWorkSchedulerState {
    pending: VecDeque<PlannedScanPlanMorsel>,
    morsels: Vec<Option<MorselState>>,
    active_morsels: usize,
    next_morsel_id: usize,
    next_emit_morsel_id: usize,
    evidence_queue: VecDeque<QueuedWork>,
    predicate_queue: VecDeque<QueuedWork>,
    projection_queue: VecDeque<QueuedWork>,
    in_flight: FuturesUnordered<BoxFuture<'static, VortexResult<WorkOutput>>>,
    completed_morsels: BTreeMap<usize, CompletedMorsel>,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
    ordered: bool,
    plan_window: usize,
    launch_window: usize,
    phase_cursor: usize,
}

const WEIGHTED_PHASES: &[ScanIoPhase] = &[
    ScanIoPhase::EvidenceProbe,
    ScanIoPhase::EvidenceProbe,
    ScanIoPhase::PredicateRead,
    ScanIoPhase::ProjectionRead,
];

fn morsel_windows(
    scheduler: &ScanScheduler,
    limited: bool,
    has_runtime_evidence: bool,
    default_window: usize,
) -> (usize, usize) {
    if limited {
        return (1, 1);
    }
    let launch_window = scheduler
        .config()
        .morsel_launch_window()
        .unwrap_or_else(|| {
            if has_runtime_evidence {
                default_window.min(DEFAULT_EVIDENCE_MORSEL_WINDOW)
            } else {
                default_window
            }
        })
        .max(1);
    let plan_window = scheduler
        .config()
        .morsel_plan_window()
        .map(|window| window.max(launch_window).max(1))
        .unwrap_or_else(|| {
            if has_runtime_evidence {
                launch_window
            } else {
                usize::MAX
            }
        });
    (plan_window, launch_window)
}

fn partition_work_stream(
    morsels: Vec<PlannedScanPlanMorsel>,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
    ordered: bool,
    plan_window: usize,
    launch_window: usize,
) -> impl futures::Stream<Item = VortexResult<ArrayRef>> + Send + 'static {
    let state = PartitionWorkSchedulerState {
        pending: VecDeque::from(morsels),
        morsels: Vec::new(),
        active_morsels: 0,
        next_morsel_id: 0,
        next_emit_morsel_id: 0,
        evidence_queue: VecDeque::new(),
        predicate_queue: VecDeque::new(),
        projection_queue: VecDeque::new(),
        in_flight: FuturesUnordered::new(),
        completed_morsels: BTreeMap::new(),
        scheduler,
        ticket,
        ordered,
        plan_window,
        launch_window,
        phase_cursor: 0,
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

            while state.in_flight.len() < state.launch_window {
                let Some(work) = state.pop_next_work() else {
                    break;
                };
                state.launch(work);
            }

            if state.in_flight.is_empty() {
                if state.is_done() {
                    return None;
                }
                let error = vortex_err!(
                    "scan2 work scheduler stalled: {} active morsels, {} pending morsels, {} evidence work items, {} predicate work items, {} projection work items",
                    state.active_morsels,
                    state.pending.len(),
                    state.evidence_queue.len(),
                    state.predicate_queue.len(),
                    state.projection_queue.len()
                );
                state.clear();
                return Some((Err(error), state));
            }

            match state.in_flight.next().await {
                Some(Ok(output)) => match state.complete_work(output) {
                    Ok(Some(array)) => return Some((Ok(array), state)),
                    Ok(None) => continue,
                    Err(error) => return Some((Err(error), state)),
                },
                Some(Err(error)) => return Some((Err(error), state)),
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
        self.next_emit_morsel_id = 0;
        self.evidence_queue.clear();
        self.predicate_queue.clear();
        self.projection_queue.clear();
        self.in_flight = FuturesUnordered::new();
        self.completed_morsels.clear();
    }

    fn is_done(&self) -> bool {
        self.pending.is_empty()
            && self.active_morsels == 0
            && self.evidence_queue.is_empty()
            && self.predicate_queue.is_empty()
            && self.projection_queue.is_empty()
            && self.in_flight.is_empty()
            && self.completed_morsels.is_empty()
    }

    fn plan_next_morsel(&mut self) -> VortexResult<()> {
        let Some(morsel) = self.pending.pop_front() else {
            return Ok(());
        };
        let morsel_id = self.next_morsel_id;
        let Some(planned) = morsel.prepared.plan_morsel(morsel_id, morsel.range)? else {
            return Ok(());
        };
        self.next_morsel_id = self.next_morsel_id.saturating_add(1);
        self.active_morsels = self.active_morsels.saturating_add(1);
        if self.morsels.len() <= morsel_id {
            self.morsels.resize_with(morsel_id + 1, || None);
        }
        self.morsels[morsel_id] = Some(planned.state);
        self.evidence_queue.extend(planned.evidence);
        if self.morsels[morsel_id]
            .as_ref()
            .is_some_and(|morsel| morsel.pending_evidence == 0)
        {
            self.enqueue_next_predicate_or_projection(morsel_id)?;
        }
        Ok(())
    }

    fn pop_next_work(&mut self) -> Option<QueuedWork> {
        for _ in 0..WEIGHTED_PHASES.len() {
            let phase = WEIGHTED_PHASES[self.phase_cursor % WEIGHTED_PHASES.len()];
            self.phase_cursor = self.phase_cursor.wrapping_add(1);
            if let Some(work) = self.pop_phase_work(phase) {
                return Some(work);
            }
        }
        [
            ScanIoPhase::EvidenceProbe,
            ScanIoPhase::PredicateRead,
            ScanIoPhase::ProjectionRead,
        ]
        .into_iter()
        .find_map(|phase| self.pop_phase_work(phase))
    }

    fn pop_phase_work(&mut self, phase: ScanIoPhase) -> Option<QueuedWork> {
        let queue = match phase {
            ScanIoPhase::EvidenceProbe | ScanIoPhase::EvidenceSetup => &mut self.evidence_queue,
            ScanIoPhase::PredicateRead => &mut self.predicate_queue,
            ScanIoPhase::ProjectionRead | ScanIoPhase::AggregateRead => &mut self.projection_queue,
        };
        while let Some(work) = queue.pop_front() {
            if self
                .morsels
                .get(work.morsel_id)
                .and_then(Option::as_ref)
                .is_some()
            {
                return Some(work);
            }
        }
        None
    }

    fn launch(&mut self, work: QueuedWork) {
        let scheduler = Arc::clone(&self.scheduler);
        let ticket = self.ticket.clone();
        self.in_flight.push(
            work.handle
                .spawn(
                    async move {
                        let _permit = scheduler.acquire(&ticket, WorkRequest::morsel()).await?;
                        work.future.await
                    }
                    .instrument(tracing::trace_span!(
                        "scan2_work",
                        phase = ?work.phase,
                        known_bytes = work.known_bytes,
                    )),
                )
                .boxed(),
        );
    }

    fn complete_work(&mut self, output: WorkOutput) -> VortexResult<Option<ArrayRef>> {
        match output {
            WorkOutput::Evidence(output) => self.complete_evidence(output),
            WorkOutput::Projection(output) => {
                Ok(self.finish_output_morsel(output.morsel_id, output.array))
            }
        }
    }

    fn complete_evidence(&mut self, output: EvidenceWorkOutput) -> VortexResult<Option<ArrayRef>> {
        let Some(morsel) = self
            .morsels
            .get_mut(output.morsel_id)
            .and_then(Option::as_mut)
        else {
            return Ok(None);
        };
        morsel.pending_evidence = morsel.pending_evidence.saturating_sub(1);
        morsel.selected = &morsel.selected & output.evidence.maybe();
        if morsel.selected.all_false() || output.evidence.all_false() {
            return Ok(self.finish_empty_morsel(output.morsel_id));
        }
        if let Some(slot) = morsel.evidence.get_mut(output.predicate_idx) {
            *slot = Some(output.evidence);
        }
        if morsel.pending_evidence == 0 {
            self.enqueue_next_predicate_or_projection(output.morsel_id)?;
        }
        Ok(None)
    }

    fn enqueue_next_predicate_or_projection(&mut self, morsel_id: usize) -> VortexResult<()> {
        loop {
            let Some(morsel) = self.morsels.get(morsel_id).and_then(Option::as_ref) else {
                return Ok(());
            };
            if morsel.pending_evidence != 0 {
                return Ok(());
            }
            if morsel.next_predicate >= morsel.prepared.predicates.len() {
                if self.enqueue_recheck_evidence(morsel_id)? {
                    return Ok(());
                }
                let Some(morsel) = self.morsels.get(morsel_id).and_then(Option::as_ref) else {
                    return Ok(());
                };
                let projection = morsel.prepared.plan_projection_work(
                    morsel_id,
                    morsel.range.clone(),
                    morsel.selected.clone(),
                )?;
                match projection {
                    Some(work) => self.projection_queue.push_back(work),
                    None => {
                        self.finish_empty_morsel(morsel_id);
                    }
                }
                return Ok(());
            }

            let predicate_idx = morsel.next_predicate;
            if morsel.evidence[predicate_idx].is_none() {
                let should_probe = {
                    let predicate = &morsel.prepared.predicates[predicate_idx];
                    !predicate.evidence.is_empty()
                        && morsel.selected.density() >= EXPR_EVAL_THRESHOLD
                };
                if should_probe {
                    let work = morsel.prepared.plan_evidence_work(
                        morsel_id,
                        predicate_idx,
                        morsel.range.clone(),
                        morsel.prepared.predicates[predicate_idx].version(),
                        EvidenceMode::Normal,
                    )?;
                    let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut)
                    else {
                        return Ok(());
                    };
                    morsel.pending_evidence = morsel.pending_evidence.saturating_add(1);
                    self.evidence_queue.push_back(work);
                    return Ok(());
                }

                let evidence = PredicateEvidence::new(
                    morsel.prepared.predicates[predicate_idx].id,
                    morsel.prepared.predicates[predicate_idx].version(),
                    morsel.range.clone(),
                )?;
                let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
                    return Ok(());
                };
                morsel.evidence[predicate_idx] = Some(evidence);
                continue;
            }
            let evidence = morsel.evidence[predicate_idx].as_ref().ok_or_else(|| {
                vortex_err!("missing evidence for predicate {predicate_idx} before residual read")
            })?;
            let need = &morsel.selected & &evidence.unproven();
            if need.all_false() {
                let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
                    return Ok(());
                };
                morsel.next_predicate = morsel.next_predicate.saturating_add(1);
                continue;
            }

            let work = morsel.prepared.plan_predicate_work(
                morsel_id,
                predicate_idx,
                morsel.range.clone(),
                need,
                morsel.prepared.predicates[predicate_idx].version(),
            )?;
            let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
                return Ok(());
            };
            morsel.next_predicate = predicate_idx.saturating_add(1);
            self.predicate_queue.push_back(work);
            return Ok(());
        }
    }

    fn enqueue_recheck_evidence(&mut self, morsel_id: usize) -> VortexResult<bool> {
        loop {
            let Some(morsel) = self.morsels.get(morsel_id).and_then(Option::as_ref) else {
                return Ok(false);
            };
            if morsel.next_recheck_predicate >= morsel.prepared.predicates.len() {
                return Ok(false);
            }

            let predicate_idx = morsel.next_recheck_predicate;
            let predicate = &morsel.prepared.predicates[predicate_idx];
            let current_version = predicate.version();
            let evidence_version = morsel.evidence[predicate_idx]
                .as_ref()
                .map(PredicateEvidence::version)
                .unwrap_or(PredicateVersion::STATIC);

            if predicate.dynamic_updates.is_some()
                && predicate.has_recheck_evidence()
                && current_version != evidence_version
            {
                let work = morsel.prepared.plan_evidence_work(
                    morsel_id,
                    predicate_idx,
                    morsel.range.clone(),
                    current_version,
                    EvidenceMode::RecheckBeforeProjection,
                )?;
                let Some(morsel) = self.morsels.get_mut(morsel_id).and_then(Option::as_mut) else {
                    return Ok(false);
                };
                morsel.pending_evidence = morsel.pending_evidence.saturating_add(1);
                self.evidence_queue.push_back(work);
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
}

struct ScanPlanPartition {
    file: VortexFile,
    request: DataSourceScanRequest,
    index: usize,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
}

impl Partition for ScanPlanPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        let Some(row_range) = self.request.row_range.as_ref() else {
            return Precision::Absent;
        };
        let row_count = row_range.end - row_range.start;
        let row_count = self.request.selection.row_count(row_count);
        let row_count = self
            .request
            .limit
            .map_or(row_count, |limit| row_count.min(limit));

        if self.request.filter.is_some() {
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
            request,
            index: _,
            scheduler,
            ticket,
        } = *self;

        let prepared = Arc::new(PreparedScanPlanFile::try_new(file, request, &ticket)?);
        let dtype = prepared.dtype.clone();
        let ranges = prepared.splits()?;
        let ordered = prepared.ordered;
        let default_window = get_available_parallelism().unwrap_or(1) * 4;
        let (plan_window, launch_window) = morsel_windows(
            &scheduler,
            prepared.limit_remaining.is_some(),
            prepared.has_runtime_evidence(),
            default_window,
        );
        let morsels = ranges
            .into_iter()
            .map(|range| PlannedScanPlanMorsel {
                prepared: Arc::clone(&prepared),
                range,
            })
            .collect::<Vec<_>>();

        let stream = partition_work_stream(
            morsels,
            scheduler,
            ticket,
            ordered,
            plan_window,
            launch_window,
        );

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

struct PlannedScanPlanScan {
    dtype: DType,
    partitions: Vec<Vec<PlannedScanPlanMorsel>>,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
    morsel_plan_window: usize,
    morsel_launch_window: usize,
}

#[derive(Clone)]
struct PlannedScanPlanMorsel {
    prepared: Arc<PreparedScanPlanFile>,
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
            row_count = row_count.saturating_add(morsel.prepared.selection.row_count(range_len));
            has_filter |= !morsel.prepared.predicates.is_empty();
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
        let scheduler = Arc::clone(&planned.scheduler);
        let ticket = planned.ticket.clone();
        let stream = partition_work_stream(
            morsels,
            scheduler,
            ticket,
            false,
            planned.morsel_plan_window,
            planned.morsel_launch_window,
        );

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

struct PreparedScanPlanFile {
    session: VortexSession,
    reader: FileReader,
    dtype: DType,
    row_range: Range<u64>,
    selection: Selection,
    ordered: bool,
    limit_remaining: Option<AtomicU64>,
    segment_source_id: SegmentSourceId,
    scheduled_segment_source: Arc<dyn ScheduledSegmentSource>,
    segment_future_cache: Arc<SegmentFutureCache>,
    split_hints: Option<Vec<u64>>,
    projection: PreparedReadRef,
    predicates: Vec<PreparedPredicate>,
}

struct PreparedPredicate {
    id: PredicateId,
    expr: Expression,
    dynamic_updates: Option<DynamicExprUpdates>,
    read: PreparedReadRef,
    evidence: Vec<PreparedEvidenceRef>,
}

impl PreparedPredicate {
    fn version(&self) -> PredicateVersion {
        self.dynamic_updates
            .as_ref()
            .map(|updates| PredicateVersion::new(updates.version()))
            .unwrap_or(PredicateVersion::STATIC)
    }

    fn has_recheck_evidence(&self) -> bool {
        self.evidence
            .iter()
            .any(|plan| plan.recheck_before_projection())
    }
}

struct RegisteredScheduledSegmentSource {
    source: Arc<dyn ScheduledSegmentSource>,
}

impl PreparedScanPlanFile {
    fn try_new(
        file: VortexFile,
        request: DataSourceScanRequest,
        ticket: &ScanTicket,
    ) -> VortexResult<Self> {
        let session = file.session().clone();
        let dtype = request.projection.return_dtype(file.dtype())?;
        let projection = request.projection.optimize_recursive(file.dtype())?;
        let filter = request
            .filter
            .map(|filter| filter.optimize_recursive(file.dtype()))
            .transpose()?;

        let root = file.scan_plan_root()?;
        let registered_source = Arc::new(RegisteredScheduledSegmentSource {
            source: file.scheduled_segment_source(),
        });
        let segment_source_id = ticket.register_segment_source(
            Arc::clone(&registered_source),
            SegmentSourceMeta {
                label: Some("vortex-file".to_string()),
            },
        );
        let scheduled_segment_source = Arc::clone(&registered_source.source);
        let segment_future_cache = file.scan_plan_segment_future_cache();
        let reader = FileReader::new(
            Arc::new(ScheduledSegmentSourceReader::new(
                segment_source_id,
                Arc::clone(&scheduled_segment_source),
                Arc::clone(&segment_future_cache),
            )),
            session.clone(),
        );

        let mut prepare_ctx =
            PrepareCtx::with_state_cache(session.clone(), file.scan_plan_state_cache());
        let projection_pushed = push_expr(&root, &projection, file.dtype(), &session)?;
        let mut split_hints = Vec::new();
        extend_split_hints(&projection_pushed, &mut split_hints);
        let projection_plan = Arc::clone(&projection_pushed)
            .prepare_read(&mut prepare_ctx)?
            .ok_or_else(|| vortex_err!("scan2 could not plan read for expression {projection}"))?;

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
                let read = Arc::clone(&pushed)
                    .prepare_read(&mut prepare_ctx)?
                    .ok_or_else(|| vortex_err!("scan2 could not plan predicate read {expr}"))?;
                let evidence = pushed.prepare_evidence(&mut prepare_ctx)?;
                let dynamic_updates = DynamicExprUpdates::new(&expr);
                Ok(PreparedPredicate {
                    id,
                    expr,
                    dynamic_updates,
                    read,
                    evidence,
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Self {
            session,
            reader,
            dtype,
            row_range: request
                .row_range
                .ok_or_else(|| vortex_err!("scan2 partition row range missing"))?,
            selection: request.selection,
            ordered: request.ordered,
            limit_remaining: request.limit.map(AtomicU64::new),
            segment_source_id,
            scheduled_segment_source,
            segment_future_cache,
            split_hints: normalize_split_hints(split_hints),
            projection: projection_plan,
            predicates,
        })
    }

    fn segment_plan_ctx(&self, phase: ScanIoPhase) -> SegmentPlanCtx {
        SegmentPlanCtx::new(
            self.segment_source_id,
            Arc::clone(&self.scheduled_segment_source),
            self.session.clone(),
        )
        .with_phase(phase)
    }

    fn submit_segment_requests(&self, requests: SegmentRequests) -> SubmittedSegmentRequests {
        submit_segment_requests_cached(
            self.segment_future_cache.as_ref(),
            self.scheduled_segment_source.as_ref(),
            requests,
        )
    }

    fn has_runtime_evidence(&self) -> bool {
        self.predicates
            .iter()
            .any(|predicate| !predicate.evidence.is_empty())
    }

    fn plan_morsel(
        self: &Arc<Self>,
        _morsel_id: usize,
        range: Range<u64>,
    ) -> VortexResult<Option<PlannedMorselWork>> {
        let selected = self.selection.row_mask(&range).mask().clone();
        if selected.all_false() {
            return Ok(None);
        }

        let state = MorselState {
            prepared: Arc::clone(self),
            range,
            selected,
            evidence: (0..self.predicates.len()).map(|_| None).collect(),
            pending_evidence: 0,
            next_predicate: 0,
            next_recheck_predicate: 0,
        };

        Ok(Some(PlannedMorselWork {
            state,
            evidence: Vec::new(),
        }))
    }

    fn plan_evidence_work(
        self: &Arc<Self>,
        morsel_id: usize,
        predicate_idx: usize,
        range: Range<u64>,
        version: PredicateVersion,
        mode: EvidenceMode,
    ) -> VortexResult<QueuedWork> {
        let predicate = &self.predicates[predicate_idx];
        let mut registered = SubmittedSegmentRequests::default();
        let req = OwnedEvidenceRequest {
            id: predicate.id,
            version,
            predicate: predicate.expr.clone(),
            range: range.clone(),
            mode,
        };
        let mut tasks = Vec::with_capacity(predicate.evidence.len());
        for plan in &predicate.evidence {
            if mode == EvidenceMode::RecheckBeforeProjection && !plan.recheck_before_projection() {
                continue;
            }
            let task = Arc::clone(plan).begin_evidence(req.clone())?;
            let mut segment_ctx = self.segment_plan_ctx(ScanIoPhase::EvidenceProbe);
            let requests = task.segment_requests(&mut segment_ctx)?;
            registered.extend(self.submit_segment_requests(requests));
            tasks.push(task);
        }

        let prepared = Arc::clone(self);
        Ok(Work::new(
            ScanIoPhase::EvidenceProbe,
            self.session.handle(),
            registered,
            async move {
                let predicate = &prepared.predicates[predicate_idx];
                let mut acc = PredicateEvidence::new(predicate.id, version, range.clone())?;
                for task in tasks {
                    for fragment in task.evidence(&prepared.reader).await? {
                        acc.absorb(fragment)?;
                    }
                    if acc.all_false() {
                        break;
                    }
                }
                Ok(EvidenceWorkOutput {
                    morsel_id,
                    predicate_idx,
                    evidence: acc,
                })
            }
            .boxed(),
        )
        .into_queued(morsel_id, WorkOutput::Evidence))
    }

    fn plan_predicate_work(
        self: &Arc<Self>,
        morsel_id: usize,
        predicate_idx: usize,
        range: Range<u64>,
        need: Mask,
        version: PredicateVersion,
    ) -> VortexResult<QueuedWork> {
        let len = range_len(&range)?;
        let predicate = &self.predicates[predicate_idx];
        let compact = need.density() < EXPR_EVAL_THRESHOLD;
        let rows = if compact {
            OwnedRowScope::selected(need.clone())
        } else {
            OwnedRowScope::try_new(Mask::new_true(len), need.clone())?
        };
        let task = Arc::clone(&predicate.read).begin_read(range.clone(), rows)?;
        let mut segment_ctx = self.segment_plan_ctx(ScanIoPhase::PredicateRead);
        let requests = task.segment_requests(&mut segment_ctx)?;
        let registered = self.submit_segment_requests(requests);

        let prepared = Arc::clone(self);
        Ok(Work::new(
            ScanIoPhase::PredicateRead,
            self.session.handle(),
            registered,
            async move {
                let predicate = &prepared.predicates[predicate_idx];
                let mut ctx = prepared.session.create_execution_ctx();
                // Filter-first: when few rows are demanded, read with selection = `need` so the leaf
                // returns the compacted (filtered) array and an expensive residual (e.g. an FSST
                // `LIKE`) evaluates over only `need.true_count()` rows. The compacted verdict is
                // scattered back into the morsel domain via `intersect_by_rank`, giving a full-length
                // mask identical to the dense path's `result & need`. Mirrors V1's flat-reader gate.
                let result = if compact {
                    let compact = task
                        .read(&prepared.reader, &mut ctx)
                        .await?
                        .null_as_false()
                        .execute(&mut ctx)?;
                    if compact.len() != need.true_count() {
                        vortex_bail!(
                            "compacted residual result length {} does not match demanded row count {}",
                            compact.len(),
                            need.true_count()
                        );
                    }
                    need.intersect_by_rank(&compact)
                } else {
                    task
                        .read(&prepared.reader, &mut ctx)
                        .await?
                        .null_as_false()
                        .execute(&mut ctx)?
                };
                if result.len() != len {
                    vortex_bail!(
                        "residual result length {} does not match morsel length {len}",
                        result.len()
                    );
                }
                let pass = &result & &need;
                let exact = !&need | &pass;
                let mut evidence = PredicateEvidence::new(predicate.id, version, range.clone())?;
                evidence.absorb(EvidenceFragment::new(
                    range,
                    PredicateEvidenceKind::ExactMask(exact),
                ))?;
                Ok(EvidenceWorkOutput {
                    morsel_id,
                    predicate_idx,
                    evidence,
                })
            }
            .boxed(),
        )
        .into_queued(morsel_id, WorkOutput::Evidence))
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

        let task =
            Arc::clone(&self.projection).begin_read(range, OwnedRowScope::selected(selected))?;
        let mut segment_ctx = self.segment_plan_ctx(ScanIoPhase::ProjectionRead);
        let requests = task.segment_requests(&mut segment_ctx)?;
        let registered = self.submit_segment_requests(requests);

        let prepared = Arc::clone(self);
        Ok(Some(
            Work::new(
                ScanIoPhase::ProjectionRead,
                self.session.handle(),
                registered,
                async move {
                    let mut ctx = prepared.session.create_execution_ctx();
                    let array = task.read(&prepared.reader, &mut ctx).await?;
                    Ok(ProjectionWorkOutput { morsel_id, array })
                }
                .boxed(),
            )
            .into_queued(morsel_id, WorkOutput::Projection),
        ))
    }

    fn splits(&self) -> VortexResult<Vec<Range<u64>>> {
        let mut points = vec![self.row_range.start];
        if let Some(hints) = &self.split_hints {
            points.extend(
                hints
                    .iter()
                    .copied()
                    .filter(|&hint| self.row_range.start < hint && hint < self.row_range.end),
            );
        }
        if points.len() == 1 {
            let mut next = self
                .row_range
                .start
                .saturating_add(FALLBACK_SPLIT_SIZE)
                .min(self.row_range.end);
            while next < self.row_range.end {
                points.push(next);
                next = next
                    .saturating_add(FALLBACK_SPLIT_SIZE)
                    .min(self.row_range.end);
            }
        }
        points.push(self.row_range.end);
        points.sort_unstable();
        points.dedup();
        Ok(points
            .windows(2)
            .filter_map(|window| {
                let range = window[0]..window[1];
                (range.start < range.end).then_some(range)
            })
            .collect())
    }
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

fn normalize_split_hints(mut hints: Vec<u64>) -> Option<Vec<u64>> {
    hints.sort_unstable();
    hints.dedup();
    (!hints.is_empty()).then_some(hints)
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
    let available = remaining.load(Ordering::Relaxed);
    if available == 0 {
        return Ok(Mask::new_false(mask.len()));
    }
    let true_count = mask.true_count();
    if true_count as u64 <= available {
        remaining.fetch_sub(true_count as u64, Ordering::Relaxed);
        return Ok(mask);
    }
    let take = usize::try_from(available).unwrap_or(usize::MAX);
    remaining.store(0, Ordering::Relaxed);
    Ok(Mask::from_indices(
        mask.len(),
        (0..mask.len()).filter(|idx| mask.value(*idx)).take(take),
    ))
}

#[cfg(test)]
mod tests {
    use vortex_array::expr::get_item;
    use vortex_array::expr::like;
    use vortex_array::expr::lit;
    use vortex_array::expr::not_eq;
    use vortex_array::expr::root;

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
}

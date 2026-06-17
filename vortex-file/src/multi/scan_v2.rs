// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ScanNode-backed multi-file data source.

use std::any::Any;
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
use vortex_layout::scan::v2::evidence::PredicateEvidence;
use vortex_layout::scan::v2::evidence::PredicateId;
use vortex_layout::scan::v2::evidence::PredicateVersion;
use vortex_layout::scan::v2::node::AggregatePlanRef;
use vortex_layout::scan::v2::node::EvidencePlanRef;
use vortex_layout::scan::v2::node::EvidenceStateKey;
use vortex_layout::scan::v2::node::ExpandCtx;
use vortex_layout::scan::v2::node::FileReader;
use vortex_layout::scan::v2::node::PlanCtx;
use vortex_layout::scan::v2::node::PushCtx;
use vortex_layout::scan::v2::node::ReadPlanRef;
use vortex_layout::scan::v2::node::RowScope;
use vortex_layout::scan::v2::node::ScanNode;
use vortex_layout::scan::v2::node::ScanNodeRef;
use vortex_layout::scan::v2::node::ScanStateCache;
use vortex_layout::scan::v2::node::ScanStateRef;
use vortex_layout::scan::v2::node::StateCtx;
use vortex_layout::scan::v2::node::StatsPlan;
use vortex_layout::scan::v2::node::StatsPlanRef;
use vortex_layout::scan::v2::request::EvidenceMode;
use vortex_layout::scan::v2::request::EvidenceRequest;
use vortex_layout::scan::v2::request::NodeRequest;
use vortex_layout::scan::v2::validate_temporal_comparisons;
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
use vortex_scan::ScanRequest;
use vortex_scan::ScanScheduler;
use vortex_scan::ScanSchedulerSessionExt;
use vortex_scan::ScanTicket;
use vortex_scan::WorkRequest;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::parallelism::get_available_parallelism;

use super::MultiFileDataSource;
use super::create_local_filesystem;
use super::open_file;
use crate::FileStatistics;
use crate::VortexFile;
use crate::VortexOpenOptions;

const DEFAULT_CONCURRENCY: usize = 8;
const FALLBACK_SPLIT_SIZE: u64 = 100_000;

struct FileStatsScanNode {
    data: ScanNodeRef,
    stats: Arc<FileStatistics>,
    fields: StructFields,
    row_count: u64,
}

struct FileStatsExprScanNode {
    data: ScanNodeRef,
    stats: Arc<FileStatistics>,
    field_idx: usize,
    field_dtype: DType,
    row_count: u64,
}

struct FileStatsPlan {
    stats: StatsSet,
    field_dtype: DType,
    row_count: u64,
    funcs: Vec<AggregateFnRef>,
}

impl FileStatsScanNode {
    fn try_new(
        data: ScanNodeRef,
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

impl ScanNode for FileStatsScanNode {
    type State = ScanStateRef;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        cx.init_node(&self.data)
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanNodeRef>> {
        let Some(data) = Arc::clone(&self.data).try_push_expr(expr, cx)? else {
            return Ok(None);
        };
        let Some((field_idx, _name, field_dtype)) = self.pushed_field(expr) else {
            return Ok(Some(data));
        };
        Ok(Some(Arc::new(FileStatsExprScanNode {
            data,
            stats: Arc::clone(&self.stats),
            field_idx,
            field_dtype,
            row_count: self.row_count,
        })))
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        Arc::clone(&self.data).plan_read(cx)
    }

    fn plan_evidence(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Vec<EvidencePlanRef>> {
        Arc::clone(&self.data).plan_evidence(cx)
    }

    fn plan_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PlanCtx,
    ) -> VortexResult<Option<AggregatePlanRef>> {
        Arc::clone(&self.data).plan_aggregate_partial(funcs, cx)
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.data.split_hints()
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.data.release(frontier, state.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "file_stats:")?;
        self.data.fmt_chain(f)
    }
}

impl ScanNode for FileStatsExprScanNode {
    type State = ScanStateRef;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        cx.init_node(&self.data)
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanNodeRef>> {
        Arc::clone(&self.data).try_push_expr(expr, cx)
    }

    fn plan_read(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        Arc::clone(&self.data).plan_read(cx)
    }

    fn plan_evidence(self: Arc<Self>, cx: &mut PlanCtx) -> VortexResult<Vec<EvidencePlanRef>> {
        Arc::clone(&self.data).plan_evidence(cx)
    }

    fn plan_aggregate_partial(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        cx: &mut PlanCtx,
    ) -> VortexResult<Option<AggregatePlanRef>> {
        Arc::clone(&self.data).plan_aggregate_partial(funcs, cx)
    }

    fn plan_stats(
        self: Arc<Self>,
        funcs: &[AggregateFnRef],
        _cx: &mut PlanCtx,
    ) -> VortexResult<Option<StatsPlanRef>> {
        let stats = self.stats.stats_sets()[self.field_idx].clone();
        Ok(Some(Arc::new(FileStatsPlan {
            stats,
            field_dtype: self.field_dtype.clone(),
            row_count: self.row_count,
            funcs: funcs.to_vec(),
        })))
    }

    fn split_hints(&self) -> Option<&[u64]> {
        self.data.split_hints()
    }

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.data.release(frontier, state.as_ref())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "file_stats_expr:")?;
        self.data.fmt_chain(f)
    }
}

impl StatsPlan for FileStatsPlan {
    type State = ();

    fn init_state(&self, _ctx: &VortexSession) -> VortexResult<Self::State> {
        Ok(())
    }

    fn stats<'a>(
        &'a self,
        range: Range<u64>,
        _io: &'a FileReader,
        _state: &'a Self::State,
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

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "file_stats")
    }
}

impl FileStatsPlan {
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
pub(super) async fn build_scan_node_data_source(
    builder: MultiFileDataSource,
) -> VortexResult<ScanNodeDataSource> {
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
            Arc::new(ScanNodeFileFactory {
                fs: Arc::clone(fs),
                file: file.clone(),
                session: builder.session.clone(),
                open_options_fn: Arc::clone(&builder.open_options_fn),
                metrics_registry: builder.metrics_registry.clone(),
            }) as Arc<dyn VortexFileFactory>
        })
        .collect();

    Ok(ScanNodeDataSource::new_with_first(
        first_file,
        factories,
        &builder.session,
    ))
}

#[async_trait]
trait VortexFileFactory: 'static + Send + Sync {
    async fn open(&self) -> VortexResult<Option<VortexFile>>;
}

struct ScanNodeFileFactory {
    fs: FileSystemRef,
    file: FileListing,
    session: VortexSession,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

#[async_trait]
impl VortexFileFactory for ScanNodeFileFactory {
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

enum ScanNodeChild {
    Opened(VortexFile),
    Deferred(Arc<dyn VortexFileFactory>),
}

/// Multi-file data source backed by scan2 ScanNode plans.
pub struct ScanNodeDataSource {
    dtype: DType,
    session: VortexSession,
    children: Vec<ScanNodeChild>,
    concurrency: usize,
}

impl ScanNodeDataSource {
    fn new_with_first(
        first: VortexFile,
        remaining: Vec<Arc<dyn VortexFileFactory>>,
        session: &VortexSession,
    ) -> Self {
        let dtype = first.dtype().clone();
        let concurrency = get_available_parallelism().unwrap_or(DEFAULT_CONCURRENCY);

        let mut children = Vec::with_capacity(1 + remaining.len());
        children.push(ScanNodeChild::Opened(first));
        children.extend(remaining.into_iter().map(ScanNodeChild::Deferred));

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
                ScanNodeChild::Opened(file) => {
                    let file = file.clone();
                    async move { Ok(Some((idx, file))) }.boxed()
                }
                ScanNodeChild::Deferred(factory) => {
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
impl DataSource for ScanNodeDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        let mut sum: u64 = 0;
        let mut opened_count: u64 = 0;
        let mut deferred_count: u64 = 0;

        for child in &self.children {
            match child {
                ScanNodeChild::Opened(file) => {
                    opened_count += 1;
                    sum = sum.saturating_add(file.row_count());
                }
                ScanNodeChild::Deferred(_) => {
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
        vortex_bail!("ScanNodeDataSource partitions are not yet serializable")
    }

    async fn plan_morsel_partitions(
        &self,
        scan_request: ScanRequest,
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
        for (partition_idx, file) in self.open_files(false).await? {
            let Some(request) = file_scan_request(partition_idx, &file, scan_request.clone())?
            else {
                continue;
            };
            let prepared = Arc::new(PreparedScanNodeFile::try_new(file, request)?);
            let ranges = prepared.splits()?;
            if ranges.is_empty() {
                continue;
            }
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
                partitions[partition].push(PlannedScanNodeMorsel {
                    prepared: Arc::clone(&prepared),
                    range,
                });
                morsel_idx = morsel_idx.saturating_add(1);
            }
        }

        let morsel_concurrency = get_available_parallelism().unwrap_or(1).saturating_mul(4);

        Ok(Some(Arc::new(PlannedScanNodeScan {
            dtype,
            partitions,
            scheduler,
            ticket,
            morsel_concurrency,
        })))
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
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
                ScanNodeChild::Opened(file) => ready.push_back(file.clone()),
                ScanNodeChild::Deferred(factory) => deferred.push_back(Arc::clone(factory)),
            }
        }

        let dtype = scan_request.projection.return_dtype(&self.dtype)?;

        Ok(Box::new(ScanNodeDataSourceScan {
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
        let ScanNodeChild::Opened(file) = &self.children[0] else {
            return Ok(absent_statistics(funcs));
        };
        scan_node_file_statistics(file.clone(), expr, funcs).await
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

struct ScanNodeDataSourceScan {
    dtype: DType,
    request: ScanRequest,
    ready: VecDeque<VortexFile>,
    deferred: VecDeque<Arc<dyn VortexFileFactory>>,
    handle: Handle,
    concurrency: usize,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
}

impl DataSourceScan for ScanNodeDataSourceScan {
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
    request: ScanRequest,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
) -> VortexResult<Option<PartitionRef>> {
    let Some(request) = file_scan_request(partition_idx, &file, request)? else {
        return Ok(None);
    };

    Ok(Some(Box::new(ScanNodePartition {
        file,
        request,
        index: partition_idx,
        scheduler,
        ticket,
    })))
}

pub(crate) fn scan_node_file_stream(
    file: VortexFile,
    request: ScanRequest,
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

pub(crate) async fn scan_node_file_statistics(
    file: VortexFile,
    expr: &Expression,
    funcs: &[AggregateFnRef],
) -> VortexResult<Vec<Precision<Scalar>>> {
    let mut stats = scan_node_file_statistics_many(file, std::slice::from_ref(expr), funcs).await?;
    Ok(stats.pop().unwrap_or_else(|| absent_statistics(funcs)))
}

pub(crate) async fn scan_node_file_statistics_many(
    file: VortexFile,
    exprs: &[Expression],
    funcs: &[AggregateFnRef],
) -> VortexResult<Vec<Vec<Precision<Scalar>>>> {
    let session = file.session().clone();
    let root = expand_file_root(&file, &session)?;
    let reader = FileReader::new(file.segment_source(), session);
    let mut result = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let pushed = push_expr(&root, expr, file.dtype(), reader.session())?;
        let Some(plan) = pushed.plan_stats(funcs, &mut PlanCtx::new(reader.session().clone()))?
        else {
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

pub(crate) fn scan_node_file_splits(file: &VortexFile) -> VortexResult<Vec<Range<u64>>> {
    let session = file.session().clone();
    let root = expand_file_root(file, &session)?;
    let row_count = file.row_count();
    let mut points = vec![0, row_count];
    if let Some(hints) = root.split_hints() {
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

fn expand_file_root(file: &VortexFile, session: &VortexSession) -> VortexResult<ScanNodeRef> {
    let mut node_request = NodeRequest::empty();
    let root = ExpandCtx::new(session.clone()).expand(file.footer().layout(), &mut node_request)?;
    Ok(match file.footer().statistics().cloned() {
        Some(stats) => FileStatsScanNode::try_new(
            Arc::clone(&root),
            Arc::new(stats),
            file.dtype(),
            file.row_count(),
        )
        .map(|node| Arc::new(node) as ScanNodeRef)
        .unwrap_or(root),
        None => root,
    })
}

fn file_scan_request(
    partition_idx: usize,
    file: &VortexFile,
    request: ScanRequest,
) -> VortexResult<Option<ScanRequest>> {
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

    Ok(Some(ScanRequest {
        row_range: Some(row_range),
        ..request
    }))
}

struct ScanNodePartition {
    file: VortexFile,
    request: ScanRequest,
    index: usize,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
}

impl Partition for ScanNodePartition {
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
        let ScanNodePartition {
            file,
            request,
            index: _,
            scheduler,
            ticket,
        } = *self;

        let prepared = Arc::new(PreparedScanNodeFile::try_new(file, request)?);
        let dtype = prepared.dtype.clone();
        let ranges = prepared.splits()?;
        let session = prepared.session.clone();
        let ordered = prepared.ordered;
        let concurrency = if ordered || prepared.limit_remaining.is_some() {
            1
        } else {
            get_available_parallelism().unwrap_or(1) * 4
        };

        let tasks = ranges
            .into_iter()
            .map(|range| {
                let prepared = Arc::clone(&prepared);
                let scheduler = Arc::clone(&scheduler);
                let ticket = ticket.clone();
                async move {
                    let _permit = scheduler.acquire(&ticket, WorkRequest::morsel()).await?;
                    prepared.read_range(range).await
                }
                .boxed()
            })
            .collect::<Vec<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>>();

        let handle = session.handle();
        let stream = stream::iter(tasks).map(move |task| handle.spawn(task));
        let stream = if ordered {
            stream.buffered(concurrency).boxed()
        } else {
            stream.buffer_unordered(concurrency).boxed()
        };
        let stream = stream.filter_map(|result| async move { result.transpose() });

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

struct PlannedScanNodeScan {
    dtype: DType,
    partitions: Vec<Vec<PlannedScanNodeMorsel>>,
    scheduler: Arc<ScanScheduler>,
    ticket: ScanTicket,
    morsel_concurrency: usize,
}

#[derive(Clone)]
struct PlannedScanNodeMorsel {
    prepared: Arc<PreparedScanNodeFile>,
    range: Range<u64>,
}

impl PlannedMorselScan for PlannedScanNodeScan {
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

        Ok(Box::new(PlannedScanNodePartition {
            planned: self,
            index: partition,
        }))
    }
}

struct PlannedScanNodePartition {
    planned: Arc<PlannedScanNodeScan>,
    index: usize,
}

impl Partition for PlannedScanNodePartition {
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
        let PlannedScanNodePartition { planned, index } = *self;
        let morsels = planned.partitions[index].clone();
        let dtype = planned.dtype.clone();
        let scheduler = Arc::clone(&planned.scheduler);
        let ticket = planned.ticket.clone();
        let concurrency = planned.morsel_concurrency;

        let stream = stream::iter(morsels).map(move |morsel| {
            let handle = morsel.prepared.session.handle();
            let scheduler = Arc::clone(&scheduler);
            let ticket = ticket.clone();
            handle.spawn(
                async move {
                    let _permit = scheduler.acquire(&ticket, WorkRequest::morsel()).await?;
                    morsel.prepared.read_range(morsel.range).await
                }
                .instrument(tracing::trace_span!("scan2_morsel")),
            )
        });

        let stream = stream
            .buffer_unordered(concurrency)
            .filter_map(|result| async move { result.transpose() });

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, stream,
        )))
    }
}

struct PreparedScanNodeFile {
    session: VortexSession,
    reader: FileReader,
    dtype: DType,
    row_range: Range<u64>,
    selection: Selection,
    ordered: bool,
    limit_remaining: Option<AtomicU64>,
    root: ScanNodeRef,
    projection: ReadPlanRef,
    projection_state: ScanStateRef,
    predicates: Vec<PredicatePlan>,
}

struct PredicatePlan {
    id: PredicateId,
    expr: Expression,
    read: ReadPlanRef,
    read_state: ScanStateRef,
    evidence: Vec<(EvidencePlanRef, ScanStateRef)>,
}

impl PreparedScanNodeFile {
    fn try_new(file: VortexFile, request: ScanRequest) -> VortexResult<Self> {
        let session = file.session().clone();
        let dtype = request.projection.return_dtype(file.dtype())?;
        let projection = request.projection.optimize_recursive(file.dtype())?;
        let filter = request
            .filter
            .map(|filter| filter.optimize_recursive(file.dtype()))
            .transpose()?;

        let root = expand_file_root(&file, &session)?;
        let reader = FileReader::new(file.segment_source(), session.clone());

        let mut node_cache = ScanStateCache::default();
        let mut state_ctx = StateCtx::new(&session, &mut node_cache);

        let projection_plan = plan_read(&root, &projection, file.dtype(), &session)?;
        let projection_state = projection_plan.init_state(&mut state_ctx)?;

        let mut evidence_state_cache: HashMap<EvidenceStateKey, ScanStateRef> = HashMap::default();
        let predicates = filter
            .as_ref()
            .map(conjuncts)
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(idx, expr)| {
                let id = PredicateId::new(
                    u32::try_from(idx).map_err(|_| vortex_err!("too many predicates"))?,
                );
                let pushed = push_expr(&root, &expr, file.dtype(), &session)?;
                let read = Arc::clone(&pushed)
                    .plan_read(&mut PlanCtx::new(session.clone()))?
                    .ok_or_else(|| vortex_err!("scan2 could not plan predicate read {expr}"))?;
                let read_state = read.init_state(&mut state_ctx)?;
                let evidence = pushed
                    .plan_evidence(&mut PlanCtx::new(session.clone()))?
                    .into_iter()
                    .map(|plan| {
                        let state = if let Some(key) = plan.state_cache_key() {
                            if let Some(state) = evidence_state_cache.get(&key) {
                                Arc::clone(state)
                            } else {
                                let state = plan.init_state(&session)?;
                                evidence_state_cache.insert(key, Arc::clone(&state));
                                state
                            }
                        } else {
                            plan.init_state(&session)?
                        };
                        Ok((plan, state))
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                Ok(PredicatePlan {
                    id,
                    expr,
                    read,
                    read_state,
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
            root,
            projection: projection_plan,
            projection_state,
            predicates,
        })
    }

    fn splits(&self) -> VortexResult<Vec<Range<u64>>> {
        let mut points = vec![self.row_range.start];
        if let Some(hints) = self.root.split_hints() {
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

    async fn read_range(&self, range: Range<u64>) -> VortexResult<Option<ArrayRef>> {
        let len = range_len(&range)?;
        let selected = self.selection.row_mask(&range).mask().clone();
        if selected.all_false() {
            return Ok(None);
        }

        let mut ctx = self.session.create_execution_ctx();
        let Some(selected) = self
            .morsel_selection(range.clone(), selected, &mut ctx)
            .await?
        else {
            return Ok(None);
        };

        if selected.all_false() {
            return Ok(None);
        }

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

        let array = self
            .projection
            .read_scoped(
                range,
                RowScope::selected(&selected),
                &self.reader,
                self.projection_state.as_ref(),
                &mut ctx,
            )
            .await?;
        Ok(Some(array))
    }

    async fn morsel_selection(
        &self,
        range: Range<u64>,
        mut selected: Mask,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<Option<Mask>> {
        let len = range_len(&range)?;
        let full_domain = Mask::new_true(len);
        let mut evidence = Vec::with_capacity(self.predicates.len());

        for predicate in &self.predicates {
            let acc = self.gather_evidence(predicate, &range).await?;
            selected = &selected & acc.maybe();
            if selected.all_false() {
                return Ok(None);
            }
            evidence.push((predicate, acc));
        }

        for (predicate, acc) in evidence {
            let need = &selected & &acc.unproven();
            if need.all_false() {
                continue;
            }
            let rows = RowScope::try_new(&full_domain, &need)?;
            let result = predicate
                .read
                .read_scoped(
                    range.clone(),
                    rows,
                    &self.reader,
                    predicate.read_state.as_ref(),
                    ctx,
                )
                .await?
                .execute::<Mask>(ctx)?;
            if result.len() != len {
                vortex_bail!(
                    "residual result length {} does not match morsel length {len}",
                    result.len()
                );
            }
            let pass = &result & &need;
            selected = &selected.bitand_not(&need) | &pass;
            if selected.all_false() {
                return Ok(None);
            }
        }
        Ok(Some(selected))
    }

    async fn gather_evidence(
        &self,
        predicate: &PredicatePlan,
        range: &Range<u64>,
    ) -> VortexResult<PredicateEvidence> {
        let mut acc =
            PredicateEvidence::new(predicate.id, PredicateVersion::STATIC, range.clone())?;
        for (plan, state) in &predicate.evidence {
            let req = EvidenceRequest {
                id: predicate.id,
                version: PredicateVersion::STATIC,
                predicate: &predicate.expr,
                range: range.clone(),
                mode: EvidenceMode::Normal,
            };
            for fragment in plan.evidence(&req, &self.reader, state.as_ref()).await? {
                acc.absorb(fragment)?;
            }
            if acc.all_false() {
                break;
            }
        }
        Ok(acc)
    }
}

fn push_expr(
    root: &ScanNodeRef,
    expr: &Expression,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<ScanNodeRef> {
    validate_temporal_comparisons(expr, dtype)?;
    Arc::clone(root)
        .try_push_expr(expr, &mut PushCtx::new(session.clone()))?
        .ok_or_else(|| vortex_err!("scan2 could not push expression {expr}"))
}

fn plan_read(
    root: &ScanNodeRef,
    expr: &Expression,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<ReadPlanRef> {
    push_expr(root, expr, dtype, session)?
        .plan_read(&mut PlanCtx::new(session.clone()))?
        .ok_or_else(|| vortex_err!("scan2 could not plan read for expression {expr}"))
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

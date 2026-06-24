// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File adapters for ScanPlan-backed scans.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::root::Root;
use vortex_array::stats::StatsSet;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::filesystem::FileListing;
use vortex_io::filesystem::FileSystemRef;
use vortex_layout::layout_v2::LayoutScanPlanCtx;
use vortex_layout::scan::v2::with_row_idx;
use vortex_metrics::MetricsRegistry;
use vortex_scan::ScanRequest as DataSourceScanRequest;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedAggregateRef;
use vortex_scan::plan::PreparedEvidenceRef;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PreparedStats;
use vortex_scan::plan::PreparedStatsRef;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::ReadContext;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanDataSource;
use vortex_scan::plan::ScanPlanFactory;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::ScanState;
use vortex_scan::plan::ScanStateRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::downcast_state;
use vortex_scan::plan::request::ScanRequest;
use vortex_scan::plan::scan_plan_projected_splits;
use vortex_scan::plan::scan_plan_split_ranges;
use vortex_scan::plan::scan_plan_statistics;
use vortex_scan::plan::scan_plan_statistics_many;
use vortex_scan::plan::scan_plan_stream;
use vortex_session::VortexSession;

use super::MultiFileDataSource;
use super::create_local_filesystem;
use super::open_file;
use crate::FileStatistics;
use crate::VortexFile;
use crate::VortexOpenOptions;

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
    fn dtype(&self) -> &DType {
        self.data.dtype()
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

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
    fn dtype(&self) -> &DType {
        &self.field_dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

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

fn absent_statistics(funcs: &[AggregateFnRef]) -> Vec<Precision<Scalar>> {
    funcs.iter().map(|_| Precision::Absent).collect()
}

/// Build a scan2 [`DataSource`](vortex_scan::DataSource) from a multi-file builder.
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
    let first_root = first_file.scan_plan_root()?;

    let factories: Vec<Arc<dyn ScanPlanFactory>> = all_files[1..]
        .iter()
        .map(|(file, fs)| {
            Arc::new(ScanPlanFileFactory {
                fs: Arc::clone(fs),
                file: file.clone(),
                session: builder.session.clone(),
                open_options_fn: Arc::clone(&builder.open_options_fn),
                metrics_registry: builder.metrics_registry.clone(),
            }) as Arc<dyn ScanPlanFactory>
        })
        .collect();

    Ok(ScanPlanDataSource::new_with_first(
        first_root,
        factories,
        &builder.session,
    ))
}

struct ScanPlanFileFactory {
    fs: FileSystemRef,
    file: FileListing,
    session: VortexSession,
    open_options_fn: Arc<dyn Fn(VortexOpenOptions) -> VortexOpenOptions + Send + Sync>,
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
}

#[async_trait]
impl ScanPlanFactory for ScanPlanFileFactory {
    async fn open(&self) -> VortexResult<Option<ScanPlanRef>> {
        let file = open_file(
            &self.fs,
            &self.file,
            &self.session,
            self.metrics_registry.as_ref(),
            self.open_options_fn.as_ref(),
        )
        .await?;
        Ok(Some(file.scan_plan_root()?))
    }
}

pub(crate) fn scan_plan_file_stream(
    file: VortexFile,
    request: DataSourceScanRequest,
) -> VortexResult<SendableArrayStream> {
    let root = file.scan_plan_root()?;
    scan_plan_stream(root, file.session().clone(), request)
}

pub(crate) async fn scan_plan_file_statistics(
    file: VortexFile,
    expr: &Expression,
    funcs: &[AggregateFnRef],
) -> VortexResult<Vec<Precision<Scalar>>> {
    let root = file.scan_plan_root()?;
    scan_plan_statistics(root, file.session().clone(), expr, funcs).await
}

pub(crate) async fn scan_plan_file_statistics_many(
    file: VortexFile,
    exprs: &[Expression],
    funcs: &[AggregateFnRef],
) -> VortexResult<Vec<Vec<Precision<Scalar>>>> {
    let root = file.scan_plan_root()?;
    scan_plan_statistics_many(root, file.session().clone(), exprs, funcs).await
}

pub(crate) fn scan_plan_file_splits(file: &VortexFile) -> VortexResult<Vec<Range<u64>>> {
    let root = file.scan_plan_root()?;
    Ok(scan_plan_split_ranges(&root))
}

pub(crate) async fn scan_plan_file_plan_splits(
    file: VortexFile,
    projection: &Expression,
) -> VortexResult<Vec<Range<u64>>> {
    let root = file.scan_plan_root()?;
    scan_plan_projected_splits(root, file.session().clone(), projection).await
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
    let root = with_row_idx(root, 0);
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

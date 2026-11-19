// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PlanDataSource`]: a [`DataSource`] over one or more [`ScanInput`]s, scanned by the
//! planner-driven executor.

use std::any::Any;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use super::ScanInput;
use super::plan_pool;
use crate::DataSource;
use crate::DataSourceScan;
use crate::DataSourceScanRef;
use crate::Partition;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanRequest;
use crate::execution::ROW_BLOCK_SIZE;
use crate::execution::driver::ScanDriver;
use crate::pool::RowLimit;
use crate::pool::ScanPoolRef;
use crate::pool::group_contiguous;
use crate::pool::trim_to_limit;
use crate::prepare::ScanBuilder;
use crate::selection::Selection;

/// Whether a file's own statistics prove a filter, bound to the file's dtype, false for every
/// row of the file.
pub type FilePruner = Arc<dyn Fn(&BoundExpression) -> VortexResult<bool> + Send + Sync>;

/// One open file of a [`PlanDataSource`]: its [`ScanInput`] and what the source knows of it
/// without reading payloads.
#[derive(Clone)]
pub struct PlanFile {
    input: ScanInput,
    row_count: Option<u64>,
    splits: Option<Arc<[u64]>>,
    pruner: Option<FilePruner>,
}

impl PlanFile {
    /// A file scanning `input`. Without a cache, the input gets a fresh one that every scan of
    /// the file shares.
    pub fn new(mut input: ScanInput) -> Self {
        input.cache.get_or_insert_default();
        Self {
            input,
            row_count: None,
            splits: None,
            pruner: None,
        }
    }

    /// Sets the file's exact row count. Without it, the source lowers the file's root plan to
    /// learn the count when a scan needs it.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        self.row_count = Some(row_count);
        self
    }

    /// Sets the row boundaries that an opened scan cuts the file's work at, such as the file's
    /// layout splits, ascending and covering the file's rows. Without them, the work is cut on
    /// the scan's chunk grid.
    pub fn with_splits(mut self, splits: Arc<[u64]>) -> Self {
        self.splits = Some(splits);
        self
    }

    /// Sets how the file's statistics rule out a scan's filter, so a scan they rule out reads
    /// nothing of the file.
    pub fn with_pruner(mut self, pruner: FilePruner) -> Self {
        self.pruner = Some(pruner);
        self
    }

    /// The input this file scans.
    pub fn input(&self) -> &ScanInput {
        &self.input
    }

    /// The file's rows, from the row count when set, else from its root plan.
    pub(super) fn rows(&self) -> VortexResult<u64> {
        match self.row_count {
            Some(rows) => Ok(rows),
            None => Ok(self.input.root.get()?.row_count()),
        }
    }

    /// The bytes of all of the file's segments.
    fn byte_size(&self) -> u64 {
        self.input
            .locations
            .iter()
            .map(|location| u64::from(location.length))
            .sum()
    }

    pub(super) fn splits(&self) -> Option<&[u64]> {
        self.splits.as_deref()
    }

    fn is_pruned(&self, filter: Option<&BoundExpression>) -> VortexResult<bool> {
        match (&self.pruner, filter) {
            (Some(pruner), Some(filter)) => pruner(filter),
            _ => Ok(false),
        }
    }
}

/// Opens a file of a [`PlanDataSource`] when a scan reaches it.
#[async_trait]
pub trait DeferredPlanFile: 'static + Send + Sync {
    /// Opens the file, or returns `None` when the scan should skip it.
    async fn open(&self) -> VortexResult<Option<PlanFile>>;
}

/// A file of a [`PlanDataSource`].
#[derive(Clone)]
pub enum PlanSourceFile {
    /// A file that is already open.
    Opened(PlanFile),
    /// A file that a scan opens when it reaches it, so a source over many files reads only the
    /// footers its scans need.
    Deferred {
        /// Opens the file.
        file: Arc<dyn DeferredPlanFile>,
        /// The file's size in bytes, if known.
        byte_size: Option<u64>,
    },
}

/// A [`DataSource`] over the files of [`ScanInput`]s, scanned by the planner-driven executor.
///
/// This is the standard way to expose sources that can name their segments, whatever their
/// format: Vortex files, Parquet files whose column chunks are placed segments, or any store that
/// can lower its data to plans. Every scan of a file shares the file's [`ScanCache`], so scans
/// that overlap in time plan the file once.
///
/// A scan's request applies to each file: its row range and row selection are in the file's own
/// rows. A scan reads its files in order.
///
/// # Splitting and opening
///
/// [`split`](DataSourceScan::split) cuts a scan on the 8,192-row block grid of its files, which
/// every chunk boundary of the executor lies on, into contiguous pieces of about equal row
/// counts. It reads nothing; a file that is not open yet is one indivisible unit.
///
/// [`open`](DataSourceScan::open) prepares each open file of the scan once and hands its parts,
/// the chunks of the scan, or the file's [splits](PlanFile::with_splits) when it has them, to
/// workers, which run several at a time on their own thread. A file that is not open yet is a
/// unit of work too: the worker that takes it opens and prepares it, and puts its parts back for
/// any worker to take. See the pool's documentation in `plan_pool.rs`.
///
/// # Partitions
///
/// A scan has one partition per file, whose [`index`](Partition::index) is the file's ordinal,
/// so a request's `partition_range` and `partition_selection` select files. A partition runs its
/// file's scan with a [`ScanDriver`] on the session's runtime.
///
/// [`ScanCache`]: crate::prepare::ScanCache
#[derive(Clone)]
pub struct PlanDataSource {
    session: VortexSession,
    dtype: DType,
    files: Arc<[PlanSourceFile]>,
    row_count: Precision<u64>,
    byte_size: Precision<u64>,
}

impl PlanDataSource {
    /// A source scanning one open `input` with `session`.
    pub fn new(session: VortexSession, input: ScanInput) -> Self {
        Self::from_file(session, PlanFile::new(input))
    }

    /// A source scanning one open `file` with `session`.
    pub fn from_file(session: VortexSession, file: PlanFile) -> Self {
        let dtype = file.input.dtype.clone();
        let row_count = file.row_count.map_or(Precision::Absent, Precision::exact);
        Self {
            session,
            dtype,
            files: Arc::new([PlanSourceFile::Opened(file)]),
            row_count,
            byte_size: Precision::Absent,
        }
    }

    /// A source scanning `files` in order with `session`. Every file must have `dtype`; a scan
    /// that opens a file of another dtype fails.
    pub fn from_files(session: VortexSession, dtype: DType, files: Vec<PlanSourceFile>) -> Self {
        let mut rows = Some(0u64);
        let mut bytes = Some(0u64);
        let mut known_rows = 0;
        let mut known_files = 0;
        for file in &files {
            match file {
                PlanSourceFile::Opened(file) => {
                    match file.row_count {
                        Some(count) => {
                            known_rows += count;
                            known_files += 1;
                        }
                        None => rows = None,
                    }
                    bytes = bytes.map(|bytes| bytes + file.byte_size());
                }
                PlanSourceFile::Deferred { byte_size, .. } => {
                    rows = None;
                    bytes = bytes.zip(*byte_size).map(|(bytes, size)| bytes + size);
                }
            }
        }
        // Files not open yet are estimated at the average row count of the open ones.
        let row_count = match rows {
            Some(rows) => Precision::exact(rows + known_rows),
            None if known_files > 0 => {
                Precision::inexact(known_rows / known_files * files.len() as u64)
            }
            None => Precision::Absent,
        };
        Self {
            session,
            dtype,
            files: files.into(),
            row_count,
            byte_size: bytes.map_or(Precision::Absent, Precision::inexact),
        }
    }

    /// Sets the exact row count of a one-file source, which scans and partitions report.
    pub fn with_row_count(mut self, row_count: u64) -> Self {
        if let [PlanSourceFile::Opened(file)] = &*self.files {
            self.files = Arc::new([PlanSourceFile::Opened(
                file.clone().with_row_count(row_count),
            )]);
        }
        self.row_count = Precision::exact(row_count);
        self
    }

    /// Sets an estimate of the source's byte size.
    pub fn with_byte_size(mut self, byte_size: Precision<u64>) -> Self {
        self.byte_size = byte_size;
        self
    }

    /// The files this source scans, in order.
    pub fn files(&self) -> &[PlanSourceFile] {
        &self.files
    }

    /// The session scans of this source run with.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

#[async_trait]
impl DataSource for PlanDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> Precision<u64> {
        self.row_count
    }

    fn byte_size(&self) -> Precision<u64> {
        self.byte_size
    }

    async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let projection = request.projection.bind(&self.dtype)?.optimize_recursive()?;
        let filter = request
            .filter
            .map(|filter| filter.bind(&self.dtype)?.optimize_recursive())
            .transpose()?;
        let mut units = Vec::new();
        let scans_rows =
            request.limit != Some(0) && !request.row_range.as_ref().is_some_and(Range::is_empty);
        for (ordinal, file) in self.files.iter().enumerate().filter(|_| scans_rows) {
            let index = ordinal as u64;
            let selected = request
                .partition_range
                .as_ref()
                .is_none_or(|range| range.contains(&index))
                && request
                    .partition_selection
                    .row_count_in_range(&(index..index + 1))
                    == 1;
            if !selected {
                continue;
            }
            let rows = match file {
                PlanSourceFile::Opened(file) => {
                    if file.is_pruned(filter.as_ref())? {
                        continue;
                    }
                    file.row_count
                        .map(|count| clamp_rows(request.row_range.clone(), count))
                }
                PlanSourceFile::Deferred { .. } => None,
            };
            if rows.as_ref().is_some_and(Range::is_empty) {
                continue;
            }
            units.push(Unit {
                file: ordinal,
                rows,
            });
        }
        let state = ScanState {
            session: self.session.clone(),
            source_dtype: self.dtype.clone(),
            files: Arc::clone(&self.files),
            dtype: projection.dtype().clone(),
            projection,
            filter,
            row_range: request.row_range,
            selection: request.selection,
            ordered: request.ordered,
            limit: request.limit,
            split_concurrency: request.split_concurrency,
        };
        Ok(Box::new(PlanScan {
            state: Arc::new(state),
            units,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

/// The request's `rows`, or all of them, clamped to a file of `count` rows.
fn clamp_rows(rows: Option<Range<u64>>, count: u64) -> Range<u64> {
    let rows = rows.unwrap_or(0..count);
    let end = rows.end.min(count);
    rows.start.min(end)..end
}

/// What every piece, partition and pool of one scan of a [`PlanDataSource`] shares.
pub(super) struct ScanState {
    pub(super) session: VortexSession,
    source_dtype: DType,
    files: Arc<[PlanSourceFile]>,
    /// The dtype of the scan's output.
    pub(super) dtype: DType,
    projection: BoundExpression,
    filter: Option<BoundExpression>,
    row_range: Option<Range<u64>>,
    pub(super) selection: Selection,
    pub(super) ordered: bool,
    pub(super) limit: Option<u64>,
    pub(super) split_concurrency: Option<NonZeroUsize>,
}

impl ScanState {
    /// The file at `ordinal` if it is open.
    pub(super) fn opened(&self, ordinal: usize) -> Option<&PlanFile> {
        match &self.files[ordinal] {
            PlanSourceFile::Opened(file) => Some(file),
            PlanSourceFile::Deferred { .. } => None,
        }
    }

    /// Opens the file at `ordinal`, or returns `None` when the file is skipped or its statistics
    /// rule out the filter. An open file was checked against the filter when the scan began.
    pub(super) async fn open_file(&self, ordinal: usize) -> VortexResult<Option<PlanFile>> {
        let file = match &self.files[ordinal] {
            PlanSourceFile::Opened(file) => return Ok(Some(file.clone())),
            PlanSourceFile::Deferred { file, .. } => file.open().await?,
        };
        let Some(file) = file else {
            return Ok(None);
        };
        if file.input.dtype != self.source_dtype {
            vortex_bail!(
                "file {ordinal} of the source has dtype {}, but the source has dtype {}",
                file.input.dtype,
                self.source_dtype
            );
        }
        if file.is_pruned(self.filter.as_ref())? {
            return Ok(None);
        }
        Ok(Some(file))
    }

    /// The rows of `file` this scan reads: the request's row range, clamped to the file.
    pub(super) fn file_rows(&self, file: &PlanFile) -> VortexResult<Range<u64>> {
        Ok(clamp_rows(self.row_range.clone(), file.rows()?))
    }

    /// A builder of this scan's `rows` of `file`.
    pub(super) fn builder(&self, file: &PlanFile, rows: Range<u64>) -> ScanBuilder {
        ScanBuilder::new(self.session.clone(), file.input.clone())
            .with_projection(self.projection.clone())
            .with_some_filter(self.filter.clone())
            .with_selection(self.selection.clone())
            .with_row_range(rows)
    }
}

/// Rows of one file that a scan reads.
#[derive(Clone, Debug)]
pub(super) struct Unit {
    /// The file's ordinal in the source.
    pub(super) file: usize,
    /// The file's rows, or `None` for the request's rows of a file whose row count is not known
    /// until it is open.
    pub(super) rows: Option<Range<u64>>,
}

/// A scan of a [`PlanDataSource`], or a piece split from one.
struct PlanScan {
    state: Arc<ScanState>,
    /// The scan's units, in file and row order.
    units: Vec<Unit>,
}

impl PlanScan {
    /// The rows of `unit` the scan selects before filtering, or `None` when not known yet.
    fn unit_rows(&self, unit: &Unit) -> Option<u64> {
        let rows = unit.rows.as_ref()?;
        Some(self.state.selection.row_count_in_range(rows))
    }

    /// The bytes of `unit`, in proportion to its share of its file's rows.
    fn unit_bytes(&self, unit: &Unit) -> Option<u64> {
        match &self.state.files[unit.file] {
            PlanSourceFile::Opened(file) => {
                let bytes = file.byte_size();
                match (&unit.rows, file.row_count) {
                    (Some(rows), Some(count)) if count > 0 => Some(
                        u64::try_from(
                            u128::from(bytes) * u128::from(rows.end - rows.start)
                                / u128::from(count),
                        )
                        .unwrap_or(u64::MAX),
                    ),
                    _ => Some(bytes),
                }
            }
            PlanSourceFile::Deferred { byte_size, .. } => *byte_size,
        }
    }
}

impl DataSourceScan for PlanScan {
    fn dtype(&self) -> &DType {
        &self.state.dtype
    }

    fn partition_count(&self) -> Precision<usize> {
        Precision::exact(self.units.len())
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        // An unordered scan's partitions share one budget; an ordered scan's each apply the limit
        // locally, and the caller trims the concatenated result.
        let shared = (!self.state.ordered)
            .then(|| self.state.limit.map(RowLimit::new))
            .flatten();
        let state = self.state;
        stream::iter(self.units.into_iter().map(move |unit| {
            let limit = shared.clone().or_else(|| state.limit.map(RowLimit::new));
            Ok(Box::new(PlanPartition {
                state: Arc::clone(&state),
                unit,
                limit,
            }) as PartitionRef)
        }))
        .boxed()
    }

    fn row_count(&self) -> Precision<u64> {
        let Some(rows) = self
            .units
            .iter()
            .map(|unit| self.unit_rows(unit))
            .sum::<Option<u64>>()
        else {
            return Precision::Absent;
        };
        let rows = self.state.limit.map_or(rows, |limit| rows.min(limit));
        if self.state.filter.is_some() {
            Precision::inexact(rows)
        } else {
            Precision::exact(rows)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        self.units
            .iter()
            .map(|unit| self.unit_bytes(unit))
            .sum::<Option<u64>>()
            .map_or(Precision::Absent, Precision::inexact)
    }

    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
        // A limit holds across the whole scan, which pieces read independently cannot share.
        if self.state.limit.is_some() || n.get() == 1 {
            return Ok(vec![self]);
        }
        let atoms = split_atoms(&self.units, n);
        if atoms.len() <= 1 {
            return Ok(vec![self]);
        }
        let known: Vec<u64> = atoms
            .iter()
            .filter_map(|unit| unit.rows.as_ref().map(|rows| rows.end - rows.start))
            .collect();
        // A file that is not open yet weighs as much as the average open unit.
        let unknown = (known.iter().sum::<u64>() / known.len().max(1) as u64).max(1);
        let weight = |unit: &Unit| {
            unit.rows
                .as_ref()
                .map_or(unknown, |rows| rows.end - rows.start)
        };
        let state = self.state;
        Ok(group_contiguous(atoms, n, weight)
            .into_iter()
            .map(|atoms| {
                Box::new(PlanScan {
                    state: Arc::clone(&state),
                    units: join_atoms(atoms),
                }) as DataSourceScanRef
            })
            .collect())
    }

    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef> {
        plan_pool::open(self.state, self.units)
    }
}

/// The pieces a split may not cut further: the scan's known rows on a grid of whole 8,192-row
/// blocks, coarse enough that each of `n` pieces gets about 64 of them, and each unit whose rows
/// are not known yet whole.
fn split_atoms(units: &[Unit], n: NonZeroUsize) -> Vec<Unit> {
    let block = ROW_BLOCK_SIZE as u64;
    let known: u64 = units
        .iter()
        .filter_map(|unit| unit.rows.as_ref().map(|rows| rows.end - rows.start))
        .sum();
    let per_atom = known.div_ceil(n.get() as u64 * 64).div_ceil(block).max(1) * block;
    let mut atoms = Vec::new();
    for unit in units {
        let Some(rows) = &unit.rows else {
            atoms.push(unit.clone());
            continue;
        };
        let mut start = rows.start;
        while start < rows.end {
            let end = ((start / per_atom + 1) * per_atom).min(rows.end);
            atoms.push(Unit {
                file: unit.file,
                rows: Some(start..end),
            });
            start = end;
        }
    }
    atoms
}

/// Joins adjacent atoms of the same file back into one unit.
fn join_atoms(atoms: Vec<Unit>) -> Vec<Unit> {
    let mut units: Vec<Unit> = Vec::new();
    for atom in atoms {
        if let Some(last) = units.last_mut()
            && last.file == atom.file
            && let (Some(rows), Some(next)) = (&mut last.rows, &atom.rows)
            && rows.end == next.start
        {
            rows.end = next.end;
            continue;
        }
        units.push(atom);
    }
    units
}

/// One file's rows of a scan of a [`PlanDataSource`].
struct PlanPartition {
    state: Arc<ScanState>,
    unit: Unit,
    limit: Option<RowLimit>,
}

impl Partition for PlanPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.unit.file
    }

    fn row_count(&self) -> Precision<u64> {
        let rows = match (&self.unit.rows, self.state.opened(self.unit.file)) {
            (Some(rows), _) => rows.clone(),
            (None, Some(file)) if file.row_count.is_some() => match self.state.file_rows(file) {
                Ok(rows) => rows,
                Err(_) => return Precision::Absent,
            },
            _ => return Precision::Absent,
        };
        let rows = self.state.selection.row_count_in_range(&rows);
        let rows = self.state.limit.map_or(rows, |limit| rows.min(limit));
        if self.state.filter.is_some() {
            Precision::inexact(rows)
        } else {
            Precision::exact(rows)
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let PlanPartition { state, unit, limit } = *self;
        let dtype = state.dtype.clone();
        let arrays = stream::once(async move {
            let Some(file) = state.open_file(unit.file).await? else {
                return Ok(stream::empty().boxed());
            };
            let rows = match unit.rows {
                Some(rows) => rows,
                None => state.file_rows(&file)?,
            };
            if rows.is_empty() {
                return Ok(stream::empty().boxed());
            }
            let driver = ScanDriver::new(state.session.handle(), 1, state.ordered)
                .with_max_in_flight(state.split_concurrency);
            VortexResult::Ok(driver.run(state.builder(&file, rows)))
        })
        .try_flatten()
        .boxed();
        let arrays = match limit {
            Some(limit) => trim_to_limit(arrays, limit),
            None => arrays,
        };
        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, arrays,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use rstest::rstest;

    use super::Unit;
    use super::join_atoms;
    use super::split_atoms;

    fn unit(file: usize, rows: std::ops::Range<u64>) -> Unit {
        Unit {
            file,
            rows: Some(rows),
        }
    }

    #[rstest]
    #[case::one_block(vec![unit(0, 0..100)], 4, 1)]
    #[case::grid_aligned(vec![unit(0, 10..40_000)], 1, 5)]
    fn atoms_lie_on_the_block_grid(
        #[case] units: Vec<Unit>,
        #[case] n: usize,
        #[case] expected: usize,
    ) {
        let n = NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN);
        let atoms = split_atoms(&units, n);
        assert_eq!(atoms.len(), expected);
        for atom in &atoms[1..] {
            let start = atom.rows.as_ref().map_or(0, |rows| rows.start);
            assert_eq!(start % 8192, 0);
        }
        let joined = join_atoms(atoms);
        assert_eq!(
            joined.iter().map(|u| u.rows.clone()).collect::<Vec<_>>(),
            units.iter().map(|u| u.rows.clone()).collect::<Vec<_>>()
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Out-of-file indexes over a table's files.
//!
//! An index file from `vortex-index` covers a set of data files and names each by URI, row count
//! and fingerprint. Here the URI is the file's manifest path and the fingerprint is its footer
//! hash, little-endian, so [`TableIndex::bind`] matches the index's files to the manifest's and
//! notices any file rewritten since the index was built.
//!
//! An index turns a predicate or a ranked query into per-file row selections, which a
//! [`TableScan`](crate::TableScan) takes through
//! [`with_file_selections`](crate::TableScan::with_file_selections). Files the index does not
//! cover, or covers in a stale version, get [`Selection::All`], so a filtered scan stays correct
//! over a table that has grown since the index was built.

use std::sync::Arc;
use std::sync::OnceLock;

use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::v2::gather;
use vortex_index::CoveredFile;
use vortex_index::FileSelection;
use vortex_index::Hit;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::IndexStep;
use vortex_io::VortexReadAt;
use vortex_layout::layouts::indexed::IndexExactness;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_scan::planning::drive::PlanRoot;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::manifest::Manifest;
use crate::manifest::ManifestEntry;
use crate::scan::TableScan;
use crate::storage::TableStorage;

/// The fingerprint an index records for a table file: its footer hash, little-endian.
pub fn fingerprint(entry: &ManifestEntry) -> Vec<u8> {
    entry.footer_hash.to_le_bytes().to_vec()
}

/// Indexes the column of every file in `manifest` into `builder`, binding each file by its
/// manifest path and footer hash.
pub async fn index_table<V: IndexVTable>(
    builder: &mut IndexFileBuilder<V>,
    manifest: &Manifest,
    storage: &dyn TableStorage,
    session: &VortexSession,
) -> VortexResult<()> {
    for entry in manifest.entries() {
        let file = session
            .open_options()
            .with_file_size(entry.byte_size)
            .open(storage.open(&entry.path)?)
            .await?;
        let covered = CoveredFile {
            uri: entry.path.clone(),
            row_count: entry.row_count,
            fingerprint: fingerprint(entry),
        };
        builder.add_covered_file(covered, &file).await?;
    }
    Ok(())
}

/// Per-file row selections an index found for a predicate.
#[derive(Debug)]
pub struct IndexSelections {
    /// Whether the selected rows are exactly the matches, or a superset the scan's filter must
    /// still check. Uncovered files are always [`Selection::All`].
    pub exactness: IndexExactness,
    /// Rows to read, by file ordinal. Files left out hold no match.
    pub selections: HashMap<u64, Selection>,
}

/// One result of a ranked search, in the table's terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TableHit {
    /// The file's ordinal in the manifest.
    pub file: u64,
    /// The row within the file.
    pub row: u64,
    /// Higher is better.
    pub score: f32,
}

/// The rows a ranked search kept, ready to scan.
#[derive(Debug)]
pub struct RankedSelections {
    /// Rows to read, by file ordinal.
    pub selections: HashMap<u64, Selection>,
    /// The hits, ordered by file and row: the order an ordered scan of
    /// [`selections`](Self::selections) yields their rows in, so the `i`-th row such a scan
    /// returns is the `i`-th hit here.
    pub hits: Vec<TableHit>,
}

impl RankedSelections {
    /// The positions of [`hits`](Self::hits), best first.
    pub fn best_first(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.hits.len()).collect();
        order.sort_by(|&a, &b| self.hits[b].score.total_cmp(&self.hits[a].score));
        order
    }
}

/// An index file bound to the manifest of the table it indexes.
pub struct TableIndex<V: IndexVTable> {
    index: IndexFile<V>,
    /// For each file the index covers, its ordinal in the manifest, or `None` when the manifest
    /// no longer holds that exact file.
    ordinals: Vec<Option<u64>>,
    /// Manifest files the index does not cover in their current version.
    uncovered: Vec<u64>,
}

impl<V: IndexVTable> TableIndex<V> {
    /// Matches the index's files to `manifest`'s by path, row count and fingerprint.
    pub fn bind(index: IndexFile<V>, manifest: &Manifest) -> Self {
        let by_path: HashMap<&str, (u64, &ManifestEntry)> = manifest
            .entries()
            .iter()
            .enumerate()
            .map(|(ordinal, entry)| (entry.path.as_str(), (ordinal as u64, entry)))
            .collect();
        let ordinals: Vec<Option<u64>> = index
            .files()
            .iter()
            .map(|covered| {
                let (ordinal, entry) = by_path.get(covered.uri.as_str())?;
                covered
                    .matches(entry.row_count, &fingerprint(entry))
                    .then_some(*ordinal)
            })
            .collect();
        let mut covered = vec![false; manifest.entries().len()];
        for ordinal in ordinals.iter().flatten() {
            covered[usize::try_from(*ordinal).unwrap_or(usize::MAX)] = true;
        }
        let uncovered = (0..manifest.entries().len() as u64)
            .filter(|&ordinal| !covered[usize::try_from(ordinal).unwrap_or(usize::MAX)])
            .collect();
        Self {
            index,
            ordinals,
            uncovered,
        }
    }

    /// The index file.
    pub fn index(&self) -> &IndexFile<V> {
        &self.index
    }

    /// Manifest files the index does not cover in their current version, which a search can't
    /// see and a filter must scan in full.
    pub fn uncovered(&self) -> &[u64] {
        &self.uncovered
    }

    /// The rows of each file that may match `expr`, an expression over the indexed column
    /// (`root()` is the column's value). `None` when the index makes no claim on `expr`.
    pub async fn locate(&self, expr: &Expression) -> VortexResult<Option<IndexSelections>> {
        let Some((exactness, found)) = self.index.locate_files(expr).await? else {
            return Ok(None);
        };
        Ok(Some(IndexSelections {
            exactness,
            selections: self.selections(found),
        }))
    }

    /// The table rows of a ranked search's `hits`, such as those of `search_bm25` or
    /// `search_vectors` on [`IndexFile::rows`]. Hits in files the manifest no longer holds are
    /// dropped.
    pub fn ranked(&self, hits: &[Hit]) -> VortexResult<RankedSelections> {
        let mut hits: Vec<TableHit> = self
            .index
            .hits_by_file(hits)?
            .into_iter()
            .filter_map(|hit| {
                Some(TableHit {
                    file: self.ordinals[hit.file]?,
                    row: hit.row,
                    score: hit.score,
                })
            })
            .collect();
        hits.sort_by_key(|hit| (hit.file, hit.row));
        hits.dedup_by_key(|hit| (hit.file, hit.row));
        let mut rows: HashMap<u64, Vec<u64>> = HashMap::default();
        for hit in &hits {
            rows.entry(hit.file).or_default().push(hit.row);
        }
        let selections = rows
            .into_iter()
            .map(|(file, rows)| {
                Ok((
                    file,
                    Selection::IncludeByIndex(StrictSortedBuffer::try_new(rows.into())?),
                ))
            })
            .collect::<VortexResult<_>>()?;
        Ok(RankedSelections { selections, hits })
    }
}

impl<V: IndexVTable> TableIndex<V> {
    /// The rows of each file `locate` would select: covered files' matches, and uncovered files
    /// in full.
    fn selections(&self, found: Vec<FileSelection>) -> HashMap<u64, Selection> {
        select_files(&self.ordinals, &self.uncovered, found)
    }

    /// Plans `scan` over the rows that may match `expr`, an expression over the indexed column,
    /// as one plan: its first planners read the index through the driver, and then open only the
    /// data files and rows the index points at.
    ///
    /// A superset index leaves rows that don't match, so give `scan` the predicate as a filter
    /// as well. When the index makes no claim on `expr`, the plan scans every file.
    pub fn plan_matching(&self, expr: &Expression, scan: TableScan) -> VortexResult<PlanRoot> {
        let ordinals = self.ordinals.clone();
        let uncovered = self.uncovered.clone();
        let step = self.index.locate_files_steps(expr)?;
        plan_after(step, scan, move |located| {
            Ok(located.map(|(_, found)| select_files(&ordinals, &uncovered, found)))
        })
    }

    /// Plans `scan` over the rows of a ranked search, such as
    /// [`search_vectors_steps`](vortex_index::IndexRows::search_vectors_steps) or
    /// [`search_bm25_steps`](vortex_index::IndexRows::search_bm25_steps) on
    /// [`IndexFile::rows`], as one plan whose first planners run the search.
    ///
    /// The returned cell holds the search's [`RankedSelections`] once the search has run, which
    /// is before the plan yields its first batch: run the plan with ordered output, and its
    /// `i`-th row is the cell's `i`-th hit.
    pub fn plan_ranked(
        self: &Arc<Self>,
        search: IndexStep<Vec<Hit>>,
        scan: TableScan,
    ) -> VortexResult<(PlanRoot, Arc<OnceLock<RankedSelections>>)> {
        let ranked = Arc::new(OnceLock::new());
        let this = Arc::clone(self);
        let cell = Arc::clone(&ranked);
        let root = plan_after(search, scan, move |hits| {
            let selections = this.ranked(&hits)?;
            let rows = selections.selections.clone();
            drop(cell.set(selections));
            Ok(Some(rows))
        })?;
        Ok((root, ranked))
    }
}

/// The selection of each manifest file, from the files an index search found.
fn select_files(
    ordinals: &[Option<u64>],
    uncovered: &[u64],
    found: Vec<FileSelection>,
) -> HashMap<u64, Selection> {
    let mut selections: HashMap<u64, Selection> = found
        .into_iter()
        .filter_map(|found| Some((ordinals[found.file]?, found.selection)))
        .collect();
    for &ordinal in uncovered {
        selections.insert(ordinal, Selection::All);
    }
    selections
}

/// Runs `step` as planners, then plans `scan` over the selections `select` makes of its answer,
/// or over every file when it makes none.
pub fn plan_after<T: Send + 'static>(
    step: IndexStep<T>,
    scan: TableScan,
    select: impl FnOnce(T) -> VortexResult<Option<HashMap<u64, Selection>>> + Send + 'static,
) -> VortexResult<PlanRoot> {
    let scope = WorkScope {
        file_ordinal: 0,
        rows: 0..scan.row_count(),
    };
    step_root(
        scope,
        step,
        Box::new(move |answer| {
            Ok(match select(answer)? {
                Some(selections) => scan.with_file_selections(selections),
                None => scan,
            }
            .plan())
        }),
    )
}

/// What a chain of index steps plans once it has its answer.
type Finish<T> = Box<dyn FnOnce(T) -> VortexResult<PlanRoot> + Send>;

/// The plan of `step`: each read a gathering planner over the index, whose continuation plans
/// the next step, and the answer handed to `finish`.
fn step_root<T: Send + 'static>(
    scope: WorkScope,
    step: IndexStep<T>,
    finish: Finish<T>,
) -> VortexResult<PlanRoot> {
    match step {
        IndexStep::Done(answer) => finish(answer),
        IndexStep::Read(read, then) => {
            let next_scope = scope.clone();
            gather(
                scope,
                *read,
                Box::new(move |batches| {
                    let next = step_root(next_scope, then(batches)?, finish)?;
                    Ok(Some(PlannerOutput::Open(next.scope, next.planner, next.io)))
                }),
            )
        }
    }
}

/// Opens an index file with its metadata, as index files need.
pub async fn open_index_file(
    session: &VortexSession,
    read: Arc<dyn VortexReadAt>,
) -> VortexResult<VortexFile> {
    session.open_options().include_metadata().open(read).await
}

#[cfg(test)]
mod tests;

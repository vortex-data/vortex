// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DuckDB.
//!
//! The driver prepares each file, prunes its rows, and builds a queue of part planners. DuckDB's
//! workers take one part at a time and drive its entire planner tree to completion. A part can
//! contain several chunks: while one chunk waits for IO, the run can advance another chunk.
//! Planners describe work; the shared queue determines which worker executes it.
//!
//! The thread initializing the file performs whole-file pruning before queueing any parts.
//! DuckDB initializes a file on one thread while its others scan other files. A filter with
//! dynamic comparisons, such as a top-N bound, is pruned again inside each part when a worker
//! takes it. The part then emits its chunk planners as ordinary children of the worker's run.
//!
//! Queued and running parts share the file execution's read coordinator. Each queued planner
//! owns its row claim so reads remain available until all possible consumers have finished.
//!
//! Parts use a static row grid, except a single projected string column can preserve whole
//! dictionary regions. This lets workers evaluate dictionary expressions without duplicating
//! them across assignment boundaries. Large regions still split to keep enough workers busy.

use std::collections::VecDeque;
use std::future;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::io::request::IoService;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::layout::scan::v2::ChunkPlanner;
use vortex::layout::scan::v2::PreparedScan;
use vortex::layout::scan::v2::PrunePlanner;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::mask::Mask;
use vortex::scan::planning::driver::FileReads;
use vortex::scan::planning::driver::Progress;
use vortex::scan::planning::driver::RowClaim;
use vortex::scan::planning::driver::Run;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::WorkScope;

use crate::RUNTIME;

/// The parts of a file's rows per thread, so threads that finish early find more work.
const PARTS_PER_THREAD: usize = 4;
/// Avoid tiny worker assignments while allowing filtered chunks to use multiple workers.
const MIN_ROWS_PER_PART: u64 = 65_536;
/// Coalesce tiny dictionaries while leaving short assignments for dictionary expressions.
const MIN_ROWS_PER_DICTIONARY_PART: u64 = 8_192;
/// Bound assignments even when the input already has enough files to fill the worker pool.
const MAX_ROWS_PER_PART: usize = 1_048_576;

fn parts_per_file(threads: usize, files: usize) -> usize {
    let threads = threads.max(1);
    let files = files.max(1);
    threads.saturating_mul(PARTS_PER_THREAD).div_ceil(files)
}

/// One file's V2 scan, shared by the threads scanning it.
pub(crate) struct FileScan {
    scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
    /// Part planners in row order that no worker has taken yet.
    work: Mutex<VecDeque<Part>>,
    /// The number of planned parts, taken or not.
    planned: usize,
    /// The threads DuckDB has given the file so far.
    claims: AtomicUsize,
    /// The rows of the parts taken so far, for DuckDB's progress bar.
    rows_taken: AtomicU64,
}

/// One worker assignment, covering a consecutive range of rows.
struct Part {
    scope: WorkScope,
    planner: Box<dyn Planner>,
}

impl FileScan {
    /// Prepares `builder`, prunes its rows, and queues planners for static row ranges.
    /// With `dynamic`, each queued part prunes again when a worker takes it.
    pub(crate) fn try_new(
        builder: ScanBuilder,
        dynamic: bool,
        threads: usize,
        files: usize,
    ) -> VortexResult<Self> {
        let parts_per_file = parts_per_file(threads, files);
        let scan = Arc::new(
            builder
                .with_max_output_rows(
                    NonZeroUsize::new(MAX_ROWS_PER_PART).vortex_expect("part rows are nonzero"),
                )
                .prepare()?,
        );
        let io = Arc::new(FileReads::new(scan.io(), scan.read_lifetimes()));
        let parts = if scan.sparse_selection() {
            // A sparse selection, such as an engine's fetch of a few rows by id, selects rows in
            // few chunks: each of them is its own part, and the rest are not planned at all, so
            // DuckDB gives the file a thread per chunk that has work, as V1 does with its splits.
            let selection = scan.selection();
            let mut chunks = scan.row_chunks();
            chunks.retain(|chunk| selection.row_count_in_range(chunk) > 0);
            chunks
        } else {
            let parts = parts(scan.rows().clone(), parts_per_file);
            match scan.string_dictionary_chunks() {
                Some(chunks) => {
                    let aligned = dictionary_parts(chunks);
                    tracing::debug!(
                        parts = parts.len(),
                        dictionary_parts = aligned.len(),
                        "DuckDB V2 dictionary assignments"
                    );
                    if aligned.len() >= parts.len() {
                        aligned
                    } else {
                        parts
                    }
                }
                None => parts,
            }
        };
        let work = plan_parts(
            &scan,
            &io,
            parts,
            dynamic,
            threads.max(1).saturating_mul(PARTS_PER_THREAD),
        )?;
        Ok(Self {
            scan,
            io,
            planned: work.len(),
            work: Mutex::new(work),
            claims: AtomicUsize::new(0),
            rows_taken: AtomicU64::new(0),
        })
    }

    /// Whether the file has work for one more thread: DuckDB gives it one thread per piece of
    /// planned work at most.
    pub(crate) fn claim(&self) -> bool {
        self.claims.fetch_add(1, Ordering::Relaxed) < self.planned
    }

    /// The percentage of the scanned rows whose chunks threads have taken.
    pub(crate) fn progress(&self) -> f64 {
        let rows = self.scan.rows();
        let total = rows.end - rows.start;
        if total == 0 || self.work.lock().is_empty() {
            return 100.0;
        }
        100.0 * self.rows_taken.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// The next array of this file for the thread whose run is `run`, or `None` once the file has
    /// no more work for the thread. Drives one part to completion before taking another, and
    /// waits for reads on the calling thread.
    pub(crate) fn next_array(&self, run: &mut Run) -> VortexResult<Option<ArrayRef>> {
        loop {
            if run.live_roots() == 0 {
                let Some(part) = self.next_part() else {
                    return Ok(None);
                };
                run.admit(part.planner, part.scope, self.io.session());
            }
            match run.advance()? {
                Progress::Batch(batch) => return Ok(Some(batch.array)),
                Progress::Spawn(..) => {
                    vortex_bail!("a queued part must emit ordinary child planners");
                }
                Progress::RootDone(_) => {}
                Progress::Idle => return Ok(None),
                Progress::Waiting => {
                    RUNTIME.block_on(future::poll_fn(|cx| run.poll_completion(cx)))?;
                }
            }
        }
    }

    /// Takes the next part no worker has claimed.
    fn next_part(&self) -> Option<Part> {
        let part = self.work.lock().pop_front();
        if let Some(part) = &part {
            self.rows_taken.fetch_add(
                part.scope.rows.end - part.scope.rows.start,
                Ordering::Relaxed,
            );
        }
        part
    }
}

/// Prunes once, then builds the worker queue without executing any of its parts.
fn plan_parts(
    scan: &Arc<PreparedScan>,
    io: &Arc<FileReads>,
    ranges: Vec<Range<u64>>,
    dynamic: bool,
    pruned_parts: usize,
) -> VortexResult<VecDeque<Part>> {
    let claim = io.claim(scan.rows().clone());
    if scan.filter().is_none() {
        return Ok(part_planners(scan, claim, ranges, None, dynamic));
    }

    let work = Arc::new(Mutex::new(VecDeque::new()));
    let next = {
        let scan = Arc::clone(scan);
        let work = Arc::clone(&work);
        move |kept: Option<Mask>, claim| {
            // Heavy pruning can leave only a few files with work. Wider scans need parallelism
            // within those files; narrow fixed-width scans keep large chunks to amortize setup.
            let ranges = if !scan.sparse_selection()
                && scan.chunk_rows() < MAX_ROWS_PER_PART
                && kept
                    .as_ref()
                    .is_some_and(|kept| kept.true_count() <= kept.len() / 2)
            {
                parts(scan.rows().clone(), pruned_parts)
            } else {
                ranges
            };
            *work.lock() = part_planners(&scan, claim, ranges, kept, dynamic);
            None
        }
    };
    let rows = scan.rows().clone();
    let pruning = PrunePlanner::new(Arc::clone(scan), claim, Box::new(next));
    let mut run = Run::new();
    run.admit(
        Box::new(pruning),
        WorkScope {
            file_ordinal: 0,
            rows,
        },
        io.session(),
    );
    loop {
        match run.advance()? {
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting => {
                RUNTIME.block_on(future::poll_fn(|cx| run.poll_completion(cx)))?;
            }
            Progress::Batch(_) | Progress::Spawn(..) => {
                vortex_bail!("whole-file pruning must finish without executing parts");
            }
        }
    }
    let work = std::mem::take(&mut *work.lock());
    Ok(work)
}

/// Builds one planner per surviving part. All children stay within the worker's run.
fn part_planners(
    scan: &Arc<PreparedScan>,
    mut claim: RowClaim,
    parts: Vec<Range<u64>>,
    kept: Option<Mask>,
    dynamic: bool,
) -> VecDeque<Part> {
    let mut work = VecDeque::new();
    for rows in parts {
        let mut part_claim = claim.split_to(rows.end);
        part_claim.restrict_to(rows.clone());
        let kept = match &kept {
            Some(kept) => {
                let offset = |row: u64| {
                    usize::try_from(row - scan.rows().start).vortex_expect("a part fits usize")
                };
                let kept = kept.slice(offset(rows.start)..offset(rows.end));
                if kept.all_false() {
                    continue;
                }
                Some(kept)
            }
            None => None,
        };
        let scope = WorkScope {
            file_ordinal: 0,
            rows,
        };
        let scan = Arc::clone(scan);
        let planner: Box<dyn Planner> = if dynamic {
            let next_scan = Arc::clone(&scan);
            let next = move |pruned: Option<Mask>, claim: RowClaim| -> Option<Box<dyn Planner>> {
                let kept = match (kept, pruned) {
                    (Some(kept), Some(pruned)) => Some(kept & &pruned),
                    (kept, pruned) => kept.or(pruned),
                };
                Some(Box::new(part_chunks(next_scan, claim, kept)))
            };
            Box::new(PrunePlanner::new(scan, part_claim, Box::new(next)))
        } else {
            Box::new(part_chunks(scan, part_claim, kept))
        };
        work.push_back(Part { scope, planner });
    }
    work
}

/// Filtering stays within the driver's bounded assignment. DuckDB can consume the resulting
/// chunked projection directly, so another cut here would repeat dictionary-expression work.
fn part_chunks(scan: Arc<PreparedScan>, claim: RowClaim, kept: Option<Mask>) -> ChunkPlanner {
    let selection = scan.selection().clone();
    if scan.filter().is_some() {
        let chunks = vec![claim.rows().clone()];
        ChunkPlanner::with_chunks(scan, claim, chunks, selection, kept)
    } else {
        ChunkPlanner::new(scan, claim, selection, kept)
    }
}

/// Divides rows on a fixed grid, aiming for `count` assignments while bounding their size.
/// A single-file scan gets enough assignments to keep its workers busy, while a many-file scan
/// avoids having every worker contend for the same reads and evaluate the same dictionaries.
fn parts(rows: Range<u64>, count: usize) -> Vec<Range<u64>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let first = rows.start / MIN_ROWS_PER_PART;
    let blocks = rows.end.div_ceil(MIN_ROWS_PER_PART) - first;
    let per_part = blocks
        .div_ceil(count.max(1) as u64)
        .min(MAX_ROWS_PER_PART as u64 / MIN_ROWS_PER_PART);
    let mut start = first * MIN_ROWS_PER_PART;
    let mut parts = Vec::new();
    while start < rows.end {
        let end = start
            .saturating_add(per_part.saturating_mul(MIN_ROWS_PER_PART))
            .min(rows.end);
        parts.push(start.max(rows.start)..end);
        start = end;
    }
    parts
}

/// Coalesces small dictionaries without cutting them. Oversized dictionaries retain the
/// driver's maximum assignment size, and the caller retains the required parallelism.
fn dictionary_parts(chunks: Vec<Range<u64>>) -> Vec<Range<u64>> {
    let mut result: Vec<Range<u64>> = Vec::new();
    for chunk in chunks {
        for chunk in parts(chunk, 1) {
            if let Some(last) = result.last_mut()
                && last.end - last.start < MIN_ROWS_PER_DICTIONARY_PART
                && chunk.end - last.start <= MAX_ROWS_PER_PART as u64
            {
                last.end = chunk.end;
            } else {
                result.push(chunk);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;

    use super::dictionary_parts;
    use super::parts;
    use super::parts_per_file;

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn dictionary_assignments_preserve_regions_and_bound_large_dictionaries() {
        assert_eq!(
            dictionary_parts(vec![0..30_000, 30_000..70_000, 70_000..100_000]),
            vec![0..30_000, 30_000..70_000, 70_000..100_000]
        );
        assert_eq!(
            dictionary_parts(vec![0..3_000, 3_000..7_000, 7_000..10_000, 10_000..12_000]),
            vec![0..10_000, 10_000..12_000]
        );
        assert_eq!(
            dictionary_parts(vec![0..2_000_000]),
            vec![0..1_048_576, 1_048_576..2_000_000]
        );
    }

    #[rstest]
    #[case(8, 1, 32)]
    #[case(8, 4, 8)]
    #[case(8, 8, 4)]
    #[case(8, 32, 1)]
    #[case(8, 100, 1)]
    #[case(96, 100, 4)]
    #[case(1, 100, 1)]
    #[case(0, 0, 4)]
    fn work_accounts_for_threads_and_files(
        #[case] threads: usize,
        #[case] files: usize,
        #[case] expected: usize,
    ) {
        assert_eq!(parts_per_file(threads, files), expected);
    }

    #[rstest]
    #[case::even(0..262_144, 2, vec![0..131_072, 131_072..262_144])]
    #[case::ragged(0..270_000, 2, vec![0..196_608, 196_608..270_000])]
    #[case::clipped(25_000..205_000, 3, vec![25_000..131_072, 131_072..205_000])]
    #[case::more_parts_than_blocks(0..100_000, 8, vec![0..65_536, 65_536..100_000])]
    #[case::empty(100..100, 4, vec![])]
    #[case::zero_count(25_000..205_000, 0, vec![25_000..205_000])]
    #[case::row_limit(u64::MAX - 1000..u64::MAX, 8, vec![u64::MAX - 1000..u64::MAX])]
    fn parts_use_a_fixed_row_grid(
        #[case] rows: Range<u64>,
        #[case] count: usize,
        #[case] expected: Vec<Range<u64>>,
    ) {
        assert_eq!(parts(rows, count), expected);
    }

    #[test]
    fn one_filtered_chunk_has_work_for_multiple_threads() {
        assert_eq!(
            parts(0..1_048_576, 16),
            (0..16)
                .map(|i| i * 65_536..(i + 1) * 65_536)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn large_files_keep_bounded_assignments() {
        assert_eq!(
            parts(0..3_000_000, 1),
            vec![0..1_048_576, 1_048_576..2_097_152, 2_097_152..3_000_000]
        );
    }
}

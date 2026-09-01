// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ScanBuilder`](vortex_layout::scan::scan_builder::ScanBuilder) integration.

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::future::join_all;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutRef;
use vortex_layout::scan::scan_builder::ScanExecutor;
use vortex_layout::scan::scan_builder::ScanRequest;
use vortex_layout::segments::SegmentSource;
use vortex_mask::AllOr;
use vortex_utils::aliases::hash_map::HashMap;

use crate::MorselScan;
use crate::build::ExecPlan;
use crate::build::build_plan;
use crate::driver::SharedMorselWorkerPool;
use crate::driver::morsels;
use crate::nodes::ConjunctMode;

type PlanCacheKey = (usize, String, Option<String>, ConjunctMode);

static PLAN_CACHE: LazyLock<Mutex<HashMap<PlanCacheKey, Weak<ExecPlan>>>> =
    LazyLock::new(Mutex::default);

/// Morsel-driven execution backend for a layout scan builder.
pub struct MorselScanExecutor {
    layout: LayoutRef,
    segments: Arc<dyn SegmentSource>,
    target_rows: u64,
    conjunct_mode: ConjunctMode,
    threads: usize,
    worker_pool: Option<Arc<SharedMorselWorkerPool>>,
}

impl MorselScanExecutor {
    /// Create an executor over a raw layout and its segment source.
    pub fn new(layout: LayoutRef, segments: Arc<dyn SegmentSource>) -> Self {
        Self {
            layout,
            segments,
            target_rows: 128 * 1024,
            conjunct_mode: ConjunctMode::Cascade,
            threads: 4,
            worker_pool: None,
        }
    }

    /// Set the target number of rows per morsel.
    pub fn with_target_rows(mut self, target_rows: u64) -> Self {
        self.target_rows = target_rows;
        self
    }

    /// Set the conjunct evaluation policy.
    pub fn with_conjunct_mode(mut self, conjunct_mode: ConjunctMode) -> Self {
        self.conjunct_mode = conjunct_mode;
        self
    }

    /// Set the number of affinity workers used by one shared scan run.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Reuse a persistent set of workers across scans.
    pub fn with_worker_pool(mut self, worker_pool: Arc<SharedMorselWorkerPool>) -> Self {
        self.threads = worker_pool.threads();
        self.worker_pool = Some(worker_pool);
        self
    }
}

impl ScanExecutor for MorselScanExecutor {
    fn build(
        &self,
        request: ScanRequest,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>> {
        if request.limit.is_some() {
            vortex_bail!("the morsel scan executor does not support limits");
        }
        if request.row_offset != 0 {
            vortex_bail!("the morsel scan executor does not support row offsets");
        }

        let projection = unbind(&request.projection)?;
        let filter = request.filter.as_ref().map(unbind).transpose()?;
        let layout_key = Arc::as_ptr(&self.layout) as *const () as usize;
        let plan_key = (
            layout_key,
            projection.to_string(),
            filter.as_ref().map(ToString::to_string),
            self.conjunct_mode,
        );
        let plan = {
            let mut cache = PLAN_CACHE.lock();
            match cache.get(&plan_key).and_then(Weak::upgrade) {
                Some(plan) => plan,
                None => {
                    let plan = Arc::new(build_plan(
                        &self.layout,
                        &projection,
                        filter.as_ref(),
                        self.conjunct_mode,
                    )?);
                    cache.insert(plan_key, Arc::downgrade(&plan));
                    plan
                }
            }
        };

        let full_range = request
            .row_range
            .clone()
            .unwrap_or_else(|| 0..plan.row_count());
        let morsels = selected_morsels(
            morsels(&plan, self.target_rows),
            &full_range,
            &request.selection,
        );

        let mut work = Vec::with_capacity(morsels.len());
        let mut outputs = Vec::with_capacity(morsels.len());
        for morsel in morsels {
            let pruning = request
                .filter
                .as_ref()
                .map(|filter| {
                    request.layout_reader.pruning_evaluation(
                        &morsel.range,
                        filter,
                        request.selection.row_mask(&morsel.range).mask().clone(),
                    )
                })
                .transpose()?;
            let (sender, receiver) = oneshot::channel();
            work.push((morsel, pruning, sender));
            outputs.push(Box::pin(async move {
                receiver
                    .await
                    .map_err(|_| vortex_err!("shared morsel scan coordinator stopped"))?
            })
                as BoxFuture<'static, VortexResult<Option<ArrayRef>>>);
        }

        let segments = Arc::clone(&self.segments);
        let session = request.session.clone();
        let handle = request.session.handle();
        let coordinator_handle = handle.clone();
        let output_dtype = plan.output_dtype().clone();
        let threads = self.threads;
        let worker_pool = self.worker_pool.clone();
        handle
            .spawn(async move {
                let prepared = join_all(work.into_iter().map(
                    |(morsel, pruning, sender)| async move {
                        let ranges = match pruning {
                            Some(pruning) => {
                                pruning.await.map(|mask| mask_ranges(&morsel.range, &mask))
                            }
                            None => Ok(morsel.selected_ranges),
                        };
                        (ranges, sender)
                    },
                ))
                .await;

                let mut ranges = Vec::new();
                let mut targets = Vec::new();
                let mut groups = Vec::new();
                for (selected_ranges, sender) in prepared {
                    let selected_ranges = match selected_ranges {
                        Ok(selected_ranges) => selected_ranges,
                        Err(err) => {
                            drop(sender.send(Err(err)));
                            continue;
                        }
                    };
                    if selected_ranges.is_empty() {
                        drop(sender.send(Ok(None)));
                        continue;
                    }
                    let group = Arc::new(OutputGroup::new(
                        selected_ranges.len(),
                        output_dtype.clone(),
                        sender,
                    ));
                    for (local_index, range) in selected_ranges.into_iter().enumerate() {
                        ranges.push(range);
                        targets.push(CompletionTarget {
                            group: Arc::clone(&group),
                            local_index,
                        });
                    }
                    groups.push(group);
                }

                if ranges.is_empty() {
                    return;
                }
                let threads = ranges.len().min(threads);
                let result = coordinator_handle
                    .spawn_blocking(move || {
                        let mut scan = MorselScan::new(plan, segments, session)
                            .with_threads(threads)
                            .with_morsels(ranges)
                            .with_completion_sink(move |index, batch| {
                                targets[index].complete(batch);
                            });
                        if let Some(worker_pool) = worker_pool {
                            scan = scan.with_worker_pool(worker_pool);
                        }
                        scan.run().map(|_| ())
                    })
                    .await;
                if let Err(err) = result {
                    let message = err.to_string();
                    for group in groups {
                        group.fail(&message);
                    }
                }
            })
            .detach();

        Ok(outputs)
    }
}

struct CompletionTarget {
    group: Arc<OutputGroup>,
    local_index: usize,
}

impl CompletionTarget {
    fn complete(&self, batch: Option<ArrayRef>) {
        self.group.complete(self.local_index, batch);
    }
}

struct OutputGroup {
    remaining: AtomicUsize,
    dtype: DType,
    batches: Mutex<Vec<(usize, ArrayRef)>>,
    sender: Mutex<Option<oneshot::Sender<VortexResult<Option<ArrayRef>>>>>,
}

impl OutputGroup {
    fn new(
        remaining: usize,
        dtype: DType,
        sender: oneshot::Sender<VortexResult<Option<ArrayRef>>>,
    ) -> Self {
        Self {
            remaining: AtomicUsize::new(remaining),
            dtype,
            batches: Mutex::new(Vec::new()),
            sender: Mutex::new(Some(sender)),
        }
    }

    fn complete(&self, index: usize, batch: Option<ArrayRef>) {
        if let Some(batch) = batch {
            self.batches.lock().push((index, batch));
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let mut batches = std::mem::take(&mut *self.batches.lock());
        batches.sort_unstable_by_key(|(index, _)| *index);
        let result = match batches.len() {
            0 => Ok(None),
            1 => Ok(batches.pop().map(|(_, batch)| batch)),
            _ => ChunkedArray::try_new(
                batches.into_iter().map(|(_, batch)| batch),
                self.dtype.clone(),
            )
            .map(|array| Some(array.into_array())),
        };
        if let Some(sender) = self.sender.lock().take() {
            drop(sender.send(result));
        }
    }

    fn fail(&self, message: &str) {
        if let Some(sender) = self.sender.lock().take() {
            drop(sender.send(Err(vortex_err!("shared morsel scan failed: {message}"))));
        }
    }
}

fn mask_ranges(range: &Range<u64>, mask: &vortex_mask::Mask) -> Vec<Range<u64>> {
    match mask.slices() {
        AllOr::All => vec![range.clone()],
        AllOr::None => Vec::new(),
        AllOr::Some(slices) => slices
            .iter()
            .map(|&(start, end)| range.start + start as u64..range.start + end as u64)
            .collect(),
    }
}

fn unbind(expr: &BoundExpression) -> VortexResult<Expression> {
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(Expression::Root);
    };
    Expression::try_new(
        scalar_fn.clone(),
        expr.children()
            .iter()
            .map(unbind)
            .collect::<VortexResult<Vec<_>>>()?,
    )
}

struct SelectedMorsel {
    range: Range<u64>,
    selected_ranges: Vec<Range<u64>>,
}

fn selected_morsels(
    morsels: Vec<Range<u64>>,
    row_range: &Range<u64>,
    selection: &vortex_scan::selection::Selection,
) -> Vec<SelectedMorsel> {
    morsels
        .into_iter()
        .filter_map(|range| {
            let start = range.start.max(row_range.start);
            let end = range.end.min(row_range.end);
            (start < end).then_some(start..end)
        })
        .filter_map(|range| {
            let mask = selection.row_mask(&range);
            let selected_ranges = match mask.mask().slices() {
                AllOr::All => vec![range.clone()],
                AllOr::None => Vec::new(),
                AllOr::Some(slices) => slices
                    .iter()
                    .map(|&(start, end)| range.start + start as u64..range.start + end as u64)
                    .collect(),
            };
            (!selected_ranges.is_empty()).then_some(SelectedMorsel {
                range,
                selected_ranges,
            })
        })
        .collect()
}

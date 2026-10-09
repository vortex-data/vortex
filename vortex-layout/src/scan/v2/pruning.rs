// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use futures::channel::oneshot;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::plan::Eval;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::Zoned;
use crate::scan::planning::ScanPlans;
use crate::scan::planning::graph::GraphStep;
use crate::scan::planning::graph::ProtocolGraph;
use crate::scan::planning::graph::ScanGraph;

pub(super) fn file_pruning_enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| env::var("VORTEX_SCAN_FILE_PRUNING").is_ok_and(|value| value == "1"));
    *ENABLED
}

pub(super) struct PrunedFile {
    pub ranges: Vec<Range<u64>>,
    pub dynamic: bool,
}

impl PrunedFile {
    /// Keeps holes inside a filter split as a mask, so disjoint surviving zones do not create
    /// separate tasks that decode the same compressed chunk independently.
    pub(super) fn select(&self, rows: Range<u64>) -> VortexResult<Option<(Range<u64>, Mask)>> {
        let lo = self.ranges.partition_point(|range| range.end <= rows.start);
        let hi = self.ranges.partition_point(|range| range.start < rows.end);
        if lo >= hi {
            return Ok(None);
        }
        let selected = &self.ranges[lo..hi];
        let rows =
            rows.start.max(selected[0].start)..rows.end.min(selected[selected.len() - 1].end);
        let len = usize::try_from(rows.end - rows.start)?;
        if selected.len() == 1 {
            return Ok(Some((rows, Mask::new_true(len))));
        }
        let mut bits = BitBufferMut::with_capacity(len);
        let mut cursor = rows.start;
        for range in selected {
            let start = rows.start.max(range.start);
            let end = rows.end.min(range.end);
            bits.append_n(false, usize::try_from(start - cursor)?);
            bits.append_n(true, usize::try_from(end - start)?);
            cursor = end;
        }
        Ok(Some((rows, Mask::from(bits.freeze()))))
    }
}

/// Loads zone proofs through the planning protocol before any data splits are announced.
pub(super) async fn prune_file(
    plans: &ScanPlans,
    proof: PlanRef,
    io: &Arc<dyn IoService>,
) -> VortexResult<PrunedFile> {
    if proof.row_count() == 0 {
        return Ok(PrunedFile {
            ranges: Vec::new(),
            dynamic: false,
        });
    }
    let mut boundaries = vec![0, proof.row_count()];
    let mut dynamic = false;
    zone_boundaries(&proof, &mut boundaries, &mut dynamic)?;
    boundaries.sort_unstable();
    boundaries.dedup();
    // A single representative row loads every complete zone table and initializes each proof's
    // cache. Expanding the proof over the file here would allocate one boolean per data row.
    let graph = ScanGraph::try_new(
        plans.session.clone(),
        &proof,
        0..1,
        Mask::new_true(1),
        plans.row_offset,
        plans.decoded.clone(),
    )?;
    let scope = WorkScope {
        file_ordinal: 0,
        rows: 0..proof.row_count(),
    };
    let (sender, receiver) = oneshot::channel();
    let planner = FilePruningPlanner {
        graph: ProtocolGraph::new(graph, Arc::clone(&plans.locations), 0),
        proof,
        plans: plans.clone(),
        boundaries,
        result: Some(sender),
    };
    let mut run = Run::new();
    run.admit(Box::new(planner), scope, io.session());
    loop {
        match run.advance()? {
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            Progress::Batch(_) => vortex_bail!("file pruning emitted a data batch"),
        }
    }
    Ok(PrunedFile {
        ranges: receiver
            .await
            .map_err(|_| vortex_err!("file pruning did not finish"))?,
        dynamic,
    })
}

struct FilePruningPlanner {
    graph: ProtocolGraph,
    proof: PlanRef,
    plans: ScanPlans,
    boundaries: Vec<u64>,
    result: Option<oneshot::Sender<Vec<Range<u64>>>>,
}

impl IoConsumer for FilePruningPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.graph.set_io_result(request, result);
    }
}

impl Planner for FilePruningPlanner {
    fn state(&self) -> State {
        if self.result.is_none() {
            State::Done
        } else if self.graph.state() == State::Done {
            State::NeedsCompute
        } else {
            self.graph.state()
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.graph.state() != State::Done {
            return Ok(match self.graph.compute()? {
                GraphStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                GraphStep::Yield | GraphStep::Piece(_) => PlannerOutput::Continue,
            });
        }
        let starts = &self.boundaries[..self.boundaries.len() - 1];
        let values = compact_proof(&self.proof, starts, &self.plans)?;
        let pruned: Mask = values
            .fill_null(false)?
            .execute(&mut self.plans.session.create_execution_ctx())?;
        let mut ranges: Vec<Range<u64>> = Vec::new();
        for (index, pair) in self.boundaries.windows(2).enumerate() {
            if pruned.value(index) {
                continue;
            }
            if let Some(last) = ranges.last_mut()
                && last.end == pair[0]
            {
                last.end = pair[1];
            } else {
                ranges.push(pair[0]..pair[1]);
            }
        }
        tracing::debug!(target: "vortex_layout::scan::v2::pruning",
            file_rows = self.proof.row_count(), intervals = starts.len(),
            surviving_rows = ranges.iter().map(|r| r.end - r.start).sum::<u64>(),
            surviving_ranges = ranges.len(), "file pruning finished");
        if let Some(result) = self.result.take() {
            drop(result.send(ranges));
        }
        Ok(PlannerOutput::Done)
    }
}

fn zone_boundaries(
    plan: &PlanRef,
    boundaries: &mut Vec<u64>,
    dynamic: &mut bool,
) -> VortexResult<()> {
    if let Some(zoned) = plan.as_opt::<Zoned>() {
        let expression = zoned
            .pruning_expression()
            .ok_or_else(|| vortex_err!("missing zone proof"))?;
        *dynamic |= DynamicExprUpdates::new(expression).is_some();
        let mut row = zoned.zone_len();
        while row < plan.row_count() {
            boundaries.push(row);
            row = row.saturating_add(zoned.zone_len());
        }
    } else {
        if let Some(eval) = plan.as_opt::<Eval>() {
            *dynamic |= DynamicExprUpdates::new(eval.expression()).is_some();
        }
        for child in plan.children().iter() {
            zone_boundaries(&child?, boundaries, dynamic)?;
        }
    }
    Ok(())
}

/// Evaluates one representative per interval in the union of all column zone boundaries. This
/// preserves compound proofs even when columns have different zone sizes; memory scales with
/// the zone count, rather than the file's row count.
fn compact_proof(plan: &PlanRef, starts: &[u64], plans: &ScanPlans) -> VortexResult<ArrayRef> {
    if let Some(zoned) = plan.as_opt::<Zoned>() {
        let expression = zoned
            .pruning_expression()
            .ok_or_else(|| vortex_err!("missing zone proof"))?;
        let mask = zoned
            .pruned_zones()
            .ok_or_else(|| vortex_err!("missing zone cache"))?
            .get(expression, &plans.session)?
            .ok_or_else(|| vortex_err!("zone proof was not loaded"))?;
        let validity = if plan.dtype().is_nullable() {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };
        return Ok(BoolArray::try_new(
            starts
                .iter()
                .map(|row| Ok(mask.value(usize::try_from(row / zoned.zone_len())?)))
                .collect::<VortexResult<_>>()?,
            validity,
        )?
        .into_array());
    }
    if let Some(eval) = plan.as_opt::<Eval>() {
        return compact_proof(&eval.child_plan()?, starts, plans)?.apply_bound(eval.expression());
    }
    if let Some(pack) = plan.as_opt::<Pack>() {
        let fields = (0..pack.nfields())
            .map(|index| compact_proof(&pack.child_required(index)?, starts, plans))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = match pack.validity()? {
            Some(validity) => Validity::Array(compact_proof(&validity, starts, plans)?),
            None => Validity::NonNullable,
        };
        return Ok(StructArray::try_new(
            pack.fields().names().clone(),
            fields,
            starts.len(),
            validity,
        )?
        .into_array());
    }
    vortex_bail!("unsupported file pruning plan: {}", plan.id())
}

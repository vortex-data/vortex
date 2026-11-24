// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads the offsets of the list columns a scan reads, once, before any chunk is planned.
//!
//! The element rows a selection of lists holds depend on the lists' offsets, which are data. A
//! chunk that had to read its offsets before it could ask for its elements would take one more
//! round of reads per level of nesting, and nothing could tell what it reads before it runs. A
//! list column's offsets are instead read whole here, ahead of the chunks, and kept with the
//! scan's whole values, where compiling a [`ListPack`] finds them. Every chunk then knows the
//! elements it reads when it is compiled, at any depth of nesting.

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_session::VortexSession;

use super::PreparedScan;
use super::adaptor::AdaptorStep;
use super::adaptor::ScanAdaptor;
use crate::execution::operators::join;
use crate::execution::pipeline::Scan;
use crate::execution::pipeline::Split;
use crate::plan::ListLengths;
use crate::plan::ListPack;
use crate::plan::PlanByPtr;
use crate::plan::PlanRef;
use crate::plan::builtins::list_pack::execute::wide_offsets_dtype;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

/// The most offsets of one list column a scan reads ahead of its chunks, unless
/// `VORTEX_LIST_OFFSETS_AHEAD_MAX` says otherwise.
///
/// Offsets are small beside the elements of long lists, and a scan reads them anyway. Those of
/// many short lists are a column as long as the lists are many, and the lists of an inner level of
/// nesting are as many as the outer level's elements: reading them whole before the first chunk
/// would delay it, hold them all in memory, and read the offsets of lists a filter drops.
const DEFAULT_OFFSETS_AHEAD_MAX: u64 = 1 << 22;

static OFFSETS_AHEAD_MAX: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("VORTEX_LIST_OFFSETS_AHEAD_MAX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_OFFSETS_AHEAD_MAX)
});

/// The most offsets of one list column a scan reads ahead of its chunks unless told otherwise.
pub(crate) fn default_offsets_ahead_max() -> u64 {
    *OFFSETS_AHEAD_MAX
}

/// The planner a list offsets planner continues into.
pub type ListOffsetsNext = Box<dyn FnOnce() -> Box<dyn Planner> + Send>;

/// Reads the offsets of every list the scan's projection and filter read, whole, and emits the
/// planner it was built to continue into. The offsets are the only reads, all asked for
/// together. A scan that reads no list, or whose offsets an earlier scan of the file read,
/// continues at once.
pub struct ListOffsetsPlanner {
    scan: Arc<PreparedScan>,
    rows: Range<u64>,
    next: Option<ListOffsetsNext>,
    /// What is being read ahead, found by the first compute.
    readings: Option<Vec<Reading>>,
    /// The requests published and not yet delivered: which reading each belongs to, and the id
    /// its adaptor knows it by. Every adaptor numbers its own requests from zero.
    requests: FxHashMap<u32, (usize, IoRequestId)>,
    next_request: u32,
    done: bool,
}

/// One list's offsets, or index of them, being read whole.
struct Reading {
    ahead: Ahead,
    adaptor: ScanAdaptor,
    pieces: Vec<ArrayRef>,
    kept: bool,
}

impl ListOffsetsPlanner {
    /// The planner reading ahead for `scan` over `rows`, continuing into `next`.
    pub fn new(scan: Arc<PreparedScan>, rows: Range<u64>, next: ListOffsetsNext) -> Self {
        Self {
            scan,
            rows,
            next: Some(next),
            readings: None,
            requests: FxHashMap::default(),
            next_request: 0,
            done: false,
        }
    }

    /// What to read ahead for the lists the scan reads, that is not read ahead yet.
    fn find(&self) -> VortexResult<Vec<Ahead>> {
        let predicates = self
            .scan
            .filter
            .iter()
            .flat_map(|filter| filter.predicates().iter());
        Ok(
            list_structures(std::iter::once(&self.scan.projection).chain(predicates))?
                .iter()
                .filter_map(|list| list.ahead(self.scan.list_offsets_ahead_max))
                .filter(|ahead| self.scan.values.get(ahead.plan()).is_none())
                .collect(),
        )
    }

    /// Starts reading everything [`find`](Self::find) found.
    fn start(&self) -> VortexResult<Vec<Reading>> {
        self.find()?
            .into_iter()
            .map(|ahead| {
                let plan = ahead.plan().clone();
                tracing::debug!(
                    rows = plan.row_count(),
                    index = matches!(ahead, Ahead::Index(_)),
                    "v2 list offsets to read ahead"
                );
                let rows = 0..plan.row_count();
                let scan = Scan::try_new(self.scan.session.clone(), plan, vec![Split::all(rows)])?
                    .with_value_cache(self.scan.values.clone());
                Ok(Reading {
                    ahead,
                    adaptor: ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)),
                    pieces: Vec::new(),
                    kept: false,
                })
            })
            .collect()
    }
}

/// Keeps what `reading` read for the chunks of `scan`.
fn keep(scan: &PreparedScan, reading: &mut Reading) -> VortexResult<()> {
    let pieces = std::mem::take(&mut reading.pieces);
    let values = reading.ahead.widen(&scan.session, pieces)?;
    scan.values.insert(reading.ahead.plan(), values);
    reading.kept = true;
    Ok(())
}

/// The plans that say where a list's elements are: its offsets, and its index of them if it has
/// one.
pub(crate) struct ListStructure {
    pub(crate) offsets: PlanRef,
    pub(crate) index: Option<PlanRef>,
    /// The elements the lists hold in all, when they are read.
    pub(crate) elements: u64,
}

/// What a scan reads whole, ahead of its chunks, to know where a list's elements are.
pub(crate) enum Ahead {
    /// The list's offsets: its lists are few enough.
    Offsets(PlanRef),
    /// The list's index of its offsets.
    Index(PlanRef),
}

impl ListStructure {
    /// What to read ahead for this list, reading at most `max` offsets whole: the offsets when
    /// they are that few, else the index if the list has one, else nothing.
    pub(crate) fn ahead(&self, max: u64) -> Option<Ahead> {
        if self.offsets.row_count() <= max {
            Some(Ahead::Offsets(self.offsets.clone()))
        } else {
            self.index.clone().map(Ahead::Index)
        }
    }
}

impl Ahead {
    pub(crate) fn plan(&self) -> &PlanRef {
        match self {
            Ahead::Offsets(plan) | Ahead::Index(plan) => plan,
        }
    }

    /// What the plan produced over all its rows, as a [`ListPack`] compiles from it.
    ///
    /// Offsets are kept as they were stored, and each chunk decodes and checks the ones it
    /// selects: a table can have hundreds of list columns, and the offsets of long, even lists
    /// are a few bytes however many they are. An index is small, and is decoded and checked
    /// here, once.
    pub(crate) fn widen(
        &self,
        session: &VortexSession,
        pieces: Vec<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        let values = join(self.plan().dtype(), pieces)?;
        match self {
            Ahead::Offsets(_) => values.optimize_ctx(session),
            Ahead::Index(_) => {
                let mut ctx = session.create_execution_ctx();
                let values = values
                    .cast(wide_offsets_dtype())?
                    .execute::<PrimitiveArray>(&mut ctx)?;
                let (lists, offsets) = values.as_slice::<u64>().split_at(values.len() / 2);
                vortex_ensure!(
                    values.len().is_multiple_of(2) && lists.is_sorted() && offsets.is_sorted(),
                    "List index is not increasing pairs"
                );
                Ok(values.into_array())
            }
        }
    }
}

/// The structure of every [`ListPack`] and [`ListLengths`] under `plans`, each once, outer lists
/// first.
pub(crate) fn list_structures<'a>(
    plans: impl Iterator<Item = &'a PlanRef>,
) -> VortexResult<Vec<ListStructure>> {
    let mut found = Vec::new();
    let mut seen = FxHashSet::default();
    for plan in plans {
        collect_list_structures(plan, &mut seen, &mut found)?;
    }
    Ok(found)
}

fn collect_list_structures(
    plan: &PlanRef,
    seen: &mut FxHashSet<PlanByPtr>,
    found: &mut Vec<ListStructure>,
) -> VortexResult<()> {
    if plan.is_data_source() || !seen.insert(PlanByPtr(plan.clone())) {
        return Ok(());
    }
    if let Some(list) = plan.as_opt::<ListPack>() {
        let offsets = list.offsets()?;
        let elements = list.elements()?;
        if seen.insert(PlanByPtr(offsets.clone())) {
            found.push(ListStructure {
                offsets,
                index: list.index()?,
                elements: elements.row_count(),
            });
        }
        // Lists nest through the elements; offsets, validity and index hold none.
        return collect_list_structures(&elements, seen, found);
    }
    if let Some(lengths) = plan.as_opt::<ListLengths>() {
        let offsets = lengths.offsets()?;
        if seen.insert(PlanByPtr(offsets.clone())) {
            // Lengths need every offset: an index is no use to them.
            found.push(ListStructure {
                offsets,
                index: None,
                elements: 0,
            });
        }
        return Ok(());
    }
    for child in plan.children() {
        collect_list_structures(child?, seen, found)?;
    }
    Ok(())
}

impl IoConsumer for ListOffsetsPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some((reading, request)) = self.requests.remove(&request.0)
            && let Some(readings) = &mut self.readings
        {
            readings[reading].adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for ListOffsetsPlanner {
    fn state(&self) -> State {
        if self.done {
            return State::Done;
        }
        let Some(readings) = &self.readings else {
            return State::NeedsCompute;
        };
        let waiting = |reading: &Reading| !reading.kept && reading.adaptor.state() == State::Waiting;
        if !readings.is_empty() && readings.iter().all(|reading| reading.kept || waiting(reading))
            && readings.iter().any(waiting)
        {
            State::Waiting
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.done {
            return Ok(PlannerOutput::Done);
        }
        if self.readings.is_none() {
            self.readings = Some(self.start()?);
        }
        let readings = self.readings.as_mut().vortex_expect("started above");
        // Every list's offsets are known from the footer, so one pass over the lists asks for
        // them all together, and a later pass decodes what has arrived.
        let mut batch = Vec::new();
        for (index, reading) in readings.iter_mut().enumerate() {
            while !reading.kept {
                match reading.adaptor.state() {
                    State::Waiting => break,
                    State::Done => keep(&self.scan, reading)?,
                    State::NeedsCompute => match reading.adaptor.compute()? {
                        AdaptorStep::Yield => {}
                        AdaptorStep::Piece(piece) => reading.pieces.push(piece.array),
                        AdaptorStep::NeedsIO(requests) => {
                            for mut request in requests {
                                self.requests
                                    .insert(self.next_request, (index, request.request));
                                request.request = IoRequestId(self.next_request);
                                self.next_request += 1;
                                batch.push(request);
                            }
                        }
                    },
                }
            }
        }
        if !batch.is_empty() {
            return Ok(PlannerOutput::NeedsIO(batch));
        }
        if readings.iter().any(|reading| !reading.kept) {
            return Ok(PlannerOutput::Continue);
        }
        tracing::debug!(lists = readings.len(), "v2 list offsets read ahead");
        self.done = true;
        let next = self
            .next
            .take()
            .ok_or_else(|| vortex_err!("the list offsets planner has nothing to continue into"))?;
        Ok(PlannerOutput::Planner(
            WorkScope {
                file_ordinal: 0,
                rows: self.rows.clone(),
            },
            next(),
        ))
    }
}

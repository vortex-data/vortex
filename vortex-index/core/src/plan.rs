// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Index searches as planner work, so one plan opens an index file, searches it and scans the
//! data rows it finds, with every read going through the plan's sources.

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_file::VortexFile;
use vortex_file::v2::OpenFilePlanner;
use vortex_file::v2::ThenFn;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSlot;
use vortex_io::request::IoTarget;
use vortex_io::request::OpenTarget;
use vortex_io::request::SourceId;
use vortex_layout::layouts::indexed::IndexVTable;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_scan::execution::driver::compose;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::CoveredFile;
use crate::FileSelection;
use crate::IndexFile;
use crate::IndexFileInfo;
use crate::IndexStep;
use crate::StepFn;
use crate::standalone::INDEX_FILE_METADATA_KEY;

/// Builds the planners that continue a search, given its answer.
pub type ThenPlanners<T> = Box<dyn FnOnce(T) -> VortexResult<Vec<Box<dyn Planner>>> + Send>;

/// Runs an index search's [`IndexStep`]s as planner work, then continues into the planners built
/// from its answer.
///
/// Each [`IndexStep::Read`] is a scan of index rows: its planners run inside this one, their
/// requests published as this planner's own, and their batches gathered in row order for the
/// step's continuation. The plan's driver performs every read, so the index is read through the
/// same sources, and the same IO session, as the data the search leads to.
pub struct IndexStepPlanner<T> {
    source: SourceId,
    scope: WorkScope,
    phase: Phase<T>,
    then: Option<ThenPlanners<T>>,
}

enum Phase<T> {
    /// Scanning index rows for a step.
    Reading(Gather, StepFn<T>),
    /// The search's answer, not yet continued.
    Answered(T),
    /// The continuation's planners, spawned one per `compute` so a driver may run them in
    /// parallel.
    Emitting(Vec<Box<dyn Planner>>),
    Done,
}

impl<T: Send + 'static> IndexStepPlanner<T> {
    /// Runs `step`, reading the index through `source`, then emits `then`'s planners over `scope`.
    pub fn new(
        source: SourceId,
        scope: WorkScope,
        step: IndexStep<T>,
        then: ThenPlanners<T>,
    ) -> VortexResult<Self> {
        let mut planner = Self {
            source,
            scope,
            phase: Phase::Done,
            then: Some(then),
        };
        planner.phase = planner.start(step)?;
        Ok(planner)
    }

    fn start(&self, step: IndexStep<T>) -> VortexResult<Phase<T>> {
        Ok(match step {
            IndexStep::Done(answer) => Phase::Answered(answer),
            IndexStep::Read(scan, then) => {
                let scan = Arc::new(scan.prepare_as(self.source)?);
                Phase::Reading(Gather::new(compose(&scan)), then)
            }
        })
    }
}

impl<T> IoConsumer for IndexStepPlanner<T> {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Phase::Reading(gather, _) = &mut self.phase {
            gather.set_io_result(request, result);
        }
    }
}

impl<T: Send + 'static> Planner for IndexStepPlanner<T> {
    fn state(&self) -> State {
        match &self.phase {
            Phase::Reading(gather, _) => gather.state(),
            Phase::Answered(_) | Phase::Emitting(_) => State::NeedsCompute,
            Phase::Done => State::Done,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::Reading(mut gather, then) => match gather.compute()? {
                Gathered::Output(output) => {
                    self.phase = Phase::Reading(gather, then);
                    Ok(output)
                }
                Gathered::Batches(batches) => {
                    self.phase = self.start(then(batches)?)?;
                    Ok(PlannerOutput::Continue)
                }
            },
            Phase::Answered(answer) => {
                let then = self
                    .then
                    .take()
                    .ok_or_else(|| vortex_err!("IndexStepPlanner continued twice"))?;
                let mut planners = then(answer)?;
                planners.reverse();
                self.phase = Phase::Emitting(planners);
                Ok(PlannerOutput::Continue)
            }
            Phase::Emitting(mut planners) => match planners.pop() {
                Some(planner) => {
                    self.phase = Phase::Emitting(planners);
                    Ok(PlannerOutput::Spawn(self.scope.clone(), planner))
                }
                None => Ok(PlannerOutput::Done),
            },
            Phase::Done => Ok(PlannerOutput::Done),
        }
    }
}

/// Opens a standalone index file inside a plan, runs a search over it, and scans the rows it
/// selects in each covered data file, opening only the files it selects.
///
/// Every open and read, of the index and of the data, is a request to the plan's sources service
/// `sources`, so one driver and one IO session serve the whole search. `open` names how to open
/// a covered file, and `configure` sets up each file's scan, such as its projection, before the
/// selection is applied.
pub struct IndexScan<V: IndexVTable> {
    /// The plan's sources service.
    pub sources: SourceId,
    /// The standalone index file.
    pub index: OpenTarget,
    /// The index kind the file holds.
    pub vtable: V,
    /// The session the files are opened and scanned with.
    pub session: VortexSession,
    /// The scope the search and its scans are emitted over.
    pub scope: WorkScope,
}

/// Opens a covered data file, by its catalog entry.
pub type OpenCovered = Arc<dyn Fn(&CoveredFile) -> VortexResult<OpenTarget> + Send + Sync>;

/// Configures the scan of one covered data file.
pub type ConfigureScan = Arc<dyn Fn(ScanBuilder) -> VortexResult<ScanBuilder> + Send + Sync>;

/// Turns an opened index into the search whose answer selects data rows.
pub type SearchFn<V> =
    Box<dyn FnOnce(&IndexFile<V>) -> VortexResult<IndexStep<Vec<FileSelection>>> + Send + 'static>;

impl<V: IndexVTable> IndexScan<V> {
    /// The plan's root work: open the index, search it with `search`, then open and scan each
    /// selected file.
    pub fn planner(
        self,
        search: SearchFn<V>,
        open: OpenCovered,
        configure: ConfigureScan,
    ) -> OpenFilePlanner {
        let Self {
            sources,
            index,
            vtable,
            session,
            scope,
        } = self;
        let data_session = session.clone();
        let index_scope = scope.clone();
        OpenFilePlanner::new(
            sources,
            index,
            session,
            scope,
            Box::new(move |file: VortexFile, source| {
                let metadata = file
                    .footer()
                    .metadata_segment(INDEX_FILE_METADATA_KEY)
                    .ok_or_else(|| {
                        vortex_err!("Not a Vortex index file: no {INDEX_FILE_METADATA_KEY} segment")
                    })?
                    .byte_range();
                let step_scope = index_scope.clone();
                let then: ThenIndex<V> = Box::new(move |index| {
                    let step = search(&index)?;
                    let then = scan_selections(
                        sources,
                        index.files().to_vec(),
                        data_session,
                        open,
                        configure,
                    );
                    Ok(
                        Box::new(IndexStepPlanner::new(source, step_scope, step, then)?)
                            as Box<dyn Planner>,
                    )
                });
                Ok(Box::new(OpenIndexPlanner {
                    file,
                    vtable: Some(vtable),
                    metadata,
                    slot: IoSlot::new(source),
                    requested: false,
                    scope: index_scope,
                    then: Some(then),
                }) as Box<dyn Planner>)
            }),
        )
    }
}

/// Continues a search into the scans of the rows it selects: one opened file each.
fn scan_selections(
    sources: SourceId,
    files: Vec<CoveredFile>,
    session: VortexSession,
    open: OpenCovered,
    configure: ConfigureScan,
) -> ThenPlanners<Vec<FileSelection>> {
    Box::new(move |selections| {
        selections
            .into_iter()
            .map(|FileSelection { file, selection }| {
                let covered = files.get(file).ok_or_else(|| {
                    vortex_err!("The search selected file {file}, which the index does not cover")
                })?;
                let configure = Arc::clone(&configure);
                let scan_session = session.clone();
                let then: ThenFn = Box::new(move |file: VortexFile, source| {
                    let builder = ScanBuilder::new(scan_session, file.scan_file());
                    let builder = configure(builder)?.with_selection(selection);
                    Ok(compose(&Arc::new(builder.prepare_as(source)?)))
                });
                let scope = WorkScope {
                    file_ordinal: file as u64,
                    rows: 0..covered.row_count,
                };
                let planner =
                    OpenFilePlanner::new(sources, open(covered)?, session.clone(), scope, then);
                Ok(Box::new(planner) as Box<dyn Planner>)
            })
            .collect()
    })
}

/// Builds the planner that continues an opened index.
type ThenIndex<V> = Box<dyn FnOnce(IndexFile<V>) -> VortexResult<Box<dyn Planner>> + Send>;

/// Reads an opened index file's description, which a file opened in a plan does not load, then
/// continues into the planner built from the index.
struct OpenIndexPlanner<V: IndexVTable> {
    file: VortexFile,
    vtable: Option<V>,
    /// Where the description is in the file.
    metadata: Range<u64>,
    slot: IoSlot,
    requested: bool,
    scope: WorkScope,
    then: Option<ThenIndex<V>>,
}

impl<V: IndexVTable> IoConsumer for OpenIndexPlanner<V> {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.slot.deliver(request, result);
    }
}

impl<V: IndexVTable> Planner for OpenIndexPlanner<V> {
    fn state(&self) -> State {
        if self.then.is_none() {
            State::Done
        } else if self.slot.is_waiting() {
            State::Waiting
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if !self.requested {
            self.requested = true;
            let len = usize::try_from(self.metadata.end - self.metadata.start)?;
            return Ok(PlannerOutput::NeedsIO(
                self.slot.issue(IoTarget::range(self.metadata.start, len)),
            ));
        }
        let bytes = match self.slot.take() {
            Some(IoResult::Bytes(bytes)) => bytes.try_to_host_sync()?,
            _ => vortex_bail!("OpenIndexPlanner: the index description was not delivered"),
        };
        let info = IndexFileInfo::decode(bytes.as_slice(), self.file.session())?;
        let vtable = self
            .vtable
            .take()
            .ok_or_else(|| vortex_err!("OpenIndexPlanner opened its index twice"))?;
        let index = IndexFile::open_with_info(vtable, &self.file, info)?;
        let then = self
            .then
            .take()
            .ok_or_else(|| vortex_err!("OpenIndexPlanner continued twice"))?;
        Ok(PlannerOutput::Planner(self.scope.clone(), then(index)?))
    }
}

/// What one [`Gather::compute`] produced.
enum Gathered {
    /// Work to hand to the plan: requests, or a checkpoint.
    Output(PlannerOutput),
    /// Every planner and morsel finished: their batches, in row order.
    Batches(Vec<ArrayRef>),
}

/// One planner or morsel of a gathered scan.
enum Item {
    Planner(Box<dyn Planner>),
    Morsel(Box<dyn Morsel>),
}

impl Item {
    fn state(&self) -> State {
        match self {
            Self::Planner(planner) => planner.state(),
            Self::Morsel(morsel) => morsel.state(),
        }
    }

    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        match self {
            Self::Planner(planner) => planner.set_io_result(request, result),
            Self::Morsel(morsel) => morsel.set_io_result(request, result),
        }
    }
}

/// Runs a scan's planners inside another planner, keeping the batches they produce.
///
/// The planners run depth first: a child runs to its end before its parent resumes, and every
/// child, independent or not, runs here. That is the emission order a driver returns batches in,
/// so the batches come out in row order. Requests are published as the gathering planner's own,
/// renumbered so the requests of different items never share an id.
struct Gather {
    /// Live items by id.
    items: HashMap<u64, Item>,
    /// Ids of the items in run order: the last is advanced next.
    stack: Vec<u64>,
    next_item: u64,
    /// Published fetches not yet delivered: the item that issued each, and its own id for it.
    pending: HashMap<IoRequestId, (u64, IoRequestId)>,
    next_request: u32,
    batches: Vec<ArrayRef>,
}

impl Gather {
    fn new(root: Box<dyn Planner>) -> Self {
        let mut gather = Self {
            items: HashMap::default(),
            stack: Vec::new(),
            next_item: 0,
            pending: HashMap::default(),
            next_request: 0,
            batches: Vec::new(),
        };
        gather.push(Item::Planner(root));
        gather
    }

    fn push(&mut self, item: Item) {
        let id = self.next_item;
        self.next_item += 1;
        self.items.insert(id, item);
        self.stack.push(id);
    }

    fn top(&self) -> Option<&Item> {
        self.stack.last().and_then(|id| self.items.get(id))
    }

    fn state(&self) -> State {
        match self.top().map(Item::state) {
            // A finished item is retired, and the scan answered, by the next compute.
            None | Some(State::Done) => State::NeedsCompute,
            Some(state) => state,
        }
    }

    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some((item, own)) = self.pending.remove(&request)
            && let Some(item) = self.items.get_mut(&item)
        {
            item.set_io_result(own, result);
        }
    }

    /// Renumbers the requests `item` published, remembering where each fetch must be delivered.
    fn publish(&mut self, item: u64, batch: IoBatch) -> VortexResult<PlannerOutput> {
        let mut published = Vec::with_capacity(batch.len());
        for mut request in batch {
            let id = IoRequestId(self.next_request);
            self.next_request = self
                .next_request
                .checked_add(1)
                .ok_or_else(|| vortex_err!("An index scan published too many requests"))?;
            if request.intent == IoIntent::Fetch {
                self.pending.insert(id, (item, request.request));
            }
            request.request = id;
            published.push(request);
        }
        Ok(PlannerOutput::NeedsIO(published))
    }

    fn compute(&mut self) -> VortexResult<Gathered> {
        let Some(&id) = self.stack.last() else {
            return Ok(Gathered::Batches(std::mem::take(&mut self.batches)));
        };
        let Some(item) = self.items.get_mut(&id) else {
            vortex_bail!("Gathered item {id} is missing");
        };
        if item.state() == State::Done {
            self.stack.pop();
            self.items.remove(&id);
            self.pending.retain(|_, (owner, _)| *owner != id);
            return Ok(Gathered::Output(PlannerOutput::Continue));
        }
        let output = match item {
            Item::Planner(planner) => match planner.compute()? {
                PlannerOutput::NeedsIO(batch) => {
                    return self.publish(id, batch).map(Gathered::Output);
                }
                PlannerOutput::Planner(_, child) | PlannerOutput::Spawn(_, child) => {
                    self.push(Item::Planner(child));
                    PlannerOutput::Continue
                }
                PlannerOutput::Morsel(_, morsel) => {
                    self.push(Item::Morsel(morsel));
                    PlannerOutput::Continue
                }
                PlannerOutput::Continue | PlannerOutput::Done => PlannerOutput::Continue,
            },
            Item::Morsel(morsel) => match morsel.compute()? {
                MorselOutput::NeedsIO(batch) => {
                    return self.publish(id, batch).map(Gathered::Output);
                }
                MorselOutput::Batch(array) => {
                    self.batches.push(array);
                    PlannerOutput::Continue
                }
                MorselOutput::Continue | MorselOutput::Done => PlannerOutput::Continue,
            },
        };
        Ok(Gathered::Output(output))
    }
}

//! Worker turn loop and per-action helpers.
//!
//! These functions take a `WorkerCtx` (a bundle of shared
//! references) plus a `WorkerId` and drive that worker's turn:
//! drain incoming wakes, update woken lanes, pop EV-ranked work,
//! run actions, try to steal from peers, and repeat until idle.
//!
//! Lane access:
//! - The current owner of a lane (recorded in `ctx.lane_owner`)
//!   has the right to lock its `Mutex<LaneRuntime>` for `update`
//!   and `run`. Locks are held for the duration of the operator
//!   call (potentially milliseconds), but only one lane is held at
//!   a time, and never while popping the worker's heap.
//! - Peers route wakes by enqueuing the lane addr onto the current
//!   owner's `incoming` queue. They never touch the `LaneRuntime`
//!   directly.
//! - Stealing transfers ownership via CAS on `lane_owner`. Stale
//!   heap entries on the previous owner are filtered at pop time
//!   (the entry's lane no longer hashes to me).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::Context;

use parking_lot::Mutex;

use crate::AsyncWorkId;
use crate::Batch;
use crate::BrokerGrant;
use crate::BrokerId;
use crate::BrokerProposal;
use crate::CompletedInterest;
use crate::EngineError;
use crate::EngineResult;
use crate::FakeIoRequest;
use crate::InputPortRef;
use crate::InterestId;
use crate::InterestSpec;
use crate::NoopMemoryHandle;
use crate::OperatorId;
use crate::RequirementCtx;
use crate::RequirementSet;
use crate::ResourceValue;
use crate::UpdateCtx;
use crate::WorkConstraints;
use crate::WorkCtx;
use crate::WorkProposal;
use crate::WorkStatus;

use super::DirtyCause;
use super::OperatorRuntime;
use super::TraceEvent;
use super::WorkerCtx;
use super::lane_waker;
use super::score_broker_proposal;
use super::score_operator_proposal;
use super::worker::HeapEntry;
use super::worker::LaneAddr;
use super::worker::LaneRuntime;
use super::worker::WorkerId;

/// Run one full turn for `worker_id`: drain its incoming queue, run
/// the EV-ranked admit-and-run loop until no more progress, and try
/// stealing from peers when local work runs dry. Returns `true` if
/// any forward progress happened.
pub(super) fn worker_turn(
    ctx: WorkerCtx<'_>,
    worker_id: WorkerId,
) -> EngineResult<bool> {
    let mut any_progress = false;
    let mut skipped: BTreeSet<OperatorId> = BTreeSet::new();
    loop {
        // Drain incoming wakes onto the heap.
        drain_incoming(ctx, worker_id)?;
        // Re-arm any deferred entries from earlier in this turn —
        // their constraint check may now pass after the most recent
        // run freed channel capacity / published a resource.
        rearm_deferred(ctx, worker_id);

        let action = pick_worker_action(ctx, worker_id, &skipped);
        let Some((lane_addr, proposal)) = action else {
            // Local heap exhausted. Try stealing from a peer.
            if try_steal(ctx, worker_id) {
                continue;
            }
            break;
        };
        let progress = run_action(ctx, lane_addr, proposal)?;
        if progress {
            any_progress = true;
            skipped.clear();
            // Run the global propagation pass so back-pressure from
            // the just-completed action (e.g. Limit consuming rows)
            // reaches upstream channels before we pick the next
            // action. Each op's pass is gated by an atomic swap on
            // `propagation_pending`, so concurrent workers each
            // process a disjoint set of ops without blocking.
            propagate_requirements(ctx)?;
            // Rebalance memory across channels (shrink/grow grants).
            // Only worker 0 runs this — task-wide state, no need to
            // contend.
            if worker_id.index() == 0 && rebalance_memory(ctx) {
                ctx.mark_all_dirty(DirtyCause::ExternalWake);
            }
            // The just-run action may have woken peer lanes (channel
            // pushes routed to peers' incoming) and our own lanes
            // (resource publishes, output capacity freed). Re-arm
            // deferred and drain incoming for the next iteration —
            // handled at the top of the loop.
        } else {
            // The action was admissible but made no progress (e.g.
            // operator declined to run for some local reason). Skip
            // this op for the rest of this turn so we don't loop
            // immediately on the same proposal.
            let lane = ctx.lanes[lane_addr].lock();
            let op = lane.op;
            drop(lane);
            skipped.insert(op);
        }
    }
    Ok(any_progress)
}

/// Drain `worker_id.incoming` onto the heap by running `update_lane`
/// for each woken lane. Loops until no more incoming, since
/// `update_lane`'s closures may push more lanes (resource publishes,
/// channel pushes, etc.) onto our own incoming.
fn drain_incoming(
    ctx: WorkerCtx<'_>,
    worker_id: WorkerId,
) -> EngineResult<()> {
    loop {
        let lane_addr = {
            let mut shared = ctx.workers[worker_id.index()].shared.lock();
            shared.incoming.pop_front()
        };
        let Some(lane_addr) = lane_addr else { break };
        // Confirm we still own this lane. If a peer stole it before
        // we got around to processing it, skip — the peer will
        // re-enqueue it on its own incoming.
        let owner = ctx.lane_owner[lane_addr].load(Ordering::Acquire);
        if owner != worker_id.index() {
            continue;
        }
        // If finished, just clear pending dirty + skip.
        if ctx.lane_finished[lane_addr].load(Ordering::Acquire) {
            // Drain to clear the pending bit; the returned Vec is
            // discarded by going out of scope.
            drop(ctx.lane_dirty[lane_addr].drain());
            continue;
        }
        let causes = ctx.lane_dirty[lane_addr].drain();
        if causes.is_empty() {
            continue;
        }
        update_lane(ctx, worker_id, lane_addr, causes)?;
    }
    Ok(())
}

fn rearm_deferred(ctx: WorkerCtx<'_>, worker_id: WorkerId) {
    let mut shared = ctx.workers[worker_id.index()].shared.lock();
    let deferred = std::mem::take(&mut shared.deferred);
    for entry in deferred {
        shared.work_heap.push(entry);
    }
}

/// Pop the highest-priority admissible action from the worker's heap.
/// Truly stale entries (epoch mismatch, lane stolen, lane finished)
/// are discarded; entries whose op is currently excluded by `skipped`
/// are buffered and re-pushed on return so they remain available once
/// `skipped` clears; entries whose constraints aren't currently
/// satisfied are moved to `deferred`.
fn pick_worker_action(
    ctx: WorkerCtx<'_>,
    worker_id: WorkerId,
    exclude_ops: &BTreeSet<OperatorId>,
) -> Option<(LaneAddr, WorkProposal)> {
    let worker_idx = worker_id.index();
    let shared_mutex = &ctx.workers[worker_idx].shared;
    let mut excluded_buf: Vec<HeapEntry> = Vec::new();
    let result = loop {
        let entry = {
            let mut shared = shared_mutex.lock();
            shared.work_heap.pop()
        };
        let Some(entry) = entry else { break None };
        // Lane ownership check (stale if stolen).
        let owner = ctx.lane_owner[entry.lane_addr].load(Ordering::Acquire);
        if owner != worker_idx {
            continue;
        }
        // Finished check.
        if ctx.lane_finished[entry.lane_addr].load(Ordering::Acquire) {
            continue;
        }
        // Lane lock (held briefly to read epoch + clone proposal).
        let lane = ctx.lanes[entry.lane_addr].lock();
        if entry.epoch != lane.epoch {
            continue; // stale: lane was re-updated since this entry was pushed
        }
        if exclude_ops.contains(&lane.op) {
            // Op is skipped this turn — buffer the entry and put it
            // back when we're done picking. This way the entry stays
            // alive across the skip; once `skipped` clears, the next
            // pick can return it.
            drop(lane);
            excluded_buf.push(entry);
            continue;
        }
        let Some(proposal) = lane.proposals.get(entry.proposal_idx).cloned() else {
            continue; // proposal vec shrank; defensive
        };
        let node = &ctx.nodes[lane.op.index()];
        if !constraints_satisfied(ctx, node, &proposal.constraints) {
            drop(lane);
            // Defer: re-armed at the top of the next iteration once
            // some action has freed capacity or published a resource.
            shared_mutex.lock().deferred.push(entry);
            continue;
        }
        break Some((entry.lane_addr, proposal));
    };
    // Push excluded entries back so they remain available once
    // `skipped` clears.
    if !excluded_buf.is_empty() {
        let mut shared = shared_mutex.lock();
        for entry in excluded_buf {
            shared.work_heap.push(entry);
        }
    }
    result
}

/// Try to steal a lane from a peer worker. On success, the stolen
/// lane is queued on our incoming with a fresh `ExternalWake` cause.
fn try_steal(ctx: WorkerCtx<'_>, worker_id: WorkerId) -> bool {
    let n = ctx.workers.len();
    if n <= 1 {
        return false;
    }
    let me = worker_id.index();
    // Try peers in round-robin order starting from me+1.
    for offset in 1..n {
        let peer_idx = (me + offset) % n;
        // Quick peek to see if peer has anything stealable. We pop
        // one entry under the peer's lock to learn a candidate lane
        // addr, then attempt the ownership CAS outside the lock.
        let candidate = {
            let mut peer_shared = ctx.workers[peer_idx].shared.lock();
            peer_shared.work_heap.pop()
        };
        let Some(entry) = candidate else { continue };
        let lane_addr = entry.lane_addr;
        // Verify the lane is currently owned by peer (and not, e.g.,
        // already stolen by yet another worker between our pop and
        // CAS).
        match ctx.lane_owner[lane_addr].compare_exchange(
            peer_idx,
            me,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                // Lane is ours now. The popped entry is technically
                // mine to keep, but the lane's proposals were generated
                // under peer's worldview — re-update from scratch on
                // our turn for hygiene. Mark the lane dirty + enqueue.
                ctx.lane_dirty[lane_addr].push(DirtyCause::ExternalWake);
                ctx.workers[me]
                    .shared
                    .lock()
                    .incoming
                    .push_back(lane_addr);
                ctx.trace(TraceEvent::OperatorMessage {
                    operator: None,
                    message: format!(
                        "steal: w{} took lane {} from w{}",
                        me, lane_addr, peer_idx
                    )
                    .into(),
                });
                return true;
            }
            Err(_) => {
                // Lost the race. The popped entry is now floating —
                // we lost it. That's OK; the new owner will re-update
                // the lane and emit fresh entries.
                continue;
            }
        }
    }
    false
}

fn constraints_satisfied(
    ctx: WorkerCtx<'_>,
    node: &OperatorRuntime,
    constraints: &WorkConstraints,
) -> bool {
    if let Some(input) = constraints.needs_input_data {
        let Some(input_ref) = node.inputs.get(input.index()) else {
            return false;
        };
        let Some(channel_index) = node.input_channel(input_ref.port()) else {
            return false;
        };
        if ctx.channels[channel_index].lock().peek(*input_ref).is_none() {
            return false;
        }
    }
    if constraints.needs_output_capacity {
        if !node.has_output {
            return false;
        }
        if !node
            .output_channel_indices()
            .iter()
            .all(|index| ctx.channels[*index].lock().has_capacity())
        {
            return false;
        }
    }
    true
}

/// Run `update` on the lane: produces fresh proposals, pushes heap
/// entries onto `worker_id`'s heap.
fn update_lane(
    ctx: WorkerCtx<'_>,
    worker_id: WorkerId,
    lane_addr: LaneAddr,
    causes: Vec<DirtyCause>,
) -> EngineResult<()> {
    let mut lane = ctx.lanes[lane_addr].lock();
    let operator_id = lane.op;
    let index = operator_id.index();
    let op = &ctx.nodes[index];
    let inputs = op.inputs.as_slice();
    let has_output = op.has_output;
    let op_input_channels = op.input_channels.as_slice();
    let op_output_channels = op.output_channels.as_slice();
    let channels = ctx.channels;
    let resources = ctx.resources;
    let async_work = ctx.async_work;
    let brokers = ctx.brokers;
    let trace = ctx.trace;
    let proposals = Rc::new(RefCell::new(Vec::<WorkProposal>::new()));

    let peek_input = |input: InputPortRef| {
        op_input_channels
            .get(input.port().index())
            .and_then(|opt| *opt)
            .and_then(|channel| channels[channel].lock().peek(input).cloned())
    };
    let input_finished = |input: InputPortRef| {
        op_input_channels
            .get(input.port().index())
            .and_then(|opt| *opt)
            .is_some_and(|channel| channels[channel].lock().is_finished_for(input))
    };
    let input_requirement = |input: InputPortRef| {
        op_input_channels
            .get(input.port().index())
            .and_then(|opt| *opt)
            .map(|channel| channels[channel].lock().requirement_for(input))
            .unwrap_or_default()
    };
    let output_requirement = || {
        let mut merged = RequirementSet::default();
        for channel in op_output_channels {
            merged.merge_from(&channels[*channel].lock().merged_requirement());
        }
        merged
    };
    let output_capacity = || {
        op_output_channels
            .iter()
            .all(|index| channels[*index].lock().has_capacity())
    };
    let resource_reader = move |id: &str| {
        resources
            .get(id)
            .and_then(|resource| resource.lock().value().cloned())
    };
    let mut take_async = move |id: AsyncWorkId| async_work.lock().take_completed(id);
    let mut cancel_async = {
        let metrics = Arc::clone(ctx.metrics);
        move |id: AsyncWorkId| {
            let Some((label, span)) = async_work.lock().cancel(id) else {
                return false;
            };
            metrics.lock().add_async_cancelled(&label);
            trace.lock().push(TraceEvent::AsyncCancelled {
                label: label.into(),
                span,
            });
            true
        }
    };
    let mut broker_register = move |broker_id: BrokerId,
                                    owner: OperatorId,
                                    spec: InterestSpec|
          -> InterestId {
        if let Some(broker) = brokers.get(broker_id.index()) {
            broker.lock().register(owner, spec)
        } else {
            InterestId::from_index(0)
        }
    };
    let mut broker_cancel = move |broker_id: BrokerId, interest: InterestId| {
        if let Some(broker) = brokers.get(broker_id.index()) {
            broker.lock().cancel(interest);
        }
    };
    let mut broker_take =
        move |broker_id: BrokerId, owner: OperatorId| -> Option<CompletedInterest> {
            brokers
                .get(broker_id.index())
                .and_then(|broker| broker.lock().take_completed(owner))
        };
    let trace_event_operator = operator_id;
    let mut trace_event = move |reason: String| {
        trace.lock().push(TraceEvent::OperatorMessage {
            operator: Some(trace_event_operator),
            message: reason.into(),
        });
    };
    let mut memory = NoopMemoryHandle;

    // Build a real waker tied to this lane's `DirtySignal` — when an
    // external future fires, the signal pushes
    // `DirtyCause::ExternalWake` and the next drain_incoming pass
    // picks the lane up.
    let signal = Arc::clone(&ctx.lane_dirty[lane_addr]);
    let waker = lane_waker(signal);
    let mut task_cx = Context::from_waker(&waker);

    {
        let mut proposals_borrow = proposals.borrow_mut();
        let mut update_ctx = UpdateCtx {
            operator: operator_id,
            inputs,
            has_output,
            causes: &causes,
            cx: &mut task_cx,
            peek_input: &peek_input,
            input_finished: &input_finished,
            input_requirement: &input_requirement,
            output_requirement: &output_requirement,
            output_capacity: &output_capacity,
            resource_reader: &resource_reader,
            take_async: &mut take_async,
            cancel_async: &mut cancel_async,
            broker_register: &mut broker_register,
            broker_cancel: &mut broker_cancel,
            broker_take: &mut broker_take,
            trace_event: &mut trace_event,
            memory: &mut memory,
            proposals: &mut proposals_borrow,
            propagation_pending: &ctx.propagation_pending[operator_id.index()],
        };
        if !lane.finished {
            ctx.nodes[index].node.erased().update(
                ctx.nodes[index].global.as_ref(),
                lane.local.as_mut(),
                &mut update_ctx,
            )?;
        }
    }

    drop(peek_input);
    drop(input_finished);
    drop(input_requirement);
    drop(output_requirement);
    drop(output_capacity);
    drop(resource_reader);
    drop(take_async);
    drop(cancel_async);
    drop(broker_register);
    drop(broker_cancel);
    drop(broker_take);
    drop(trace_event);

    let collected = Rc::try_unwrap(proposals)
        .map_err(|_| EngineError::message("proposal handle still shared"))?
        .into_inner();
    lane.proposals.clear();
    lane.proposals.extend(collected);
    // Bump the epoch so any pre-existing heap entries pointing into
    // the now-replaced proposal vector get discarded as stale by the
    // heap pop loop.
    lane.epoch = lane.epoch.wrapping_add(1);
    let new_epoch = lane.epoch;
    let n_proposals = lane.proposals.len();
    let entries: Vec<HeapEntry> = (0..n_proposals)
        .map(|proposal_idx| HeapEntry {
            score: score_operator_proposal(&lane.proposals[proposal_idx]),
            lane_addr,
            proposal_idx,
            epoch: new_epoch,
        })
        .collect();
    drop(lane);
    let mut shared = ctx.workers[worker_id.index()].shared.lock();
    for entry in entries {
        shared.work_heap.push(entry);
    }
    Ok(())
}

/// Run `run` on the operator. Routes wakes from channel pushes /
/// pops / resource publishes to the appropriate lane owners' incoming
/// queues.
fn run_action(
    ctx: WorkerCtx<'_>,
    lane_addr: LaneAddr,
    proposal: WorkProposal,
) -> EngineResult<bool> {
    let mut lane = ctx.lanes[lane_addr].lock();
    let operator_id = lane.op;
    let index = operator_id.index();
    let op = &ctx.nodes[index];
    let inputs = op.inputs.as_slice();
    let has_output = op.has_output;
    let op_input_channels = op.input_channels.as_slice();
    let op_output_channels = op.output_channels.as_slice();
    let channels = ctx.channels;
    let resources = ctx.resources;
    let async_work = ctx.async_work;
    let trace = ctx.trace;
    let progress_flag = Rc::new(RefCell::new(false));
    let resource_changed: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let dirty_buf: Rc<RefCell<Vec<(OperatorId, DirtyCause)>>> =
        Rc::new(RefCell::new(Vec::new()));

    let label: Arc<str> = ctx.nodes[index].node.spec().label.clone().into();
    let score = proposal.class.priority() * 1_000_000 + proposal.ev_score();
    ctx.trace(TraceEvent::OperatorRun {
        operator: operator_id,
        label,
        class: proposal.class,
        score,
    });

    let peek_input = |input: InputPortRef| {
        op_input_channels
            .get(input.port().index())
            .and_then(|opt| *opt)
            .and_then(|channel| channels[channel].lock().peek(input).cloned())
    };
    let mut pop_input = {
        let progress_flag = Rc::clone(&progress_flag);
        let dirty_buf = Rc::clone(&dirty_buf);
        move |input: InputPortRef| {
            let channel = (*op_input_channels.get(input.port().index())?)?;
            let mut ch = channels[channel].lock();
            let popped = ch.pop(input);
            if popped.is_some() {
                *progress_flag.borrow_mut() = true;
                let producers: Vec<OperatorId> = ch.spec().from.clone();
                drop(ch);
                let mut dirty = dirty_buf.borrow_mut();
                for producer in producers {
                    dirty.push((producer, DirtyCause::OutputCapacityFreed));
                }
            }
            popped
        }
    };
    let input_finished = |input: InputPortRef| {
        op_input_channels
            .get(input.port().index())
            .and_then(|opt| *opt)
            .is_some_and(|channel| channels[channel].lock().is_finished_for(input))
    };
    let input_requirement = |input: InputPortRef| {
        op_input_channels
            .get(input.port().index())
            .and_then(|opt| *opt)
            .map(|channel| channels[channel].lock().requirement_for(input))
            .unwrap_or_default()
    };
    let output_requirement = || {
        let mut merged = RequirementSet::default();
        for channel in op_output_channels {
            merged.merge_from(&channels[*channel].lock().merged_requirement());
        }
        merged
    };
    let output_capacity = || {
        op_output_channels
            .iter()
            .all(|index| channels[*index].lock().has_capacity())
    };
    let mut push_output = {
        let progress_flag = Rc::clone(&progress_flag);
        let dirty_buf = Rc::clone(&dirty_buf);
        move |batch: Batch| {
            for index in op_output_channels {
                let mut ch = channels[*index].lock();
                ch.push(operator_id, batch.clone())?;
                let consumer_ports: Vec<InputPortRef> = ch.spec().to.clone();
                drop(ch);
                let mut dirty = dirty_buf.borrow_mut();
                for input_ref in consumer_ports {
                    dirty.push((
                        input_ref.operator(),
                        DirtyCause::InputArrived {
                            port: input_ref.port(),
                        },
                    ));
                }
            }
            *progress_flag.borrow_mut() = true;
            Ok(())
        }
    };
    let mut seal_output = {
        let progress_flag = Rc::clone(&progress_flag);
        let dirty_buf = Rc::clone(&dirty_buf);
        move || {
            for index in op_output_channels {
                let mut ch = channels[*index].lock();
                // Only fully sealing the channel (last producer to
                // seal) propagates `InputSealed` to consumers. Earlier
                // per-producer seals are no-ops for the consumer side.
                let fully_sealed = ch.seal_from(operator_id)?;
                if fully_sealed {
                    let consumer_ports: Vec<InputPortRef> = ch.spec().to.clone();
                    drop(ch);
                    let mut dirty = dirty_buf.borrow_mut();
                    for input_ref in consumer_ports {
                        dirty.push((
                            input_ref.operator(),
                            DirtyCause::InputSealed {
                                port: input_ref.port(),
                            },
                        ));
                    }
                }
            }
            *progress_flag.borrow_mut() = true;
            Ok(())
        }
    };
    let resource_reader = move |id: &str| {
        resources
            .get(id)
            .and_then(|resource| resource.lock().value().cloned())
    };
    let mut resource_writer = {
        let resource_changed = Rc::clone(&resource_changed);
        let progress_flag = Rc::clone(&progress_flag);
        move |id: &str, value: ResourceValue| {
            let Some(resource) = resources.get(id) else {
                return Err(EngineError::message(format!("missing resource {id}")));
            };
            if resource.lock().publish(value) {
                resource_changed.borrow_mut().push(id.to_string());
                *progress_flag.borrow_mut() = true;
                trace.lock().push(TraceEvent::ResourcePublished {
                    id: Arc::<str>::from(id),
                });
            }
            Ok(())
        }
    };
    let mut spawn_fake_io = {
        let metrics = Arc::clone(ctx.metrics);
        let owner = ctx.nodes[index].id;
        move |request: FakeIoRequest| {
            metrics.lock().add_async_started(&request.label);
            trace.lock().push(TraceEvent::AsyncSubmitted {
                label: request.label.clone().into(),
                span: request.span,
            });
            Ok(async_work.lock().spawn(owner, request))
        }
    };
    let mut take_async = {
        let progress_flag = Rc::clone(&progress_flag);
        move |id: AsyncWorkId| {
            let taken = async_work.lock().take_completed(id);
            if taken.is_some() {
                *progress_flag.borrow_mut() = true;
            }
            taken
        }
    };
    let mut cancel_async = {
        let metrics = Arc::clone(ctx.metrics);
        move |id: AsyncWorkId| {
            let Some((label, span)) = async_work.lock().cancel(id) else {
                return false;
            };
            metrics.lock().add_async_cancelled(&label);
            trace.lock().push(TraceEvent::AsyncCancelled {
                label: label.into(),
                span,
            });
            true
        }
    };
    let trace_event_owner = ctx.nodes[index].id;
    let mut trace_event = move |reason: String| {
        trace.lock().push(TraceEvent::OperatorMessage {
            operator: Some(trace_event_owner),
            message: reason.into(),
        });
    };
    let mut memory = NoopMemoryHandle;

    let status = {
        let mut work_ctx = WorkCtx {
            inputs,
            has_output,
            peek_input: &peek_input,
            pop_input: &mut pop_input,
            input_finished: &input_finished,
            input_requirement: &input_requirement,
            output_requirement: &output_requirement,
            output_capacity: &output_capacity,
            push_output: &mut push_output,
            seal_output: &mut seal_output,
            resource_reader: &resource_reader,
            resource_writer: &mut resource_writer,
            spawn_fake_io: &mut spawn_fake_io,
            take_async: &mut take_async,
            cancel_async: &mut cancel_async,
            trace_event: &mut trace_event,
            memory: &mut memory,
            propagation_pending: &ctx.propagation_pending[operator_id.index()],
        };
        ctx.nodes[index].node.erased().run(
            ctx.nodes[index].global.as_ref(),
            lane.local.as_mut(),
            proposal.key.clone(),
            &mut work_ctx,
        )?
    };

    let made_progress = *progress_flag.borrow();
    drop(peek_input);
    drop(pop_input);
    drop(input_finished);
    drop(input_requirement);
    drop(output_requirement);
    drop(output_capacity);
    drop(push_output);
    drop(seal_output);
    drop(resource_reader);
    drop(resource_writer);
    drop(spawn_fake_io);
    drop(take_async);
    drop(cancel_async);
    drop(trace_event);
    let resource_changed_ids = std::mem::take(&mut *resource_changed.borrow_mut());
    drop(resource_changed);

    let finished_now = status == WorkStatus::Finished;
    if finished_now {
        lane.finished = true;
        lane.proposals.clear();
    }
    drop(lane);
    if finished_now {
        ctx.lane_finished[lane_addr].store(true, Ordering::Release);
    }

    // Dispatch each (op, cause) the closures recorded.
    let collected = Rc::try_unwrap(dirty_buf)
        .map_err(|_| EngineError::message("dirty buffer still shared"))?
        .into_inner();
    for (op, cause) in collected {
        ctx.mark_op_dirty(op, cause);
    }
    ctx.mark_op_dirty(operator_id, DirtyCause::ExternalWake);
    // Post-run propagation re-arm is gated on the operator's
    // `propagation_depends_on_state` opt-in. Pure-of-output operators
    // (the default) skip this; their translation only changes on T2
    // (downstream merged-requirement change) or T3 (explicit
    // `WorkCtx::request_propagation` self-call), so a per-batch
    // re-fire is wasted work.
    if ctx.nodes[operator_id.index()]
        .node
        .erased()
        .propagation_depends_on_state()
    {
        ctx.propagation_pending[operator_id.index()].store(true, Ordering::Release);
    }
    for id in resource_changed_ids {
        ctx.mark_all_dirty(DirtyCause::ResourceUpdated { id });
    }

    Ok(made_progress || finished_now)
}

/// Per-op propagation pass. Walks every operator that has its
/// `propagation_pending` flag set; for each, gathers the merged
/// output requirement and pushes it back into input channels. If a
/// channel's merged requirement changes, the producer is woken.
///
/// Returns `true` if any channel's merged requirement changed.
pub(super) fn propagate_requirements(
    ctx: WorkerCtx<'_>,
) -> EngineResult<bool> {
    // Cheap scan: if no operator has its propagation flag set, this
    // pass has nothing to do. Return before paying the
    // resource-snapshot allocation. For queries that reach a steady
    // state — e.g. q20 after `zone_sink` publishes the prune mask
    // and `ZoneMapOperator::update`'s version-poll has caught up —
    // every subsequent `propagate_requirements` call from the
    // driver / mid-worker loop hits this fast path.
    if !ctx
        .propagation_pending
        .iter()
        .any(|flag| flag.load(Ordering::Acquire))
    {
        return Ok(false);
    }
    // Resources snapshot is the same for every operator on this pass;
    // build it once outside the per-op loop. We only allocate when
    // at least one op is actually going to translate.
    let resources = ctx
        .resources
        .iter()
        .filter_map(|(id, resource)| {
            resource.lock().value().cloned().map(|value| (id.clone(), value))
        })
        .collect::<BTreeMap<_, _>>();
    let req_ctx = RequirementCtx {
        resource_reader: &|id| resources.get(id).cloned(),
    };
    let mut any_changed = false;
    // Walk in reverse-topological order (sinks first) so that flags
    // a consumer's `propagate_requirements` sets on its producer
    // are picked up later in the same pass — the demand cascade
    // completes in one traversal instead of one DAG layer per
    // propagate call.
    for &index in ctx.propagation_order {
        let op_id = ctx.nodes[index].id;
        if !ctx.propagation_pending[op_id.index()].swap(false, Ordering::AcqRel) {
            continue;
        }
        if ctx.op_lane_count[op_id.index()] == 0 {
            continue;
        }
        // Use lane 0 of this op as the propagation lane (matches old
        // behaviour). Lock briefly to invoke `propagate_requirements`.
        let lane_addr = ctx.op_lane_offset[op_id.index()];
        let mut lane = ctx.lanes[lane_addr].lock();

        let op_output_channels = ctx.nodes[index].output_channels.as_slice();
        let mut output_requirement = RequirementSet::default();
        for channel in op_output_channels {
            output_requirement.merge_from(
                &ctx.channels[*channel].lock().merged_requirement(),
            );
        }
        for slot in lane.propagate_inputs_buffer.iter_mut() {
            slot.clear();
        }
        // Split-borrow `lane` so `local` and `propagate_inputs_buffer`
        // can both be passed `&mut`.
        let LaneRuntime {
            local,
            propagate_inputs_buffer,
            ..
        } = &mut *lane;
        ctx.trace(TraceEvent::PropagateRequirementsRan { operator: op_id });
        ctx.nodes[index].node.erased().propagate_requirements(
            ctx.nodes[index].global.as_ref(),
            local.as_mut(),
            &output_requirement,
            propagate_inputs_buffer.as_mut_slice(),
            &req_ctx,
        )?;
        let inputs_snapshot = ctx.nodes[index].inputs.as_slice();
        let op_input_channels = ctx.nodes[index].input_channels.as_slice();
        for input_index in 0..inputs_snapshot.len() {
            let requirement = &mut lane.propagate_inputs_buffer[input_index];
            if requirement.is_empty() {
                continue;
            }
            let input = inputs_snapshot[input_index];
            let Some(channel_index) = op_input_channels[input_index] else {
                return Err(EngineError::message("missing input channel"));
            };
            trace_requirement_change(ctx, requirement, index);
            let mut channel = ctx.channels[channel_index].lock();
            if channel.set_requirement(input, requirement)? {
                any_changed = true;
                let producers: Vec<OperatorId> = channel.spec().from.clone();
                let is_spmc = channel.spec().topology == crate::ChannelTopology::Spmc;
                drop(channel);
                for producer in &producers {
                    ctx.propagation_pending[producer.index()]
                        .store(true, Ordering::Release);
                    ctx.mark_op_dirty(*producer, DirtyCause::OutputRequirementChanged);
                }
                if is_spmc {
                    ctx.trace(TraceEvent::RequirementSpmcMerged);
                }
                ctx.trace
                    .lock()
                    .push(TraceEvent::RequirementChanged { input });
            }
        }
        drop(lane);
    }
    Ok(any_changed)
}

fn trace_requirement_change(
    ctx: WorkerCtx<'_>,
    requirement: &RequirementSet,
    node_index: usize,
) {
    let required = requirement.required_count_from_zero();
    if required > 0 {
        ctx.trace
            .lock()
            .push(TraceEvent::RequirementRootRequired { rows: required });
    }
    if let Some((start, end)) =
        super::contiguous_presence(requirement, crate::RowDemand::NotNeeded)
    {
        let mut trace = ctx.trace.lock();
        trace.push(TraceEvent::RequirementNotNeeded { start, end });
        let label = &ctx.nodes[node_index].node.spec().label;
        if label.contains("aggregate") {
            trace.push(TraceEvent::AggregateLimitSealed);
        }
        if label.contains("late_dynamic_filter") {
            trace.push(TraceEvent::LateFilterMarkedSuffix);
        }
    }
}

/// Memory rebalance. Returns `true` if any channel's grant changed.
pub(super) fn rebalance_memory(ctx: WorkerCtx<'_>) -> bool {
    let used = ctx
        .channels
        .iter()
        .map(|c| c.lock().retained_bytes())
        .sum::<usize>()
        .saturating_add(ctx.async_work.lock().retained_bytes());
    ctx.metrics.lock().observe_memory_bytes(used);
    let mut changed = false;
    if used >= ctx.memory_limit_bytes {
        for channel_mutex in ctx.channels {
            let mut channel = channel_mutex.lock();
            let min_bytes = channel.spec().buffer.min_bytes();
            if channel.set_current_capacity(min_bytes) {
                changed = true;
            }
        }
        if changed {
            ctx.trace(TraceEvent::MemoryGrantShrink);
        }
        return changed;
    }
    if used <= ctx.memory_limit_bytes / 2 {
        for channel_mutex in ctx.channels {
            let mut channel = channel_mutex.lock();
            let target_bytes = channel.spec().buffer.target_bytes();
            if channel.set_current_capacity(target_bytes) {
                changed = true;
            }
        }
        if changed {
            ctx.trace(TraceEvent::MemoryGrantGrow);
        }
    }
    changed
}

/// Hide unused-warning if the broker_proposals field isn't yet used
/// for something more sophisticated than `proposals()` calls.
#[allow(dead_code)]
fn _broker_proposals_anchor(_: &Mutex<Vec<BrokerProposal>>) {}

#[allow(dead_code)]
fn _grant_anchor(_: BrokerGrant) {}

#[allow(dead_code)]
fn _lane_anchor(_: &Mutex<LaneRuntime>) {}

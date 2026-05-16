use serde::{Deserialize, Serialize};
use vortex_trace_format::framing::RecordKind;
use vortex_trace_format::record::{TracePayload, TraceRecord};
use vortex_trace_format::snapshot::TurnSnapshot;

use crate::error::ReplayError;
use crate::file::read_record_header;

/// Cheap per-turn aggregates for the scrub bar.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimelineSummary {
    /// Total event count per turn (excludes the snapshot record).
    pub events_per_turn: Vec<u32>,
    /// Per-turn counts of variant categories used by the scrub bar
    /// heatmap.
    pub category_counts: Vec<TurnCategoryCounts>,
    pub total_events: u64,
    pub total_snapshots: u64,
    pub turns: u32,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct TurnCategoryCounts {
    pub turn_phase: u32,
    pub demand: u32,
    pub heap: u32,
    pub channel: u32,
    pub async_io: u32,
    pub broker: u32,
    pub resource_other: u32,
}

/// Internal index of a trace file. Built once at load time.
pub(crate) struct TraceIndex {
    /// `(turn, byte_offset_of_record_header)` for every snapshot.
    pub snapshot_offsets: Vec<(u32, u64)>,
    /// `byte_offset_of_first_event_record_header` for each turn.
    pub turn_event_offsets: Vec<u64>,
    /// Number of events recorded in each turn (not counting the
    /// snapshot record).
    pub events_per_turn: Vec<u32>,
}

impl TraceIndex {
    pub fn build(
        bytes: &[u8],
        start_offset: u64,
        _operator_count: usize,
    ) -> Result<(Self, TimelineSummary), ReplayError> {
        let mut snapshot_offsets = Vec::new();
        let mut turn_event_offsets: Vec<u64> = Vec::new();
        let mut events_per_turn: Vec<u32> = Vec::new();
        let mut category_counts: Vec<TurnCategoryCounts> = Vec::new();

        let mut offset = start_offset;
        let len = bytes.len() as u64;

        // Turn lifecycle: a turn opens at `TurnBegin` and closes at
        // the next `Snapshot` record. `events_per_turn` is appended
        // to at close time; the entry covers every event from
        // `TurnBegin` through `TurnEnd` inclusive.
        let mut in_turn = false;
        let mut events_this_turn: u32 = 0;
        let mut counts_this_turn = TurnCategoryCounts::default();

        let mut total_events = 0u64;
        let mut total_snapshots = 0u64;

        while offset < len {
            let (rec_header, payload_offset) = read_record_header(bytes, offset)?;
            // record_len covers kind byte + payload
            let payload_len = (rec_header.record_len as u64).saturating_sub(1);
            let payload_end = payload_offset + payload_len;
            if payload_end > len {
                return Err(ReplayError::Truncated {
                    offset: payload_end,
                });
            }

            match rec_header.kind {
                RecordKind::Event => {
                    let payload = &bytes[payload_offset as usize..payload_end as usize];
                    let record: TraceRecord = postcard::from_bytes(payload)?;

                    if record.payload.is_turn_begin() && !in_turn {
                        in_turn = true;
                        turn_event_offsets.push(offset);
                        events_this_turn = 0;
                        counts_this_turn = TurnCategoryCounts::default();
                    } else if !in_turn {
                        // Event before any TurnBegin: open implicitly.
                        in_turn = true;
                        turn_event_offsets.push(offset);
                    }

                    events_this_turn = events_this_turn.saturating_add(1);
                    bump_category(&mut counts_this_turn, &record.payload);
                    total_events += 1;
                }
                RecordKind::Snapshot => {
                    let payload = &bytes[payload_offset as usize..payload_end as usize];
                    let snap: TurnSnapshot = postcard::from_bytes(payload)?;
                    snapshot_offsets.push((snap.turn, offset));
                    total_snapshots += 1;
                    if in_turn {
                        events_per_turn.push(events_this_turn);
                        category_counts.push(counts_this_turn);
                        events_this_turn = 0;
                        counts_this_turn = TurnCategoryCounts::default();
                        in_turn = false;
                    }
                }
            }

            offset = payload_end;
        }

        // Flush trailing partial turn if the file ended without a
        // closing snapshot.
        if in_turn && events_this_turn > 0 {
            events_per_turn.push(events_this_turn);
            category_counts.push(counts_this_turn);
        }

        let summary = TimelineSummary {
            events_per_turn: events_per_turn.clone(),
            category_counts,
            total_events,
            total_snapshots,
            turns: events_per_turn.len() as u32,
        };

        Ok((
            Self {
                snapshot_offsets,
                turn_event_offsets,
                events_per_turn,
            },
            summary,
        ))
    }

    /// Find the snapshot whose recorded turn is the largest
    /// `<= target_turn`, returning its byte offset.
    ///
    /// Returns `None` if no such snapshot exists (target is before
    /// the first snapshot).
    pub fn snapshot_at_or_before(&self, target_turn: u32) -> Option<(u32, u64)> {
        let mut best: Option<(u32, u64)> = None;
        for &(turn, offset) in &self.snapshot_offsets {
            if turn <= target_turn {
                if best.map(|(t, _)| t).unwrap_or(0) <= turn {
                    best = Some((turn, offset));
                }
            } else {
                break;
            }
        }
        best
    }

    pub fn first_event_offset(&self, turn: u32) -> Option<u64> {
        self.turn_event_offsets.get(turn as usize).copied()
    }
}

fn bump_category(c: &mut TurnCategoryCounts, p: &TracePayload) {
    match p {
        TracePayload::TurnBegin
        | TracePayload::PhaseBegin { .. }
        | TracePayload::PhaseEnd { .. }
        | TracePayload::TurnEnd { .. }
        | TracePayload::WorkerStepBegin { .. }
        | TracePayload::WorkerStepEnd { .. } => c.turn_phase = c.turn_phase.saturating_add(1),
        TracePayload::RequirementChanged { .. }
        | TracePayload::RequirementSpmcMerged { .. }
        | TracePayload::RequirementRootRequired { .. }
        | TracePayload::RequirementNotNeeded { .. }
        | TracePayload::AggregateLimitSealed { .. }
        | TracePayload::LateFilterMarkedSuffix { .. } => c.demand = c.demand.saturating_add(1),
        TracePayload::ProposalsCleared { .. }
        | TracePayload::ProposalEnqueued { .. }
        | TracePayload::ProposalPopped { .. }
        | TracePayload::OperatorRun { .. } => c.heap = c.heap.saturating_add(1),
        TracePayload::ChannelPush { .. }
        | TracePayload::ChannelPop { .. }
        | TracePayload::ChannelSeal { .. } => c.channel = c.channel.saturating_add(1),
        TracePayload::AsyncSubmitted { .. }
        | TracePayload::AsyncWake { .. }
        | TracePayload::AsyncCancelled { .. } => c.async_io = c.async_io.saturating_add(1),
        TracePayload::BrokerProposalEnqueued { .. }
        | TracePayload::BrokerSubmit { .. }
        | TracePayload::BrokerPull { .. }
        | TracePayload::SubstrateComplete { .. } => c.broker = c.broker.saturating_add(1),
        TracePayload::ResourcePublished { .. }
        | TracePayload::MemoryGrantChanged { .. }
        | TracePayload::OperatorMessage { .. } => {
            c.resource_other = c.resource_other.saturating_add(1)
        }
    }
}

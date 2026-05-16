use vortex_trace_format::framing::RecordKind;
use vortex_trace_format::record::TraceRecord;
use vortex_trace_format::snapshot::TurnSnapshot;

use crate::error::ReplayError;
use crate::file::{TraceFile, read_record_header};
use crate::state::SchedulerState;

/// A timeline coordinate.
///
/// `event_in_turn = 0` means "before any event of this turn"
/// (immediately after the snapshot for the prior turn was consumed).
/// `event_in_turn = K` means "after K events of this turn"; the
/// final event of a turn is `TurnEnd`, so reaching
/// `events_in_turn(T)` means the entire turn has been replayed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct TimelinePos {
    pub turn: u32,
    pub event_in_turn: u32,
}

impl TimelinePos {
    fn as_tuple(self) -> (u32, u32) {
        (self.turn, self.event_in_turn)
    }
}

/// Forward seeks within this many turns of the current position
/// step incrementally. Larger jumps go through a snapshot restore.
const FORWARD_STEP_TURN_BUDGET: u32 = 4;

pub struct ReplayCursor<'a> {
    file: &'a TraceFile,
    position: TimelinePos,
    state: SchedulerState,
    /// Byte offset of the *next* record to read. Records strictly
    /// before this offset have been applied to `state` (or absorbed
    /// by a snapshot we loaded directly).
    next_offset: u64,
}

impl<'a> ReplayCursor<'a> {
    pub fn new(file: &'a TraceFile) -> Self {
        let state = SchedulerState::from_header(file.header());
        let next_offset = file.index().first_event_offset(0).unwrap_or(0);
        Self {
            file,
            position: TimelinePos::default(),
            state,
            next_offset,
        }
    }

    pub fn position(&self) -> TimelinePos {
        self.position
    }

    pub fn state(&self) -> &SchedulerState {
        &self.state
    }

    /// Seek to `target`.
    ///
    /// Strategy:
    /// - If we're already there, do nothing.
    /// - If `target` is forward of the current position and within
    ///   `FORWARD_STEP_TURN_BUDGET`, step incrementally — no
    ///   snapshot decode, just one record read + apply per event.
    /// - Otherwise restore from the nearest preceding snapshot and
    ///   walk forward to `target`.
    ///
    /// Drag-scrubbing the timeline is therefore O(events traversed)
    /// over the whole drag, not O(events_in_turn) per
    /// `step_forward` call.
    pub fn seek(&mut self, target: TimelinePos) -> Result<(), ReplayError> {
        let turns = self.file.turns();
        if target.turn >= turns {
            return Err(ReplayError::InvalidPosition(target));
        }
        let events_in_target_turn = self.file.events_in_turn(target.turn);
        if target.event_in_turn > events_in_target_turn {
            return Err(ReplayError::InvalidPosition(target));
        }
        if target == self.position {
            return Ok(());
        }

        let going_forward = target.as_tuple() > self.position.as_tuple();
        let large_jump = !going_forward
            || target.turn.saturating_sub(self.position.turn) > FORWARD_STEP_TURN_BUDGET;

        if large_jump {
            self.restore_to_turn_start(target.turn)?;
        }

        // Step forward until we hit `target`. The advance loop
        // skips snapshot records (they roll the position to the
        // next turn) and applies events.
        while self.position != target {
            if !self.advance_one_record()? {
                // EOF reached before target — should not happen
                // because we validated bounds above.
                return Err(ReplayError::InvalidPosition(target));
            }
        }
        Ok(())
    }

    /// Advance the cursor by exactly one *event*, consuming any
    /// snapshot records that bracket turns. Returns the applied
    /// event, or `None` at end of trace.
    ///
    /// This is the O(1) primitive: one record-header read, one
    /// postcard parse of the payload, and one in-place state mutate.
    /// When the next record is a snapshot the cursor reloads state
    /// from it (cheap parse, one allocation, but always bounded by
    /// the snapshot size — see `advance_one_record`).
    pub fn step_forward(&mut self) -> Result<Option<TraceRecord>, ReplayError> {
        loop {
            let kind = self.peek_next_kind();
            match kind {
                None => return Ok(None),
                Some(RecordKind::Snapshot) => {
                    if !self.advance_one_record()? {
                        return Ok(None);
                    }
                    continue;
                }
                Some(RecordKind::Event) => {
                    let bytes = self.file.bytes();
                    let (rec_header, payload_offset) =
                        read_record_header(bytes, self.next_offset)?;
                    let payload_len = (rec_header.record_len as u64).saturating_sub(1);
                    let payload_end = payload_offset + payload_len;
                    let payload = &bytes[payload_offset as usize..payload_end as usize];
                    let record: TraceRecord = postcard::from_bytes(payload)?;
                    self.state.apply(&record);
                    self.next_offset = payload_end;
                    self.position.event_in_turn = self.position.event_in_turn.saturating_add(1);
                    return Ok(Some(record));
                }
            }
        }
    }

    /// Read up to `count` consecutive events ahead of the cursor
    /// without mutating cursor state. Used by the event-log panel.
    pub fn peek_forward(&self, count: u32) -> Result<Vec<TraceRecord>, ReplayError> {
        let bytes = self.file.bytes();
        let len = bytes.len() as u64;
        let mut out = Vec::with_capacity(count as usize);
        let mut offset = self.next_offset;
        while (out.len() as u32) < count && offset < len {
            let (rec_header, payload_offset) = read_record_header(bytes, offset)?;
            let payload_len = (rec_header.record_len as u64).saturating_sub(1);
            let payload_end = payload_offset + payload_len;
            if rec_header.kind == RecordKind::Event {
                let payload = &bytes[payload_offset as usize..payload_end as usize];
                let record: TraceRecord = postcard::from_bytes(payload)?;
                out.push(record);
            }
            offset = payload_end;
        }
        Ok(out)
    }

    // ---- helpers --------------------------------------------------

    fn peek_next_kind(&self) -> Option<RecordKind> {
        let bytes = self.file.bytes();
        if self.next_offset >= bytes.len() as u64 {
            return None;
        }
        read_record_header(bytes, self.next_offset)
            .ok()
            .map(|(h, _)| h.kind)
    }

    /// Advance past exactly one record. Mutates state, position, and
    /// next_offset. Returns false at EOF.
    fn advance_one_record(&mut self) -> Result<bool, ReplayError> {
        let bytes = self.file.bytes();
        if self.next_offset >= bytes.len() as u64 {
            return Ok(false);
        }
        let (rec_header, payload_offset) = read_record_header(bytes, self.next_offset)?;
        let payload_len = (rec_header.record_len as u64).saturating_sub(1);
        let payload_end = payload_offset + payload_len;
        match rec_header.kind {
            RecordKind::Event => {
                let payload = &bytes[payload_offset as usize..payload_end as usize];
                let record: TraceRecord = postcard::from_bytes(payload)?;
                self.state.apply(&record);
                self.next_offset = payload_end;
                self.position.event_in_turn = self.position.event_in_turn.saturating_add(1);
            }
            RecordKind::Snapshot => {
                // Snapshots are authoritative at turn boundaries.
                // The recorder samples real scheduler state at the
                // end of every turn; some fields (lane bitsets,
                // memory_used_bytes, channel buffered reduction,
                // output_requirement merges) aren't reconstructible
                // from events alone. Reload state from the snapshot
                // so sequential stepping converges with
                // snapshot-restore seek to byte-identical state.
                let payload = &bytes[payload_offset as usize..payload_end as usize];
                let snap: TurnSnapshot = postcard::from_bytes(payload)?;
                self.state.load_snapshot(&snap);
                self.next_offset = payload_end;
                self.position.turn = snap.turn.saturating_add(1);
                self.position.event_in_turn = 0;
            }
        }
        Ok(true)
    }

    /// Reset state and cursor to the start of `target_turn` (i.e.
    /// `(turn=target_turn, event_in_turn=0)`), using the nearest
    /// preceding snapshot as a keyframe.
    fn restore_to_turn_start(&mut self, target_turn: u32) -> Result<(), ReplayError> {
        self.state = SchedulerState::from_header(self.file.header());
        let bytes = self.file.bytes();
        let index = self.file.index();

        // Snapshot at turn N closes turn N; loading it positions us
        // at the start of turn N+1.
        let needed_snapshot_turn = target_turn.checked_sub(1);
        let snapshot = needed_snapshot_turn.and_then(|t| index.snapshot_at_or_before(t));

        if let Some((snap_turn, offset)) = snapshot {
            let (rec_header, payload_offset) = read_record_header(bytes, offset)?;
            if rec_header.kind != RecordKind::Snapshot {
                return Err(ReplayError::InvalidRecordKind(rec_header.kind as u8));
            }
            let payload_len = (rec_header.record_len as u64).saturating_sub(1);
            let payload_end = payload_offset + payload_len;
            let payload = &bytes[payload_offset as usize..payload_end as usize];
            let snap: TurnSnapshot = postcard::from_bytes(payload)?;
            self.state.load_snapshot(&snap);

            // First event of (snap_turn + 1) is where we resume.
            let resume_turn = snap_turn + 1;
            self.position = TimelinePos {
                turn: resume_turn,
                event_in_turn: 0,
            };
            self.next_offset = index
                .first_event_offset(resume_turn)
                .unwrap_or(payload_end);
        } else {
            // No usable snapshot: start from t=0.
            self.position = TimelinePos::default();
            self.next_offset = index.first_event_offset(0).unwrap_or(0);
        }

        // If target_turn is *before* where we just landed (impossible
        // unless the file is malformed), bail out.
        if self.position.turn > target_turn {
            return Err(ReplayError::InvalidPosition(TimelinePos {
                turn: target_turn,
                event_in_turn: 0,
            }));
        }

        // Walk forward to the start of target_turn by consuming any
        // intervening turns' events + their closing snapshot.
        while self.position.turn < target_turn {
            if !self.advance_one_record()? {
                return Err(ReplayError::InvalidPosition(TimelinePos {
                    turn: target_turn,
                    event_in_turn: 0,
                }));
            }
        }
        Ok(())
    }
}

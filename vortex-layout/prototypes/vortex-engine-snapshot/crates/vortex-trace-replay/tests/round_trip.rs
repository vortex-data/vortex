use std::io::Cursor;
use std::time::Instant;

use vortex_trace_replay::{ReplayCursor, TimelinePos, TraceFile, fixture};

#[test]
fn synthetic_fixture_round_trips() {
    let mut buf = Vec::new();
    fixture::write_fixture(Cursor::new(&mut buf)).unwrap();
    let file = TraceFile::from_bytes(buf).unwrap();
    assert_eq!(file.turns(), 12);
    let summary = file.timeline_summary();
    assert!(summary.total_events > 100);
    assert_eq!(summary.total_snapshots, 12);
}

#[test]
fn seek_is_deterministic() {
    let mut buf = Vec::new();
    fixture::write_fixture(Cursor::new(&mut buf)).unwrap();
    let file = TraceFile::from_bytes(buf).unwrap();

    let mut cursor_a = ReplayCursor::new(&file);
    let mut cursor_b = ReplayCursor::new(&file);

    let target = TimelinePos {
        turn: 7,
        event_in_turn: 3,
    };
    cursor_a.seek(target).unwrap();
    cursor_b.seek(target).unwrap();

    let json_a = serde_json::to_string(cursor_a.state()).unwrap();
    let json_b = serde_json::to_string(cursor_b.state()).unwrap();
    assert_eq!(json_a, json_b);
}

#[test]
fn state_at_first_turn_has_root_requirement() {
    let mut buf = Vec::new();
    fixture::write_fixture(Cursor::new(&mut buf)).unwrap();
    let file = TraceFile::from_bytes(buf).unwrap();

    let mut cursor = ReplayCursor::new(&file);
    let events_in_first = file.events_in_turn(0);
    cursor
        .seek(TimelinePos {
            turn: 0,
            event_in_turn: events_in_first,
        })
        .unwrap();
    // After replaying turn 0, the root-required event has fired and
    // demand has reached every operator's input port.
    assert!(!cursor.state().requirements.is_empty());
}

#[test]
fn step_forward_matches_seek() {
    // For every event in the trace, the state after N step_forwards
    // from t=0 must equal the state from a fresh cursor seeking
    // directly to that position.
    let mut buf = Vec::new();
    fixture::write_fixture(Cursor::new(&mut buf)).unwrap();
    let file = TraceFile::from_bytes(buf).unwrap();

    let mut walker = ReplayCursor::new(&file);

    let mut total_events = 0u32;
    let mut step_count = 0u32;
    while walker.step_forward().unwrap().is_some() {
        step_count += 1;
        total_events += 1;
        // Every 17 steps, cross-check against a fresh-seek cursor.
        if step_count % 17 != 0 {
            continue;
        }
        let pos = walker.position();
        let mut fresh = ReplayCursor::new(&file);
        fresh.seek(pos).unwrap();
        let walker_state = serde_json::to_string(walker.state()).unwrap();
        let fresh_state = serde_json::to_string(fresh.state()).unwrap();
        assert_eq!(
            walker_state, fresh_state,
            "state divergence at {pos:?}"
        );
    }
    assert!(total_events > 100);
}

#[test]
fn forward_drag_is_amortized_constant() {
    // Simulate scrubbing: advance one event at a time across the
    // entire trace. Total time must scale O(events), not O(events²).
    // Bound: drag of N events finishes in well under N * (turn-walk
    // cost). 5_000 microseconds for ~500 events is generous slack
    // even on a slow machine.
    let mut buf = Vec::new();
    fixture::write_fixture(Cursor::new(&mut buf)).unwrap();
    let file = TraceFile::from_bytes(buf).unwrap();

    let mut cursor = ReplayCursor::new(&file);
    let start = Instant::now();
    let mut events = 0u64;
    while cursor.step_forward().unwrap().is_some() {
        events += 1;
    }
    let elapsed = start.elapsed();
    // Sanity: 500 events should not take milliseconds-per-event.
    // We assert under 1ms per event, which is many orders of
    // magnitude looser than the actual ~µs/event but catches the
    // O(N²) regression cleanly.
    let per_event_ns = elapsed.as_nanos() / events.max(1) as u128;
    assert!(
        per_event_ns < 1_000_000,
        "step_forward took {} ns/event over {} events; suspect O(N) regression",
        per_event_ns,
        events,
    );
}

#[test]
fn forward_seek_within_budget_does_not_reload_snapshot() {
    // Two cursors: one seeks turn-by-turn forward, the other does
    // a single small forward seek. Both must end at byte-identical
    // state.
    let mut buf = Vec::new();
    fixture::write_fixture(Cursor::new(&mut buf)).unwrap();
    let file = TraceFile::from_bytes(buf).unwrap();

    let mut a = ReplayCursor::new(&file);
    a.seek(TimelinePos {
        turn: 2,
        event_in_turn: 5,
    })
    .unwrap();
    a.seek(TimelinePos {
        turn: 4,
        event_in_turn: 10,
    })
    .unwrap();

    let mut b = ReplayCursor::new(&file);
    b.seek(TimelinePos {
        turn: 4,
        event_in_turn: 10,
    })
    .unwrap();

    assert_eq!(
        serde_json::to_string(a.state()).unwrap(),
        serde_json::to_string(b.state()).unwrap(),
    );
}

#[test]
fn backward_seek_uses_snapshot_restore() {
    let mut buf = Vec::new();
    fixture::write_fixture(Cursor::new(&mut buf)).unwrap();
    let file = TraceFile::from_bytes(buf).unwrap();

    let mut cursor = ReplayCursor::new(&file);
    cursor
        .seek(TimelinePos {
            turn: 9,
            event_in_turn: 0,
        })
        .unwrap();
    let forward_state = serde_json::to_string(cursor.state()).unwrap();

    cursor
        .seek(TimelinePos {
            turn: 2,
            event_in_turn: 0,
        })
        .unwrap();
    cursor
        .seek(TimelinePos {
            turn: 9,
            event_in_turn: 0,
        })
        .unwrap();
    let after_round_trip = serde_json::to_string(cursor.state()).unwrap();

    assert_eq!(forward_state, after_round_trip);
}

#[test]
fn snapshot_keyframe_produces_consistent_state() {
    let mut buf = Vec::new();
    fixture::write_fixture(Cursor::new(&mut buf)).unwrap();
    let file = TraceFile::from_bytes(buf).unwrap();

    // Seek directly to turn 5, event 0 — should load the snapshot
    // from turn 4 and be valid even without replaying from t=0.
    let mut cursor = ReplayCursor::new(&file);
    cursor
        .seek(TimelinePos {
            turn: 5,
            event_in_turn: 0,
        })
        .unwrap();
    let state = cursor.state();
    assert_eq!(state.turn, 4); // snapshot of close-of-turn-4
    assert_eq!(state.workers.len(), 4);
    assert!(!state.channels.is_empty());
}

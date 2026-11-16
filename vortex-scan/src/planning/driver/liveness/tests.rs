// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;

use vortex_error::VortexResult;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;

use super::*;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::morsel::MorselOutput;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::tests::scripted;
use crate::planning::tests::scripted::MorselStep;
use crate::planning::tests::scripted::RecordingIoSource;
use crate::planning::tests::scripted::ScriptedMorsel;
use crate::planning::tests::scripted::array_of;
use crate::planning::tests::scripted::scope;

/// Ordered record of batches and of pinned values being dropped.
#[derive(Clone, Default)]
struct Events(Arc<Mutex<Vec<String>>>);

impl Events {
    fn push(&self, event: impl Into<String>) {
        self.0.lock().push(event.into());
    }

    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.0.lock())
    }
}

/// A pinned value that records when it is dropped.
struct Tracked(&'static str, Events);

impl Drop for Tracked {
    fn drop(&mut self) {
        self.1.push(format!("drop {}", self.0));
    }
}

fn pin_rows(liveness: &Liveness, events: &Events, name: &'static str, rows: Range<u64>) -> bool {
    liveness.pin(Lifetime::Rows(rows), Tracked(name, events.clone()))
}

#[test]
fn pin_lives_until_its_rows_are_released() {
    let (liveness, events) = (Liveness::new(), Events::default());
    let claim = liveness.claim(0..8);
    assert!(pin_rows(&liveness, &events, "a", 0..8));
    assert!(events.take().is_empty());
    drop(claim);
    assert_eq!(events.take(), ["drop a"]);
    assert_eq!(liveness.pinned(), 0);
}

#[test]
fn overlapping_claims_keep_a_pin_until_both_release() {
    let (liveness, events) = (Liveness::new(), Events::default());
    let first = liveness.claim(0..8);
    let second = liveness.claim(4..12);
    pin_rows(&liveness, &events, "a", 4..8);
    drop(first);
    assert!(events.take().is_empty());
    drop(second);
    assert_eq!(events.take(), ["drop a"]);
}

#[test]
fn restricting_a_claim_drops_pins_only_outside_the_remaining_rows() {
    let (liveness, events) = (Liveness::new(), Events::default());
    let mut claim = liveness.claim(0..12);
    pin_rows(&liveness, &events, "head", 0..4);
    pin_rows(&liveness, &events, "middle", 4..8);
    pin_rows(&liveness, &events, "tail", 8..12);
    claim.restrict_to(4..8);
    let mut dropped = events.take();
    dropped.sort();
    assert_eq!(dropped, ["drop head", "drop tail"]);
    assert_eq!(claim.rows(), &(4..8));
    drop(claim);
    assert_eq!(events.take(), ["drop middle"]);
}

#[test]
fn a_pin_spanning_live_rows_survives_a_partial_release() {
    let (liveness, events) = (Liveness::new(), Events::default());
    let left = liveness.claim(0..4);
    let right = liveness.claim(4..8);
    pin_rows(&liveness, &events, "a", 2..6);
    drop(left);
    assert!(events.take().is_empty());
    assert!(liveness.is_claimed(&(2..6)));
    drop(right);
    assert_eq!(events.take(), ["drop a"]);
}

#[test]
fn pinning_unclaimed_rows_drops_the_value_at_once() {
    let (liveness, events) = (Liveness::new(), Events::default());
    let _claim = liveness.claim(0..4);
    assert!(!pin_rows(&liveness, &events, "a", 4..8));
    assert_eq!(events.take(), ["drop a"]);
    assert_eq!(liveness.pinned(), 0);
}

#[test]
fn a_file_lifetime_lasts_until_no_claim_is_live() {
    let (liveness, events) = (Liveness::new(), Events::default());
    assert!(!liveness.pin(Lifetime::File, Tracked("early", events.clone())));
    assert_eq!(events.take(), ["drop early"]);

    let first = liveness.claim(0..4);
    let second = liveness.claim(100..104);
    assert!(liveness.pin(Lifetime::File, Tracked("file", events.clone())));
    drop(first);
    assert!(events.take().is_empty());
    drop(second);
    assert_eq!(events.take(), ["drop file"]);
}

/// Cuts its scope into chunks and emits one single-batch morsel per chunk, pinning a value to
/// each chunk first. Its remaining rows are the chunks it has not emitted.
struct Chunker {
    chunks: VecDeque<Range<u64>>,
    end: u64,
    liveness: Liveness,
    events: Events,
    names: VecDeque<&'static str>,
    log: scripted::Log,
}

impl IoConsumer for Chunker {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {
        unreachable!("the chunker publishes no IO")
    }
}

impl Planner for Chunker {
    fn state(&self) -> State {
        if self.chunks.is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let chunk = self.chunks.pop_front().vortex_expect("a chunk is left");
        let name = self.names.pop_front().vortex_expect("a name is left");
        assert!(pin_rows(&self.liveness, &self.events, name, chunk.clone()));
        let morsel = ScriptedMorsel::new(
            "morsel",
            vec![MorselStep::Compute(MorselOutput::Batch(array_of(
                chunk.end - chunk.start,
            )))],
            &self.log,
        );
        Ok(PlannerOutput::Morsel(scope(chunk), Box::new(morsel)))
    }

    fn remaining_rows(&self) -> Option<Range<u64>> {
        Some(self.chunks.front().map_or(self.end, |chunk| chunk.start)..self.end)
    }
}

fn chunker(liveness: &Liveness, events: &Events) -> Box<dyn Planner> {
    Box::new(Chunker {
        chunks: VecDeque::from([0..4, 4..8, 8..12]),
        end: 12,
        liveness: liveness.clone(),
        events: events.clone(),
        names: VecDeque::from(["0", "1", "2"]),
        log: scripted::Log::default(),
    })
}

fn io() -> Arc<dyn IoSource> {
    Arc::new(RecordingIoSource::default())
}

#[test]
fn the_driver_drops_each_chunks_values_once_its_morsel_retires() -> VortexResult<()> {
    let (liveness, events) = (Liveness::new(), Events::default());
    let mut run = Run::new().with_step_limit(100);
    run.admit_with_liveness(
        chunker(&liveness, &events),
        scope(0..12),
        io(),
        liveness.clone(),
    );
    loop {
        match run.advance()? {
            Progress::Batch(batch) => events.push(format!("batch {:?}", batch.scope.rows)),
            Progress::RootDone(_) | Progress::Idle => break,
            Progress::Waiting | Progress::Spawn(..) => unreachable!("no IO and no spawns"),
        }
    }
    assert_eq!(
        events.take(),
        [
            "batch 0..4",
            "drop 0",
            "batch 4..8",
            "drop 1",
            "batch 8..12",
            "drop 2"
        ]
    );
    assert_eq!(liveness.pinned(), 0);
    assert!(!liveness.is_claimed(&(0..12)));
    Ok(())
}

#[test]
fn cancelling_a_root_drops_the_values_its_work_held() -> VortexResult<()> {
    let (liveness, events) = (Liveness::new(), Events::default());
    let mut run = Run::new().with_step_limit(100);
    let root = run.admit_with_liveness(
        chunker(&liveness, &events),
        scope(0..12),
        io(),
        liveness.clone(),
    );
    assert!(matches!(run.advance()?, Progress::Batch(_)));
    assert!(events.take().is_empty());
    run.cancel(root);
    assert_eq!(events.take(), ["drop 0"]);
    assert!(!liveness.is_claimed(&(0..12)));
    Ok(())
}

#[test]
fn a_scan_claim_keeps_values_after_the_work_finishes() -> VortexResult<()> {
    let (liveness, events) = (Liveness::new(), Events::default());
    let scan = liveness.claim(0..12);
    assert!(liveness.pin(Lifetime::File, Tracked("dictionary", events.clone())));
    let mut run = Run::new().with_step_limit(100);
    run.admit_with_liveness(
        chunker(&liveness, &events),
        scope(0..12),
        io(),
        liveness.clone(),
    );
    while !matches!(run.advance()?, Progress::Idle) {}
    assert!(events.take().is_empty());
    drop(scan);
    let mut dropped = events.take();
    dropped.sort();
    assert_eq!(dropped, ["drop 0", "drop 1", "drop 2", "drop dictionary"]);
    Ok(())
}

/// Publishes one fetch and waits for it, reporting remaining rows that reach past the scope it
/// was admitted with.
struct Widening {
    published: bool,
}

impl IoConsumer for Widening {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Widening {
    fn state(&self) -> State {
        if self.published {
            State::Waiting
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        self.published = true;
        Ok(PlannerOutput::NeedsIO(vec![IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(0),
            target: IoTarget::range(0, 4),
        }]))
    }

    fn remaining_rows(&self) -> Option<Range<u64>> {
        Some(4..16)
    }
}

#[test]
fn remaining_rows_are_clamped_to_the_claim() -> VortexResult<()> {
    let liveness = Liveness::new();
    let mut run = Run::new().with_step_limit(100);
    run.admit_with_liveness(
        Box::new(Widening { published: false }),
        scope(0..8),
        io(),
        liveness.clone(),
    );
    // The planner narrows, then parks on its fetch, which the source never answers.
    assert!(matches!(run.advance()?, Progress::Waiting));
    assert!(!liveness.is_claimed(&(0..4)));
    assert!(liveness.is_claimed(&(4..8)));
    assert!(!liveness.is_claimed(&(8..16)));
    Ok(())
}

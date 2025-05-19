use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use derivative::Derivative;
use parking_lot::Mutex;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::VortexExpect;

use crate::segments::SegmentId;

#[derive(Derivative)]
#[derivative(PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SequenceId {
    id: Vec<usize>,
    #[derivative(
        PartialEq = "ignore",
        PartialOrd = "ignore",
        Ord = "ignore",
        Hash = "ignore",
        Debug = "ignore"
    )]
    universe: Arc<Mutex<SequenceUniverse>>,
}

impl SequenceId {
    /// Create a new Sequence from root. No ordering guarantees exists for separate instances created
    /// using this method.
    pub fn root() -> SequencePointer {
        SequencePointer {
            pointer: vec![0],
            universe: Default::default(),
        }
    }

    /// Create a sub sequence starting from this [SequenceId]. If Self has an id of [1, 2],
    /// This method would return its first child [1, 2, 0] as well as the [SequencePointer]
    /// to create siblings [1, 2, [1, ..)]
    pub fn descend(self) -> (Self, SequencePointer) {
        let mut id = self.id.clone();
        id.push(0);

        let mut pointer = SequencePointer {
            pointer: id,
            universe: self.universe.clone(),
        };

        let first_child = pointer.advance();
        (first_child, pointer)
    }

    /// Await until all id's in this universe that are strictly less than self are dropped.
    /// Returns a monotonically increasing [SegmentId]
    pub async fn collapse(self) -> SegmentId {
        WaitSequenceFuture {
            id: self.id.clone(),
            universe: self.universe.clone(),
        }
        .await
    }

    /// This is intentionally not pub. [SequencePointer::advance] is the only allowed way to create
    /// [SequenceId] instances
    fn new(id: Vec<usize>, universe: Arc<Mutex<SequenceUniverse>>) -> Self {
        // NOTE: This is the only place we construct a SequenceId, and
        // we immediately add it to the universe.
        let res = Self { id, universe };
        res.universe.lock().add(&res);
        res
    }
}

impl Drop for SequenceId {
    fn drop(&mut self) {
        self.universe.lock().remove(self);
    }
}

pub struct SequencePointer {
    pointer: Vec<usize>,
    universe: Arc<Mutex<SequenceUniverse>>,
}

impl SequencePointer {
    pub fn advance(&mut self) -> SequenceId {
        let id = self.pointer.clone();

        // increment x.y.z -> x.y.(z + 1)
        let last = self.pointer.last_mut();
        let last = last.vortex_expect("must have at least one element");
        *last += 1;

        SequenceId::new(id, self.universe.clone())
    }
}

#[derive(Default)]
struct SequenceUniverse {
    active: BTreeSet<Vec<usize>>,
    wakers: HashMap<Vec<usize>, Waker>,
    next_segment_id: SegmentId,
}

impl SequenceUniverse {
    fn add(&mut self, sequence_id: &SequenceId) {
        self.active.insert(sequence_id.id.clone());
    }

    fn remove(&mut self, sequence_id: &SequenceId) {
        self.active.remove(&sequence_id.id);
        let Some(first) = self.active.first() else {
            // last sequence finished, we must have no pending futures
            assert!(self.wakers.is_empty(), "all wakers must have been removed");
            return;
        };
        if let Some(waker) = self.wakers.remove(first) {
            waker.wake_by_ref();
        }
    }

    pub fn next_segment_id(&mut self) -> SegmentId {
        let res = self.next_segment_id;
        self.next_segment_id = SegmentId::from(*res + 1);
        res
    }
}

struct WaitSequenceFuture {
    id: Vec<usize>,
    universe: Arc<Mutex<SequenceUniverse>>,
}

impl Future for WaitSequenceFuture {
    type Output = SegmentId;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut guard = self.universe.lock();
        let current_first = guard
            .active
            .first()
            .cloned()
            .vortex_expect("if we have a future, we must have at least one active sequence");
        if self.id == current_first {
            return Poll::Ready(guard.next_segment_id());
        }
        guard.wakers.insert(self.id.clone(), cx.waker().clone());
        Poll::Pending
    }
}

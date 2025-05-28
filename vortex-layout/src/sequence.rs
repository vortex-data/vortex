use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::VortexExpect;

use crate::segments::SegmentId;

pub struct SequenceId {
    id: Vec<usize>,
    universe: Arc<Mutex<SequenceUniverse>>,
}

impl PartialEq for SequenceId {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for SequenceId {}

impl PartialOrd for SequenceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl Ord for SequenceId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl Hash for SequenceId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl fmt::Debug for SequenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SequenceId").field("id", &self.id).finish()
    }
}

impl SequenceId {
    /// Create a new Sequence from root. No ordering guarantees exists for separate instances created
    /// using this method.
    pub fn root() -> SequencePointer {
        SequencePointer(SequenceId::new(vec![0], Default::default()))
    }

    /// Create a sub sequence starting from this [SequenceId]. If Self has an id of [1, 2],
    /// This method would return its first child [1, 2, 0] as well as the [SequencePointer]
    /// to create siblings [1, 2, [1, ..)]
    pub fn descend(self) -> SequencePointer {
        let mut id = self.id.clone();
        id.push(0);
        SequencePointer(SequenceId::new(id, self.universe.clone()))
    }

    /// Await until all id's in this universe that are strictly less than self are dropped.
    /// Returns a monotonically increasing [SegmentId]
    pub async fn collapse(self) -> SegmentId {
        WaitSequenceFuture(self).await
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
        let waker = self.universe.lock().remove(self);
        if let Some(w) = waker {
            w.wake();
        }
    }
}

// TODO(os): make this !Send to prevent holding this over await points
pub struct SequencePointer(SequenceId);

impl SequencePointer {
    pub fn advance(&mut self) -> SequenceId {
        let mut next_id = self.0.id.clone();

        // increment x.y.z -> x.y.(z + 1)
        let last = next_id.last_mut();
        let last = last.vortex_expect("must have at least one element");
        *last += 1;
        let next_sibling = SequenceId::new(next_id, self.0.universe.clone());
        std::mem::replace(&mut self.0, next_sibling)
    }

    pub fn downgrade(self) -> SequenceId {
        self.0
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

    fn remove(&mut self, sequence_id: &SequenceId) -> Option<Waker> {
        self.active.remove(&sequence_id.id);
        let Some(first) = self.active.first() else {
            // last sequence finished, we must have no pending futures
            assert!(self.wakers.is_empty(), "all wakers must have been removed");
            return None;
        };
        self.wakers.remove(first)
    }

    pub fn next_segment_id(&mut self) -> SegmentId {
        let res = self.next_segment_id;
        self.next_segment_id = SegmentId::from(*res + 1);
        res
    }
}

struct WaitSequenceFuture(SequenceId);

impl Future for WaitSequenceFuture {
    type Output = SegmentId;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut guard = self.0.universe.lock();
        let current_first = guard
            .active
            .first()
            .cloned()
            .vortex_expect("if we have a future, we must have at least one active sequence");
        if self.0.id == current_first {
            return Poll::Ready(guard.next_segment_id());
        }
        guard.wakers.insert(self.0.id.clone(), cx.waker().clone());
        Poll::Pending
    }
}

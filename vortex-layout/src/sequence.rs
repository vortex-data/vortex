use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;
use vortex_error::VortexExpect;
use vortex_utils::aliases::hash_map::HashMap;

use crate::segments::SegmentId;

/// A hierarchical sequence identifier that exists within a shared universe.
///
/// SequenceIds form a collision-free universe where each ID is represented as a vector
/// of indices (e.g., `[0, 1, 2]`). The API design prevents collisions by only allowing
/// new IDs to be created through controlled advancement or descent operations.
///
/// # Hierarchy and Ordering
///
/// IDs are hierarchical and lexicographically ordered:
/// - `[0]` < `[0, 0]` < `[0, 1]` < `[1]` < `[1, 0]`
/// - A parent ID like `[0, 1]` can spawn children `[0, 1, 0]`, `[0, 1, 1]`, etc.
/// - Sibling IDs are created by advancing: `[0, 0]` → `[0, 1]` → `[0, 2]`
///
/// # Drop Ordering
///
/// When a SequenceId is dropped, it may wake futures waiting for ordering guarantees.
/// The `collapse()` method leverages this to provide deterministic ordering of
/// recursively created sequence IDs.
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
        Some(self.cmp(other))
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
    /// Creates a new root sequence universe starting with ID `[0]`.
    ///
    /// Each call to `root()` creates an independent universe with no ordering
    /// guarantees between separate root instances. Within a single universe,
    /// all IDs are strictly ordered.
    pub fn root() -> SequencePointer {
        SequencePointer(SequenceId::new(vec![0], Default::default()))
    }

    /// Creates a child sequence by descending one level in the hierarchy.
    ///
    /// If this SequenceId has ID `[1, 2]`, this method creates the first child
    /// `[1, 2, 0]` and returns a `SequencePointer` that can generate siblings
    /// `[1, 2, 1]`, `[1, 2, 2]`, etc.
    ///
    /// # Ownership
    ///
    /// This method consumes `self`, as the parent ID is no longer needed once
    /// we've descended to work with its children.
    pub fn descend(self) -> SequencePointer {
        let mut id = self.id.clone();
        id.push(0);
        SequencePointer(SequenceId::new(id, self.universe.clone()))
    }

    /// Waits until all SequenceIds with IDs lexicographically smaller than this one are dropped.
    ///
    /// This async method provides ordering guarantees by ensuring all "prior" sequences
    /// in the universe have been dropped before returning. Combined with the collision-free
    /// API, this guarantees that for this universe no sequences lexicographically smaller than
    /// this one will ever be created again.
    ///
    /// # Ordering Guarantee
    ///
    /// Once `collapse()` returns, you can be certain that:
    /// - All sequences with smaller IDs have been dropped
    /// - No new sequences with smaller IDs can ever be created (due to collision prevention)
    /// - The returned `SegmentId` is monotonically increasing within this universe
    ///
    /// # Use Cases
    ///
    /// This is particularly useful for ordering recursively created work:
    /// - Recursive algorithms that spawn child tasks
    /// - Ensuring deterministic processing order across concurrent operations  
    /// - Converting hierarchical sequence identifiers to linear segment identifiers
    ///
    /// # Returns
    ///
    /// A monotonically increasing `SegmentId` that can be used for ordered storage
    /// or processing. Each successful collapse within a universe produces a larger
    /// `SegmentId` than the previous one.
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

/// A pointer that can advance through sibling sequence IDs.
///
/// SequencePointer is the only mechanism for creating new SequenceIds within
/// a universe.
pub struct SequencePointer(SequenceId);

impl SequencePointer {
    /// Advances to the next sibling sequence and returns the current one.
    ///
    /// # Ownership
    ///
    /// This method requires `&mut self` because it advances the internal state
    /// to point to the next sibling position.
    pub fn advance(&mut self) -> SequenceId {
        let mut next_id = self.0.id.clone();

        // increment x.y.z -> x.y.(z + 1)
        let last = next_id.last_mut();
        let last = last.vortex_expect("must have at least one element");
        *last += 1;
        let next_sibling = SequenceId::new(next_id, self.0.universe.clone());
        std::mem::replace(&mut self.0, next_sibling)
    }

    /// Converts this pointer into its current SequenceId, consuming the pointer.
    ///
    /// This method is useful when you want to access the current SequenceId
    /// without advancing to the next sibling. Once downgraded, you cannot
    /// create additional siblings from this pointer.
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

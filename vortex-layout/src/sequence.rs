// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::Stream;
use futures::StreamExt;
use parking_lot::Mutex;
use pin_project_lite::pin_project;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStream;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

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
    /// The [`SequenceId`] once all other segment IDs before it have been dropped. The caller can hold
    /// onto the sequence ID essentially as a lock on future calls to [`SequenceId::collapse`]
    /// in order to perform ordered operations.
    pub async fn collapse(&mut self) {
        WaitSequenceFuture(self).await;
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
#[derive(Debug)]
pub struct SequencePointer(SequenceId);

impl SequencePointer {
    /// Splits this pointer into two, where the second is strictly greater than the first.
    pub fn split(mut self) -> (SequencePointer, SequencePointer) {
        (self.split_off(), self)
    }

    /// Split off a pointer to appear before the current one.
    ///
    /// The current pointer is advanced to the next sibling, and we return a new pointer.
    pub fn split_off(&mut self) -> SequencePointer {
        // Advance ourselves to the next sibling, and return a new pointer to the previous one.
        self.advance().descend()
    }

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
}

struct WaitSequenceFuture<'a>(&'a mut SequenceId);

impl Future for WaitSequenceFuture<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut guard = self.0.universe.lock();
        let current_first = guard
            .active
            .first()
            .cloned()
            .vortex_expect("if we have a future, we must have at least one active sequence");
        if self.0.id == current_first {
            guard.wakers.remove(&self.0.id);
            return Poll::Ready(());
        }

        guard.wakers.insert(self.0.id.clone(), cx.waker().clone());
        Poll::Pending
    }
}

/// If the future itself is dropped, we don't want to orphan the waker
impl Drop for WaitSequenceFuture<'_> {
    fn drop(&mut self) {
        self.0.universe.lock().wakers.remove(&self.0.id);
    }
}

pub trait SequentialStream: Stream<Item = VortexResult<(SequenceId, ArrayRef)>> {
    fn dtype(&self) -> &DType;
}

pub type SendableSequentialStream = Pin<Box<dyn SequentialStream + Send>>;

impl SequentialStream for SendableSequentialStream {
    fn dtype(&self) -> &DType {
        (**self).dtype()
    }
}

pub trait SequentialStreamExt: SequentialStream {
    // not named boxed to prevent clashing with StreamExt
    fn sendable(self) -> SendableSequentialStream
    where
        Self: Sized + Send + 'static,
    {
        Box::pin(self)
    }
}

impl<S: SequentialStream> SequentialStreamExt for S {}

pin_project! {
    pub struct SequentialStreamAdapter<S> {
        dtype: DType,
        #[pin]
        inner: S,
    }
}

impl<S> SequentialStreamAdapter<S> {
    pub fn new(dtype: DType, inner: S) -> Self {
        Self { dtype, inner }
    }
}

impl<S> SequentialStream for SequentialStreamAdapter<S>
where
    S: Stream<Item = VortexResult<(SequenceId, ArrayRef)>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<S> Stream for SequentialStreamAdapter<S>
where
    S: Stream<Item = VortexResult<(SequenceId, ArrayRef)>>,
{
    type Item = VortexResult<(SequenceId, ArrayRef)>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let array = futures::ready!(this.inner.poll_next(cx));
        if let Some(Ok((_, array))) = array.as_ref() {
            assert_eq!(
                array.dtype(),
                this.dtype,
                "Sequential stream of {} got chunk of {}.",
                array.dtype(),
                this.dtype
            );
        }

        Poll::Ready(array)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

pub trait SequentialArrayStreamExt: ArrayStream {
    /// Converts the stream to a [`SendableSequentialStream`].
    fn sequenced(self, mut pointer: SequencePointer) -> SendableSequentialStream
    where
        Self: Sized + Send + 'static,
    {
        Box::pin(SequentialStreamAdapter::new(
            self.dtype().clone(),
            StreamExt::map(self, move |item| {
                item.map(|array| (pointer.advance(), array))
            }),
        ))
    }
}

impl<S: ArrayStream> SequentialArrayStreamExt for S {}

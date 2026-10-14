// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row demand that travels alongside a child stream written under a nullable container.
//!
//! Container writers store their children unmasked: a struct field holds a value in rows where the
//! struct is null, and a list's elements include the elements of null lists. A [`DemandChannel`]
//! carries, for every row of such a child stream, whether the logical column contains it, so that
//! writers recording statistics aggregate only the demanded rows (see
//! [`DynAccumulator::accumulate_demanded`]).
//!
//! The container publishes the demand on each chunk before sending the chunk to its child, and the
//! child takes the demand on each chunk it receives in stream order. Demand is matched to rows by
//! position, not by chunk, so strategies between the two may re-chunk the stream freely.
//!
//! [`DynAccumulator::accumulate_demanded`]: vortex_array::aggregate_fn::DynAccumulator::accumulate_demanded

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_error::VortexExpect;
use vortex_mask::Mask;

/// The number of demand masks a channel buffers before a consumer claims it.
///
/// A consumer claims the channel when its write starts, before it receives any rows, so only the
/// chunks already in flight are published before the claim. A channel that buffers more than this
/// has no consumer, and drops its demand rather than holding a mask for every row of the column.
const MAX_UNCLAIMED_MASKS: usize = 64;

/// The producing side of the row demand on a child stream.
///
/// Clones publish to the same channel, which a single consumer claims with [`Self::claim`].
#[derive(Clone, Default)]
pub struct DemandChannel(Arc<Mutex<DemandState>>);

#[derive(Default)]
struct DemandState {
    /// Published masks not yet taken, in row order.
    masks: VecDeque<Mask>,
    /// Whether a consumer has claimed the channel.
    claimed: bool,
    /// Whether the demand is no longer known, so no consumer can rely on it.
    lost: bool,
}

impl DemandChannel {
    /// Creates a channel with no published demand.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a channel whose demand is unknown, for children whose rows no writer can track,
    /// such as each run of dictionary codes.
    pub fn lost() -> Self {
        let channel = Self::new();
        channel.lose();
        channel
    }

    /// Publishes the demand on the next `demand.len()` rows of the child stream.
    ///
    /// Must be called before those rows are sent to the child.
    pub fn push(&self, demand: Mask) {
        let mut state = self.0.lock();
        if state.lost {
            return;
        }
        if !state.claimed && state.masks.len() >= MAX_UNCLAIMED_MASKS {
            state.lose();
            return;
        }
        state.masks.push_back(demand);
    }

    /// Marks the demand on the remaining rows as unknown, e.g. because the producer's own demand
    /// was lost.
    pub fn lose(&self) {
        self.0.lock().lose();
    }

    /// Claims the channel for a consumer.
    ///
    /// Returns `None` if the channel was already claimed or its demand was lost.
    pub fn claim(&self) -> Option<DemandReceiver> {
        let mut state = self.0.lock();
        if state.claimed || state.lost {
            return None;
        }
        state.claimed = true;
        Some(DemandReceiver(Arc::clone(&self.0)))
    }
}

impl DemandState {
    fn lose(&mut self) {
        self.lost = true;
        self.masks.clear();
    }
}

/// The row demand on a write, claimed from its context with
/// [`LayoutWriterContext::claim_demand`](crate::LayoutWriterContext::claim_demand).
pub enum ClaimedDemand {
    /// Every row of the write is demanded.
    All,
    /// The demand on each row of the write, taken chunk by chunk in stream order.
    Rows(DemandReceiver),
    /// Which rows are demanded is not known, e.g. because another writer already claimed the
    /// demand. Statistics over the write's rows would not be exact.
    Unknown,
}

impl ClaimedDemand {
    /// Whether every row of the write is demanded.
    pub fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    /// A channel on which a container publishes the demand on a child that shares or derives from
    /// these rows. The channel starts lost if this demand is unknown.
    pub fn child_channel(&self) -> DemandChannel {
        match self {
            Self::Unknown => DemandChannel::lost(),
            Self::All | Self::Rows(_) => DemandChannel::new(),
        }
    }

    /// Takes the demand on the next `len` rows, or `None` if it is unknown.
    ///
    /// Must be called once per chunk, in stream order.
    pub fn take(&self, len: usize) -> Option<Mask> {
        match self {
            Self::All => Some(Mask::new_true(len)),
            Self::Rows(receiver) => receiver.take(len),
            Self::Unknown => None,
        }
    }
}

/// The consuming side of the row demand on a child stream, returned by [`DemandChannel::claim`].
pub struct DemandReceiver(Arc<Mutex<DemandState>>);

impl DemandReceiver {
    /// Takes the demand on the next `len` rows of the child stream.
    ///
    /// Must be called once per received chunk, in stream order. Returns `None` once the demand is
    /// lost, including when fewer rows were published than are taken.
    pub fn take(&self, len: usize) -> Option<Mask> {
        let mut state = self.0.lock();
        if state.lost {
            return None;
        }
        let published: usize = state.masks.iter().map(Mask::len).sum();
        if published < len {
            state.lose();
            return None;
        }

        let mut parts = Vec::new();
        let mut remaining = len;
        while remaining > 0 {
            let front = state
                .masks
                .pop_front()
                .vortex_expect("published enough rows");
            if front.len() <= remaining {
                remaining -= front.len();
                parts.push(front);
            } else {
                parts.push(front.slice(..remaining));
                state.masks.push_front(front.slice(remaining..));
                remaining = 0;
            }
        }

        match parts.len() {
            0 => Some(Mask::new_true(0)),
            1 => parts.pop(),
            _ => Some(Mask::concat(parts.iter()).vortex_expect("masks concatenate")),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;

    use super::*;

    #[test]
    fn take_realigns_demand_to_new_chunks() {
        let channel = DemandChannel::new();
        let receiver = channel.claim().expect("unclaimed");
        channel.push(Mask::from_iter([true, false, true]));
        channel.push(Mask::from_iter([false, true]));

        assert_eq!(receiver.take(2), Some(Mask::from_iter([true, false])));
        assert_eq!(receiver.take(3), Some(Mask::from_iter([true, false, true])));
        assert_eq!(receiver.take(0), Some(Mask::new_true(0)));
    }

    #[test]
    fn a_channel_has_a_single_consumer() {
        let channel = DemandChannel::new();
        let shared = channel.clone();
        assert!(channel.claim().is_some());
        assert!(shared.claim().is_none());
    }

    #[test]
    fn taking_unpublished_rows_loses_the_demand() {
        let channel = DemandChannel::new();
        let receiver = channel.claim().expect("unclaimed");
        channel.push(Mask::new_true(2));

        assert_eq!(receiver.take(3), None);
        channel.push(Mask::new_true(2));
        assert_eq!(receiver.take(1), None);
    }

    #[test]
    fn an_unclaimed_channel_drops_its_demand() {
        let channel = DemandChannel::new();
        for _ in 0..=MAX_UNCLAIMED_MASKS {
            channel.push(Mask::new_true(1));
        }
        assert!(channel.claim().is_none());
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! When a file's in-memory scan state can be dropped, decided by the work still live in the
//! driver.
//!
//! A [`Liveness`] tracks which root rows of one file live work may still touch. A [`Run`] that
//! admits a root with [`admit_with_liveness`] gives every planner and morsel descended from it a
//! [`RowClaim`] over its scope's rows: the claim is taken when the work is spawned, narrowed to
//! the work's [`remaining_rows`], and released when the work retires or is cancelled.
//!
//! Anything decoded for those rows, such as arrays, dictionary values, or buffers, can be pinned
//! to the rows it serves with [`Liveness::pin`]. The pin holds the value until no claim covers
//! any of its rows, and then drops it. A cache that wants to find such a value again keeps a weak
//! reference to it: the pin is the only strong reference that keeps it alive.
//!
//! [`Run`]: super::Run
//! [`admit_with_liveness`]: super::Run::admit_with_liveness
//! [`remaining_rows`]: crate::planning::planner::Planner::remaining_rows

mod coverage;
#[cfg(test)]
mod tests;

use std::any::Any;
use std::collections::BTreeMap;
use std::ops::Bound::Excluded;
use std::ops::Bound::Included;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_error::VortexExpect;
use vortex_utils::aliases::hash_map::HashMap;

use self::coverage::Coverage;

/// The root rows whose work may still need a pinned value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// Only work whose claim overlaps these rows can need the value.
    Rows(Range<u64>),
    /// No tighter bound is known: keep the value while any work on the file is live.
    File,
}

/// Which root rows of one file live work still claims, and the values pinned to them.
///
/// Clones share state, so the runs of every thread that scans the file can report to one
/// liveness. Nothing is pinned while no claim is live: hold a [`claim`](Self::claim) over the
/// whole scan before admitting its roots when values must outlive the gap between one root
/// finishing and the next being admitted.
#[derive(Clone, Default)]
pub struct Liveness {
    state: Arc<Mutex<State>>,
}

#[derive(Default)]
struct State {
    coverage: Coverage,
    pins: HashMap<u64, Pin>,
    next_pin: u64,
    /// Pins with a row lifetime, by the end of their rows.
    expiry: BTreeMap<u64, Vec<u64>>,
}

struct Pin {
    rows: Option<Range<u64>>,
    /// Never read: dropping it is the point.
    _value: Box<dyn Any + Send>,
}

impl Liveness {
    /// Creates liveness for one file with nothing claimed or pinned.
    pub fn new() -> Self {
        Self::default()
    }

    /// Claims `rows` until the returned claim is dropped.
    pub fn claim(&self, rows: Range<u64>) -> RowClaim {
        RowClaim::new(self.clone(), rows)
    }

    /// Whether live work claims any of `rows`.
    pub fn is_claimed(&self, rows: &Range<u64>) -> bool {
        self.state.lock().coverage.overlaps(rows)
    }

    /// Holds `value` until no live claim covers any row of `lifetime`.
    ///
    /// Returns `false`, dropping `value` at once, when no claim covers those rows already.
    pub fn pin<T: Send + 'static>(&self, lifetime: Lifetime, value: T) -> bool {
        let mut state = self.state.lock();
        let rows = match lifetime {
            Lifetime::Rows(rows) => {
                if !state.coverage.overlaps(&rows) {
                    drop(state);
                    drop(value);
                    return false;
                }
                Some(rows)
            }
            Lifetime::File => {
                if state.coverage.is_empty() {
                    drop(state);
                    drop(value);
                    return false;
                }
                None
            }
        };
        let id = state.next_pin;
        state.next_pin += 1;
        if let Some(rows) = &rows {
            state.expiry.entry(rows.end).or_default().push(id);
        }
        state.pins.insert(
            id,
            Pin {
                rows,
                _value: Box::new(value),
            },
        );
        true
    }

    /// The number of values currently pinned.
    pub fn pinned(&self) -> usize {
        self.state.lock().pins.len()
    }

    fn add(&self, rows: &Range<u64>) {
        self.state.lock().coverage.add(rows);
    }

    /// Releases one claim on `rows`. Evicted values are dropped after the lock is released, so
    /// a value's destructor may use this liveness.
    fn release(&self, rows: &Range<u64>) {
        let evicted = self.state.lock().release(rows);
        drop(evicted);
    }
}

impl State {
    fn release(&mut self, rows: &Range<u64>) -> Vec<Pin> {
        let gaps = self.coverage.release(rows);
        if self.coverage.is_empty() {
            self.expiry.clear();
            return self.pins.drain().map(|(_, pin)| pin).collect();
        }
        let mut evicted = Vec::new();
        for gap in gaps {
            let ends: Vec<_> = self
                .expiry
                .range((Excluded(gap.start), Included(gap.end)))
                .map(|(&end, _)| end)
                .collect();
            for end in ends {
                let ids = self.expiry.get_mut(&end).vortex_expect("expiry exists");
                ids.retain(|id| {
                    let pin = self.pins.get(id).vortex_expect("pin exists");
                    if pin
                        .rows
                        .as_ref()
                        .is_some_and(|rows| rows.start >= gap.start)
                    {
                        evicted.push(self.pins.remove(id).vortex_expect("pin exists"));
                        false
                    } else {
                        true
                    }
                });
                if ids.is_empty() {
                    self.expiry.remove(&end);
                }
            }
        }
        evicted
    }
}

/// An owned claim on root rows that live work may still touch.
///
/// The driver holds one for every planner and morsel of a root admitted with liveness, so
/// planners do not handle claims themselves. An engine can hold one over a whole scan so values
/// pinned by its first roots survive until its last root is admitted. Dropping the claim releases
/// its remaining rows.
#[must_use = "dropping a row claim releases its rows"]
pub struct RowClaim {
    liveness: Liveness,
    rows: Range<u64>,
}

impl RowClaim {
    fn new(liveness: Liveness, rows: Range<u64>) -> Self {
        assert!(rows.start <= rows.end, "claim rows must be ordered");
        liveness.add(&rows);
        Self { liveness, rows }
    }

    /// The rows this claim still covers.
    pub fn rows(&self) -> &Range<u64> {
        &self.rows
    }

    /// The liveness this claim reports to.
    pub fn liveness(&self) -> &Liveness {
        &self.liveness
    }

    /// Releases every row outside `rows`, which must lie within the claim.
    ///
    /// # Panics
    ///
    /// Panics if `rows` is unordered or not contained in the claim.
    pub fn restrict_to(&mut self, rows: Range<u64>) {
        assert!(
            self.rows.start <= rows.start && rows.start <= rows.end && rows.end <= self.rows.end,
            "remaining rows must be contained in the claim"
        );
        if self.rows == rows {
            return;
        }
        let evicted = {
            let mut state = self.liveness.state.lock();
            let mut evicted = state.release(&(self.rows.start..rows.start));
            evicted.extend(state.release(&(rows.end..self.rows.end)));
            evicted
        };
        self.rows = rows;
        drop(evicted);
    }

    /// Splits the claim's rows before `end` off into a claim of their own, keeping the rest.
    ///
    /// The rows stay claimed throughout, so this never takes the liveness lock.
    ///
    /// # Panics
    ///
    /// Panics if `end` is outside the claim.
    pub fn split_front(&mut self, end: u64) -> RowClaim {
        assert!(
            self.rows.start <= end && end <= self.rows.end,
            "the split must lie within the claim"
        );
        let front = RowClaim {
            liveness: self.liveness.clone(),
            rows: self.rows.start..end,
        };
        self.rows.start = end;
        front
    }
}

impl Drop for RowClaim {
    fn drop(&mut self) {
        self.liveness.release(&self.rows);
    }
}

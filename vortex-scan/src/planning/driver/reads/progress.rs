// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Owned row claims travel with queued and running work.

use std::collections::BTreeMap;
use std::ops::Bound::Excluded;
use std::ops::Bound::Included;
use std::ops::Range;

use vortex_error::VortexExpect;

use super::FileReads;
use super::ReadEntry;
use super::ReadState;

/// Disjoint spans with the number of owners covering each span. Splitting ownership does not
/// change coverage; only sharing and releasing rows touch this index.
#[derive(Default)]
pub(super) struct Claims(BTreeMap<u64, Span>);

#[derive(Clone, Copy)]
struct Span {
    end: u64,
    owners: usize,
}

impl Claims {
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn split_at(&mut self, row: u64) {
        if let Some((&start, &span)) = self.0.range(..row).next_back()
            && row < span.end
        {
            self.0.get_mut(&start).vortex_expect("span exists").end = row;
            self.0.insert(row, span);
        }
    }

    fn join_at(&mut self, row: u64) {
        if let Some((&start, &left)) = self.0.range(..row).next_back()
            && let Some(&right) = self.0.get(&row)
            && left.end == row
            && left.owners == right.owners
        {
            self.0.get_mut(&start).vortex_expect("span exists").end = right.end;
            self.0.remove(&row);
        }
    }

    fn add(&mut self, rows: &Range<u64>) {
        if rows.is_empty() {
            return;
        }
        self.split_at(rows.start);
        self.split_at(rows.end);
        let starts: Vec<_> = self
            .0
            .range(rows.clone())
            .map(|(&start, _)| start)
            .collect();
        let mut cursor = rows.start;
        for start in starts {
            if cursor < start {
                self.0.insert(
                    cursor,
                    Span {
                        end: start,
                        owners: 1,
                    },
                );
            }
            let span = self.0.get_mut(&start).vortex_expect("span exists");
            span.owners += 1;
            cursor = span.end;
        }
        if cursor < rows.end {
            self.0.insert(
                cursor,
                Span {
                    end: rows.end,
                    owners: 1,
                },
            );
        }
        self.join_at(rows.start);
        self.join_at(rows.end);
    }

    /// Returns newly uncovered gaps, including adjacent rows already released by other parts.
    fn release(&mut self, rows: &Range<u64>) -> Vec<Range<u64>> {
        if rows.is_empty() {
            return Vec::new();
        }
        self.split_at(rows.start);
        self.split_at(rows.end);
        let starts: Vec<_> = self
            .0
            .range(rows.clone())
            .map(|(&start, _)| start)
            .collect();
        let mut removed = Vec::new();
        for start in starts {
            let span = self.0.get_mut(&start).vortex_expect("span exists");
            span.owners -= 1;
            if span.owners == 0 {
                self.0.remove(&start);
                removed.push(start);
            }
        }
        self.join_at(rows.start);
        self.join_at(rows.end);
        let mut gaps = Vec::new();
        for row in removed {
            let start = self
                .0
                .range(..row)
                .next_back()
                .map_or(0, |(_, span)| span.end);
            let end = self
                .0
                .range(row..)
                .next()
                .map_or(u64::MAX, |(&start, _)| start);
            let gap = start..end;
            if gaps.last() != Some(&gap) {
                gaps.push(gap);
            }
        }
        gaps
    }
}

impl ReadState {
    /// Return evicted entries so transport cancellation runs outside the state lock.
    fn release(&mut self, rows: &Range<u64>) -> Vec<ReadEntry> {
        let gaps = self.claims.release(rows);
        if self.claims.is_empty() {
            self.expiry.clear();
            return self.entries.drain().map(|(_, entry)| entry).collect();
        }
        let mut evicted = Vec::new();
        for gap in gaps {
            let ends: Vec<_> = self
                .expiry
                .range((Excluded(gap.start), Included(gap.end)))
                .map(|(&end, _)| end)
                .collect();
            for end in ends {
                let targets = self.expiry.get_mut(&end).vortex_expect("expiry exists");
                targets.retain(|target| {
                    let entry = self.entries.get(target).vortex_expect("read exists");
                    if entry
                        .rows
                        .as_ref()
                        .is_some_and(|rows| rows.start >= gap.start)
                    {
                        evicted.push(self.entries.remove(target).vortex_expect("read exists"));
                        false
                    } else {
                        true
                    }
                });
                if targets.is_empty() {
                    self.expiry.remove(&end);
                }
            }
        }
        evicted
    }
}

/// An owned reservation for rows that may still read from a file execution.
///
/// Move the claim with work through queues and planner handoffs. Split it before handing off
/// disjoint children; dropping it marks its remaining rows finished or cancelled. Output batches
/// must not own claims, since retaining a batch does not imply any future file reads.
#[must_use = "dropping a row claim releases its rows"]
pub struct RowClaim {
    file: FileReads,
    rows: Range<u64>,
}

impl RowClaim {
    pub(super) fn new(file: FileReads, rows: Range<u64>) -> Self {
        assert!(rows.start <= rows.end, "claim rows must be ordered");
        file.state.lock().claims.add(&rows);
        Self { file, rows }
    }

    /// The file-local rows this claim still reserves.
    pub fn rows(&self) -> &Range<u64> {
        &self.rows
    }

    /// Transfers the prefix ending at `end` into a child claim, keeping the suffix in `self`.
    /// Coverage is unchanged, so splitting requires no lock or registration updates.
    /// Panics if `end` is outside this claim's range.
    pub fn split_to(&mut self, end: u64) -> Self {
        assert!(
            self.rows.contains(&end) || end == self.rows.end,
            "split must lie within the claim"
        );
        let rows = self.rows.start..end;
        self.rows.start = end;
        Self {
            file: self.file.clone(),
            rows,
        }
    }

    /// Releases rows proven unnecessary, retaining only the given subrange.
    /// Panics if the range is not ordered and contained in this claim.
    pub fn restrict_to(&mut self, rows: Range<u64>) {
        assert!(
            self.rows.start <= rows.start && rows.start <= rows.end && rows.end <= self.rows.end,
            "remaining rows must be contained in the claim"
        );
        if self.rows == rows {
            return;
        }
        let evicted = {
            let mut state = self.file.state.lock();
            let mut evicted = state.release(&(self.rows.start..rows.start));
            evicted.extend(state.release(&(rows.end..self.rows.end)));
            self.rows = rows;
            evicted
        };
        drop(evicted);
    }

    /// Creates another owner for the same rows, for consumers that overlap rather than split.
    /// Each owner's remaining rows stay registered until that owner releases them.
    pub fn share(&self) -> Self {
        Self::new(self.file.clone(), self.rows.clone())
    }
}

impl Drop for RowClaim {
    fn drop(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        let evicted = {
            let mut state = self.file.state.lock();
            state.release(&self.rows)
        };
        drop(evicted);
    }
}

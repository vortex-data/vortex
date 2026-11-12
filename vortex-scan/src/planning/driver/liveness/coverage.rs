// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reference-counted coverage of a file's root rows by live claims.

use std::collections::BTreeMap;
use std::ops::Range;

use vortex_error::VortexExpect;

/// Disjoint spans with the number of claims covering each span.
#[derive(Default)]
pub(super) struct Coverage(BTreeMap<u64, Span>);

#[derive(Clone, Copy)]
struct Span {
    end: u64,
    owners: usize,
}

impl Coverage {
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether any claim covers part of `rows`.
    pub(super) fn overlaps(&self, rows: &Range<u64>) -> bool {
        !rows.is_empty()
            && self
                .0
                .range(..rows.end)
                .next_back()
                .is_some_and(|(_, span)| span.end > rows.start)
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

    pub(super) fn add(&mut self, rows: &Range<u64>) {
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

    /// Releases one claim on `rows`, returning the maximal uncovered gaps around every span that
    /// became uncovered, including adjacent rows other claims released earlier.
    pub(super) fn release(&mut self, rows: &Range<u64>) -> Vec<Range<u64>> {
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

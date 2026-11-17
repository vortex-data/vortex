// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan requests.

use std::num::NonZeroUsize;
use std::ops::Range;

use vortex_array::expr::Expression;
use vortex_array::expr::root;

use crate::selection::Selection;

/// A request to scan a data source.
#[derive(Debug, Clone)]
pub struct ScanRequest {
    /// Projection expression. Defaults to `root()` which returns all columns.
    pub projection: Expression,
    /// Filter expression, `None` implies no filter.
    pub filter: Option<Expression>,
    /// The per-partition row range to read. Row range will be applied
    /// over every partition you scan.
    pub row_range: Option<Range<u64>>,
    /// The per-partition row selection to read. Row selection will be applied
    /// over every partition you scan.
    pub selection: Selection,
    /// Partition selection to scan, which allows readers to skip unwanted partitions.
    pub partition_selection: Selection,
    /// Partition range to scan, which allows readers to skip unwanted partitions.
    pub partition_range: Option<Range<u64>>,
    /// Whether the scan should preserve row order. If false, the scan may produce rows in any
    /// order, for example to enable parallel execution across partitions.
    pub ordered: bool,
    /// Optional limit on the number of rows returned, applied after filtering and row selection.
    ///
    /// For an unordered scan the limit is global: partitions share one budget, so together they
    /// return no more than `limit` rows. An ordered scan cannot share a budget whose reservation
    /// order is completion order, so each partition applies the limit locally and the caller must
    /// trim the concatenated result.
    pub limit: Option<u64>,
    /// The most splits of each partition to decode at once, in all.
    ///
    /// `None` keeps the reader's default, which scales with the machine's cores. Each split being
    /// decoded holds the decoded chunks of every column it overlaps, so a scan that runs alongside
    /// others, such as one task of several on a machine, can set this to bound its memory. It does
    /// not bound I/O: segment reads for later splits may still be coalesced ahead of them.
    pub split_concurrency: Option<NonZeroUsize>,
}

impl Default for ScanRequest {
    fn default() -> Self {
        Self {
            projection: root(),
            filter: None,
            row_range: None,
            selection: Selection::default(),
            partition_selection: Selection::default(),
            ordered: false,
            limit: None,
            partition_range: None,
            split_concurrency: None,
        }
    }
}

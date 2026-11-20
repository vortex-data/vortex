// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) use vortex_scan::pool::RowLimit;

/// The row limit of a scan.
#[derive(Clone)]
pub(crate) enum ScanLimit {
    /// A fresh budget of this many rows for every execution of the scan.
    PerExecution(u64),
    /// A budget shared with the sibling partitions of an unordered scan.
    Shared(RowLimit),
}

impl ScanLimit {
    /// The budget for one execution of the scan.
    pub(crate) fn budget(&self) -> RowLimit {
        match self {
            Self::PerExecution(limit) => RowLimit::new(*limit),
            Self::Shared(limit) => limit.clone(),
        }
    }
}

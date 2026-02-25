// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::scan::ScanBuilder;
use vortex::scan::Selection;

/// Custom Vortex-specific information that can be provided by external indexes or other sources.
///
/// This is intended as a low-level interface for users building their own data systems, see the [advance index] example from the DataFusion repo for a similar usage with Parquet.
///
/// [advance index]: https://github.com/apache/datafusion/blob/47df535d2cd5aac5ad5a92bdc837f38e05ea0f0f/datafusion-examples/examples/data_io/parquet_advanced_index.rs
#[derive(Default)]
pub struct VortexAccessPlan {
    selection: Option<Selection>,
}

impl VortexAccessPlan {
    /// Sets a [`Selection`] for this plan.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = Some(selection);
        self
    }
}

impl VortexAccessPlan {
    /// Returns the selection, if one was set.
    pub fn selection(&self) -> Option<&Selection> {
        self.selection.as_ref()
    }

    /// Apply the plan to the scan's builder.
    pub fn apply_to_builder<A>(&self, mut scan_builder: ScanBuilder<A>) -> ScanBuilder<A>
    where
        A: 'static + Send,
    {
        let Self { selection } = self;

        if let Some(selection) = selection {
            scan_builder = scan_builder.with_selection(selection.clone());
        }

        scan_builder
    }
}

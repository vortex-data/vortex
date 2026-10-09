// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::scan::selection::Selection;

/// Additional Vortex-specific scan constraints attached to a
/// [`PartitionedFile`].
///
/// `VortexAccessPlan` is the hook to use when an external index or planner
/// already knows that only part of a file needs to be scanned. The plan is
/// attached as `extensions` on `PartitionedFile`, and the internal
/// `VortexOpener` applies it before building the Vortex scan.
///
/// The current access plan surface is intentionally small: it lets callers
/// provide a [`Selection`] that narrows the rows considered by the scan.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use datafusion_datasource::PartitionedFile;
/// # use vortex::scan::selection::Selection;
/// use vortex_datafusion::VortexAccessPlan;
///
/// # let selection: Selection = todo!();
/// let file = PartitionedFile::new("metrics.vortex", 1024).with_extensions(Arc::new(
///     VortexAccessPlan::default().with_selection(selection),
/// ));
/// # let _ = file;
/// ```
///
/// This is a low-level integration point for systems building their own access
/// paths on top of DataFusion. For a conceptually similar Parquet example, see
/// DataFusion's
/// [`parquet_advanced_index`].
///
/// [`PartitionedFile`]: datafusion_datasource::PartitionedFile
/// [`parquet_advanced_index`]: https://github.com/apache/datafusion/blob/47df535d2cd5aac5ad5a92bdc837f38e05ea0f0f/datafusion-examples/examples/data_io/parquet_advanced_index.rs
#[derive(Default)]
pub struct VortexAccessPlan {
    selection: Option<Selection>,
}

impl VortexAccessPlan {
    /// Returns the selection, if one was set.
    pub fn selection(&self) -> Option<&Selection> {
        self.selection.as_ref()
    }

    /// Sets the row [`Selection`] to apply when the file is opened.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = Some(selection);
        self
    }

    /// Applies this access plan to a [`ScanBuilder`].
    ///
    /// This is used internally by the file opener after it has translated a
    /// `PartitionedFile` into a Vortex scan.
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

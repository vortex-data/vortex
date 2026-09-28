// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Composable footer opening, file-statistics pruning, and filter-and-project over natural
//! splits.
//!
//! These stages use the explicit planning protocol independently of the existing
//! file-opening and scan entry points. Survivors pass to a caller-supplied continuation, and
//! [`plan_scan`] composes the whole chain for one file.

pub mod filter_project;
pub mod footer_open;
pub mod footer_prune;

use std::sync::Arc;

use vortex_array::expr::Expression;
use vortex_io::VortexReadAt;
use vortex_layout::scan::planning::SegmentLocation;
use vortex_layout::scan::v2::ScanFile;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::next::PendingPlanner;
use vortex_scan::planning::next::next_fn;
use vortex_scan::planning::next::pending;
use vortex_scan::planning::planner::Planner;
use vortex_session::VortexSession;

use crate::Footer;
use crate::VortexFile;
use crate::planning::filter_project::FilterProject;
use crate::planning::footer_open::DEFAULT_INITIAL_READ_SIZE;
use crate::planning::footer_open::FooterOpen;
use crate::planning::footer_prune::FooterPrune;

/// A file to open: what is already known about it and how to read the rest.
pub struct FileSource {
    /// Byte source for the file.
    pub read: Arc<dyn VortexReadAt>,
    /// File size in bytes, if known; unknown sizes cost one `Size` request.
    pub size: Option<u64>,
    /// A cached footer, which makes the open stage need no IO.
    pub footer: Option<Footer>,
}

/// A file whose footer is parsed and validated against its size.
pub struct OpenedFile {
    /// Byte source for the file.
    pub read: Arc<dyn VortexReadAt>,
    /// File size in bytes.
    pub size: u64,
    /// The parsed footer.
    pub footer: Footer,
}

/// The location of every segment the footer describes, indexed by segment id.
pub(crate) fn segment_locations(footer: &Footer) -> Arc<[SegmentLocation]> {
    footer
        .segment_specs_with_metadata()
        .iter()
        .map(|spec| SegmentLocation {
            offset: spec.offset,
            length: spec.length,
            alignment: spec.alignment,
        })
        .collect()
}

/// The parts of an opened file that the `VORTEX_SCAN_V2` executor reads through.
pub fn scan_file(file: &VortexFile) -> ScanFile {
    ScanFile {
        layout: Arc::clone(file.footer().layout()),
        locations: segment_locations(file.footer()),
        segments: file.segment_source(),
    }
}

/// Composes footer opening and file-statistics pruning for one file.
/// Surviving files pass to `next`; rejected and empty files produce no child work.
pub fn plan_file(
    source: FileSource,
    filter: Option<Expression>,
    session: VortexSession,
    next: Next<OpenedFile>,
) -> Box<dyn PendingPlanner> {
    let after_open: Next<OpenedFile> = {
        let session = session.clone();
        next_fn(move |opened| {
            Ok(FooterPrune::new(
                opened,
                filter.clone(),
                session.clone(),
                Arc::clone(&next),
            ))
        })
    };
    pending(move || {
        Ok(Box::new(FooterOpen::new(
            source,
            DEFAULT_INITIAL_READ_SIZE,
            session,
            after_open,
        )) as Box<dyn Planner>)
    })
}

/// Composes footer opening, file-statistics pruning, and filter-and-project for one file.
/// Every surviving natural split becomes one morsel whose data reads go through the protocol.
pub fn plan_scan(
    source: FileSource,
    filter: Option<Expression>,
    projection: Expression,
    session: VortexSession,
) -> Box<dyn PendingPlanner> {
    let leaf: Next<OpenedFile> = {
        let filter = filter.clone();
        let session = session.clone();
        next_fn(move |opened| {
            Ok(FilterProject::new(
                opened,
                filter.clone(),
                projection.clone(),
                session.clone(),
            ))
        })
    };
    plan_file(source, filter, session, leaf)
}

#[cfg(test)]
mod tests;

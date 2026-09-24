// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File stages: footer open, file-statistics pruning, and filter-and-project over natural
//! splits, plus the types they pass between each other.

pub mod filter_project;
pub mod footer_open;
pub mod footer_prune;
pub mod range_morsel;

use std::sync::Arc;

use vortex_array::expr::Expression;
use vortex_file::Footer;
use vortex_io::VortexReadAt;
use vortex_session::VortexSession;

use crate::next::Next;
use crate::next::PendingPlanner;
use crate::next::next_fn;
use crate::next::pending;
use crate::planner::Planner;
use crate::stages::filter_project::FilterProject;
use crate::stages::footer_open::DEFAULT_INITIAL_READ_SIZE;
use crate::stages::footer_open::FooterOpen;
use crate::stages::footer_prune::FooterPrune;
use crate::stages::range_morsel::RangeOnly;

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

/// What a scan asks of a file.
pub struct ScanQuery {
    /// Row predicate, or none to keep every row.
    pub filter: Option<Expression>,
    /// Output expression, or none to emit the diagnostic range morsel instead of data.
    pub projection: Option<Expression>,
}

/// Composes the stages for one file into pending root work: footer open, file-statistics
/// pruning, then filter-and-project when the query has a projection or the diagnostic range
/// morsel otherwise.
///
/// The session is captured by the `Next` closures, not passed between stages. The runtime that
/// drives data reads is created by `FilterProject` on its worker.
pub fn plan_file(
    source: FileSource,
    query: ScanQuery,
    session: VortexSession,
) -> Box<dyn PendingPlanner> {
    let ScanQuery { filter, projection } = query;
    let after_prune: Next<OpenedFile> = match projection {
        Some(projection) => {
            let filter = filter.clone();
            let session = session.clone();
            next_fn(move |opened: OpenedFile| {
                Ok(FilterProject::new(
                    opened,
                    filter.clone(),
                    projection.clone(),
                    session.clone(),
                ))
            })
        }
        None => next_fn(|opened: OpenedFile| Ok(RangeOnly::new(opened))),
    };
    let after_open: Next<OpenedFile> = {
        let session = session.clone();
        next_fn(move |opened: OpenedFile| {
            Ok(FooterPrune::new(
                opened,
                filter.clone(),
                session.clone(),
                Arc::clone(&after_prune),
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

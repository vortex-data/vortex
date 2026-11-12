// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building a manifest from data files that are already written.

use vortex_error::VortexResult;
use vortex_file::Footer;
use vortex_io::VortexReadAt;
use vortex_session::VortexSession;

use crate::footer::read_footer_region;
use crate::manifest::Manifest;
use crate::manifest::ManifestEntry;
use crate::stats::TableStats;

/// Builds a manifest one data file at a time, reading each file's footer region.
///
/// The manifest records every file's column statistics as one table, when all files share a
/// dtype.
pub struct ManifestBuilder {
    session: VortexSession,
    inline_footers: bool,
    entries: Vec<ManifestEntry>,
    footers: Vec<Footer>,
}

impl ManifestBuilder {
    /// A builder that inlines every file's footer region into the manifest, so scans open files
    /// without reading their footers.
    pub fn new(session: VortexSession) -> Self {
        Self {
            session,
            inline_footers: true,
            entries: Vec::new(),
            footers: Vec::new(),
        }
    }

    /// Whether to inline footer regions. Without, a scan reads each file's region once, at the
    /// offset and length the manifest records.
    pub fn with_inline_footers(mut self, inline: bool) -> Self {
        self.inline_footers = inline;
        self
    }

    /// Appends the file at `path`, whose bytes `read` serves, as the table's next file.
    pub async fn add(
        &mut self,
        path: impl Into<String>,
        read: &dyn VortexReadAt,
    ) -> VortexResult<()> {
        let size = read.size().await?;
        let region = read_footer_region(read, size, &self.session).await?;
        self.entries.push(ManifestEntry::try_new(
            path,
            size,
            &region,
            self.inline_footers,
        )?);
        self.footers.push(region.footer);
        Ok(())
    }

    /// The manifest of the files added, in the order they were added.
    pub fn finish(self) -> VortexResult<Manifest> {
        let manifest = Manifest::new(self.entries);
        let Some(first) = self.footers.first() else {
            return Ok(manifest);
        };
        let dtype = first.dtype().clone();
        if self.footers.iter().any(|footer| footer.dtype() != &dtype) {
            return Ok(manifest);
        }
        manifest.with_stats(TableStats::try_new(&self.footers, &dtype, &self.session)?)
    }
}

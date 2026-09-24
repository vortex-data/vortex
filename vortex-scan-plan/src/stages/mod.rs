// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File stages: footer open, file-statistics pruning, and filter-and-project over natural
//! splits, plus the types they pass between each other.

pub mod footer_open;

use std::sync::Arc;

use vortex_array::expr::Expression;
use vortex_file::Footer;
use vortex_io::VortexReadAt;

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

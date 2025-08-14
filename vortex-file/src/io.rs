// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use object_store::ObjectStore;
use object_store::path::Path;
use std::sync::Arc;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::VortexResult;

/// A trait used for reading Vortex files from file-like sources.
pub trait FileReader: 'static + Send + Sync {
    /// Returns the total size of the underlying file.
    fn size(&self) -> u64;

    /// Reads multiple ranges of bytes from the underlying file.
    fn read_ranges(&self, ranges: Vec<ReadRange>) -> BoxFuture<'_, VortexResult<Vec<ByteBuffer>>>;
}

pub struct ReadRange {
    pub offset: u64,
    pub length: usize,
    pub alignment: Alignment,
}

pub struct ObjectStoreFileReader {
    object_store: Arc<dyn ObjectStore>,
    location: Path,
}

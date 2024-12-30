#![allow(dead_code)]
#![allow(unused_variables)]
use std::ops::Range;
use std::sync::{Arc, RwLock};

use vortex_array::ArrayData;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;

use crate::{InitialRead, LayoutMessageCache, VortexReadArrayStream};

pub struct VortexFileArrayReader<R> {
    read: R,
    initial: InitialRead,

    message_cache: Arc<RwLock<LayoutMessageCache>>,
}

impl<R: VortexReadAt> VortexFileArrayReader<R> {
    pub fn row_count(&self) -> usize {
        self.initial.fb_layout().row_count() as usize
    }

    /// Read a single row range from the Vortex file.
    pub async fn read_range(&self, _row_range: Range<usize>) -> VortexResult<ArrayData> {
        todo!()
    }

    /// Stream the chunks of the Vortex file.
    pub fn into_stream(self) -> VortexReadArrayStream<R> {
        todo!()
    }
}

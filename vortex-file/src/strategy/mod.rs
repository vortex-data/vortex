//! This module defines the default layout strategy for a Vortex file.

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::chunked::writer::{ChunkedLayoutOptions, ChunkedLayoutWriter};
use crate::layouts::struct_::writer::StructLayoutWriter;
use crate::strategies::{LayoutStrategy, LayoutWriter, LayoutWriterExt};

/// The default Vortex file layout strategy.
///
/// The current implementation is a placeholder and needs to be fleshed out.
pub struct VortexLayoutStrategy;

impl LayoutStrategy for VortexLayoutStrategy {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        if dtype.is_struct() {
            StructLayoutWriter::try_new_with_factory(dtype, VortexLayoutStrategy).map(|w| w.boxed())
        } else {
            Ok(ChunkedLayoutWriter::new(dtype, ChunkedLayoutOptions::default()).boxed())
        }
    }
}

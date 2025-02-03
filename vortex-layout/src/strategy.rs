//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::Layout`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::flat::writer::FlatLayoutWriter;
use crate::layouts::flat::FlatLayout;
use crate::writer::{LayoutWriter, LayoutWriterExt};

/// A trait for creating new layout writers given a DType.
pub trait LayoutStrategy: Send + Sync {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>>;
}

/// Implement the [`LayoutStrategy`] trait for the [`FlatLayout`] for easy use.
impl LayoutStrategy for FlatLayout {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(FlatLayoutWriter::new(dtype.clone(), Default::default()).boxed())
    }
}

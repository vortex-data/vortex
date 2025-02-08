//! This is a collection of built-in layout strategies designed to be used in conjunction with one
//! another to develop an overall strategy.
//!
//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`Layout`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::chunked::writer::{ChunkedLayoutOptions, ChunkedLayoutWriter};
use crate::layouts::flat::writer::FlatLayoutWriter;
use crate::layouts::flat::FlatLayout;
use crate::layouts::struct_::writer::StructLayoutWriter;
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

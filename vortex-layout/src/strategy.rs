//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::Layout`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use vortex_array::ArrayContext;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::flat::writer::FlatLayoutWriter;
use crate::layouts::struct_::writer::StructLayoutWriter;
use crate::writer::{LayoutWriter, LayoutWriterExt};

/// A trait for creating new layout writers given a DType.
pub trait LayoutStrategy: 'static + Send + Sync {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>>;
}

/// A layout strategy that preserves struct arrays and writes everything else as flat.
pub struct StructStrategy;

impl LayoutStrategy for StructStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        if let DType::Struct(..) = dtype {
            StructLayoutWriter::try_new_with_strategy(ctx, dtype, StructStrategy).map(|w| w.boxed())
        } else {
            Ok(FlatLayoutWriter::new(ctx.clone(), dtype.clone(), Default::default()).boxed())
        }
    }
}

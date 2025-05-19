//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::LayoutData`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::NewLayoutWriter;
use crate::segments::NewSegmentWriter;
use crate::sequence::SequenceId;
use crate::writer::LayoutWriter;

/// A trait for creating new layout writers given a DType.
pub trait LayoutStrategy: 'static + Send + Sync {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>>;
}

pub type SequentialArrayStream =
    Pin<Box<dyn Stream<Item = VortexResult<(SequenceId, ArrayRef)>> + Send>>;

pub trait NewLayoutStrategy: 'static + Send + Sync {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn NewSegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn NewLayoutWriter>>;
}

/// A layout strategy that preserves struct arrays and writes everything else as flat.
pub struct StructStrategy;

impl LayoutStrategy for StructStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        if let DType::Struct(..) = dtype {
            StructLayoutWriter::try_new_with_strategy(ctx, dtype, &StructStrategy)
                .map(|w| w.boxed())
        } else {
            Ok(FlatLayoutWriter::new(ctx.clone(), dtype.clone(), Default::default()).boxed())
        }
    }
}

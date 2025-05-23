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

use crate::LayoutWriter;
use crate::segments::SegmentWriter;
use crate::sequence::SequenceId;

pub type SequentialArrayStream =
    Pin<Box<dyn Stream<Item = VortexResult<(SequenceId, ArrayRef)>> + Send>>;

pub trait LayoutStrategy: 'static + Send + Sync {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn SegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn LayoutWriter>>;
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

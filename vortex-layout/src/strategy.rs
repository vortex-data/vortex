//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::Layout`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::chunked::writer::{ChunkedLayoutOptions, ChunkedLayoutWriter};
use crate::layouts::flat::writer::FlatLayoutWriter;
use crate::layouts::flat::FlatLayout;
use crate::layouts::struct_::writer::StructLayoutWriter;
use crate::writer::{LayoutWriter, LayoutWriterExt};

/// A trait for creating new layout writers given a DType.
pub trait LayoutStrategy: 'static + Send + Sync {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>>;
}

/// Implement the [`LayoutStrategy`] trait for the [`FlatLayout`] for easy use.
impl LayoutStrategy for FlatLayout {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(FlatLayoutWriter::new(dtype.clone(), Default::default()).boxed())
    }
}

/// A layout strategy that preserves struct arrays and writes everything else as flat.
pub struct StructStrategy;

impl LayoutStrategy for StructStrategy {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        if let DType::Struct(..) = dtype {
            StructLayoutWriter::try_new_with_factory(dtype, StructStrategy).map(|w| w.boxed())
        } else {
            Ok(FlatLayoutWriter::new(dtype.clone(), Default::default()).boxed())
        }
    }
}

/// A layout strategy that preserves each chunk as-given.
pub struct ChunkedStrategy {
    pub chunk_strategy: Arc<dyn LayoutStrategy>,
}

impl Default for ChunkedStrategy {
    fn default() -> Self {
        Self {
            chunk_strategy: Arc::new(StructStrategy),
        }
    }
}

impl LayoutStrategy for ChunkedStrategy {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(ChunkedLayoutWriter::new(
            dtype,
            ChunkedLayoutOptions {
                chunk_strategy: self.chunk_strategy.clone(),
                ..Default::default()
            },
        )
        .boxed())
    }
}

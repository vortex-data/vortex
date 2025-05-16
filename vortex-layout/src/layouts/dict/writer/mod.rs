use arcref::ArcRef;
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dict::DictEncoding;
use vortex_dict::builders::{DictConstraints, DictEncoder, dict_encoder};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

mod repeating;

use crate::{LayoutRef, LayoutStrategy, LayoutWriter, LayoutWriterExt};

#[derive(Clone)]
pub struct DictLayoutOptions {
    pub constraints: DictConstraints,
}

impl Default for DictLayoutOptions {
    fn default() -> Self {
        Self {
            constraints: DictConstraints {
                max_bytes: 1024 * 1024,
                max_len: u16::MAX as usize,
            },
        }
    }
}

/// A layout strategy that encodes chunk into values and codes, if found
/// appropriate by the btrblocks compressor. Current implementation only
/// checks the first chunk to decide whether to apply dict layout and
/// encodes chunks into dictionaries. When the dict constraints are hit, a
/// new dictionary is created.
#[derive(Clone)]
pub struct DictStrategy {
    pub options: DictLayoutOptions,
    pub codes: ArcRef<dyn LayoutStrategy>,
    pub values: ArcRef<dyn LayoutStrategy>,
    pub fallback: ArcRef<dyn LayoutStrategy>,
}

impl LayoutStrategy for DictStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        if !dict_layout_supported(dtype) {
            return self.fallback.new_writer(ctx, dtype);
        }
        Ok(DelegatingDictLayoutWriter {
            ctx: ctx.clone(),
            strategy: self.clone(),
            dtype: dtype.clone(),
            writer: None,
        }
        .boxed())
    }
}

pub fn dict_layout_supported(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
    )
}

struct DelegatingDictLayoutWriter {
    ctx: ArrayContext,
    strategy: DictStrategy,
    dtype: DType,
    writer: Option<Box<dyn LayoutWriter>>,
}

impl LayoutWriter for DelegatingDictLayoutWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn crate::segments::SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        assert_eq!(
            chunk.dtype(),
            &self.dtype,
            "Can't push chunks of the wrong dtype into a LayoutWriter. Pushed {} but expected {}.",
            chunk.dtype(),
            self.dtype
        );
        match self.writer.as_mut() {
            Some(writer) => writer.push_chunk(segment_writer, chunk),
            None => {
                let compressed = BtrBlocksCompressor.compress(&chunk)?;
                let mut writer = if !compressed.is_encoding(DictEncoding.id()) {
                    self.strategy.fallback.new_writer(&self.ctx, &self.dtype)?
                } else {
                    repeating::DictLayoutWriter::new(
                        self.ctx.clone(),
                        &self.dtype,
                        self.strategy.clone(),
                    )
                    .boxed()
                };
                writer.push_chunk(segment_writer, chunk)?;
                self.writer = Some(writer);
                Ok(())
            }
        }
    }

    fn flush(
        &mut self,
        segment_writer: &mut dyn crate::segments::SegmentWriter,
    ) -> VortexResult<()> {
        match self.writer.as_mut() {
            None => vortex_bail!("flush called before push_chunk"),
            Some(writer) => writer.flush(segment_writer),
        }
    }

    fn finish(
        &mut self,
        segment_writer: &mut dyn crate::segments::SegmentWriter,
    ) -> VortexResult<LayoutRef> {
        match self.writer.as_mut() {
            None => vortex_bail!("finish called before push_chunk"),
            Some(writer) => writer.finish(segment_writer),
        }
    }
}

enum EncodingState {
    Continue((Box<dyn DictEncoder>, ArrayRef)),
    // (values, encoded, unencoded)
    Done((ArrayRef, ArrayRef, ArrayRef)),
}

fn start_encoding(constraints: &DictConstraints, chunk: &dyn Array) -> VortexResult<EncodingState> {
    let encoder = dict_encoder(chunk, constraints)?;
    encode_chunk(encoder, chunk)
}

fn encode_chunk(
    mut encoder: Box<dyn DictEncoder>,
    chunk: &dyn Array,
) -> VortexResult<EncodingState> {
    let encoded = encoder.encode(chunk)?;
    Ok(match remainder(chunk, encoded.len())? {
        None => EncodingState::Continue((encoder, encoded)),
        Some(unencoded) => EncodingState::Done((encoder.values()?, encoded, unencoded)),
    })
}

fn remainder(array: &dyn Array, encoded_len: usize) -> VortexResult<Option<ArrayRef>> {
    (encoded_len < array.len())
        .then(|| array.slice(encoded_len, array.len()))
        .transpose()
}

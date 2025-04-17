use vortex_array::arcref::ArcRef;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dict::builders::{DictConstraints, DictEncoder};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use super::{EncodingState, dict_layout, encode_chunk, start_encoding};
use crate::layouts::chunked::writer::chunked_layout;
use crate::segments::SegmentWriter;
use crate::{Layout, LayoutStrategy, LayoutWriter, LayoutWriterExt as _};

pub struct DictLayoutWriter {
    ctx: ArrayContext,
    constraints: DictConstraints,
    child_strategy: ArcRef<dyn LayoutStrategy>,
    values_strategy: ArcRef<dyn LayoutStrategy>,
    dtype: DType,
    state: State,
}

impl DictLayoutWriter {
    pub fn new(
        ctx: ArrayContext,
        dtype: &DType,
        child_strategy: ArcRef<dyn LayoutStrategy>,
        values_strategy: ArcRef<dyn LayoutStrategy>,
        constraints: DictConstraints,
    ) -> Self {
        Self {
            ctx,
            constraints,
            child_strategy,
            values_strategy,
            dtype: dtype.clone(),
            state: State::default(),
        }
    }
}

enum State {
    Uninit,
    Codes(Codes),
    Fallback(Fallback),
}

impl Default for State {
    fn default() -> Self {
        Self::Uninit
    }
}

impl State {
    fn dict_values(&mut self) -> VortexResult<ArrayRef> {
        match self {
            Self::Uninit => Err(vortex_err!("push_chunk not called yet")),
            Self::Codes(codes) => codes.encoder.values(),
            Self::Fallback(fallback) => Ok(fallback.dict_values.clone()),
        }
    }
}

struct Codes {
    encoder: Box<dyn DictEncoder>,
    writer: Box<dyn LayoutWriter>,
}

struct Fallback {
    dict_values: ArrayRef,
    codes_writer: Box<dyn LayoutWriter>,
    fallback_writer: Box<dyn LayoutWriter>,
}

impl DictLayoutWriter {
    fn create_codes_writer(
        &self,
        segments: &mut dyn SegmentWriter,
        encoded: ArrayRef,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        let codes_dtype = if self.dtype.is_nullable() {
            encoded.dtype().as_nullable()
        } else {
            encoded.dtype().clone()
        };
        let mut writer = self.child_strategy.new_writer(&self.ctx, &codes_dtype)?;
        writer.push_chunk(segments, encoded)?;
        Ok(writer)
    }

    fn create_fallback_writer(
        &self,
        segments: &mut dyn SegmentWriter,
        unencoded: ArrayRef,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        let mut fallback = self.child_strategy.new_writer(&self.ctx, &self.dtype)?;
        fallback.push_chunk(segments, unencoded)?;
        Ok(fallback)
    }
}

impl LayoutWriter for DictLayoutWriter {
    fn push_chunk(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        self.state = match std::mem::take(&mut self.state) {
            State::Uninit => match start_encoding(&self.constraints, &chunk)? {
                EncodingState::Continue((encoder, encoded)) => {
                    let writer = self.create_codes_writer(segments, encoded)?;
                    State::Codes(Codes { encoder, writer })
                }
                EncodingState::Done((values, encoded, unencoded)) => {
                    let mut writer = self.create_codes_writer(segments, encoded)?;
                    writer.flush(segments)?;
                    State::Fallback(Fallback {
                        dict_values: values,
                        codes_writer: writer,
                        fallback_writer: self.create_fallback_writer(segments, unencoded)?,
                    })
                }
            },
            State::Codes(Codes {
                encoder,
                mut writer,
            }) => match encode_chunk(encoder, &chunk)? {
                EncodingState::Continue((encoder, encoded)) => {
                    writer.push_chunk(segments, encoded)?;
                    State::Codes(Codes { encoder, writer })
                }
                EncodingState::Done((values, encoded, unencoded)) => {
                    writer.push_chunk(segments, encoded)?;
                    writer.flush(segments)?;
                    State::Fallback(Fallback {
                        dict_values: values,
                        codes_writer: writer,
                        fallback_writer: self.create_fallback_writer(segments, unencoded)?,
                    })
                }
            },
            State::Fallback(mut fallback) => {
                fallback.fallback_writer.push_chunk(segments, chunk)?;
                State::Fallback(fallback)
            }
        };
        Ok(())
    }

    fn flush(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        match &mut self.state {
            State::Uninit => Err(vortex_err!(
                "DictLayoutWriter flush called before push_chunk"
            )),
            State::Codes(codes) => codes.writer.flush(segment_writer),
            State::Fallback(state) => state.fallback_writer.flush(segment_writer),
        }
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        let mut dict_writer = self.values_strategy.new_writer(&self.ctx, &self.dtype)?;
        let values_layout = dict_writer.push_one(segments, self.state.dict_values()?)?;

        match std::mem::take(&mut self.state) {
            State::Uninit => vortex_bail!("DictLayoutWriter finish called before push_chunk"),
            State::Codes(mut codes) => {
                let codes_layout = codes.writer.finish(segments)?;
                dict_layout(values_layout, codes_layout)
            }
            State::Fallback(mut state) => {
                let codes_layout = state.codes_writer.finish(segments)?;
                let fallback_layout = state.fallback_writer.finish(segments)?;
                Ok(chunked_layout(
                    self.dtype.clone(),
                    codes_layout.row_count() + fallback_layout.row_count(),
                    vec![dict_layout(values_layout, codes_layout)?, fallback_layout],
                ))
            }
        }
    }
}

use bytes::Bytes;
use vortex_array::arcref::ArcRef;
use vortex_array::compute::slice;
use vortex_array::{Array, ArrayContext, ArrayRef, RkyvMetadata, SerializeMetadata};
use vortex_dict::builders::{DictEncoder, dict_encoder};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::layouts::chunked::writer::chunked_layout;
use crate::layouts::dict::DictLayout;
use crate::segments::SegmentWriter;
use crate::{Layout, LayoutStrategy, LayoutVTableRef, LayoutWriter, LayoutWriterExt};

pub struct DictLayoutOptions {
    /// max dictionary size in bytes, uncompressed
    pub max_dict_size_bytes: usize,
    pub max_dict_len: usize,
}

impl Default for DictLayoutOptions {
    fn default() -> Self {
        Self {
            max_dict_size_bytes: 1024 * 1024,
            max_dict_len: u16::MAX as usize,
        }
    }
}

pub struct DictLayoutWriter {
    ctx: ArrayContext,
    options: DictLayoutOptions,
    child_strategy: ArcRef<dyn LayoutStrategy>,
    dict_strategy: ArcRef<dyn LayoutStrategy>,
    dtype: DType,
    state: State,
}

pub fn dict_layout_supported(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
    )
}

impl DictLayoutWriter {
    pub fn try_new(
        ctx: ArrayContext,
        dtype: &DType,
        child_strategy: ArcRef<dyn LayoutStrategy>,
        dict_strategy: ArcRef<dyn LayoutStrategy>,
        options: DictLayoutOptions,
    ) -> VortexResult<Self> {
        Ok(Self {
            ctx: ctx.clone(),
            options,
            child_strategy: child_strategy.clone(),
            dict_strategy,
            dtype: dtype.clone(),
            state: State::default(),
        })
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

impl Codes {
    fn transition(
        mut self,
        chunk: &dyn Array,
        codes_len: usize,
        segments: &mut dyn SegmentWriter,
        child_strategy: ArcRef<dyn LayoutStrategy>,
        ctx: &ArrayContext,
        dtype: &DType,
    ) -> VortexResult<State> {
        if let Some(remainder) = remainder(chunk, codes_len)? {
            self.writer.flush(segments)?;
            let mut fallback = child_strategy.new_writer(ctx, dtype)?;
            fallback.push_chunk(segments, remainder)?;
            Ok(State::Fallback(Fallback {
                dict_values: self.encoder.values()?,
                codes_writer: self.writer,
                fallback_writer: fallback,
            }))
        } else {
            Ok(State::Codes(self))
        }
    }
}

struct Fallback {
    dict_values: ArrayRef,
    codes_writer: Box<dyn LayoutWriter>,
    fallback_writer: Box<dyn LayoutWriter>,
}

impl LayoutWriter for DictLayoutWriter {
    fn push_chunk(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        self.state = match std::mem::take(&mut self.state) {
            State::Uninit => {
                let mut encoder = dict_encoder(&chunk, self.options.max_dict_size_bytes)?;
                let codes = encoder.encode(&chunk)?;
                let codes_len = codes.len();

                // match values nullability
                let codes_dtype = if self.dtype.is_nullable() {
                    codes.dtype().as_nullable()
                } else {
                    codes.dtype().clone()
                };
                let mut writer = self.child_strategy.new_writer(&self.ctx, &codes_dtype)?;
                writer.push_chunk(segments, codes)?;

                Codes { encoder, writer }.transition(
                    &chunk,
                    codes_len,
                    segments,
                    self.child_strategy.clone(),
                    &self.ctx,
                    &self.dtype,
                )?
            }
            State::Codes(mut codes) => {
                let chunk_codes = codes.encoder.encode(&chunk)?;
                let codes_len = chunk_codes.len();

                codes.writer.push_chunk(segments, chunk_codes)?;
                codes.transition(
                    &chunk,
                    codes_len,
                    segments,
                    self.child_strategy.clone(),
                    &self.ctx,
                    &self.dtype,
                )?
            }
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
        let mut dict_writer = self.dict_strategy.new_writer(&self.ctx, &self.dtype)?;
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

fn remainder(array: &dyn Array, encoded_len: usize) -> VortexResult<Option<ArrayRef>> {
    (encoded_len < array.len())
        .then(|| slice(array, encoded_len, array.len()))
        .transpose()
}

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct DictLayoutMetadata {
    pub codes_ptype: PType,
}

fn dict_layout(values: Layout, codes: Layout) -> VortexResult<Layout> {
    let codes_ptype = codes.dtype().try_into()?;
    let metadata = Bytes::copy_from_slice(
        &RkyvMetadata(DictLayoutMetadata { codes_ptype })
            .serialize()
            .ok_or_else(|| vortex_err!("could not serialize dict layout metadata"))?,
    );
    Ok(Layout::new_owned(
        "dict".into(),
        LayoutVTableRef::new_ref(&DictLayout),
        values.dtype().clone(),
        codes.row_count(),
        vec![],
        vec![values, codes],
        Some(metadata),
    ))
}

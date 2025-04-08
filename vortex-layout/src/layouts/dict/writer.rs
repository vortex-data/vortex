use vortex_array::arcref::ArcRef;
use vortex_array::compute::slice;
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_dict::builders::{DictEncoder, dict_encoder};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::layouts::dict::DictLayout;
use crate::segments::SegmentWriter;
use crate::{Layout, LayoutStrategy, LayoutVTableRef, LayoutWriter, LayoutWriterExt};

pub struct DictLayoutOptions {
    /// max dictionary size in bytes
    pub max_dict_size_bytes: usize,
}

impl Default for DictLayoutOptions {
    fn default() -> Self {
        Self {
            max_dict_size_bytes: 1024 * 1024,
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
                let mut writer = self.child_strategy.new_writer(&self.ctx, codes.dtype())?;
                writer.push_chunk(segments, codes)?;
                State::Codes(Codes { encoder, writer })
            }
            State::Codes(mut codes) => {
                let chunk_codes = codes.encoder.encode(&chunk)?;
                let codes_len = chunk_codes.len();
                codes.writer.push_chunk(segments, chunk_codes)?;
                if codes_len <= chunk.len() {
                    codes.writer.flush(segments)?;
                    let mut fallback = self.child_strategy.new_writer(&self.ctx, &self.dtype)?;
                    let remaining = slice(&chunk, codes_len, chunk.len())?;
                    fallback.push_chunk(segments, remaining)?;
                    State::Fallback(Fallback {
                        dict_values: codes.encoder.values()?,
                        codes_writer: codes.writer,
                        fallback_writer: fallback,
                    })
                } else {
                    State::Codes(codes)
                }
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
        let dict = dict_writer.push_one(segments, self.state.dict_values()?)?;

        let (row_count, children) = match std::mem::take(&mut self.state) {
            State::Uninit => vortex_bail!("DictLayoutWriter finish called before push_chunk"),
            State::Codes(mut codes) => {
                let codes_layout = codes.writer.finish(segments)?;
                (codes_layout.row_count(), vec![dict, codes_layout])
            }
            State::Fallback(mut state) => {
                let codes_layout = state.codes_writer.finish(segments)?;
                let fallback_layout = state.fallback_writer.finish(segments)?;
                (
                    codes_layout.row_count() + fallback_layout.row_count(),
                    vec![dict, codes_layout, fallback_layout],
                )
            }
        };

        Ok(Layout::new_owned(
            "dict".into(),
            LayoutVTableRef::new_ref(&DictLayout),
            self.dtype.clone(),
            row_count,
            vec![],
            children,
            None,
        ))
    }
}

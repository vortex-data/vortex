use vortex_array::arcref::ArcRef;
use vortex_array::compute::slice;
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_dict::builders::{DictEncoder, dict_encoder};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::layouts::dict::DictLayout;
use crate::segments::SegmentWriter;
use crate::{Layout, LayoutStrategy, LayoutVTableRef, LayoutWriter};

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
    dict_encoder: Option<Box<dyn DictEncoder>>,
    codes_writer: Box<dyn LayoutWriter>,
    fallback_writer: Option<Box<dyn LayoutWriter>>,
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
            dict_encoder: None,
            codes_writer: child_strategy.new_writer(&ctx, dtype)?,
            fallback_writer: None,
        })
    }
}

impl LayoutWriter for DictLayoutWriter {
    fn push_chunk(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        if let Some(fallback) = self.fallback_writer.as_mut() {
            return fallback.push_chunk(segments, chunk);
        }

        if self.dict_encoder.is_none() {
            self.dict_encoder = Some(dict_encoder(&chunk, self.options.max_dict_size_bytes)?);
        }
        let encoder = self.dict_encoder.as_mut().vortex_expect("can't be None");
        let chunk_codes = encoder.encode(&chunk)?;
        let codes_len = chunk_codes.len();

        self.codes_writer.push_chunk(segments, chunk_codes)?;
        if codes_len <= chunk.len() {
            let mut fallback = self.child_strategy.new_writer(&self.ctx, &self.dtype)?;
            let remaining = slice(&chunk, codes_len, chunk.len())?;
            fallback.push_chunk(segments, remaining)?;
            self.fallback_writer = Some(fallback);
        }
        Ok(())
    }

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        // TODO(os): write dict

        let codes = self.codes_writer.finish(segments)?;
        let mut row_count = codes.row_count();
        let mut children = vec![codes];
        if let Some(fallback) = self.fallback_writer.as_mut() {
            let fallback_layout = fallback.finish(segments)?;
            row_count += fallback_layout.row_count();
            children.push(fallback_layout);
        }

        Ok(Layout::new_owned(
            "dict".into(),
            LayoutVTableRef::new_ref(&DictLayout),
            self.dtype,
            row_count,
            vec![],
            children,
            metadata,
        ))
    }
}

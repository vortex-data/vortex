use async_trait::async_trait;
use itertools::Itertools;
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_dict::builders::DictEncoder;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult, vortex_bail};

use super::{DictStrategy, EncodingState, encode_chunk, start_encoding};
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::DictLayout;
use crate::segments::ConcurrentSegmentWriter;
use crate::{IntoLayout, LayoutRef, LayoutWriter};

pub struct DictLayoutWriter {
    ctx: ArrayContext,
    strategy: DictStrategy,
    dtype: DType,
    writers: Vec<(Box<dyn LayoutWriter>, Box<dyn LayoutWriter>)>,
    encoder: Option<Box<dyn DictEncoder>>,
}

impl DictLayoutWriter {
    pub fn new(ctx: ArrayContext, dtype: &DType, strategy: DictStrategy) -> Self {
        Self {
            ctx,
            strategy,
            dtype: dtype.clone(),
            writers: vec![],
            encoder: None,
        }
    }
}

impl DictLayoutWriter {
    async fn flush_last(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<()> {
        if let Some((last_values, last_codes)) = self.writers.last_mut() {
            last_values.flush(segment_writer).await?;
            last_codes.flush(segment_writer).await?;
        }
        Ok(())
    }

    async fn new_dict(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
        encoded_dtype: &DType,
    ) -> VortexResult<()> {
        self.flush_last(segment_writer).await?;

        let codes_dtype = if self.dtype.is_nullable() {
            encoded_dtype.as_nullable()
        } else {
            encoded_dtype.clone()
        };
        let codes_writer = self.strategy.codes.new_writer(&self.ctx, &codes_dtype)?;
        let values_writer = self.strategy.values.new_writer(&self.ctx, &self.dtype)?;
        self.writers.push((values_writer, codes_writer));
        Ok(())
    }

    async fn push_encoded(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        match self.writers.last_mut() {
            Some((_, codes)) => codes.push_chunk(segment_writer, chunk).await,
            None => vortex_bail!("no active codes writer"),
        }
    }

    async fn push_values(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
        values: ArrayRef,
    ) -> VortexResult<()> {
        match self.writers.last_mut() {
            Some((values_writer, _)) => values_writer.push_chunk(segment_writer, values).await,
            None => vortex_bail!("no active values writer"),
        }
    }
}

#[async_trait]
impl LayoutWriter for DictLayoutWriter {
    async fn push_chunk(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        assert_eq!(
            chunk.dtype(),
            &self.dtype,
            "Can't push chunks of the wrong dtype into a LayoutWriter. Pushed {} but expected {}.",
            chunk.dtype(),
            self.dtype
        );
        let mut to_be_encoded = Some(chunk);
        while let Some(remaining) = to_be_encoded.take() {
            match self.encoder.take() {
                None => match start_encoding(&self.strategy.options.constraints, &remaining)? {
                    EncodingState::Continue((encoder, encoded)) => {
                        self.new_dict(segment_writer, encoded.dtype()).await?;
                        self.push_encoded(segment_writer, encoded).await?;
                        self.encoder = Some(encoder);
                    }
                    EncodingState::Done((values, encoded, unencoded)) => {
                        self.new_dict(segment_writer, encoded.dtype()).await?;
                        self.push_encoded(segment_writer, encoded).await?;
                        self.push_values(segment_writer, values).await?;
                        to_be_encoded = Some(unencoded);
                    }
                },
                Some(encoder) => match encode_chunk(encoder, &remaining)? {
                    EncodingState::Continue((encoder, encoded)) => {
                        self.push_encoded(segment_writer, encoded).await?;
                        self.encoder = Some(encoder);
                    }
                    EncodingState::Done((values, encoded, unencoded)) => {
                        self.push_encoded(segment_writer, encoded).await?;
                        self.push_values(segment_writer, values).await?;
                        to_be_encoded = Some(unencoded);
                    }
                },
            }
        }
        Ok(())
    }

    async fn flush(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<()> {
        if let Some(mut encoder) = self.encoder.take() {
            self.push_values(segment_writer, encoder.values()?).await?;
            self.flush_last(segment_writer).await?;
        }
        Ok(())
    }

    async fn finish(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<LayoutRef> {
        if self.encoder.is_some() {
            vortex_bail!("flush not called before finish")
        }

        let mut children: Vec<LayoutRef> = self
            .writers
            .iter_mut()
            .map(|(values, codes)| {
                Ok::<_, VortexError>(
                    DictLayout::new(
                        values.finish(segment_writer)?,
                        codes.finish(segment_writer)?,
                    )
                    .into_layout(),
                )
            })
            .try_collect()?;

        if children.len() == 1 {
            return Ok(children.remove(0));
        }

        let row_count = children.iter().map(|child| child.row_count()).sum();
        Ok(ChunkedLayout::new(
            row_count,
            self.dtype.clone(),
            OwnedLayoutChildren::layout_children(children),
        )
        .into_layout())
    }
}

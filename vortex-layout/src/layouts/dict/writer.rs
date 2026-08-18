// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::builders::dict::DictConstraints;
use vortex_array::builders::dict::DictEncoder;
use vortex_array::builders::dict::dict_encoder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::compressed::CompressorPlugin;
use crate::layouts::dict::DictLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;

/// Constraints for dictionary layout encoding.
///
/// Note that [`max_len`](Self::max_len) is limited to `u16` (65,535 entries) by design. Since
/// layout chunks are typically ~8k elements, having more than 64k unique values in a dictionary
/// means dictionary encoding provides little compression benefit. If a column has very high
/// cardinality, the fallback encoding strategy should be used instead.
#[derive(Clone)]
pub struct DictLayoutConstraints {
    /// Maximum size of the dictionary in bytes.
    pub max_bytes: usize,
    /// Maximum dictionary length. Limited to `u16` because dictionaries with more than 64k unique
    /// values provide diminishing compression returns given typical chunk sizes (~8k elements).
    ///
    /// The codes dtype is determined upfront from this constraint:
    /// - [`PType::U8`] when max_len <= 255
    /// - [`PType::U16`] when max_len > 255
    ///
    /// Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for external compatibility.
    pub max_len: u16,
}

impl From<DictLayoutConstraints> for DictConstraints {
    fn from(value: DictLayoutConstraints) -> Self {
        DictConstraints {
            max_bytes: value.max_bytes,
            max_len: value.max_len as usize,
        }
    }
}

impl Default for DictLayoutConstraints {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024,
            max_len: u16::MAX,
        }
    }
}

#[derive(Clone, Default)]
pub struct DictLayoutOptions {
    pub constraints: DictLayoutConstraints,
}

/// A layout strategy that encodes chunk into values and codes, if found
/// appropriate by the btrblocks compressor. Current implementation only
/// checks the first chunk to decide whether to apply dict layout and
/// encodes chunks into dictionaries. When the dict constraints are hit, a
/// new dictionary is created.
#[derive(Clone)]
pub struct DictStrategy {
    codes: Arc<dyn LayoutStrategy>,
    values: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
    options: DictLayoutOptions,
    probe_compressor: Arc<dyn CompressorPlugin>,
}

impl DictStrategy {
    pub fn new<Codes: LayoutStrategy, Values: LayoutStrategy, Fallback: LayoutStrategy>(
        codes: Codes,
        values: Values,
        fallback: Fallback,
        options: DictLayoutOptions,
        probe_compressor: Arc<dyn CompressorPlugin>,
    ) -> Self {
        Self {
            codes: Arc::new(codes),
            values: Arc::new(values),
            fallback: Arc::new(fallback),
            options,
            probe_compressor,
        }
    }
}

impl LayoutStrategy for DictStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn crate::LayoutWriter>> {
        let mode = if dict_layout_supported(&dtype) {
            None
        } else {
            Some(DictWriterMode::Fallback(self.fallback.new_writer(
                ctx.clone(),
                Arc::clone(&segment_sink),
                dtype.clone(),
                session,
            )?))
        };
        Ok(Box::new(DictLayoutWriter {
            codes: Arc::clone(&self.codes),
            values: Arc::clone(&self.values),
            fallback: Arc::clone(&self.fallback),
            probe_compressor: Arc::clone(&self.probe_compressor),
            constraints: self.options.constraints.clone().into(),
            ctx,
            segment_sink,
            dtype,
            session: session.clone(),
            mode,
        }))
    }
}

enum DictWriterMode {
    Fallback(Box<dyn crate::LayoutWriter>),
    Dictionary(DictionaryLayoutWriter),
}

struct DictLayoutWriter {
    codes: Arc<dyn LayoutStrategy>,
    values: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
    probe_compressor: Arc<dyn CompressorPlugin>,
    constraints: DictConstraints,
    ctx: LayoutWriterContext,
    segment_sink: SegmentSinkRef,
    dtype: DType,
    session: VortexSession,
    mode: Option<DictWriterMode>,
}

impl DictLayoutWriter {
    fn initialize(&mut self, first: &ArrayRef) -> VortexResult<()> {
        let compressed = self
            .probe_compressor
            .compress_chunk(first, &mut self.session.create_execution_ctx())?;
        self.mode = Some(if compressed.is::<Dict>() {
            DictWriterMode::Dictionary(DictionaryLayoutWriter {
                codes: Arc::clone(&self.codes),
                values: Arc::clone(&self.values),
                ctx: self.ctx.clone(),
                segment_sink: Arc::clone(&self.segment_sink),
                dtype: self.dtype.clone(),
                session: self.session.clone(),
                encoder: DictStreamState {
                    encoder: None,
                    constraints: self.constraints.clone(),
                },
                active_codes: None,
                child_layouts: Vec::new(),
            })
        } else {
            DictWriterMode::Fallback(self.fallback.new_writer(
                self.ctx.clone(),
                Arc::clone(&self.segment_sink),
                self.dtype.clone(),
                &self.session,
            )?)
        });
        Ok(())
    }
}

#[async_trait]
impl crate::LayoutWriter for DictLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        if self.mode.is_none() {
            self.initialize(&chunk)?;
        }
        match self.mode.as_mut().vortex_expect("writer mode initialized") {
            DictWriterMode::Fallback(writer) => writer.write(sequence_id, chunk).await,
            DictWriterMode::Dictionary(writer) => writer.write(sequence_id, chunk).await,
        }
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        if self.mode.is_none() {
            self.mode = Some(DictWriterMode::Fallback(self.fallback.new_writer(
                self.ctx.clone(),
                Arc::clone(&self.segment_sink),
                self.dtype.clone(),
                &self.session,
            )?));
        }
        match self.mode.as_mut().vortex_expect("writer mode initialized") {
            DictWriterMode::Fallback(writer) => writer.finish(sequence_id).await,
            DictWriterMode::Dictionary(writer) => writer.finish(sequence_id).await,
        }
    }

    async fn close(mut self: Box<Self>) -> VortexResult<LayoutRef> {
        match self.mode.take().vortex_expect("writer mode initialized") {
            DictWriterMode::Fallback(writer) => writer.close().await,
            DictWriterMode::Dictionary(writer) => Box::new(writer).close().await,
        }
    }
}

struct DictionaryLayoutWriter {
    codes: Arc<dyn LayoutStrategy>,
    values: Arc<dyn LayoutStrategy>,
    ctx: LayoutWriterContext,
    segment_sink: SegmentSinkRef,
    dtype: DType,
    session: VortexSession,
    encoder: DictStreamState,
    active_codes: Option<Box<dyn crate::LayoutWriter>>,
    child_layouts: Vec<LayoutRef>,
}

impl DictionaryLayoutWriter {
    async fn process(&mut self, chunk: DictionaryChunk) -> VortexResult<()> {
        match chunk {
            DictionaryChunk::Codes {
                sequence_id,
                codes,
                codes_ptype,
            } => {
                if self.active_codes.is_none() {
                    self.active_codes = Some(self.codes.new_writer(
                        self.ctx.clone(),
                        Arc::clone(&self.segment_sink),
                        DType::Primitive(codes_ptype, Nullability::NonNullable),
                        &self.session,
                    )?);
                }
                self.active_codes
                    .as_mut()
                    .vortex_expect("codes writer active")
                    .write(sequence_id, codes)
                    .await
            }
            DictionaryChunk::Values(sequence_id, values) => {
                let mut sequence = sequence_id.descend();
                let mut codes = self
                    .active_codes
                    .take()
                    .vortex_expect("values follow codes");
                let mut values_writer = self.values.new_writer(
                    self.ctx.clone(),
                    Arc::clone(&self.segment_sink),
                    self.dtype.clone(),
                    &self.session,
                )?;
                codes.finish(sequence.advance()).await?;
                let codes_layout = codes.close().await?;
                values_writer.write(sequence.advance(), values).await?;
                values_writer.finish(sequence.advance()).await?;
                let values_layout = values_writer.close().await?;
                self.child_layouts
                    .push(DictLayout::new(values_layout, codes_layout).into_layout());
                Ok(())
            }
        }
    }

    async fn write_chunk(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let mut labeler = DictChunkLabeler::new(sequence_id);
        let chunks = self.encoder.encode(
            &mut labeler,
            chunk,
            &mut self.session.create_execution_ctx(),
        )?;
        for chunk in chunks {
            self.process(chunk).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl crate::LayoutWriter for DictionaryLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        self.write_chunk(sequence_id, chunk).await
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        let mut labeler = DictChunkLabeler::new(sequence_id);
        for chunk in self.encoder.drain_values(&mut labeler) {
            self.process(chunk).await?;
        }
        if self.active_codes.is_some() {
            return Err(vortex_err!("incomplete dictionary run"));
        }
        Ok(())
    }

    async fn close(mut self: Box<Self>) -> VortexResult<LayoutRef> {
        if self.child_layouts.len() == 1 {
            return Ok(self.child_layouts.pop().vortex_expect("one child layout"));
        }
        let row_count = self
            .child_layouts
            .iter()
            .map(|child| child.row_count())
            .sum();
        Ok(ChunkedLayout::new(
            row_count,
            self.dtype,
            OwnedLayoutChildren::layout_children(self.child_layouts),
        )
        .into_layout())
    }
}

enum DictionaryChunk {
    Codes {
        sequence_id: SequenceId,
        codes: ArrayRef,
        codes_ptype: PType,
    },
    Values(SequenceId, ArrayRef),
}

struct DictStreamState {
    encoder: Option<Box<dyn DictEncoder>>,
    constraints: DictConstraints,
}

impl DictStreamState {
    fn encode(
        &mut self,
        labeler: &mut DictChunkLabeler,
        chunk: ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<DictionaryChunk>> {
        let mut res = Vec::new();
        let mut to_be_encoded = Some(chunk);
        while let Some(remaining) = to_be_encoded.take() {
            match self.encoder.take() {
                None => match start_encoding(&self.constraints, &remaining, exec_ctx)? {
                    EncodingState::Continue((encoder, encoded)) => {
                        let ptype = encoder.codes_ptype();
                        res.push(labeler.codes(encoded, ptype));
                        self.encoder = Some(encoder);
                    }
                    EncodingState::Done((values, encoded, unencoded)) => {
                        // Encoder was created and consumed within start_encoding
                        let ptype = PType::try_from(encoded.dtype())
                            .vortex_expect("codes should be primitive");
                        res.push(labeler.codes(encoded, ptype));
                        res.push(labeler.values(values));
                        to_be_encoded = Some(unencoded);
                    }
                },
                Some(encoder) => {
                    let ptype = encoder.codes_ptype();
                    match encode_chunk(encoder, &remaining, exec_ctx)? {
                        EncodingState::Continue((encoder, encoded)) => {
                            res.push(labeler.codes(encoded, ptype));
                            self.encoder = Some(encoder);
                        }
                        EncodingState::Done((values, encoded, unencoded)) => {
                            res.push(labeler.codes(encoded, ptype));
                            res.push(labeler.values(values));
                            to_be_encoded = Some(unencoded);
                        }
                    }
                }
            }
        }
        Ok(res)
    }

    fn drain_values(&mut self, labeler: &mut DictChunkLabeler) -> Vec<DictionaryChunk> {
        match self.encoder.take() {
            None => Vec::new(),
            Some(mut encoder) => vec![labeler.values(encoder.reset())],
        }
    }
}

struct DictChunkLabeler {
    sequence: SequencePointer,
}

impl DictChunkLabeler {
    fn new(sequence_id: SequenceId) -> Self {
        Self {
            sequence: sequence_id.descend(),
        }
    }

    fn codes(&mut self, codes: ArrayRef, codes_ptype: PType) -> DictionaryChunk {
        DictionaryChunk::Codes {
            sequence_id: self.sequence.advance(),
            codes,
            codes_ptype,
        }
    }

    fn values(&mut self, values: ArrayRef) -> DictionaryChunk {
        DictionaryChunk::Values(self.sequence.advance(), values)
    }
}

pub fn dict_layout_supported(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
    )
}

#[derive(prost::Message)]
pub struct DictLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    // i32 is required for proto, use the generated getter to read this field.
    codes_ptype: i32,
}

impl DictLayoutMetadata {
    pub fn new(codes_ptype: PType) -> Self {
        let mut metadata = Self::default();
        metadata.set_codes_ptype(codes_ptype);
        metadata
    }
}

enum EncodingState {
    Continue((Box<dyn DictEncoder>, ArrayRef)),
    // (values, encoded, unencoded)
    Done((ArrayRef, ArrayRef, ArrayRef)),
}

fn start_encoding(
    constraints: &DictConstraints,
    chunk: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<EncodingState> {
    let encoder = dict_encoder(chunk, constraints);
    encode_chunk(encoder, chunk, ctx)
}

fn encode_chunk(
    mut encoder: Box<dyn DictEncoder>,
    chunk: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<EncodingState> {
    let encoded = encoder.encode(chunk, ctx)?.into_array();
    match remainder(chunk, encoded.len())? {
        None => Ok(EncodingState::Continue((encoder, encoded))),
        Some(unencoded) => Ok(EncodingState::Done((encoder.reset(), encoded, unencoded))),
    }
}

fn remainder(array: &ArrayRef, encoded_len: usize) -> VortexResult<Option<ArrayRef>> {
    if encoded_len < array.len() {
        Ok(Some(array.slice(encoded_len..array.len())?))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::builders::dict::DictConstraints;
    use vortex_array::dtype::PType;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::DictChunkLabeler;
    use super::DictStreamState;
    use super::DictionaryChunk;
    use crate::sequence::SequenceId;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn encoded_codes_ptype(
        arr: vortex_array::ArrayRef,
        constraints: DictConstraints,
    ) -> VortexResult<PType> {
        let mut labeler = DictChunkLabeler::new(SequenceId::root().downgrade());
        let chunks = DictStreamState {
            encoder: None,
            constraints,
        }
        .encode(&mut labeler, arr, &mut SESSION.create_execution_ctx())?;
        chunks
            .into_iter()
            .find_map(|chunk| match chunk {
                DictionaryChunk::Codes { codes_ptype, .. } => Some(codes_ptype),
                DictionaryChunk::Values(..) => None,
            })
            .ok_or_else(|| vortex_err!("dictionary encoder produced no codes"))
    }

    /// Regression test for selecting U8 codes when the configured dictionary fits in U8.
    #[test]
    fn test_dict_writer_uses_u8_for_small_dictionaries() -> VortexResult<()> {
        // Use max_len = 100 to force U8 codes (since 100 <= 255).
        let constraints = DictConstraints {
            max_bytes: 1024 * 1024,
            max_len: 100,
        };

        // Create a simple string array with a few unique values.
        let arr = VarBinArray::from(vec!["hello", "world", "hello", "world"]).into_array();

        assert_eq!(
            encoded_codes_ptype(arr, constraints)?,
            PType::U8,
            "codes should use U8 for small dictionaries"
        );
        Ok(())
    }

    /// Test that the codes use U16 when the dictionary may contain more than 255 entries.
    #[test]
    fn test_dict_writer_uses_u16_for_large_dictionaries() -> VortexResult<()> {
        // Use max_len = 1000 to allow U16 codes (since 1000 > 255).
        let constraints = DictConstraints {
            max_bytes: 1024 * 1024,
            max_len: 1000,
        };

        // Create an array with more than 255 distinct values to force U16 codes.
        let values: Vec<String> = (0..300).map(|i| format!("value_{i}")).collect();
        let arr =
            VarBinArray::from(values.iter().map(|s| s.as_str()).collect::<Vec<_>>()).into_array();

        assert_eq!(
            encoded_codes_ptype(arr, constraints)?,
            PType::U16,
            "codes should use U16 for large dictionaries"
        );
        Ok(())
    }
}

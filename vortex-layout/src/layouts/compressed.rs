// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::stats::Stat;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A boxed compressor function from arrays into compressed arrays.
///
/// API consumers are free to implement this trait to provide new plugin compressors.
pub trait CompressorPlugin: Send + Sync + 'static {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;
}

impl CompressorPlugin for Arc<dyn CompressorPlugin> {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.as_ref().compress_chunk(chunk, ctx)
    }
}

impl<F> CompressorPlugin for F
where
    F: Fn(&ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef> + Send + Sync + 'static,
{
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self(chunk, ctx)
    }
}

impl CompressorPlugin for BtrBlocksCompressor {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.compress(chunk, ctx)
    }
}

/// Keeps arrays of allowed encodings at given paths of a column, compressing only their children.
///
/// Each path is relative to the column being compressed: the root path names the column itself,
/// and deeper paths step through struct fields and list elements. An array at such a path whose
/// encoding is allowed there keeps that encoding, and each of its child slots is compressed with
/// the inner compressor. For example, a dictionary keeps its codes and values as a dictionary, so
/// a reader exporting the field to an Arrow dictionary gets the stored dictionary back instead of
/// re-encoding the decoded values. Everything else goes to the inner compressor.
pub struct AllowedEncodingsCompressor {
    inner: Arc<dyn CompressorPlugin>,
    allowed: Arc<[(FieldPath, Vec<ArrayId>)]>,
}

impl AllowedEncodingsCompressor {
    /// Creates a compressor that keeps the `allowed` encodings at their paths, compressing
    /// everything with `inner`.
    pub fn new<C: CompressorPlugin>(
        inner: C,
        allowed: impl IntoIterator<Item = (FieldPath, Vec<ArrayId>)>,
    ) -> Self {
        Self {
            inner: Arc::new(inner),
            allowed: allowed.into_iter().collect(),
        }
    }

    fn compress_at(
        &self,
        array: &ArrayRef,
        path: &FieldPath,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if self
            .allowed
            .iter()
            .any(|(at, ids)| at == path && ids.contains(&array.encoding_id()))
        {
            let slots = array
                .slots()
                .iter()
                .map(|slot| {
                    slot.as_ref()
                        .map(|child| self.inner.compress_chunk(child, ctx))
                        .transpose()
                })
                .collect::<VortexResult<ArraySlots>>()?;
            // SAFETY: compression rewrites each child slot to an equivalent physical
            // representation, preserving the parent array's logical values.
            return unsafe { array.clone().with_slots(slots) };
        }
        let below = |field: Field| {
            let child = path.clone().push(field);
            self.allowed
                .iter()
                .any(|(at, _)| at.parts().starts_with(child.parts()))
        };

        match array.dtype() {
            DType::Struct(fields, _)
                if fields
                    .names()
                    .iter()
                    .any(|name| below(Field::Name(name.clone()))) =>
            {
                let array = array.clone().execute::<StructArray>(ctx)?;
                let compressed = fields
                    .names()
                    .iter()
                    .enumerate()
                    .map(|(idx, name)| {
                        let field = array.unmasked_field(idx);
                        if below(Field::Name(name.clone())) {
                            self.compress_at(field, &path.clone().push(name.clone()), ctx)
                        } else {
                            self.inner.compress_chunk(field, ctx)
                        }
                    })
                    .collect::<VortexResult<Vec<_>>>()?;
                Ok(StructArray::try_new(
                    fields.names().clone(),
                    compressed,
                    array.len(),
                    array.validity()?,
                )?
                .into_array())
            }
            DType::List(..) if below(Field::ElementType) => {
                let list = match array.as_opt::<List>() {
                    Some(list) => list.into_owned(),
                    None => list_from_list_view(array.clone().execute::<ListViewArray>(ctx)?, ctx)?,
                };
                let elements =
                    self.compress_at(list.elements(), &path.clone().push(Field::ElementType), ctx)?;
                let offsets = self.inner.compress_chunk(list.offsets(), ctx)?;
                Ok(ListArray::try_new(elements, offsets, list.validity()?, ctx)?.into_array())
            }
            _ => self.inner.compress_chunk(array, ctx),
        }
    }
}

impl CompressorPlugin for AllowedEncodingsCompressor {
    fn compress_chunk(&self, chunk: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        self.compress_at(chunk, &FieldPath::root(), ctx)
    }
}

/// A layout writer that compresses chunks.
#[derive(Clone)]
pub struct CompressingStrategy {
    child: Arc<dyn LayoutStrategy>,
    compressor: Arc<dyn CompressorPlugin>,
    stats: Arc<[Stat]>,
    concurrency: usize,
}

impl CompressingStrategy {
    /// Create a new compressing layout strategy with the given child strategy and compressor.
    pub fn new<S: LayoutStrategy, C: CompressorPlugin>(child: S, compressor: C) -> Self {
        Self {
            child: Arc::new(child),
            compressor: Arc::new(compressor),
            stats: Stat::all().collect(),
            concurrency: get_available_parallelism().unwrap_or(1),
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Override the set of statistics computed on each chunk before compression.
    /// Defaults to `Stat::all()`.
    pub fn with_stats(mut self, stats: &[Stat]) -> Self {
        self.stats = stats.into();
        self
    }
}

#[async_trait]
impl LayoutStrategy for CompressingStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let compressor = Arc::clone(&self.compressor);
        let stats = Arc::clone(&self.stats);
        let session = session.clone();
        let compute_session = session.clone();

        let handle = session.handle();
        let stream = stream
            .map(move |chunk| {
                let compressor = Arc::clone(&compressor);
                let stats = Arc::clone(&stats);
                let session = compute_session.clone();
                handle.spawn_cpu(move || {
                    let (sequence_id, chunk) = chunk?;
                    let mut ctx = session.create_execution_ctx();
                    // Compute the stats for the chunk prior to compression. The exact
                    // uncompressed size is measured once here and cached for the compressor.
                    chunk.statistics().compute_all(&stats, &mut ctx)?;
                    if UncompressedSizeInBytes::supports_dtype(chunk.dtype()) {
                        uncompressed_size_in_bytes(&chunk, &mut ctx)?;
                    }
                    Ok((sequence_id, compressor.compress_chunk(&chunk, &mut ctx)?))
                })
            })
            .buffered(self.concurrency);

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
                eof,
                &session,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VTable;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_btrblocks::BtrBlocksCompressorBuilder;

    use super::*;
    use crate::test::new_session;

    /// A dictionary inside a list's elements keeps its encoding only where it is allowed.
    #[rstest]
    #[case::not_allowed(false)]
    #[case::allowed(true)]
    fn allowed_dict_in_list_elements_is_kept(#[case] allow: bool) -> VortexResult<()> {
        let mut ctx = new_session().create_execution_ctx();
        let values: Vec<String> = (0..6).map(|i| format!("value_{i}")).collect();
        let tags = DictArray::try_new(
            PrimitiveArray::from_iter(0..6u32).into_array(),
            VarBinViewArray::from_iter_str(values).into_array(),
        )?
        .into_array();
        let elements = StructArray::from_fields(&[("tag", tags)])?.into_array();
        let list = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter([0u32, 2, 6]).into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();

        let tag_path = FieldPath::root().push(Field::ElementType).push("tag");
        let allowed = allow.then(|| (tag_path, vec![Dict.id()]));
        let compressor =
            AllowedEncodingsCompressor::new(BtrBlocksCompressorBuilder::empty().build(), allowed);
        let compressed = compressor.compress_chunk(&list, &mut ctx)?;

        let tag = compressed
            .as_::<List>()
            .elements()
            .clone()
            .execute::<StructArray>(&mut ctx)?
            .unmasked_field_by_name("tag")?
            .clone();
        assert_eq!(tag.is::<Dict>(), allow, "{}", compressed.display_tree());
        assert_arrays_eq!(&compressed, &list, &mut ctx);
        Ok(())
    }
}

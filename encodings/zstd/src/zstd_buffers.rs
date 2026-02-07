// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use prost::Message as _;
use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::ArrayChildren;
use vortex_array::session::ArraySessionExt;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::VisitorVTable;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;

use crate::ZstdBuffersMetadata;

vtable!(ZstdBuffers);

#[derive(Debug)]
pub struct ZstdBuffersVTable;

impl ZstdBuffersVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.zstd_buffers");
}

/// An encoding that ZSTD-compresses the buffers of any wrapped array.
///
/// Unlike [`ZstdArray`](crate::ZstdArray), which interleaves string lengths with content bytes,
/// `ZstdBuffersArray` compresses each buffer independently. This enables zero-conversion
/// GPU decompression since the original buffer layout is preserved after decompression.
#[derive(Clone, Debug)]
pub struct ZstdBuffersArray {
    inner_encoding_id: ArrayId,
    inner_metadata: Vec<u8>,
    compressed_buffers: Vec<ByteBuffer>,
    uncompressed_sizes: Vec<u64>,
    buffer_alignments: Vec<u32>,
    children: Vec<ArrayRef>,
    dtype: DType,
    len: usize,
    stats_set: ArrayStats,
}

impl ZstdBuffersArray {
    /// Compresses the buffers of the given array using ZSTD.
    ///
    /// Each buffer of the input array is independently ZSTD-compressed. The children
    /// and metadata of the input array are preserved as-is.
    pub fn compress(array: &ArrayRef, level: i32) -> VortexResult<Self> {
        let encoding_id = array.encoding_id();
        let metadata = array
            .metadata()?
            .ok_or_else(|| vortex_err!("Array does not support serialization"))?;
        let buffer_handles = array.buffer_handles();
        let children = array.children();

        let mut compressed_buffers = Vec::with_capacity(buffer_handles.len());
        let mut uncompressed_sizes = Vec::with_capacity(buffer_handles.len());
        let mut buffer_alignments = Vec::with_capacity(buffer_handles.len());

        let mut compressor = zstd::bulk::Compressor::new(level)?;
        for handle in &buffer_handles {
            buffer_alignments.push(
                u32::try_from(*handle.alignment())
                    .map_err(|_| vortex_err!("Buffer alignment too large for u32"))?,
            );
            let host_buf = handle.clone().try_to_host_sync()?;
            uncompressed_sizes.push(host_buf.len() as u64);
            let compressed = compressor.compress(&host_buf)?;
            compressed_buffers.push(ByteBuffer::from(compressed));
        }

        Ok(Self {
            inner_encoding_id: encoding_id,
            inner_metadata: metadata,
            compressed_buffers,
            uncompressed_sizes,
            buffer_alignments,
            children,
            dtype: array.dtype().clone(),
            len: array.len(),
            stats_set: Default::default(),
        })
    }

    fn decompress_buffers(&self) -> VortexResult<Vec<ByteBuffer>> {
        let mut decompressor = zstd::bulk::Decompressor::new()?;
        let mut result = Vec::with_capacity(self.compressed_buffers.len());
        for (i, (buf, &uncompressed_size)) in self
            .compressed_buffers
            .iter()
            .zip(&self.uncompressed_sizes)
            .enumerate()
        {
            let size = usize::try_from(uncompressed_size)?;
            let alignment = self.buffer_alignments.get(i).copied().unwrap_or(1);

            let decompressed = decompressor.decompress(buf, size)?;

            let aligned = Alignment::new(alignment as usize);
            let mut output = ByteBufferMut::with_capacity_aligned(size, aligned);
            output.extend_from_slice(&decompressed);
            result.push(output.freeze());
        }
        Ok(result)
    }

    fn rebuild_inner(
        &self,
        registry: &vortex_array::session::ArrayRegistry,
    ) -> VortexResult<ArrayRef> {
        let decompressed_buffers = self.decompress_buffers()?;
        let buffer_handles: Vec<BufferHandle> = decompressed_buffers
            .into_iter()
            .map(BufferHandle::new_host)
            .collect();

        let inner_vtable = registry
            .find(&self.inner_encoding_id)
            .ok_or_else(|| vortex_err!("Unknown inner encoding: {}", self.inner_encoding_id))?;

        let children_slice = self.children.as_slice();
        inner_vtable.build(
            self.inner_encoding_id.clone(),
            &self.dtype,
            self.len,
            &self.inner_metadata,
            &buffer_handles,
            &children_slice,
        )
    }
}

fn array_id_from_string(s: &str) -> ArrayId {
    ArrayId::new_arc(Arc::from(s))
}

impl VTable for ZstdBuffersVTable {
    type Array = ZstdBuffersArray;

    type Metadata = ProstMetadata<ZstdBuffersMetadata>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(array: &ZstdBuffersArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ZstdBuffersMetadata {
            inner_encoding_id: array.inner_encoding_id.to_string(),
            inner_metadata: array.inner_metadata.clone(),
            uncompressed_sizes: array.uncompressed_sizes.clone(),
            buffer_alignments: array.buffer_alignments.clone(),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.0.encode_to_vec()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ZstdBuffersMetadata::decode(buffer)?))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ZstdBuffersArray> {
        let compressed_buffers: Vec<ByteBuffer> = buffers
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;

        let child_arrays: Vec<ArrayRef> = (0..children.len())
            .map(|i| children.get(i, dtype, len))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(ZstdBuffersArray {
            inner_encoding_id: array_id_from_string(&metadata.0.inner_encoding_id),
            inner_metadata: metadata.0.inner_metadata.clone(),
            compressed_buffers,
            uncompressed_sizes: metadata.0.uncompressed_sizes.clone(),
            buffer_alignments: metadata.0.buffer_alignments.clone(),
            children: child_arrays,
            dtype: dtype.clone(),
            len,
            stats_set: Default::default(),
        })
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        array.children = children;
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let registry = ctx.session().arrays().registry().clone();
        let inner_array = array.rebuild_inner(&registry)?;
        inner_array.execute::<ArrayRef>(ctx)
    }
}

impl BaseArrayVTable<ZstdBuffersVTable> for ZstdBuffersVTable {
    fn len(array: &ZstdBuffersArray) -> usize {
        array.len
    }

    fn dtype(array: &ZstdBuffersArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ZstdBuffersArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &ZstdBuffersArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.inner_encoding_id.hash(state);
        array.inner_metadata.hash(state);
        for buf in &array.compressed_buffers {
            buf.array_hash(state, precision);
        }
        array.uncompressed_sizes.hash(state);
        array.buffer_alignments.hash(state);
        array.dtype.hash(state);
        array.len.hash(state);
        for child in &array.children {
            child.array_hash(state, precision);
        }
    }

    fn array_eq(array: &ZstdBuffersArray, other: &ZstdBuffersArray, precision: Precision) -> bool {
        array.inner_encoding_id == other.inner_encoding_id
            && array.inner_metadata == other.inner_metadata
            && array.compressed_buffers.len() == other.compressed_buffers.len()
            && array
                .compressed_buffers
                .iter()
                .zip(&other.compressed_buffers)
                .all(|(a, b)| a.array_eq(b, precision))
            && array.uncompressed_sizes == other.uncompressed_sizes
            && array.buffer_alignments == other.buffer_alignments
            && array.dtype == other.dtype
            && array.len == other.len
            && array.children.len() == other.children.len()
            && array
                .children
                .iter()
                .zip(&other.children)
                .all(|(a, b)| a.array_eq(b, precision))
    }
}

impl OperationsVTable<ZstdBuffersVTable> for ZstdBuffersVTable {
    fn scalar_at(array: &ZstdBuffersArray, index: usize) -> VortexResult<Scalar> {
        let session = vortex_array::session::ArraySession::default();
        let inner_array = array.rebuild_inner(session.registry())?;
        inner_array.scalar_at(index)
    }
}

impl ValidityVTable<ZstdBuffersVTable> for ZstdBuffersVTable {
    fn validity(array: &ZstdBuffersArray) -> VortexResult<vortex_array::validity::Validity> {
        Ok(vortex_array::validity::Validity::from(
            array.dtype.nullability(),
        ))
    }
}

impl VisitorVTable<ZstdBuffersVTable> for ZstdBuffersVTable {
    fn visit_buffers(array: &ZstdBuffersArray, visitor: &mut dyn ArrayBufferVisitor) {
        for (i, buffer) in array.compressed_buffers.iter().enumerate() {
            visitor.visit_buffer_handle(
                &format!("compressed_{i}"),
                &BufferHandle::new_host(buffer.clone()),
            );
        }
    }

    fn visit_children(array: &ZstdBuffersArray, visitor: &mut dyn ArrayChildVisitor) {
        for (i, child) in array.children.iter().enumerate() {
            visitor.visit_child(&format!("child_{i}"), child);
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;

    use super::*;

    fn make_primitive_array() -> ArrayRef {
        PrimitiveArray::from_iter(0i32..100).into_array()
    }

    fn make_varbinview_array() -> ArrayRef {
        VarBinViewArray::from_iter_str(["hello", "world", "foo", "bar", "a longer string here"])
            .into_array()
    }

    fn make_nullable_primitive_array() -> ArrayRef {
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]).into_array()
    }

    fn make_nullable_varbinview_array() -> ArrayRef {
        VarBinViewArray::from_iter_nullable_str([
            Some("hello"),
            None,
            Some("world"),
            None,
            Some("a moderately long string for testing"),
        ])
        .into_array()
    }

    fn make_empty_primitive_array() -> ArrayRef {
        PrimitiveArray::from_iter(Vec::<i32>::new()).into_array()
    }

    fn make_inlined_varbinview_array() -> ArrayRef {
        VarBinViewArray::from_iter_str(["hi", "ok", "yes", "no"]).into_array()
    }

    #[rstest]
    #[case::primitive(make_primitive_array())]
    #[case::varbinview(make_varbinview_array())]
    #[case::nullable_primitive(make_nullable_primitive_array())]
    #[case::nullable_varbinview(make_nullable_varbinview_array())]
    #[case::empty_primitive(make_empty_primitive_array())]
    #[case::inlined_varbinview(make_inlined_varbinview_array())]
    fn test_roundtrip(#[case] input: ArrayRef) -> VortexResult<()> {
        let compressed = ZstdBuffersArray::compress(&input, 3)?;

        assert_eq!(compressed.len, input.len());
        assert_eq!(&compressed.dtype, input.dtype());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decompressed = compressed.into_array().execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(input, decompressed);
        Ok(())
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::Precision;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::session::ArraySessionExt;
use vortex_array::vtable;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ZstdBuffersMetadata;

vtable!(ZstdBuffers, ZstdBuffers, ZstdBuffersData);

#[derive(Clone, Debug)]
pub struct ZstdBuffers;

impl ZstdBuffers {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.zstd_buffers");

    pub fn try_new(
        dtype: DType,
        len: usize,
        data: ZstdBuffersData,
    ) -> VortexResult<ZstdBuffersArray> {
        Array::try_from_parts(ArrayParts::new(ZstdBuffers, dtype, len, data))
    }

    pub fn compress(array: &ArrayRef, level: i32) -> VortexResult<ZstdBuffersArray> {
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
        // Compression is currently CPU-only, so we gather all buffers on the host.
        for handle in &buffer_handles {
            buffer_alignments.push(u32::from(handle.alignment()));
            let host_buf = handle.clone().try_to_host_sync()?;
            uncompressed_sizes.push(host_buf.len() as u64);
            let compressed = compressor.compress(&host_buf)?;
            compressed_buffers.push(BufferHandle::new_host(ByteBuffer::from(compressed)));
        }

        let data = ZstdBuffersData {
            inner_encoding_id: encoding_id,
            inner_metadata: metadata,
            compressed_buffers,
            uncompressed_sizes,
            buffer_alignments,
        };
        let slots = children.into_iter().map(Some).collect();
        let compressed = Array::try_from_parts(
            ArrayParts::new(ZstdBuffers, array.dtype().clone(), array.len(), data)
                .with_slots(slots),
        )?;
        compressed.statistics().inherit_from(array.statistics());
        Ok(compressed)
    }

    pub fn build_inner(
        array: &ZstdBuffersArray,
        buffer_handles: &[BufferHandle],
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let registry = session.arrays().registry().clone();
        let inner_vtable = registry
            .find(&array.data().inner_encoding_id)
            .ok_or_else(|| {
                vortex_err!("Unknown inner encoding: {}", array.data().inner_encoding_id)
            })?;

        let children: Vec<ArrayRef> = array.slots().iter().flatten().cloned().collect();
        inner_vtable.deserialize(
            array.dtype(),
            array.len(),
            &array.data().inner_metadata,
            buffer_handles,
            &children.as_slice(),
            session,
        )
    }

    fn decompress_and_build_inner(
        array: &ZstdBuffersArray,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let decompressed_buffers = array.data().decompress_buffers()?;
        Self::build_inner(array, &decompressed_buffers, session)
    }
}

/// An encoding that ZSTD-compresses the buffers of any wrapped array.
///
/// Unlike [`ZstdArray`](crate::ZstdArray), which interleaves string lengths with content bytes,
/// `ZstdBuffersArray` compresses each buffer independently. This enables zero-conversion
/// GPU decompression since the original buffer layout is preserved after decompression.
#[derive(Clone, Debug)]
pub struct ZstdBuffersData {
    inner_encoding_id: ArrayId,
    inner_metadata: Vec<u8>,
    compressed_buffers: Vec<BufferHandle>,
    uncompressed_sizes: Vec<u64>,
    buffer_alignments: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ZstdBuffersDecodePlan {
    compressed_buffers: Vec<BufferHandle>,
    frame_sizes: Arc<[usize]>,
    output_sizes: Arc<[usize]>,
    output_offsets: Vec<usize>,
    output_alignments: Vec<Alignment>,
    output_size_total: usize,
    output_size_max: usize,
}

impl ZstdBuffersDecodePlan {
    pub fn compressed_buffers(&self) -> &[BufferHandle] {
        &self.compressed_buffers
    }

    pub fn frame_sizes(&self) -> Arc<[usize]> {
        Arc::clone(&self.frame_sizes)
    }

    pub fn output_sizes(&self) -> Arc<[usize]> {
        Arc::clone(&self.output_sizes)
    }

    pub fn output_offsets(&self) -> &[usize] {
        &self.output_offsets
    }

    pub fn output_size_total(&self) -> usize {
        self.output_size_total
    }

    pub fn output_size_max(&self) -> usize {
        self.output_size_max
    }

    pub fn num_frames(&self) -> usize {
        self.compressed_buffers.len()
    }

    /// Split a contiguous decompressed output buffer into per-buffer handles using planned
    /// offsets/sizes and enforce each buffer's required alignment.
    pub fn split_output_handle(
        &self,
        output_handle: &BufferHandle,
    ) -> VortexResult<Vec<BufferHandle>> {
        self.output_offsets
            .iter()
            .zip(self.output_sizes.iter())
            .zip(self.output_alignments.iter())
            .map(|((&offset, &size), &alignment)| {
                output_handle
                    .slice(offset..offset + size)
                    .ensure_aligned(alignment)
            })
            .collect::<VortexResult<Vec<_>>>()
    }
}

impl ZstdBuffersData {
    fn validate(&self) -> VortexResult<()> {
        vortex_ensure_eq!(
            self.compressed_buffers.len(),
            self.uncompressed_sizes.len(),
            "zstd_buffers metadata mismatch: {} compressed buffers vs {} sizes",
            self.compressed_buffers.len(),
            self.uncompressed_sizes.len()
        );
        vortex_ensure_eq!(
            self.compressed_buffers.len(),
            self.buffer_alignments.len(),
            "zstd_buffers metadata mismatch: {} compressed buffers vs {} alignments",
            self.compressed_buffers.len(),
            self.buffer_alignments.len()
        );
        Ok(())
    }

    fn decompress_buffers(&self) -> VortexResult<Vec<BufferHandle>> {
        // CPU decode path: zstd::bulk works on host bytes, so compressed buffers are
        // materialized on the host via `try_to_host_sync`.
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

            let aligned = Alignment::try_from(alignment)?;
            let mut output = ByteBufferMut::with_capacity_aligned(size, aligned);
            let spare = output.spare_capacity_mut();

            // This is currently guaranteed, but still good to check because
            // of the unsafe calls below.
            if spare.len() < size {
                return Err(vortex_err!(
                    "Insufficient output capacity: expected at least {}, got {}",
                    size,
                    spare.len()
                ));
            }
            // SAFETY: we only expose the first `size` bytes and mark them initialized via
            // `set_len(size)` after zstd reports how many bytes were written.
            let dst =
                unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<u8>(), size) };
            let compressed = buf.clone().try_to_host_sync()?;
            let written = decompressor.decompress_to_buffer(compressed.as_slice(), dst)?;
            if written != size {
                return Err(vortex_err!(
                    "Decompressed size mismatch: expected {}, got {}",
                    size,
                    written
                ));
            }
            // SAFETY: zstd wrote exactly `size` initialized bytes into `dst`.
            unsafe { output.set_len(size) };
            result.push(BufferHandle::new_host(output.freeze()));
        }
        Ok(result)
    }

    pub fn decode_plan(&self) -> VortexResult<ZstdBuffersDecodePlan> {
        // If invariants are somehow broken, device decompression could have UB, so ensure
        // they still hold.
        self.validate()?;

        let output_sizes = self
            .uncompressed_sizes
            .iter()
            .map(|&size| usize::try_from(size))
            .collect::<Result<Vec<_>, _>>()?;
        let output_size_max = output_sizes.iter().copied().max().unwrap_or(0);

        let output_alignments = self
            .buffer_alignments
            .iter()
            .map(|&alignment| Alignment::try_from(alignment))
            .collect::<VortexResult<Vec<_>>>()?;

        let (output_offsets, output_size_total) =
            compute_output_layout(&output_sizes, &output_alignments);

        let compressed_buffers = self.compressed_buffers.clone();
        let frame_sizes: Arc<[usize]> = compressed_buffers
            .iter()
            .map(BufferHandle::len)
            .collect::<Vec<_>>()
            .into();
        let output_sizes: Arc<[usize]> = output_sizes.into();

        Ok(ZstdBuffersDecodePlan {
            compressed_buffers,
            frame_sizes,
            output_sizes,
            output_offsets,
            output_alignments,
            output_size_total,
            output_size_max,
        })
    }
}

fn compute_output_layout(
    output_sizes: &[usize],
    output_alignments: &[Alignment],
) -> (Vec<usize>, usize) {
    // Compute aligned offsets for each decompressed buffer in one contiguous output allocation.
    // Each buffer starts at the next multiple of its required alignment.
    let mut offsets = Vec::with_capacity(output_sizes.len());
    let mut total_size = 0usize;

    for (&size, &alignment) in output_sizes.iter().zip(output_alignments.iter()) {
        total_size = total_size.next_multiple_of(*alignment);
        offsets.push(total_size);
        total_size += size;
    }

    (offsets, total_size)
}

fn array_id_from_string(s: &str) -> ArrayId {
    ArrayId::new_arc(Arc::from(s))
}

impl ArrayHash for ZstdBuffersData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.inner_encoding_id.hash(state);
        self.inner_metadata.hash(state);
        for buf in &self.compressed_buffers {
            buf.array_hash(state, precision);
        }
        self.uncompressed_sizes.hash(state);
        self.buffer_alignments.hash(state);
    }
}

impl ArrayEq for ZstdBuffersData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.inner_encoding_id == other.inner_encoding_id
            && self.inner_metadata == other.inner_metadata
            && self.compressed_buffers.len() == other.compressed_buffers.len()
            && self
                .compressed_buffers
                .iter()
                .zip(&other.compressed_buffers)
                .all(|(a, b)| a.array_eq(b, precision))
            && self.uncompressed_sizes == other.uncompressed_sizes
            && self.buffer_alignments == other.buffer_alignments
    }
}

impl VTable for ZstdBuffers {
    type ArrayData = ZstdBuffersData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        _dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate()
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.compressed_buffers.len()
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        array.compressed_buffers[idx].clone()
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        Some(format!("compressed_{idx}"))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        format!("child_{idx}")
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            ZstdBuffersMetadata {
                inner_encoding_id: array.inner_encoding_id.to_string(),
                inner_metadata: array.inner_metadata.clone(),
                uncompressed_sizes: array.uncompressed_sizes.clone(),
                buffer_alignments: array.buffer_alignments.clone(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = ZstdBuffersMetadata::decode(metadata)?;
        let compressed_buffers: Vec<BufferHandle> = buffers.to_vec();

        let slots: Vec<Option<ArrayRef>> = (0..children.len())
            .map(|i| children.get(i, dtype, len).map(Some))
            .collect::<VortexResult<Vec<_>>>()?;

        let data = ZstdBuffersData {
            inner_encoding_id: array_id_from_string(&metadata.inner_encoding_id),
            inner_metadata: metadata.inner_metadata.clone(),
            compressed_buffers,
            uncompressed_sizes: metadata.uncompressed_sizes.clone(),
            buffer_alignments: metadata.buffer_alignments.clone(),
        };

        data.validate()?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    // with_slots handles child replacement via the slots mechanism

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let session = ctx.session();
        let inner_array = ZstdBuffers::decompress_and_build_inner(&array, session)?;
        inner_array
            .execute::<ArrayRef>(ctx)
            .map(ExecutionResult::done)
    }
}

impl OperationsVTable<ZstdBuffers> for ZstdBuffers {
    fn scalar_at(
        array: ArrayView<'_, ZstdBuffers>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // TODO(os): maybe we should not support scalar_at, it is really slow, and adding a cache
        // layer here is weird. Valid use of zstd buffers array would be by executing it first into
        // canonical
        let inner_array = ZstdBuffers::decompress_and_build_inner(
            &array.into_owned(),
            &vortex_array::LEGACY_SESSION,
        )?;
        inner_array.scalar_at(index)
    }
}

impl ValidityVTable<ZstdBuffers> for ZstdBuffers {
    fn validity(
        array: ArrayView<'_, ZstdBuffers>,
    ) -> VortexResult<vortex_array::validity::Validity> {
        if !array.dtype().is_nullable() {
            return Ok(vortex_array::validity::Validity::NonNullable);
        }

        let inner_array = ZstdBuffers::decompress_and_build_inner(
            &array.into_owned(),
            &vortex_array::LEGACY_SESSION,
        )?;
        inner_array.validity()
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
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProvider;
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
        let compressed = ZstdBuffers::compress(&input, 3)?;

        assert_eq!(compressed.len(), input.len());
        assert_eq!(compressed.dtype(), input.dtype());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decompressed = compressed.into_array().execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(input, decompressed);
        Ok(())
    }

    #[test]
    fn test_compress_inherits_stats() -> VortexResult<()> {
        let input = make_primitive_array();
        input.statistics().set(Stat::Min, Precision::exact(0i32));

        let compressed = ZstdBuffers::compress(&input, 3)?;

        assert!(compressed.statistics().get(Stat::Min).is_some());
        Ok(())
    }

    #[test]
    fn test_validity_delegates_for_nullable_input() -> VortexResult<()> {
        let input = make_nullable_primitive_array();
        let compressed = ZstdBuffers::compress(&input, 3)?.into_array();

        assert_eq!(compressed.all_valid()?, input.all_valid()?);
        assert_eq!(compressed.all_invalid()?, input.all_invalid()?);

        for i in 0..input.len() {
            assert_eq!(compressed.is_valid(i)?, input.is_valid(i)?);
        }

        Ok(())
    }
}

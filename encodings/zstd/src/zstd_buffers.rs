// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use prost::Message as _;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::session::ArraySessionExt;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
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

vtable!(ZstdBuffers);

#[derive(Clone, Debug)]
pub struct ZstdBuffers;

impl ZstdBuffers {
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
    compressed_buffers: Vec<BufferHandle>,
    uncompressed_sizes: Vec<u64>,
    buffer_alignments: Vec<u32>,
    pub(crate) slots: Vec<Option<ArrayRef>>,
    dtype: DType,
    len: usize,
    stats_set: ArrayStats,
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

impl ZstdBuffersArray {
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
        // Compression is currently CPU-only, so we gather all buffers on the host.
        for handle in &buffer_handles {
            buffer_alignments.push(u32::from(handle.alignment()));
            let host_buf = handle.clone().try_to_host_sync()?;
            uncompressed_sizes.push(host_buf.len() as u64);
            let compressed = compressor.compress(&host_buf)?;
            compressed_buffers.push(BufferHandle::new_host(ByteBuffer::from(compressed)));
        }

        let compressed = Self {
            inner_encoding_id: encoding_id,
            inner_metadata: metadata,
            compressed_buffers,
            uncompressed_sizes,
            buffer_alignments,
            slots: children.into_iter().map(Some).collect(),
            dtype: array.dtype().clone(),
            len: array.len(),
            stats_set: Default::default(),
        };
        compressed
            .stats_set
            .to_ref(compressed.as_ref())
            .inherit_from(array.statistics());
        Ok(compressed)
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

    fn decompress_and_build_inner(&self, session: &VortexSession) -> VortexResult<ArrayRef> {
        let decompressed_buffers = self.decompress_buffers()?;
        self.build_inner(&decompressed_buffers, session)
    }

    // This is exposed to help non-CPU executors pass uncompressed buffer handles
    // to build the inner array.
    pub fn build_inner(
        &self,
        buffer_handles: &[BufferHandle],
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let registry = session.arrays().registry().clone();
        let inner_vtable = registry
            .find(&self.inner_encoding_id)
            .ok_or_else(|| vortex_err!("Unknown inner encoding: {}", self.inner_encoding_id))?;

        let children: Vec<ArrayRef> = self.slots.iter().flatten().cloned().collect();
        inner_vtable.build(
            self.inner_encoding_id.clone(),
            &self.dtype,
            self.len,
            &self.inner_metadata,
            buffer_handles,
            &children.as_slice(),
            session,
        )
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

impl VTable for ZstdBuffers {
    type Array = ZstdBuffersArray;

    type Metadata = ProstMetadata<ZstdBuffersMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::Array) -> &Self {
        &ZstdBuffers
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

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
        for child in array.slots.iter().flatten() {
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
            && array.slots.len() == other.slots.len()
            && array
                .slots
                .iter()
                .flatten()
                .zip(other.slots.iter().flatten())
                .all(|(a, b)| a.array_eq(b, precision))
    }

    fn nbuffers(array: &ZstdBuffersArray) -> usize {
        array.compressed_buffers.len()
    }

    fn buffer(array: &ZstdBuffersArray, idx: usize) -> BufferHandle {
        array.compressed_buffers[idx].clone()
    }

    fn buffer_name(_array: &ZstdBuffersArray, idx: usize) -> Option<String> {
        Some(format!("compressed_{idx}"))
    }

    fn slots(array: &ZstdBuffersArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &ZstdBuffersArray, idx: usize) -> String {
        format!("child_{idx}")
    }

    fn with_slots(array: &mut ZstdBuffersArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        array.slots = slots;
        Ok(())
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

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ZstdBuffersMetadata::decode(bytes)?))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ZstdBuffersArray> {
        let compressed_buffers: Vec<BufferHandle> = buffers.to_vec();

        let child_arrays: Vec<Option<ArrayRef>> = (0..children.len())
            .map(|i| children.get(i, dtype, len).map(Some))
            .collect::<VortexResult<Vec<_>>>()?;

        let array = ZstdBuffersArray {
            inner_encoding_id: array_id_from_string(&metadata.0.inner_encoding_id),
            inner_metadata: metadata.0.inner_metadata.clone(),
            compressed_buffers,
            uncompressed_sizes: metadata.0.uncompressed_sizes.clone(),
            buffer_alignments: metadata.0.buffer_alignments.clone(),
            slots: child_arrays,
            dtype: dtype.clone(),
            len,
            stats_set: Default::default(),
        };

        array.validate()?;
        Ok(array)
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let session = ctx.session();
        let inner_array = array.decompress_and_build_inner(session)?;
        inner_array
            .execute::<ArrayRef>(ctx)
            .map(ExecutionResult::done)
    }
}

impl OperationsVTable<ZstdBuffers> for ZstdBuffers {
    fn scalar_at(
        array: &ZstdBuffersArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // TODO(os): maybe we should not support scalar_at, it is really slow, and adding a cache
        // layer here is weird. Valid use of zstd buffers array would be by executing it first into
        // canonical
        let inner_array = array.decompress_and_build_inner(&vortex_array::LEGACY_SESSION)?;
        inner_array.scalar_at(index)
    }
}

impl ValidityVTable<ZstdBuffers> for ZstdBuffers {
    fn validity(array: &ZstdBuffersArray) -> VortexResult<vortex_array::validity::Validity> {
        if !array.dtype.is_nullable() {
            return Ok(vortex_array::validity::Validity::NonNullable);
        }

        let inner_array = array.decompress_and_build_inner(&vortex_array::LEGACY_SESSION)?;
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
        let compressed = ZstdBuffersArray::compress(&input, 3)?;

        assert_eq!(compressed.len, input.len());
        assert_eq!(&compressed.dtype, input.dtype());

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let decompressed = compressed.into_array().execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(input, decompressed);
        Ok(())
    }

    #[test]
    fn test_compress_inherits_stats() -> VortexResult<()> {
        let input = make_primitive_array();
        input.statistics().set(Stat::Min, Precision::exact(0i32));

        let compressed = ZstdBuffersArray::compress(&input, 3)?;

        assert!(compressed.statistics().get(Stat::Min).is_some());
        Ok(())
    }

    #[test]
    fn test_validity_delegates_for_nullable_input() -> VortexResult<()> {
        let input = make_nullable_primitive_array();
        let compressed = ZstdBuffersArray::compress(&input, 3)?.into_array();

        assert_eq!(compressed.all_valid()?, input.all_valid()?);
        assert_eq!(compressed.all_invalid()?, input.all_invalid()?);

        for i in 0..input.len() {
            assert_eq!(compressed.is_valid(i)?, input.is_valid(i)?);
        }

        Ok(())
    }
}

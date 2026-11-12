// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::decompressor::with_decompressor;
use crate::validate_frame_content_size;

mod plugin;

/// A [`ZstdBuffers`]-encoded Vortex array.
pub type ZstdBuffersArray = Array<ZstdBuffers>;

#[derive(Clone, Debug)]
/// Encoding marker for buffer-level zstd compression.
pub struct ZstdBuffers;

impl ZstdBuffers {
    /// Construct a [`ZstdBuffersArray`] from compressed buffer data.
    pub fn try_new(
        dtype: DType,
        len: usize,
        data: ZstdBuffersData,
    ) -> VortexResult<ZstdBuffersArray> {
        Array::try_from_parts(ArrayParts::new(
            ZstdBuffers,
            dtype,
            len,
            data,
            ArraySlots::new(),
        ))
    }

    /// Compress every top-level buffer of `array` independently with zstd.
    ///
    /// The wrapped array's serialized representation is captured so it can be rebuilt after
    /// decompression, including any buffers or children selected by its serializer.
    pub fn compress(
        array: &ArrayRef,
        level: i32,
        session: &VortexSession,
    ) -> VortexResult<ZstdBuffersArray> {
        let serialization = session
            .array_serialize(array)?
            .ok_or_else(|| vortex_err!("[ZstdBuffers]: Array does not support serialization"))?;

        let mut compressed_buffers = Vec::with_capacity(serialization.buffers.len());
        let mut uncompressed_sizes = Vec::with_capacity(serialization.buffers.len());
        let mut buffer_alignments = Vec::with_capacity(serialization.buffers.len());

        let mut compressor = zstd::bulk::Compressor::new(level)?;
        for buffer in &serialization.buffers {
            buffer_alignments.push(u32::from(buffer.alignment()));
            uncompressed_sizes.push(buffer.len() as u64);
            let mut compressed = compressor.compress(buffer)?;
            compressed.shrink_to_fit();
            compressed_buffers.push(BufferHandle::new_host(ByteBuffer::from(compressed)));
        }

        let mut slots: Vec<Option<ArrayRef>> =
            serialization.children.into_iter().map(Some).collect();
        let inner_child_count = slots.len();
        let validity_child = lift_validity(&array.validity()?, array.len(), &mut slots);

        let data = ZstdBuffersData {
            inner_encoding_id: serialization.serialized_id,
            inner_metadata: serialization.metadata,
            compressed_buffers,
            uncompressed_sizes,
            buffer_alignments,
            inner_child_count,
            validity_child,
        };
        let slots: ArraySlots = slots.into();
        let compressed = Array::try_from_parts(ArrayParts::new(
            ZstdBuffers,
            array.dtype().clone(),
            array.len(),
            data,
            slots,
        ))?;
        compressed.statistics().inherit_from(array.statistics());
        Ok(compressed)
    }

    /// Rebuild the wrapped array from decompressed buffer handles.
    pub fn build_inner(
        array: &ZstdBuffersArray,
        buffer_handles: &[BufferHandle],
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let inner_vtable = session
            .arrays()
            .registry()
            .get(&array.data().inner_encoding_id)
            .ok_or_else(|| {
                vortex_err!("Unknown inner encoding: {}", array.data().inner_encoding_id)
            })?;

        let children: Vec<ArrayRef> = array.slots()[..array.data().inner_child_count]
            .iter()
            .flatten()
            .cloned()
            .collect();
        inner_vtable.deserialize(
            ArrayDeserialization::new(
                array.data().inner_encoding_id,
                array.dtype(),
                array.len(),
                &array.data().inner_metadata,
                buffer_handles,
                &children.as_slice(),
            ),
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
    /// Number of leading slots that are children of the wrapped array.
    inner_child_count: usize,
    /// Slot holding the wrapped array's validity, if it is not implied by the dtype.
    ///
    /// This is either one of the wrapped array's children, or one extra slot after them.
    validity_child: Option<usize>,
}

impl Display for ZstdBuffersData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "inner_encoding: {}", self.inner_encoding_id)
    }
}

#[derive(Clone, Debug)]
/// Decode plan for buffer-level zstd decompression.
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
    /// Compressed buffers to decode.
    pub fn compressed_buffers(&self) -> &[BufferHandle] {
        &self.compressed_buffers
    }

    /// Compressed frame sizes in bytes.
    pub fn frame_sizes(&self) -> Arc<[usize]> {
        Arc::clone(&self.frame_sizes)
    }

    /// Decompressed output size for each buffer.
    pub fn output_sizes(&self) -> Arc<[usize]> {
        Arc::clone(&self.output_sizes)
    }

    /// Byte offsets of each decompressed buffer in one contiguous output allocation.
    pub fn output_offsets(&self) -> &[usize] {
        &self.output_offsets
    }

    /// Total byte size of the planned contiguous output allocation.
    pub fn output_size_total(&self) -> usize {
        self.output_size_total
    }

    /// Largest single decompressed buffer size.
    pub fn output_size_max(&self) -> usize {
        self.output_size_max
    }

    /// Number of compressed frames/buffers in the plan.
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
            "zstd_buffers metadata mismatch"
        );
        vortex_ensure_eq!(
            self.compressed_buffers.len(),
            self.buffer_alignments.len(),
            "zstd_buffers metadata mismatch"
        );
        Ok(())
    }

    fn decompress_buffers(&self) -> VortexResult<Vec<BufferHandle>> {
        // CPU decode path: zstd::bulk works on host bytes, so compressed buffers are
        // materialized on the host via `try_to_host_sync`.
        with_decompressor(|decompressor| {
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
                let compressed = buf.clone().try_to_host_sync()?;
                validate_frame_content_size(compressed.as_slice(), uncompressed_size, i)?;
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
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(spare.as_mut_ptr().cast::<u8>(), size)
                };
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
        })
    }

    /// Build a decode plan for external or device decompression.
    pub fn decode_plan(&self) -> VortexResult<ZstdBuffersDecodePlan> {
        // If invariants are somehow broken, device decompression could have UB, so ensure
        // they still hold.
        self.validate()?;
        // Only host-resident frames are checked: reading a device frame's header would cost a D2H
        // copy per frame, and the device output allocation is fallible so it cannot abort.
        for (index, (buffer, &metadata_size)) in self
            .compressed_buffers
            .iter()
            .zip(&self.uncompressed_sizes)
            .enumerate()
        {
            if let Some(frame) = buffer.as_host_opt() {
                validate_frame_content_size(frame.as_slice(), metadata_size, index)?;
            }
        }

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
        total_size = total_size.next_multiple_of(alignment.as_usize());
        offsets.push(total_size);
        total_size += size;
    }

    (offsets, total_size)
}

/// Store `validity` in `slots`, reusing a child of the wrapped array when it already holds it.
///
/// Returns the index of the slot holding the validity, if it is not implied by the dtype.
fn lift_validity(
    validity: &Validity,
    len: usize,
    slots: &mut Vec<Option<ArrayRef>>,
) -> Option<usize> {
    let validity = validity_to_child(validity, len)?;
    let shared = slots.iter().position(|slot| {
        slot.as_ref()
            .is_some_and(|c| ArrayRef::ptr_eq(c, &validity))
    });
    Some(shared.unwrap_or_else(|| {
        slots.push(Some(validity));
        slots.len() - 1
    }))
}

#[expect(clippy::disallowed_methods, reason = "interning a dynamic id")]
fn array_id_from_string(s: &str) -> ArrayId {
    ArrayId::new(s)
}

impl ArrayHash for ZstdBuffersData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.inner_encoding_id.hash(state);
        self.inner_metadata.hash(state);
        for buf in &self.compressed_buffers {
            buf.array_hash(state, accuracy);
        }
        self.uncompressed_sizes.hash(state);
        self.buffer_alignments.hash(state);
        self.inner_child_count.hash(state);
        self.validity_child.hash(state);
    }
}

impl ArrayEq for ZstdBuffersData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.inner_encoding_id == other.inner_encoding_id
            && self.inner_metadata == other.inner_metadata
            && self.compressed_buffers.len() == other.compressed_buffers.len()
            && self
                .compressed_buffers
                .iter()
                .zip(&other.compressed_buffers)
                .all(|(a, b)| a.array_eq(b, accuracy))
            && self.uncompressed_sizes == other.uncompressed_sizes
            && self.buffer_alignments == other.buffer_alignments
            && self.inner_child_count == other.inner_child_count
            && self.validity_child == other.validity_child
    }
}

impl VTable for ZstdBuffers {
    type TypedArrayData = ZstdBuffersData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.zstd_buffers");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate()?;
        vortex_ensure!(
            data.inner_child_count <= slots.len(),
            "ZstdBuffers has {} slots but {} inner children",
            slots.len(),
            data.inner_child_count
        );
        let extra_slots = slots.len() - data.inner_child_count;
        match data.validity_child {
            Some(idx) => {
                vortex_ensure!(
                    dtype.is_nullable(),
                    "ZstdBuffers validity requires a nullable dtype"
                );
                vortex_ensure!(
                    idx < slots.len(),
                    "ZstdBuffers validity slot {idx} out of bounds"
                );
                vortex_ensure!(
                    extra_slots == usize::from(idx == data.inner_child_count),
                    "ZstdBuffers has {extra_slots} unexpected extra slots"
                );
                let validity = slots[idx]
                    .as_ref()
                    .ok_or_else(|| vortex_err!("ZstdBuffers validity slot {idx} is empty"))?;
                vortex_ensure_eq!(validity.dtype(), &Validity::DTYPE);
                vortex_ensure_eq!(validity.len(), len);
            }
            None => {
                vortex_ensure_eq!(extra_slots, 0, "ZstdBuffers has unexpected extra slots")
            }
        }
        Ok(())
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let mut data = array.data().clone();
        data.compressed_buffers = buffers.to_vec();
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
            array.slots().iter().cloned().collect(),
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        format!("child_{idx}")
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
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, ZstdBuffers>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // TODO(os): maybe we should not support scalar_at, it is really slow, and adding a cache
        // layer here is weird. Valid use of zstd buffers array would be by executing it first into
        // canonical
        let inner_array =
            ZstdBuffers::decompress_and_build_inner(&array.into_owned(), ctx.session())?;
        inner_array.execute_scalar(index, ctx)
    }
}

impl ValidityVTable<ZstdBuffers> for ZstdBuffers {
    fn validity(array: ArrayView<'_, ZstdBuffers>) -> VortexResult<Validity> {
        let child = array
            .data()
            .validity_child
            .and_then(|idx| array.slots()[idx].as_ref());
        Ok(child_to_validity(child, array.dtype().nullability()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use prost::Message;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::stats::Precision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProvider;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::ZstdBuffersMetadata;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

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
        let compressed = ZstdBuffers::compress(&input, 3, &SESSION)?;

        assert_eq!(compressed.len(), input.len());
        assert_eq!(compressed.dtype(), input.dtype());

        let mut ctx = SESSION.create_execution_ctx();

        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::primitive(make_primitive_array())]
    #[case::varbinview(make_varbinview_array())]
    #[case::nullable_primitive(make_nullable_primitive_array())]
    #[case::nullable_varbinview(make_nullable_varbinview_array())]
    #[case::empty_primitive(make_empty_primitive_array())]
    #[case::inlined_varbinview(make_inlined_varbinview_array())]
    fn test_serde_roundtrip(#[case] input: ArrayRef) -> VortexResult<()> {
        let session = VortexSession::empty();
        session.arrays().register(ZstdBuffers);

        let compressed = ZstdBuffers::compress(&input, 3, &session)?.into_array();
        let dtype = compressed.dtype().clone();
        let len = compressed.len();

        let array_ctx = ArrayContext::empty();
        let serialized =
            compressed.serialize(&array_ctx, &session, &SerializeOptions::default())?;

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze())?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;

        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(decoded, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_compress_inherits_stats() -> VortexResult<()> {
        let input = make_primitive_array();
        input.statistics().set(Stat::Min, Precision::exact(0i32));

        let compressed = ZstdBuffers::compress(&input, 3, &SESSION)?;

        assert_eq!(
            compressed.statistics().get(Stat::Min),
            Precision::exact(0i32)
        );
        Ok(())
    }

    #[test]
    fn test_rejects_mismatched_frame_content_size_before_output_allocation() -> VortexResult<()> {
        let compressed = ZstdBuffers::compress(&make_primitive_array(), 3, &SESSION)?;
        let mut data = compressed.data().clone();
        data.uncompressed_sizes[0] = 16 * 1024 * 1024 * 1024;

        for error in [
            data.decompress_buffers().unwrap_err(),
            data.decode_plan().unwrap_err(),
        ] {
            assert!(
                error.to_string().contains("metadata declares"),
                "unexpected error: {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_validity_delegates_for_nullable_input() -> VortexResult<()> {
        let input = make_nullable_primitive_array();
        let compressed = ZstdBuffers::compress(&input, 3, &SESSION)?.into_array();

        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(compressed.all_valid(&mut ctx)?, input.all_valid(&mut ctx)?);
        assert_eq!(
            compressed.all_invalid(&mut ctx)?,
            input.all_invalid(&mut ctx)?
        );

        for i in 0..input.len() {
            assert_eq!(
                compressed.is_valid(i, &mut ctx)?,
                input.is_valid(i, &mut ctx)?
            );
        }

        Ok(())
    }

    #[test]
    fn test_validity_shares_inner_child() -> VortexResult<()> {
        let compressed = ZstdBuffers::compress(&make_nullable_primitive_array(), 3, &SESSION)?;

        assert_eq!(
            compressed.slots().len(),
            compressed.data().inner_child_count
        );
        assert!(compressed.data().validity_child.is_some());
        Ok(())
    }

    #[test]
    fn test_all_invalid_validity_uses_extra_slot() -> VortexResult<()> {
        let input = PrimitiveArray::new(vortex_buffer::buffer![1i32, 2, 3], Validity::AllInvalid)
            .into_array();
        let compressed = ZstdBuffers::compress(&input, 3, &SESSION)?;

        let inner_child_count = compressed.data().inner_child_count;
        assert_eq!(compressed.slots().len(), inner_child_count + 1);
        assert_eq!(compressed.data().validity_child, Some(inner_child_count));

        let mut ctx = SESSION.create_execution_ctx();
        assert!(compressed.into_array().all_invalid(&mut ctx)?);
        Ok(())
    }

    #[test]
    fn test_deserialize_without_lifted_validity() -> VortexResult<()> {
        let session = VortexSession::empty();
        let input = make_nullable_primitive_array();
        let compressed = ZstdBuffers::compress(&input, 3, &session)?;

        let mut metadata = ZstdBuffersMetadata::decode(
            ArrayPlugin::serialize(&ZstdBuffers, compressed.as_array(), &session)?
                .vortex_expect("metadata")
                .metadata
                .as_slice(),
        )?;
        metadata.inner_child_count = None;
        metadata.validity_child = None;

        let children: Vec<ArrayRef> = compressed.slots().iter().flatten().cloned().collect();
        let decoded = ArrayPlugin::deserialize(
            &ZstdBuffers,
            ArrayDeserialization::new(
                VTable::id(&ZstdBuffers),
                input.dtype(),
                input.len(),
                &metadata.encode_to_vec(),
                &compressed.data().compressed_buffers,
                &children.as_slice(),
            ),
            &session,
        )?;
        assert!(decoded.as_::<ZstdBuffers>().data().validity_child.is_some());

        let mut ctx = session.create_execution_ctx();
        for i in 0..input.len() {
            assert_eq!(decoded.is_valid(i, &mut ctx)?, input.is_valid(i, &mut ctx)?);
        }
        Ok(())
    }
}

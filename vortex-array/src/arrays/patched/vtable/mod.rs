// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernels;
mod operations;
mod slice;

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::DeserializeMetadata;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::PrimitiveArray;
use crate::arrays::patched::PatchedArray;
use crate::arrays::patched::compute::rules::PARENT_RULES;
use crate::arrays::patched::vtable::kernels::PARENT_KERNELS;
use crate::arrays::primitive::PrimitiveArrayParts;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityChild;
use crate::vtable::ValidityVTableFromChild;

vtable!(Patched);

#[derive(Clone, Debug)]
pub struct Patched;

impl ValidityChild<Patched> for Patched {
    fn validity_child(array: &PatchedArray) -> &ArrayRef {
        &array.inner
    }
}

#[derive(Clone, prost::Message)]
pub struct PatchedMetadata {
    /// A bitfield packed into a single u64 containing all the metadata needed to decode a
    /// serialized `PatchedArray`.
    ///
    /// See [`PatchedMetadataFields`].
    #[prost(uint64, tag = "1")]
    pub(crate) packed: u64,
}

/// A bitfield implemented on top of a `u64` containing the necessary metadata for reading a
/// serialized `PatchedArray`.
///
/// The bit fields are in the following order:
///
/// * `offset`: 10 bits (always < 1024). An offset into the first chunk's patches that should be
///   considered in-view.
/// * `n_lanes_exp`: 3 bits. The binary exponent of `n_lanes`, which must be a power of two.
///   A stored value of 0b000 represents n_lanes=1, and 0b111 represents n_lanes=128.
/// * `n_patches`: 23 bits. The number of total patches, and the length of the indices and values
///   child arrays.
///
/// The remaining bits 36..64 are reserved for future use.
pub(crate) struct PatchedMetadataFields(u64);

impl std::fmt::Debug for PatchedMetadataFields {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatchedMetadataFields")
            .field("offset", &self.offset())
            .field("n_lanes", &self.n_lanes())
            .field("n_patches", &self.n_patches())
            .finish()
    }
}

impl PatchedMetadataFields {
    const OFFSET_BITS: u32 = 10;
    const N_LANES_EXP_BITS: u32 = 3;
    const N_PATCHES_BITS: u32 = 23;

    const OFFSET_MASK: u64 = (1 << Self::OFFSET_BITS) - 1;
    const N_LANES_EXP_MASK: u64 = (1 << Self::N_LANES_EXP_BITS) - 1;
    const N_PATCHES_MASK: u64 = (1 << Self::N_PATCHES_BITS) - 1;

    const OFFSET_SHIFT: u32 = 0;
    const N_LANES_EXP_SHIFT: u32 = Self::OFFSET_BITS;
    const N_PATCHES_SHIFT: u32 = Self::OFFSET_BITS + Self::N_LANES_EXP_BITS;

    /// Create a new `PatchedMetadataFields` from the component values.
    ///
    /// # Errors
    ///
    /// Returns an error if any value exceeds its bit width:
    /// - `offset` must be < 1024 (10 bits)
    /// - `n_lanes` must be a power of two between 1 and 128 inclusive
    /// - `n_patches` must be < 8388608 (23 bits)
    pub fn new(offset: usize, n_lanes: usize, n_patches: usize) -> VortexResult<Self> {
        vortex_ensure!(
            offset < (1 << Self::OFFSET_BITS),
            "offset must be < 1024, got {offset}"
        );
        vortex_ensure!(
            n_lanes.is_power_of_two() && n_lanes <= 128,
            "n_lanes must be a power of two between 1 and 128, got {n_lanes}"
        );
        vortex_ensure!(
            n_patches < (1 << Self::N_PATCHES_BITS),
            "n_patches must be < 8388608, got {n_patches}"
        );

        let n_lanes_exp = n_lanes.trailing_zeros() as u64;

        let flags = (offset as u64)
            | (n_lanes_exp << Self::N_LANES_EXP_SHIFT)
            | ((n_patches as u64) << Self::N_PATCHES_SHIFT);
        Ok(Self(flags))
    }

    /// Extract the offset field (bits 0..10).
    pub fn offset(&self) -> usize {
        ((self.0 >> Self::OFFSET_SHIFT) & Self::OFFSET_MASK) as usize
    }

    /// Extract the n_lanes field (bits 10..13), converted from the stored exponent.
    pub fn n_lanes(&self) -> usize {
        let exp = (self.0 >> Self::N_LANES_EXP_SHIFT) & Self::N_LANES_EXP_MASK;
        1 << exp
    }

    /// Extract the n_patches field (bits 13..36).
    pub fn n_patches(&self) -> usize {
        ((self.0 >> Self::N_PATCHES_SHIFT) & Self::N_PATCHES_MASK) as usize
    }

    /// Return the underlying u64 representation.
    pub fn into_inner(self) -> u64 {
        self.0
    }
}

impl From<u64> for PatchedMetadataFields {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl VTable for Patched {
    type Array = PatchedArray;
    type Metadata = ProstMetadata<PatchedMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &Patched
    }

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.patched")
    }

    fn len(array: &Self::Array) -> usize {
        array.len
    }

    fn dtype(array: &Self::Array) -> &DType {
        array.inner.dtype()
    }

    fn stats(array: &Self::Array) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &Self::Array, state: &mut H, precision: Precision) {
        array.len.hash(state);
        array.offset.hash(state);
        array.n_lanes.hash(state);
        array.inner.array_hash(state, precision);
        array.lane_offsets.array_hash(state, precision);
        array.indices.array_hash(state, precision);
        array.values.array_hash(state, precision);
    }

    fn array_eq(array: &Self::Array, other: &Self::Array, precision: Precision) -> bool {
        array.len == other.len
            && array.offset == other.offset
            && array.n_lanes == other.n_lanes
            && array.inner.array_eq(&other.inner, precision)
            && array.lane_offsets.array_eq(&other.lane_offsets, precision)
            && array.indices.array_eq(&other.indices, precision)
            && array.values.array_eq(&other.values, precision)
    }

    fn nbuffers(_array: &Self::Array) -> usize {
        0
    }

    fn buffer(_array: &Self::Array, idx: usize) -> BufferHandle {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn buffer_name(_array: &Self::Array, idx: usize) -> Option<String> {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn nchildren(_array: &Self::Array) -> usize {
        4
    }

    fn child(array: &Self::Array, idx: usize) -> ArrayRef {
        match idx {
            0 => array.inner.clone(),
            1 => array.lane_offsets.clone(),
            2 => array.indices.clone(),
            3 => array.values.clone(),
            _ => vortex_panic!("invalid child index for PatchedArray: {idx}"),
        }
    }

    fn child_name(_array: &Self::Array, idx: usize) -> String {
        match idx {
            0 => "inner".to_string(),
            1 => "lane_offsets".to_string(),
            2 => "patch_indices".to_string(),
            3 => "patch_values".to_string(),
            _ => vortex_panic!("invalid child index for PatchedArray: {idx}"),
        }
    }

    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata> {
        let fields = PatchedMetadataFields::new(array.offset, array.n_lanes, array.indices.len())?;

        Ok(ProstMetadata(PatchedMetadata {
            packed: fields.into_inner(),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let inner = <ProstMetadata<PatchedMetadata> as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(inner))
    }

    fn append_to_builder(
        array: &Self::Array,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let dtype = array.dtype();

        if !dtype.is_primitive() {
            // Default pathway: canonicalize and propagate.
            let canonical = array
                .clone()
                .into_array()
                .execute::<Canonical>(ctx)?
                .into_array();
            builder.extend_from_array(&canonical);
            return Ok(());
        }

        let ptype = dtype.as_ptype();

        let len = array.len();

        array.inner.append_to_builder(builder, ctx)?;

        let offset = array.offset;
        let lane_offsets = array.lane_offsets.clone().execute::<PrimitiveArray>(ctx)?;
        let indices = array.indices.clone().execute::<PrimitiveArray>(ctx)?;
        let values = array.values.clone().execute::<PrimitiveArray>(ctx)?;

        match_each_native_ptype!(ptype, |V| {
            let typed_builder = builder
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<V>>()
                .vortex_expect("correctly typed builder");

            // Overwrite the last `len` elements of the builder. These would have been
            // populated by the inner.append_to_builder() call above.
            let output = typed_builder.values_mut();
            let trailer = output.len() - len;

            apply_patches_primitive::<V>(
                &mut output[trailer..],
                offset,
                len,
                array.n_lanes,
                lane_offsets.as_slice::<u32>(),
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
            );
        });

        Ok(())
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<PatchedArray> {
        let fields = PatchedMetadataFields::from(metadata.packed);
        let offset = fields.offset();
        let n_lanes = fields.n_lanes();
        let n_patches = fields.n_patches();

        // n_chunks should correspond to the chunk in the `inner`.
        // After slicing when offset > 0, there may be additional chunks.
        let n_chunks = (len + offset).div_ceil(1024);

        let inner = children.get(0, dtype, len)?;
        let lane_offsets = children.get(1, PType::U32.into(), n_chunks * n_lanes + 1)?;
        let indices = children.get(2, PType::U16.into(), n_patches)?;
        let values = children.get(3, dtype, n_patches)?;

        Ok(PatchedArray {
            inner,
            n_lanes,
            offset,
            len,
            lane_offsets,
            indices,
            values,
            stats_set: ArrayStats::default(),
        })
    }

    fn with_children(array: &mut Self::Array, mut children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 4,
            "PatchedArray must have exactly 4 children"
        );

        let inner = children.remove(0);
        let lane_offsets = children.remove(0);
        let indices = children.remove(0);
        let values = children.remove(0);

        // We only execute these checks on debug builds, since this method is called in a tight
        //  loop during optimization, and we want to avoid the overhead of repeatedly checking the
        //  component types.
        if cfg!(debug_assertions) {
            vortex_ensure!(
                array.dtype().eq_with_nullability_superset(inner.dtype()),
                "inner array DType must match outer array DType"
            );

            vortex_ensure_eq!(
                lane_offsets.dtype(),
                &DType::from(PType::U32),
                "lane_offsets must have u32 type"
            );

            vortex_ensure_eq!(
                indices.dtype(),
                &DType::from(PType::U16),
                "indices must be u16 type"
            );
        }

        array.inner = inner;
        array.lane_offsets = lane_offsets;
        array.indices = indices;
        array.values = values;

        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let inner = array
            .inner
            .clone()
            .execute::<Canonical>(ctx)?
            .into_primitive();

        let PrimitiveArrayParts {
            buffer,
            ptype,
            validity,
        } = inner.into_parts();

        let lane_offsets = array.lane_offsets.clone().execute::<PrimitiveArray>(ctx)?;
        let indices = array.indices.clone().execute::<PrimitiveArray>(ctx)?;

        // TODO(aduffy): add support for non-primitive PatchedArray patches application (?)
        let values = array.values.clone().execute::<PrimitiveArray>(ctx)?;

        let patched_values = match_each_native_ptype!(values.ptype(), |V| {
            let offset = array.offset;
            let len = array.len;

            let mut output = Buffer::<V>::from_byte_buffer(buffer.unwrap_host()).into_mut();

            apply_patches_primitive::<V>(
                &mut output,
                offset,
                len,
                array.n_lanes,
                lane_offsets.as_slice::<u32>(),
                indices.as_slice::<u16>(),
                values.as_slice::<V>(),
            );

            let output = output.freeze();

            PrimitiveArray::from_byte_buffer(output.into_byte_buffer(), ptype, validity)
        });

        Ok(ExecutionResult::done(patched_values.into_array()))
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Apply patches on top of the existing value types.
#[allow(clippy::too_many_arguments)]
fn apply_patches_primitive<V: NativePType>(
    output: &mut [V],
    offset: usize,
    len: usize,
    n_lanes: usize,
    lane_offsets: &[u32],
    indices: &[u16],
    values: &[V],
) {
    let n_chunks = (offset + len).div_ceil(1024);
    for chunk in 0..n_chunks {
        let start = lane_offsets[chunk * n_lanes] as usize;
        let stop = lane_offsets[chunk * n_lanes + n_lanes] as usize;

        for idx in start..stop {
            // the indices slice is measured as an offset into the 1024-value chunk.
            let index = chunk * 1024 + indices[idx] as usize;
            if index < offset || index >= offset + len {
                continue;
            }

            let value = values[idx];
            output[index - offset] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::Canonical;
    use crate::DynArray;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::Patched;
    use crate::arrays::PatchedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity;
    use crate::patches::Patches;
    use crate::serde::ArrayParts;
    use crate::serde::SerializeOptions;
    use crate::validity::Validity;

    #[test]
    fn test_execute() {
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let executed = array
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_primitive()
            .into_buffer::<u16>();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 1;
        expected[2] = 1;
        expected[3] = 1;

        assert_eq!(executed, expected.freeze());
    }

    #[test]
    fn test_execute_sliced() {
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .slice(3..1024)
            .unwrap();

        let executed = array
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_primitive()
            .into_buffer::<u16>();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 1;

        assert_eq!(executed, expected.freeze());
    }

    #[test]
    fn test_append_to_builder_non_nullable() {
        let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 10;
        expected[2] = 20;
        expected[3] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_append_to_builder_sliced() {
        let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .slice(3..1024)
            .unwrap();

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(expected, result);
    }

    #[test]
    fn test_append_to_builder_with_validity() {
        // Create inner array with nulls at indices 0 and 5.
        let validity = Validity::from_iter((0..10).map(|i| i != 0 && i != 5));
        let values = PrimitiveArray::new(buffer![0u16; 10], validity).into_array();

        // Apply patches at indices 1, 2, 3.
        let patches = Patches::new(
            10,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
        )
        .unwrap();

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        let array = PatchedArray::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let mut builder = builder_with_capacity(array.dtype(), array.len());
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        // Expected: null at 0, patched 10/20/30 at 1/2/3, zero at 4, null at 5, zeros at 6-9.
        let expected = PrimitiveArray::from_option_iter([
            None,
            Some(10u16),
            Some(20),
            Some(30),
            Some(0),
            None,
            Some(0),
            Some(0),
            Some(0),
            Some(0),
        ])
        .into_array();

        assert_arrays_eq!(expected, result);
    }

    fn make_patched_array(
        inner: impl IntoIterator<Item = u16>,
        patch_indices: &[u32],
        patch_values: &[u16],
    ) -> VortexResult<PatchedArray> {
        let values: Vec<u16> = inner.into_iter().collect();
        let len = values.len();
        let array = PrimitiveArray::from_iter(values).into_array();

        let indices = PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array();
        let patch_vals = PrimitiveArray::from_iter(patch_values.iter().copied()).into_array();

        let patches = Patches::new(len, 0, indices, patch_vals, None)?;

        let session = VortexSession::empty();
        let mut ctx = ExecutionCtx::new(session);

        PatchedArray::from_array_and_patches(array, &patches, &mut ctx)
    }

    #[rstest]
    #[case::basic(
        make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30]).unwrap().into_array()
    )]
    #[case::multi_chunk(
        make_patched_array(vec![0u16; 4096], &[100, 1500, 2500, 3500], &[11, 22, 33, 44]).unwrap().into_array()
    )]
    #[case::sliced({
        let arr = make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30]).unwrap();
        arr.slice(2..1024).unwrap()
    })]
    fn test_serde_roundtrip(#[case] array: crate::ArrayRef) {
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array
            .clone()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        // Concat into a single buffer.
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();
        let decoded = parts
            .decode(
                &dtype,
                len,
                &ReadContext::new(ctx.to_ids()),
                &LEGACY_SESSION,
            )
            .unwrap();

        assert!(decoded.is::<Patched>());
        assert_eq!(
            array.display_values().to_string(),
            decoded.display_values().to_string()
        );
    }

    #[test]
    fn test_with_children_basic() -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30])?;

        // Get original children via direct field access
        let inner = array.inner.clone();
        let lane_offsets = array.lane_offsets.clone();
        let indices = array.indices.clone();
        let values = array.values.clone();

        // Create new PatchedArray with same children using with_children
        let new_array =
            array
                .clone()
                .into_array()
                .with_children(vec![inner, lane_offsets, indices, values])?;

        assert!(new_array.is::<Patched>());
        assert_eq!(array.len(), new_array.len());
        assert_eq!(array.dtype(), new_array.dtype());

        // Execute both and compare results
        let mut ctx = ExecutionCtx::new(VortexSession::empty());
        let original_executed = array
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_primitive();
        let new_executed = new_array.execute::<Canonical>(&mut ctx)?.into_primitive();

        assert_arrays_eq!(original_executed, new_executed);

        Ok(())
    }

    #[test]
    fn test_with_children_modified_inner() -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; 10], &[1, 2, 3], &[10, 20, 30])?;

        // Create a different inner array (all 5s instead of 0s)
        let new_inner = PrimitiveArray::from_iter(vec![5u16; 10]).into_array();
        let lane_offsets = array.lane_offsets.clone();
        let indices = array.indices.clone();
        let values = array.values.clone();

        let new_array =
            array
                .into_array()
                .with_children(vec![new_inner, lane_offsets, indices, values])?;

        // Execute and verify the inner values changed (except at patch positions)
        let mut ctx = ExecutionCtx::new(VortexSession::empty());
        let executed = new_array.execute::<Canonical>(&mut ctx)?.into_primitive();

        // Expected: all 5s except indices 1, 2, 3 which are patched to 10, 20, 30
        let expected = PrimitiveArray::from_iter([5u16, 10, 20, 30, 5, 5, 5, 5, 5, 5]);
        assert_arrays_eq!(expected, executed);

        Ok(())
    }

    mod metadata_fields_tests {
        use vortex_error::VortexResult;

        use super::super::PatchedMetadataFields;

        #[test]
        fn test_roundtrip_min_values() -> VortexResult<()> {
            let fields = PatchedMetadataFields::new(0, 1, 0)?;
            assert_eq!(fields.offset(), 0);
            assert_eq!(fields.n_lanes(), 1);
            assert_eq!(fields.n_patches(), 0);
            assert_eq!(fields.into_inner(), 0);
            Ok(())
        }

        #[test]
        fn test_roundtrip_typical_values() -> VortexResult<()> {
            let fields = PatchedMetadataFields::new(512, 16, 1000)?;
            assert_eq!(fields.offset(), 512);
            assert_eq!(fields.n_lanes(), 16);
            assert_eq!(fields.n_patches(), 1000);
            Ok(())
        }

        #[test]
        fn test_roundtrip_max_values() -> VortexResult<()> {
            let max_offset = (1 << 10) - 1; // 1023
            let max_n_lanes = 128; // 2^7
            let max_n_patches = (1 << 23) - 1; // 8388607

            let fields = PatchedMetadataFields::new(max_offset, max_n_lanes, max_n_patches)?;
            assert_eq!(fields.offset(), max_offset);
            assert_eq!(fields.n_lanes(), max_n_lanes);
            assert_eq!(fields.n_patches(), max_n_patches);
            Ok(())
        }

        #[test]
        fn test_all_valid_n_lanes() -> VortexResult<()> {
            for exp in 0..=7 {
                let n_lanes = 1 << exp;
                let fields = PatchedMetadataFields::new(0, n_lanes, 0)?;
                assert_eq!(fields.n_lanes(), n_lanes);
            }
            Ok(())
        }

        #[test]
        fn test_from_u64() {
            // n_lanes=16 means exp=4, stored in bits 10..13
            let n_lanes_exp = 4u64; // log2(16)
            let raw: u64 = 512 | (n_lanes_exp << 10) | (1000 << 13);
            let fields = PatchedMetadataFields::from(raw);
            assert_eq!(fields.offset(), 512);
            assert_eq!(fields.n_lanes(), 16);
            assert_eq!(fields.n_patches(), 1000);
        }

        #[test]
        fn test_offset_overflow() {
            let result = PatchedMetadataFields::new(1024, 1, 0);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("offset must be < 1024")
            );
        }

        #[test]
        fn test_n_lanes_not_power_of_two() {
            let result = PatchedMetadataFields::new(0, 3, 0);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("n_lanes must be a power of two")
            );
        }

        #[test]
        fn test_n_lanes_overflow() {
            let result = PatchedMetadataFields::new(0, 256, 0);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("n_lanes must be a power of two between 1 and 128")
            );
        }

        #[test]
        fn test_n_lanes_zero() {
            let result = PatchedMetadataFields::new(0, 0, 0);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("n_lanes must be a power of two")
            );
        }

        #[test]
        fn test_n_patches_overflow() {
            let result = PatchedMetadataFields::new(0, 1, 1 << 23);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("n_patches must be < 8388608")
            );
        }
    }
}

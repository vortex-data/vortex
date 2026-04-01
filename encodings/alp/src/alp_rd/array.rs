// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::require_child;
use vortex_array::require_patches;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::alp_rd::kernel::PARENT_KERNELS;
use crate::alp_rd::rules::RULES;
use crate::alp_rd_decode;

vtable!(ALPRD);

#[derive(Clone, prost::Message)]
pub struct ALPRDMetadata {
    #[prost(uint32, tag = "1")]
    right_bit_width: u32,
    #[prost(uint32, tag = "2")]
    dict_len: u32,
    #[prost(uint32, repeated, tag = "3")]
    dict: Vec<u32>,
    #[prost(enumeration = "PType", tag = "4")]
    left_parts_ptype: i32,
    #[prost(message, tag = "5")]
    patches: Option<PatchesMetadata>,
}

impl VTable for ALPRD {
    type Array = ALPRDArray;

    type Metadata = ProstMetadata<ALPRDMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &ALPRD
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ALPRDArray) -> usize {
        array.left_parts().len()
    }

    fn dtype(array: &ALPRDArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ALPRDArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ALPRDArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.left_parts().array_hash(state, precision);
        array.left_parts_dictionary.array_hash(state, precision);
        array.right_parts().array_hash(state, precision);
        array.right_bit_width.hash(state);
        array.left_parts_patches.array_hash(state, precision);
    }

    fn array_eq(array: &ALPRDArray, other: &ALPRDArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.left_parts().array_eq(other.left_parts(), precision)
            && array
                .left_parts_dictionary
                .array_eq(&other.left_parts_dictionary, precision)
            && array.right_parts().array_eq(other.right_parts(), precision)
            && array.right_bit_width == other.right_bit_width
            && array
                .left_parts_patches
                .array_eq(&other.left_parts_patches, precision)
    }

    fn nbuffers(_array: &ALPRDArray) -> usize {
        0
    }

    fn buffer(_array: &ALPRDArray, idx: usize) -> BufferHandle {
        vortex_panic!("ALPRDArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ALPRDArray, _idx: usize) -> Option<String> {
        None
    }

    fn metadata(array: &ALPRDArray) -> VortexResult<Self::Metadata> {
        let dict = array
            .left_parts_dictionary()
            .iter()
            .map(|&i| i as u32)
            .collect::<Vec<_>>();

        Ok(ProstMetadata(ALPRDMetadata {
            right_bit_width: array.right_bit_width() as u32,
            dict_len: array.left_parts_dictionary().len() as u32,
            dict,
            left_parts_ptype: array.left_parts().dtype().as_ptype() as i32,
            patches: array
                .left_parts_patches()
                .map(|p| p.to_metadata(array.len(), array.left_parts().dtype()))
                .transpose()?,
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
        Ok(ProstMetadata(
            <ProstMetadata<ALPRDMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ALPRDArray> {
        if children.len() < 2 {
            vortex_bail!(
                "Expected at least 2 children for ALPRD encoding, found {}",
                children.len()
            );
        }

        let left_parts_dtype = DType::Primitive(metadata.0.left_parts_ptype(), dtype.nullability());
        let left_parts = children.get(0, &left_parts_dtype, len)?;
        let left_parts_dictionary: Buffer<u16> = metadata.0.dict.as_slice()
            [0..metadata.0.dict_len as usize]
            .iter()
            .map(|&i| {
                u16::try_from(i)
                    .map_err(|_| vortex_err!("left_parts_dictionary code {i} does not fit in u16"))
            })
            .try_collect()?;

        let right_parts_dtype = match &dtype {
            DType::Primitive(PType::F32, _) => {
                DType::Primitive(PType::U32, Nullability::NonNullable)
            }
            DType::Primitive(PType::F64, _) => {
                DType::Primitive(PType::U64, Nullability::NonNullable)
            }
            _ => vortex_bail!("Expected f32 or f64 dtype, got {:?}", dtype),
        };
        let right_parts = children.get(1, &right_parts_dtype, len)?;

        let left_parts_patches = metadata
            .0
            .patches
            .map(|p| {
                let indices = children.get(2, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(3, &left_parts_dtype, p.len()?)?;

                Patches::new(
                    len,
                    p.offset()?,
                    indices,
                    values,
                    // TODO(0ax1): handle chunk offsets
                    None,
                )
            })
            .transpose()?;

        ALPRDArray::try_new(
            dtype.clone(),
            left_parts,
            left_parts_dictionary,
            right_parts,
            u8::try_from(metadata.0.right_bit_width).map_err(|_| {
                vortex_err!(
                    "right_bit_width {} out of u8 range",
                    metadata.0.right_bit_width
                )
            })?,
            left_parts_patches,
        )
    }

    fn slots(array: &ALPRDArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &ALPRDArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut ALPRDArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ALPRDArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );

        // Reconstruct patches from slots + existing metadata
        array.left_parts_patches =
            match (&slots[LP_PATCH_INDICES_SLOT], &slots[LP_PATCH_VALUES_SLOT]) {
                (Some(indices), Some(values)) => {
                    let old = array
                        .left_parts_patches
                        .as_ref()
                        .vortex_expect("ALPRDArray had patch slots but no patches metadata");
                    Some(Patches::new(
                        old.array_len(),
                        old.offset(),
                        indices.clone(),
                        values.clone(),
                        slots[LP_PATCH_CHUNK_OFFSETS_SLOT].clone(),
                    )?)
                }
                _ => None,
            };
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.left_parts(), 0 => Primitive);
        let array = require_child!(array, array.right_parts(), 1 => Primitive);
        require_patches!(
            array,
            array.left_parts_patches(),
            LP_PATCH_INDICES_SLOT,
            LP_PATCH_VALUES_SLOT,
            LP_PATCH_CHUNK_OFFSETS_SLOT
        );

        let right_bit_width = array.right_bit_width();
        let ALPRDArrayParts {
            left_parts,
            right_parts,
            left_parts_dictionary,
            left_parts_patches,
            dtype,
            ..
        } = Arc::unwrap_or_clone(array).into_inner().into_parts();
        let ptype = dtype.as_ptype();

        let left_parts = left_parts
            .try_into::<Primitive>()
            .ok()
            .vortex_expect("ALPRD execute: left_parts is primitive");
        let right_parts = right_parts
            .try_into::<Primitive>()
            .ok()
            .vortex_expect("ALPRD execute: right_parts is primitive");

        // Decode the left_parts using our builtin dictionary.
        let left_parts_dict = left_parts_dictionary;
        let validity = left_parts.validity_mask()?;

        let decoded_array = if ptype == PType::F32 {
            PrimitiveArray::new(
                alp_rd_decode::<f32>(
                    left_parts.into_buffer::<u16>(),
                    &left_parts_dict,
                    right_bit_width,
                    right_parts.into_buffer::<u32>(),
                    left_parts_patches,
                    ctx,
                )?,
                Validity::from_mask(validity, dtype.nullability()),
            )
        } else {
            PrimitiveArray::new(
                alp_rd_decode::<f64>(
                    left_parts.into_buffer::<u16>(),
                    &left_parts_dict,
                    right_bit_width,
                    right_parts.into_buffer::<u64>(),
                    left_parts_patches,
                    ctx,
                )?,
                Validity::from_mask(validity, dtype.nullability()),
            )
        };

        Ok(ExecutionResult::done(decoded_array.into_array()))
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

/// The left (most significant) parts of the real-double encoded values.
pub(super) const LEFT_PARTS_SLOT: usize = 0;
/// The right (least significant) parts of the real-double encoded values.
pub(super) const RIGHT_PARTS_SLOT: usize = 1;
/// The indices of left-parts exception values that could not be dictionary-encoded.
pub(super) const LP_PATCH_INDICES_SLOT: usize = 2;
/// The exception values for left-parts that could not be dictionary-encoded.
pub(super) const LP_PATCH_VALUES_SLOT: usize = 3;
/// Chunk offsets for the left-parts patch indices/values.
pub(super) const LP_PATCH_CHUNK_OFFSETS_SLOT: usize = 4;
pub(super) const NUM_SLOTS: usize = 5;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = [
    "left_parts",
    "right_parts",
    "patch_indices",
    "patch_values",
    "patch_chunk_offsets",
];

#[derive(Clone, Debug)]
pub struct ALPRDArray {
    dtype: DType,
    slots: Vec<Option<ArrayRef>>,
    left_parts_patches: Option<Patches>,
    left_parts_dictionary: Buffer<u16>,
    right_bit_width: u8,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ALPRDArrayParts {
    pub dtype: DType,
    pub left_parts: ArrayRef,
    pub left_parts_patches: Option<Patches>,
    pub left_parts_dictionary: Buffer<u16>,
    pub right_parts: ArrayRef,
}

#[derive(Clone, Debug)]
pub struct ALPRD;

impl ALPRD {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.alprd");
}

impl ALPRDArray {
    /// Build a new `ALPRDArray` from components.
    pub fn try_new(
        dtype: DType,
        left_parts: ArrayRef,
        left_parts_dictionary: Buffer<u16>,
        right_parts: ArrayRef,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
    ) -> VortexResult<Self> {
        if !dtype.is_float() {
            vortex_bail!("ALPRDArray given invalid DType ({dtype})");
        }

        let len = left_parts.len();
        if right_parts.len() != len {
            vortex_bail!(
                "left_parts (len {}) and right_parts (len {}) must be of same length",
                len,
                right_parts.len()
            );
        }

        if !left_parts.dtype().is_unsigned_int() {
            vortex_bail!("left_parts dtype must be uint");
        }
        // we delegate array validity to the left_parts child
        if dtype.is_nullable() != left_parts.dtype().is_nullable() {
            vortex_bail!(
                "ALPRDArray dtype nullability ({}) must match left_parts dtype nullability ({})",
                dtype,
                left_parts.dtype()
            );
        }

        // we enforce right_parts to be non-nullable uint
        if !right_parts.dtype().is_unsigned_int() || right_parts.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable uint", right_parts.dtype());
        }

        let left_parts_patches = left_parts_patches
            .map(|patches| {
                if !patches.values().all_valid()? {
                    vortex_bail!("patches must be all valid: {}", patches.values());
                }
                // TODO(ngates): assert the DType, don't cast it.
                // TODO(joe): assert the DType, don't cast it in the next PR.
                let mut patches = patches.cast_values(left_parts.dtype())?;
                // Force execution of the lazy cast so patch values are materialized
                // before serialization.
                *patches.values_mut() = patches.values().to_canonical()?.into_array();
                Ok(patches)
            })
            .transpose()?;

        let slots = Self::make_slots(&left_parts, &right_parts, &left_parts_patches);

        Ok(Self {
            dtype,
            slots,
            left_parts_dictionary,
            right_bit_width,
            left_parts_patches,
            stats_set: Default::default(),
        })
    }

    /// Build a new `ALPRDArray` from components. This does not perform any validation, and instead
    /// it constructs it from parts.
    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        left_parts: ArrayRef,
        left_parts_dictionary: Buffer<u16>,
        right_parts: ArrayRef,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
    ) -> Self {
        let slots = Self::make_slots(&left_parts, &right_parts, &left_parts_patches);

        Self {
            dtype,
            slots,
            left_parts_patches,
            left_parts_dictionary,
            right_bit_width,
            stats_set: Default::default(),
        }
    }

    fn make_slots(
        left_parts: &ArrayRef,
        right_parts: &ArrayRef,
        patches: &Option<Patches>,
    ) -> Vec<Option<ArrayRef>> {
        let (pi, pv, pco) = match patches {
            Some(p) => (
                Some(p.indices().clone()),
                Some(p.values().clone()),
                p.chunk_offsets().clone(),
            ),
            None => (None, None, None),
        };
        vec![
            Some(left_parts.clone()),
            Some(right_parts.clone()),
            pi,
            pv,
            pco,
        ]
    }

    /// Return all the owned parts of the array
    pub fn into_parts(mut self) -> ALPRDArrayParts {
        let left_parts = self.slots[LEFT_PARTS_SLOT]
            .take()
            .vortex_expect("ALPRDArray left_parts slot");
        let right_parts = self.slots[RIGHT_PARTS_SLOT]
            .take()
            .vortex_expect("ALPRDArray right_parts slot");
        ALPRDArrayParts {
            dtype: self.dtype,
            left_parts,
            left_parts_patches: self.left_parts_patches,
            left_parts_dictionary: self.left_parts_dictionary,
            right_parts,
        }
    }

    /// Returns true if logical type of the array values is f32.
    ///
    /// Returns false if the logical type of the array values is f64.
    #[inline]
    pub fn is_f32(&self) -> bool {
        matches!(&self.dtype, DType::Primitive(PType::F32, _))
    }

    /// The leftmost (most significant) bits of the floating point values stored in the array.
    ///
    /// These are bit-packed and dictionary encoded, and cannot directly be interpreted without
    /// the metadata of this array.
    pub fn left_parts(&self) -> &ArrayRef {
        self.slots[LEFT_PARTS_SLOT]
            .as_ref()
            .vortex_expect("ALPRDArray left_parts slot")
    }

    /// The rightmost (least significant) bits of the floating point values stored in the array.
    pub fn right_parts(&self) -> &ArrayRef {
        self.slots[RIGHT_PARTS_SLOT]
            .as_ref()
            .vortex_expect("ALPRDArray right_parts slot")
    }

    #[inline]
    pub fn right_bit_width(&self) -> u8 {
        self.right_bit_width
    }

    /// Patches of left-most bits.
    pub fn left_parts_patches(&self) -> Option<Patches> {
        self.left_parts_patches.clone()
    }

    /// The dictionary that maps the codes in `left_parts` into bit patterns.
    #[inline]
    pub fn left_parts_dictionary(&self) -> &Buffer<u16> {
        &self.left_parts_dictionary
    }

    pub fn replace_left_parts_patches(&mut self, patches: Option<Patches>) {
        // Update both the patches and the corresponding slots to keep them in sync.
        let (pi, pv, pco) = match &patches {
            Some(p) => (
                Some(p.indices().clone()),
                Some(p.values().clone()),
                p.chunk_offsets().clone(),
            ),
            None => (None, None, None),
        };
        self.slots[LP_PATCH_INDICES_SLOT] = pi;
        self.slots[LP_PATCH_VALUES_SLOT] = pv;
        self.slots[LP_PATCH_CHUNK_OFFSETS_SLOT] = pco;
        self.left_parts_patches = patches;
    }
}

impl ValidityChild<ALPRD> for ALPRD {
    fn validity_child(array: &ALPRDArray) -> &ArrayRef {
        array.left_parts()
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ProstMetadata;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;

    use super::ALPRDMetadata;
    use crate::ALPRDFloat;
    use crate::alp_rd;

    #[rstest]
    #[case(vec![0.1f32.next_up(); 1024], 1.123_848_f32)]
    #[case(vec![0.1f64.next_up(); 1024], 1.123_848_591_110_992_f64)]
    fn test_array_encode_with_nulls_and_patches<T: ALPRDFloat>(
        #[case] reals: Vec<T>,
        #[case] seed: T,
    ) {
        assert_eq!(reals.len(), 1024, "test expects 1024-length fixture");
        // Null out some of the values.
        let mut reals: Vec<Option<T>> = reals.into_iter().map(Some).collect();
        reals[1] = None;
        reals[5] = None;
        reals[900] = None;

        // Create a new array from this.
        let real_array = PrimitiveArray::from_option_iter(reals.iter().cloned());

        // Pick a seed that we know will trigger lots of patches.
        let encoder: alp_rd::RDEncoder = alp_rd::RDEncoder::new(&[seed.powi(-2)]);

        let rd_array = encoder.encode(&real_array);

        let decoded = rd_array.to_primitive();

        assert_arrays_eq!(decoded, PrimitiveArray::from_option_iter(reals));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alprd_metadata() {
        check_metadata(
            "alprd.metadata",
            ProstMetadata(ALPRDMetadata {
                right_bit_width: u32::MAX,
                patches: Some(PatchesMetadata::new(
                    usize::MAX,
                    usize::MAX,
                    PType::U64,
                    None,
                    None,
                    None,
                )),
                dict: Vec::new(),
                left_parts_ptype: PType::U64 as i32,
                dict_len: 8,
            }),
        );
    }
}

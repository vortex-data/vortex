// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use itertools::Itertools;
use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::VortexSessionExecute;
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
use vortex_array::validity::Validity;
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
use vortex_session::registry::CachedId;

use crate::alp_rd::kernel::PARENT_KERNELS;
use crate::alp_rd::rules::RULES;
use crate::alp_rd_decode;

/// A [`ALPRD`]-encoded Vortex array.
pub type ALPRDArray = Array<ALPRD>;

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

impl ArrayHash for ALPRDData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.left_parts_dictionary.array_hash(state, precision);
        self.right_bit_width.hash(state);
        self.patch_offset.hash(state);
        self.patch_offset_within_chunk.hash(state);
    }
}

impl ArrayEq for ALPRDData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.left_parts_dictionary
            .array_eq(&other.left_parts_dictionary, precision)
            && self.right_bit_width == other.right_bit_width
            && self.patch_offset == other.patch_offset
            && self.patch_offset_within_chunk == other.patch_offset_within_chunk
    }
}

impl VTable for ALPRD {
    type ArrayData = ALPRDData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.alprd");
        *ID
    }

    fn validate(
        &self,
        data: &ALPRDData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        validate_parts(
            dtype,
            len,
            left_parts_from_slots(slots),
            right_parts_from_slots(slots),
            patches_from_slots(
                slots,
                data.patch_offset,
                data.patch_offset_within_chunk,
                len,
            )
            .as_ref(),
        )
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ALPRDArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let dict = array
            .left_parts_dictionary()
            .iter()
            .map(|&i| i as u32)
            .collect::<Vec<_>>();

        Ok(Some(
            ALPRDMetadata {
                right_bit_width: array.right_bit_width() as u32,
                dict_len: array.left_parts_dictionary().len() as u32,
                dict,
                left_parts_ptype: array.left_parts().dtype().as_ptype() as i32,
                patches: array
                    .left_parts_patches()
                    .map(|p| p.to_metadata(array.len(), p.dtype()))
                    .transpose()?,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = ALPRDMetadata::decode(metadata)?;
        if children.len() < 2 {
            vortex_bail!(
                "Expected at least 2 children for ALPRD encoding, found {}",
                children.len()
            );
        }

        let left_parts_dtype = DType::Primitive(metadata.left_parts_ptype(), dtype.nullability());
        let left_parts = children.get(0, &left_parts_dtype, len)?;
        let left_parts_dictionary: Buffer<u16> = metadata.dict.as_slice()
            [0..metadata.dict_len as usize]
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
            .patches
            .map(|p| {
                let indices = children.get(2, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(3, &left_parts_dtype.as_nonnullable(), p.len()?)?;

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
        // NOTE: `VTable::deserialize` has a fixed trait signature without `ExecutionCtx`, so we
        // cannot plumb a ctx in here. We construct a legacy ctx locally at this trait boundary.
        let left_parts_patches = ALPRDData::canonicalize_patches(
            &left_parts,
            left_parts_patches,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        let slots = ALPRDData::make_slots(&left_parts, &right_parts, &left_parts_patches);
        let data = ALPRDData::new(
            left_parts_dictionary,
            u8::try_from(metadata.right_bit_width).map_err(|_| {
                vortex_err!(
                    "right_bit_width {} out of u8 range",
                    metadata.right_bit_width
                )
            })?,
            left_parts_patches,
        );
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.left_parts(), 0 => Primitive);
        let array = require_child!(array, array.right_parts(), 1 => Primitive);
        require_patches!(
            array,
            LP_PATCH_INDICES_SLOT,
            LP_PATCH_VALUES_SLOT,
            LP_PATCH_CHUNK_OFFSETS_SLOT
        );

        let dtype = array.dtype().clone();
        let right_bit_width = array.right_bit_width();
        let ALPRDDataParts {
            left_parts,
            right_parts,
            left_parts_dictionary,
            left_parts_patches,
        } = ALPRDArrayOwnedExt::into_data_parts(array);
        let ptype = dtype.as_ptype();

        let left_parts = left_parts
            .try_downcast::<Primitive>()
            .ok()
            .vortex_expect("ALPRD execute: left_parts is primitive");
        let right_parts = right_parts
            .try_downcast::<Primitive>()
            .ok()
            .vortex_expect("ALPRD execute: right_parts is primitive");

        // Decode the left_parts using our builtin dictionary.
        let left_parts_dict = left_parts_dictionary;
        let validity = left_parts
            .as_ref()
            .validity()?
            .to_mask(left_parts.as_ref().len(), ctx)?;

        let decoded_array = if ptype == PType::F32 {
            PrimitiveArray::new(
                alp_rd_decode::<f32>(
                    left_parts.into_buffer_mut::<u16>(),
                    &left_parts_dict,
                    right_bit_width,
                    right_parts.into_buffer_mut::<u32>(),
                    left_parts_patches,
                    ctx,
                )?,
                Validity::from_mask(validity, dtype.nullability()),
            )
        } else {
            PrimitiveArray::new(
                alp_rd_decode::<f64>(
                    left_parts.into_buffer_mut::<u16>(),
                    &left_parts_dict,
                    right_bit_width,
                    right_parts.into_buffer_mut::<u64>(),
                    left_parts_patches,
                    ctx,
                )?,
                Validity::from_mask(validity, dtype.nullability()),
            )
        };

        Ok(ExecutionResult::done(decoded_array.into_array()))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
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
pub struct ALPRDData {
    patch_offset: Option<usize>,
    patch_offset_within_chunk: Option<usize>,
    left_parts_dictionary: Buffer<u16>,
    right_bit_width: u8,
}

impl Display for ALPRDData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "right_bit_width: {}", self.right_bit_width)?;
        if let Some(offset) = self.patch_offset {
            write!(f, ", patch_offset: {offset}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ALPRDDataParts {
    pub left_parts: ArrayRef,
    pub left_parts_patches: Option<Patches>,
    pub left_parts_dictionary: Buffer<u16>,
    pub right_parts: ArrayRef,
}

#[derive(Clone, Debug)]
pub struct ALPRD;

impl ALPRD {
    pub fn try_new(
        dtype: DType,
        left_parts: ArrayRef,
        left_parts_dictionary: Buffer<u16>,
        right_parts: ArrayRef,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ALPRDArray> {
        let len = left_parts.len();
        let left_parts_patches =
            ALPRDData::canonicalize_patches(&left_parts, left_parts_patches, ctx)?;
        let slots = ALPRDData::make_slots(&left_parts, &right_parts, &left_parts_patches);
        let data = ALPRDData::new(left_parts_dictionary, right_bit_width, left_parts_patches);
        Array::try_from_parts(ArrayParts::new(ALPRD, dtype, len, data).with_slots(slots))
    }

    /// # Safety
    /// See [`ALPRD::try_new`] for preconditions.
    pub unsafe fn new_unchecked(
        dtype: DType,
        left_parts: ArrayRef,
        left_parts_dictionary: Buffer<u16>,
        right_parts: ArrayRef,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
    ) -> ALPRDArray {
        let len = left_parts.len();
        let slots = ALPRDData::make_slots(&left_parts, &right_parts, &left_parts_patches);
        let data = unsafe {
            ALPRDData::new_unchecked(left_parts_dictionary, right_bit_width, left_parts_patches)
        };
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(ALPRD, dtype, len, data).with_slots(slots))
        }
    }
}

impl ALPRDData {
    fn canonicalize_patches(
        left_parts: &ArrayRef,
        left_parts_patches: Option<Patches>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Patches>> {
        left_parts_patches
            .map(|patches| {
                if !patches.values().all_valid(ctx)? {
                    vortex_bail!("patches must be all valid: {}", patches.values());
                }
                // TODO(ngates): assert the DType, don't cast it.
                // TODO(joe): assert the DType, don't cast it in the next PR.
                let mut patches = patches.cast_values(&left_parts.dtype().as_nonnullable())?;
                // Force execution of the lazy cast so patch values are materialized
                // before serialization.
                let canonical = patches.values().clone().execute::<Canonical>(ctx)?;
                *patches.values_mut() = canonical.into_array();
                Ok(patches)
            })
            .transpose()
    }

    /// Build a new `ALPRDArray` from components.
    pub fn new(
        left_parts_dictionary: Buffer<u16>,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
    ) -> Self {
        let (patch_offset, patch_offset_within_chunk) = match &left_parts_patches {
            Some(patches) => (Some(patches.offset()), patches.offset_within_chunk()),
            None => (None, None),
        };

        Self {
            patch_offset,
            patch_offset_within_chunk,
            left_parts_dictionary,
            right_bit_width,
        }
    }

    /// Build a new `ALPRDArray` from components. This does not perform any validation, and instead
    /// it constructs it from parts.
    pub(crate) unsafe fn new_unchecked(
        left_parts_dictionary: Buffer<u16>,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
    ) -> Self {
        Self::new(left_parts_dictionary, right_bit_width, left_parts_patches)
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
    pub fn into_parts(self, left_parts: ArrayRef, right_parts: ArrayRef) -> ALPRDDataParts {
        ALPRDDataParts {
            left_parts,
            left_parts_patches: None,
            left_parts_dictionary: self.left_parts_dictionary,
            right_parts,
        }
    }

    #[inline]
    pub fn right_bit_width(&self) -> u8 {
        self.right_bit_width
    }

    /// The dictionary that maps the codes in `left_parts` into bit patterns.
    #[inline]
    pub fn left_parts_dictionary(&self) -> &Buffer<u16> {
        &self.left_parts_dictionary
    }
}

fn left_parts_from_slots(slots: &[Option<ArrayRef>]) -> &ArrayRef {
    slots[LEFT_PARTS_SLOT]
        .as_ref()
        .vortex_expect("ALPRDArray left_parts slot")
}

fn right_parts_from_slots(slots: &[Option<ArrayRef>]) -> &ArrayRef {
    slots[RIGHT_PARTS_SLOT]
        .as_ref()
        .vortex_expect("ALPRDArray right_parts slot")
}

fn patches_from_slots(
    slots: &[Option<ArrayRef>],
    patch_offset: Option<usize>,
    patch_offset_within_chunk: Option<usize>,
    len: usize,
) -> Option<Patches> {
    match (&slots[LP_PATCH_INDICES_SLOT], &slots[LP_PATCH_VALUES_SLOT]) {
        (Some(indices), Some(values)) => {
            let patch_offset = patch_offset.vortex_expect("ALPRDArray patch slots without offset");
            Some(unsafe {
                Patches::new_unchecked(
                    len,
                    patch_offset,
                    indices.clone(),
                    values.clone(),
                    slots[LP_PATCH_CHUNK_OFFSETS_SLOT].clone(),
                    patch_offset_within_chunk,
                )
            })
        }
        _ => None,
    }
}

fn validate_parts(
    dtype: &DType,
    len: usize,
    left_parts: &ArrayRef,
    right_parts: &ArrayRef,
    left_parts_patches: Option<&Patches>,
) -> VortexResult<()> {
    if !dtype.is_float() {
        vortex_bail!("ALPRDArray given invalid DType ({dtype})");
    }

    vortex_ensure!(
        left_parts.len() == len,
        "left_parts len {} != outer len {len}",
        left_parts.len(),
    );
    vortex_ensure!(
        right_parts.len() == len,
        "right_parts len {} != outer len {len}",
        right_parts.len(),
    );

    if !left_parts.dtype().is_unsigned_int() {
        vortex_bail!("left_parts dtype must be uint");
    }
    if dtype.is_nullable() != left_parts.dtype().is_nullable() {
        vortex_bail!(
            "ALPRDArray dtype nullability ({}) must match left_parts dtype nullability ({})",
            dtype,
            left_parts.dtype()
        );
    }

    let expected_right_parts_dtype = match dtype {
        DType::Primitive(PType::F32, _) => DType::Primitive(PType::U32, Nullability::NonNullable),
        DType::Primitive(PType::F64, _) => DType::Primitive(PType::U64, Nullability::NonNullable),
        _ => vortex_bail!("Expected f32 or f64 dtype, got {:?}", dtype),
    };
    vortex_ensure!(
        right_parts.dtype() == &expected_right_parts_dtype,
        "right_parts dtype {} does not match expected {}",
        right_parts.dtype(),
        expected_right_parts_dtype,
    );

    if let Some(patches) = left_parts_patches {
        vortex_ensure!(
            patches.array_len() == len,
            "patches array_len {} != outer len {len}",
            patches.array_len(),
        );
        vortex_ensure!(
            patches.dtype().eq_ignore_nullability(left_parts.dtype()),
            "patches dtype {} does not match left_parts dtype {}",
            patches.dtype(),
            left_parts.dtype(),
        );
        vortex_ensure!(
            patches
                .values()
                .all_valid(&mut LEGACY_SESSION.create_execution_ctx())?,
            "patches must be all valid: {}",
            patches.values()
        );
    }

    Ok(())
}

pub trait ALPRDArrayExt: TypedArrayRef<ALPRD> {
    fn left_parts(&self) -> &ArrayRef {
        left_parts_from_slots(self.as_ref().slots())
    }

    fn right_parts(&self) -> &ArrayRef {
        right_parts_from_slots(self.as_ref().slots())
    }

    fn right_bit_width(&self) -> u8 {
        ALPRDData::right_bit_width(self)
    }

    fn left_parts_patches(&self) -> Option<Patches> {
        patches_from_slots(
            self.as_ref().slots(),
            self.patch_offset,
            self.patch_offset_within_chunk,
            self.as_ref().len(),
        )
    }

    fn left_parts_dictionary(&self) -> &Buffer<u16> {
        ALPRDData::left_parts_dictionary(self)
    }
}
impl<T: TypedArrayRef<ALPRD>> ALPRDArrayExt for T {}

pub trait ALPRDArrayOwnedExt {
    fn into_data_parts(self) -> ALPRDDataParts;
}

impl ALPRDArrayOwnedExt for Array<ALPRD> {
    fn into_data_parts(self) -> ALPRDDataParts {
        let left_parts_patches = self.left_parts_patches();
        let left_parts = self.left_parts().clone();
        let right_parts = self.right_parts().clone();
        let mut parts = ALPRDDataParts {
            left_parts,
            left_parts_patches: None,
            left_parts_dictionary: self.left_parts_dictionary().clone(),
            right_parts,
        };
        parts.left_parts_patches = left_parts_patches;
        parts
    }
}

impl ValidityChild<ALPRD> for ALPRD {
    fn validity_child(array: ArrayView<'_, ALPRD>) -> ArrayRef {
        array.left_parts().clone()
    }
}

#[cfg(test)]
mod test {
    use prost::Message;
    use rstest::rstest;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
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
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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

        let rd_array = encoder.encode(real_array.as_view(), &mut ctx);

        let decoded = rd_array
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(decoded, PrimitiveArray::from_option_iter(reals));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alprd_metadata() {
        check_metadata(
            "alprd.metadata",
            &ALPRDMetadata {
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
            }
            .encode_to_vec(),
        );
    }
}

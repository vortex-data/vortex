// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use ::alp::MAX_DICT_SIZE;
use itertools::Itertools;
use prost::Message;
use vortex_array::Array;
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
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::patches::PatchSlotIndices;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::require_child;
use vortex_array::require_patches;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.left_parts_dictionary.array_hash(state, accuracy);
        self.right_bit_width.hash(state);
        self.patches_data.hash(state);
    }
}

impl ArrayEq for ALPRDData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.left_parts_dictionary
            .array_eq(&other.left_parts_dictionary, accuracy)
            && self.right_bit_width == other.right_bit_width
            && self.patches_data == other.patches_data
    }
}

impl VTable for ALPRD {
    type TypedArrayData = ALPRDData;

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
        let alprd_slots = ALPRDSlotsView::from_slots(slots);
        validate_parts(
            data,
            dtype,
            len,
            alprd_slots.left_parts,
            alprd_slots.right_parts,
            patches_from_slots(slots, data.patches_data.as_ref(), len).as_ref(),
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
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
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = ALPRDMetadata::decode(metadata)?;
        if children.len() < 2 {
            vortex_bail!(
                "Expected at least 2 children for ALPRD encoding, found {}",
                children.len()
            );
        }

        // Stored metadata is untrusted: dict_len must stay within the stored entries.
        if metadata.dict_len as usize > metadata.dict.len() {
            vortex_bail!(
                "ALP-RD dictionary length {} exceeds the {} stored entries",
                metadata.dict_len,
                metadata.dict.len()
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
                    &mut session.create_execution_ctx(),
                )
            })
            .transpose()?;
        let slots = ALPRDData::make_slots(&left_parts, &right_parts, left_parts_patches.as_ref());
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
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ALPRDSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.left_parts(), ALPRDSlots::LEFT_PARTS => Primitive);
        let array =
            require_child!(array, array.right_parts(), ALPRDSlots::RIGHT_PARTS => Primitive);
        require_patches!(
            array,
            ALPRDSlots::PATCH_INDICES,
            ALPRDSlots::PATCH_VALUES,
            ALPRDSlots::PATCH_CHUNK_OFFSETS
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
            .execute_mask(left_parts.as_ref().len(), ctx)?;

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
}

#[array_slots(ALPRD)]
pub struct ALPRDSlots {
    /// The left (most significant) parts of the real-double encoded values.
    #[slot(0)]
    pub left_parts: ArrayRef,
    /// The right (least significant) parts of the real-double encoded values.
    #[slot(1)]
    pub right_parts: ArrayRef,
    /// The indices of left-parts exception values that could not be dictionary-encoded.
    #[slot(2)]
    pub patch_indices: Option<ArrayRef>,
    /// The exception values for left-parts that could not be dictionary-encoded.
    #[slot(3)]
    pub patch_values: Option<ArrayRef>,
    /// Chunk offsets for the left-parts patch indices/values.
    #[slot(4)]
    pub patch_chunk_offsets: Option<ArrayRef>,
}

const LP_PATCH_SLOTS: PatchSlotIndices = PatchSlotIndices {
    indices: ALPRDSlots::PATCH_INDICES,
    values: ALPRDSlots::PATCH_VALUES,
    chunk_offsets: ALPRDSlots::PATCH_CHUNK_OFFSETS,
};

#[derive(Clone, Debug)]
pub struct ALPRDData {
    patches_data: Option<PatchesData>,
    left_parts_dictionary: Buffer<u16>,
    right_bit_width: u8,
}

impl Display for ALPRDData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "right_bit_width: {}", self.right_bit_width)?;
        if let Some(pd) = &self.patches_data {
            write!(f, ", patch_offset: {}", pd.offset())?;
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
    ) -> VortexResult<ALPRDArray> {
        let len = left_parts.len();
        let slots = ALPRDData::make_slots(&left_parts, &right_parts, left_parts_patches.as_ref());
        let data = ALPRDData::new(left_parts_dictionary, right_bit_width, left_parts_patches);
        Array::try_from_parts(ArrayParts::new(ALPRD, dtype, len, data, slots))
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
        let slots = ALPRDData::make_slots(&left_parts, &right_parts, left_parts_patches.as_ref());
        let data = unsafe {
            ALPRDData::new_unchecked(left_parts_dictionary, right_bit_width, left_parts_patches)
        };
        unsafe { Array::from_parts_unchecked(ArrayParts::new(ALPRD, dtype, len, data, slots)) }
    }
}

impl ALPRDData {
    /// Build a new `ALPRDArray` from components.
    pub fn new(
        left_parts_dictionary: Buffer<u16>,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
    ) -> Self {
        Self {
            patches_data: left_parts_patches.as_ref().map(PatchesData::from_patches),
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
        patches: Option<&Patches>,
    ) -> ArraySlots {
        let mut slots: ArraySlots = smallvec![Some(left_parts.clone()), Some(right_parts.clone())];
        PatchesData::push_slots(&mut slots, patches);
        slots
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

fn patches_from_slots(
    slots: &[Option<ArrayRef>],
    patches_data: Option<&PatchesData>,
    len: usize,
) -> Option<Patches> {
    PatchesData::patches_from_slots(patches_data, len, slots, LP_PATCH_SLOTS)
}

#[allow(clippy::disallowed_methods)]
fn validate_parts(
    data: &ALPRDData,
    dtype: &DType,
    len: usize,
    left_parts: &ArrayRef,
    right_parts: &ArrayRef,
    left_parts_patches: Option<&Patches>,
) -> VortexResult<()> {
    if !dtype.is_float() {
        vortex_bail!("ALPRDArray given invalid DType ({dtype})");
    }

    vortex_ensure_eq!(left_parts.len(), len);
    vortex_ensure_eq!(right_parts.len(), len);

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

    let (expected_right_parts_dtype, float_bit_width) = match dtype {
        DType::Primitive(PType::F32, _) => (
            DType::Primitive(PType::U32, Nullability::NonNullable),
            u32::BITS,
        ),
        DType::Primitive(PType::F64, _) => (
            DType::Primitive(PType::U64, Nullability::NonNullable),
            u64::BITS,
        ),
        _ => vortex_bail!("Expected f32 or f64 dtype, got {:?}", dtype),
    };
    vortex_ensure_eq!(right_parts.dtype(), &expected_right_parts_dtype);

    // Stored metadata is untrusted: the alp decoder masks codes into an at-most-MAX_DICT_SIZE
    // dictionary, and the decode shift overflows the float's bit pattern once right_bit_width
    // reaches the width of the float.
    vortex_ensure!(
        data.left_parts_dictionary().len() <= MAX_DICT_SIZE as usize,
        "ALP-RD dictionary holds {} entries, more than MAX_DICT_SIZE ({})",
        data.left_parts_dictionary().len(),
        MAX_DICT_SIZE
    );
    vortex_ensure!(
        u32::from(data.right_bit_width()) < float_bit_width,
        "ALP-RD right_bit_width {} must be below the {}-bit float width",
        data.right_bit_width(),
        float_bit_width
    );

    if let Some(patches) = left_parts_patches {
        vortex_ensure_eq!(patches.array_len(), len);
        // Left-parts exceptions are always all-valid and are stored as the non-nullable left-parts
        // dtype. Requiring that exact dtype (rather than ignoring nullability) means each
        // construction path must produce correct patches, removing the need to normalize them.
        // Non-nullable also implies all-valid, so no separate validity check is required.
        let expected = left_parts.dtype().as_nonnullable();
        vortex_ensure_eq!(
            patches.dtype(),
            &expected,
            "patches dtype must be the non-nullable left_parts dtype"
        );
    }

    Ok(())
}

pub trait ALPRDArrayExt: ALPRDArraySlotsExt {
    fn right_bit_width(&self) -> u8 {
        ALPRDData::right_bit_width(self)
    }

    fn left_parts_patches(&self) -> Option<Patches> {
        patches_from_slots(self.slots(), self.patches_data.as_ref(), self.len())
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
    use std::sync::LazyLock;

    use prost::Message;
    use rstest::rstest;
    use vortex_array::ArrayDeserialization;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::ALPRD;
    use super::ALPRDMetadata;
    use crate::ALPRDFloat;
    use crate::RDEncoderExt;
    use crate::alp_rd;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case(vec![0.1f32.next_up(); 1024], 1.123_848_f32)]
    #[case(vec![0.1f64.next_up(); 1024], 1.123_848_591_110_992_f64)]
    fn test_array_encode_with_nulls_and_patches<T: ALPRDFloat + NativePType>(
        #[case] reals: Vec<T>,
        #[case] seed: T,
    ) {
        let mut ctx = SESSION.create_execution_ctx();
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

        let rd_array = encoder.encode(real_array.as_view());

        assert_arrays_eq!(rd_array, PrimitiveArray::from_option_iter(reals), &mut ctx);
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

    /// Deserialize ALP-RD parts the way a file read does, through the plugin door, so both
    /// deserialize-time and validate-time rejections surface as errors.
    fn deserialize_alprd(
        dtype: &DType,
        len: usize,
        metadata: &ALPRDMetadata,
        children: Vec<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        ALPRD.deserialize(
            ArrayDeserialization::new(
                ALPRD.id(),
                dtype,
                len,
                &metadata.encode_to_vec(),
                &[],
                &children,
            ),
            &SESSION,
        )
    }

    /// The stored dictionary is untrusted, so `dict_len` past the end of the stored entries
    /// must be a deserialize error, not a slice panic (#10756).
    #[test]
    fn test_deserialize_rejects_dict_len_past_the_stored_dictionary() {
        let dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let metadata = ALPRDMetadata {
            right_bit_width: 32,
            dict_len: 8,
            dict: vec![0, 1],
            left_parts_ptype: PType::U16 as i32,
            patches: None,
        };
        let children = vec![buffer![0u16].into_array(), buffer![0u32].into_array()];

        let error = deserialize_alprd(&dtype, 1, &metadata, children)
            .expect_err("dict_len past the stored dictionary must be rejected");
        assert!(error.to_string().contains("dictionary"));
    }

    /// The `alp` decoder's lane lookup masks codes into an at-most-`MAX_DICT_SIZE` dictionary,
    /// so a longer stored dictionary must be rejected rather than panicking at decode time.
    #[test]
    fn test_deserialize_rejects_dictionary_larger_than_max_dict_size() {
        let dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let metadata = ALPRDMetadata {
            right_bit_width: 16,
            dict_len: 9,
            dict: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            left_parts_ptype: PType::U16 as i32,
            patches: None,
        };
        let children = vec![buffer![0u16].into_array(), buffer![0u32].into_array()];

        let error = deserialize_alprd(&dtype, 1, &metadata, children)
            .expect_err("a dictionary larger than MAX_DICT_SIZE must be rejected");
        assert!(error.to_string().contains("MAX_DICT_SIZE"));
    }

    /// The decode shift `left << right_bit_width` overflows the float's bit pattern once
    /// `right_bit_width` reaches the width of the float, so the boundary value is rejected.
    #[rstest]
    #[case::f32(PType::F32, 32)]
    #[case::f64(PType::F64, 64)]
    fn test_deserialize_rejects_right_bit_width_at_the_float_width(
        #[case] float_ptype: PType,
        #[case] right_bit_width: u32,
    ) {
        let dtype = DType::Primitive(float_ptype, Nullability::NonNullable);
        let metadata = ALPRDMetadata {
            right_bit_width,
            dict_len: 2,
            dict: vec![0, 1],
            left_parts_ptype: PType::U16 as i32,
            patches: None,
        };
        let right_parts = match float_ptype {
            PType::F32 => buffer![0u32].into_array(),
            _ => buffer![0u64].into_array(),
        };
        let children = vec![buffer![0u16].into_array(), right_parts];

        let error = deserialize_alprd(&dtype, 1, &metadata, children)
            .expect_err("right_bit_width at the float width must be rejected");
        assert!(error.to_string().contains("right_bit_width"));
    }

    /// Metadata inside the format's bounds still deserializes and decodes: code 1 maps through
    /// the dictionary and the shift stays below the f32 width.
    #[test]
    fn test_deserialize_accepts_metadata_within_bounds() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let metadata = ALPRDMetadata {
            right_bit_width: 16,
            dict_len: 2,
            dict: vec![10, 20],
            left_parts_ptype: PType::U16 as i32,
            patches: None,
        };
        let children = vec![buffer![1u16].into_array(), buffer![0u32].into_array()];

        let deserialized = deserialize_alprd(&dtype, 1, &metadata, children)?;
        let decoded = deserialized.execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(
            decoded,
            PrimitiveArray::from_iter([f32::from_bits(20u32 << 16)]),
            &mut ctx
        );
        Ok(())
    }
}

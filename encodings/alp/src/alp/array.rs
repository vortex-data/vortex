// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::require_child;
use vortex_array::require_patches;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ALPFloat;
use crate::alp::Exponents;
use crate::alp::decompress::execute_decompress;
use crate::alp::rules::PARENT_KERNELS;
use crate::alp::rules::RULES;

/// A [`ALP`]-encoded Vortex array.
pub type ALPArray = Array<ALP>;

impl ArrayHash for ALPData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.exponents.hash(state);
        self.patch_offset.hash(state);
        self.patch_offset_within_chunk.hash(state);
    }
}

impl ArrayEq for ALPData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.exponents == other.exponents
            && self.patch_offset == other.patch_offset
            && self.patch_offset_within_chunk == other.patch_offset_within_chunk
    }
}

impl VTable for ALP {
    type ArrayData = ALPData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.alp");
        *ID
    }

    fn validate(
        &self,
        data: &ALPData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = ALPSlotsView::from_slots(slots);
        validate_parts(
            dtype,
            len,
            data.exponents,
            slots.encoded,
            patches_from_slots(
                &slots,
                data.patch_offset,
                data.patch_offset_within_chunk,
                len,
            ),
        )
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ALPArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let exponents = array.exponents();
        Ok(Some(
            ALPMetadata {
                exp_e: exponents.e as u32,
                exp_f: exponents.f as u32,
                patches: array
                    .patches()
                    .map(|p| p.to_metadata(array.len(), array.dtype()))
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
        let metadata = ALPMetadata::decode(metadata)?;
        let encoded_ptype = match &dtype {
            DType::Primitive(PType::F32, n) => DType::Primitive(PType::I32, *n),
            DType::Primitive(PType::F64, n) => DType::Primitive(PType::I64, *n),
            d => vortex_bail!(MismatchedTypes: "f32 or f64", d),
        };
        let encoded = children.get(0, &encoded_ptype, len)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(1, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(2, dtype, p.len()?)?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()?
                    .map(|dtype| children.get(3, &dtype, usize::try_from(p.chunk_offsets_len())?))
                    .transpose()?;

                Patches::new(len, p.offset()?, indices, values, chunk_offsets)
            })
            .transpose()?;

        let slots = ALPData::make_slots(&encoded, &patches);
        let data = ALPData::new(
            Exponents {
                e: u8::try_from(metadata.exp_e)?,
                f: u8::try_from(metadata.exp_f)?,
            },
            patches,
        );
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        ALPSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.encoded(), ALPSlots::ENCODED => Primitive);
        require_patches!(
            array,
            ALPSlots::PATCH_INDICES,
            ALPSlots::PATCH_VALUES,
            ALPSlots::PATCH_CHUNK_OFFSETS
        );

        Ok(ExecutionResult::done(
            execute_decompress(array, ctx)?.into_array(),
        ))
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

#[array_slots(ALP)]
pub struct ALPSlots {
    /// The ALP-encoded values array.
    pub encoded: ArrayRef,
    /// The indices of exception values that could not be ALP-encoded.
    pub patch_indices: Option<ArrayRef>,
    /// The exception values that could not be ALP-encoded.
    pub patch_values: Option<ArrayRef>,
    /// Chunk offsets for the patch indices/values.
    pub patch_chunk_offsets: Option<ArrayRef>,
}

#[derive(Clone, Debug)]
pub struct ALPData {
    patch_offset: Option<usize>,
    patch_offset_within_chunk: Option<usize>,
    exponents: Exponents,
}

impl Display for ALPData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "exponents: {}", self.exponents)?;
        if let Some(offset) = self.patch_offset {
            write!(f, ", patch_offset: {offset}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ALP;

#[derive(Clone, prost::Message)]
pub struct ALPMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) exp_e: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) exp_f: u32,
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

impl ALPData {
    fn validate_components(
        encoded: &ArrayRef,
        exponents: Exponents,
        patches: Option<&Patches>,
    ) -> VortexResult<()> {
        vortex_ensure!(
            matches!(
                encoded.dtype(),
                DType::Primitive(PType::I32 | PType::I64, _)
            ),
            "ALP encoded ints have invalid DType {}",
            encoded.dtype(),
        );

        // Validate exponents are in-bounds for the float, and that patches have the proper
        // length and type.
        let Exponents { e, f } = exponents;
        match encoded.dtype().as_ptype() {
            PType::I32 => {
                vortex_ensure!(exponents.e <= f32::MAX_EXPONENT, "e out of bounds: {e}");
                vortex_ensure!(exponents.f <= f32::MAX_EXPONENT, "f out of bounds: {f}");
                if let Some(patches) = patches {
                    Self::validate_patches::<f32>(patches, encoded)?;
                }
            }
            PType::I64 => {
                vortex_ensure!(e <= f64::MAX_EXPONENT, "e out of bounds: {e}");
                vortex_ensure!(f <= f64::MAX_EXPONENT, "f out of bounds: {f}");

                if let Some(patches) = patches {
                    Self::validate_patches::<f64>(patches, encoded)?;
                }
            }
            _ => unreachable!(),
        }

        // Validate patches
        if let Some(patches) = patches {
            vortex_ensure!(
                patches.array_len() == encoded.len(),
                "patches array_len != encoded len: {} != {}",
                patches.array_len(),
                encoded.len()
            );

            // Verify that the patches DType are of the proper DType.
        }

        Ok(())
    }

    fn logical_dtype(encoded: &ArrayRef) -> VortexResult<DType> {
        match encoded.dtype() {
            DType::Primitive(PType::I32, nullability) => {
                Ok(DType::Primitive(PType::F32, *nullability))
            }
            DType::Primitive(PType::I64, nullability) => {
                Ok(DType::Primitive(PType::F64, *nullability))
            }
            _ => vortex_bail!("ALP encoded ints have invalid DType {}", encoded.dtype(),),
        }
    }

    /// Validate that any patches provided are valid for the ALPArray.
    fn validate_patches<T: ALPFloat>(patches: &Patches, encoded: &ArrayRef) -> VortexResult<()> {
        vortex_ensure!(
            patches.array_len() == encoded.len(),
            "patches array_len != encoded len: {} != {}",
            patches.array_len(),
            encoded.len()
        );

        let expected_type = DType::Primitive(T::PTYPE, encoded.dtype().nullability());
        vortex_ensure!(
            patches.dtype() == &expected_type,
            "Expected patches type {expected_type}, actual {}",
            patches.dtype(),
        );

        Ok(())
    }
}

impl ALPData {
    /// Build a new `ALPArray` from components, panicking on validation failure.
    ///
    /// See [`ALP::try_new`] for reference on preconditions that must pass before
    /// calling this method.
    pub fn new(exponents: Exponents, patches: Option<Patches>) -> Self {
        let (patch_offset, patch_offset_within_chunk) = match &patches {
            Some(p) => (Some(p.offset()), p.offset_within_chunk()),
            None => (None, None),
        };

        Self {
            patch_offset,
            patch_offset_within_chunk,
            exponents,
        }
    }

    /// Build a new `ALPArray` from components:
    ///
    /// * `encoded` contains the ALP-encoded ints. Any null values are replaced with placeholders
    /// * `exponents` are the ALP exponents, valid range depends on the data type
    /// * `patches` are any patch values that don't cleanly encode using the ALP conversion function
    ///
    /// Build a new `ALPArray` from components without validation.
    ///
    /// See [`ALP::try_new`] for information about the preconditions that should be checked
    /// **before** calling this method.
    pub(crate) unsafe fn new_unchecked(exponents: Exponents, patches: Option<Patches>) -> Self {
        Self::new(exponents, patches)
    }
}

/// Constructors for [`ALPArray`].
impl ALP {
    pub fn new(encoded: ArrayRef, exponents: Exponents, patches: Option<Patches>) -> ALPArray {
        let dtype = ALPData::logical_dtype(&encoded).vortex_expect("ALP encoded dtype");
        let len = encoded.len();
        let slots = ALPData::make_slots(&encoded, &patches);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(ALP, dtype, len, ALPData::new(exponents, patches))
                    .with_slots(slots),
            )
        }
    }

    pub fn try_new(
        encoded: ArrayRef,
        exponents: Exponents,
        patches: Option<Patches>,
    ) -> VortexResult<ALPArray> {
        let dtype = ALPData::logical_dtype(&encoded)?;
        let len = encoded.len();
        let slots = ALPData::make_slots(&encoded, &patches);
        let data = ALPData::new(exponents, patches);
        Array::try_from_parts(ArrayParts::new(ALP, dtype, len, data).with_slots(slots))
    }

    /// # Safety
    /// See [`ALP::try_new`] for preconditions.
    pub unsafe fn new_unchecked(
        encoded: ArrayRef,
        exponents: Exponents,
        patches: Option<Patches>,
    ) -> ALPArray {
        let dtype = ALPData::logical_dtype(&encoded).vortex_expect("ALP encoded dtype");
        let len = encoded.len();
        let slots = ALPData::make_slots(&encoded, &patches);
        let data = unsafe { ALPData::new_unchecked(exponents, patches) };
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(ALP, dtype, len, data).with_slots(slots))
        }
    }
}

impl ALPData {
    fn make_slots(encoded: &ArrayRef, patches: &Option<Patches>) -> Vec<Option<ArrayRef>> {
        let (patch_indices, patch_values, patch_chunk_offsets) = match patches {
            Some(p) => (
                Some(p.indices().clone()),
                Some(p.values().clone()),
                p.chunk_offsets().clone(),
            ),
            None => (None, None, None),
        };
        vec![
            Some(encoded.clone()),
            patch_indices,
            patch_values,
            patch_chunk_offsets,
        ]
    }

    #[inline]
    pub fn exponents(&self) -> Exponents {
        self.exponents
    }
}

pub trait ALPArrayExt: ALPArraySlotsExt {
    fn exponents(&self) -> Exponents {
        self.exponents
    }

    fn patches(&self) -> Option<Patches> {
        patches_from_slots(
            &self.slots_view(),
            self.patch_offset,
            self.patch_offset_within_chunk,
            self.as_ref().len(),
        )
    }
}

fn patches_from_slots(
    slots: &ALPSlotsView,
    patch_offset: Option<usize>,
    patch_offset_within_chunk: Option<usize>,
    len: usize,
) -> Option<Patches> {
    match (slots.patch_indices, slots.patch_values) {
        (Some(indices), Some(values)) => {
            let patch_offset = patch_offset.vortex_expect("has patch slots but no patch_offset");
            Some(unsafe {
                Patches::new_unchecked(
                    len,
                    patch_offset,
                    indices.clone(),
                    values.clone(),
                    slots.patch_chunk_offsets.cloned(),
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
    exponents: Exponents,
    encoded: &ArrayRef,
    patches: Option<Patches>,
) -> VortexResult<()> {
    let logical_dtype = ALPData::logical_dtype(encoded)?;
    ALPData::validate_components(encoded, exponents, patches.as_ref())?;
    vortex_ensure!(
        encoded.len() == len,
        "ALP encoded len {} != outer len {len}",
        encoded.len(),
    );
    vortex_ensure!(
        &logical_dtype == dtype,
        "ALP dtype {} does not match encoded logical dtype {}",
        dtype,
        logical_dtype,
    );
    Ok(())
}

impl<T: TypedArrayRef<ALP>> ALPArrayExt for T {}

pub trait ALPArrayOwnedExt {
    fn into_parts(self) -> (ArrayRef, Exponents, Option<Patches>);
}

impl ALPArrayOwnedExt for Array<ALP> {
    #[inline]
    fn into_parts(self) -> (ArrayRef, Exponents, Option<Patches>) {
        let patches = self.patches();
        let exponents = self.exponents();
        let encoded = self.encoded().clone();
        (encoded, exponents, patches)
    }
}

impl ValidityChild<ALP> for ALP {
    fn validity_child(array: ArrayView<'_, ALP>) -> ArrayRef {
        array.encoded().clone()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::alp_encode;
    use crate::decompress_into_array;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_execute_f32(#[case] size: usize) {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..size).map(|i| i as f32));
        let encoded = alp_encode(values.as_view(), None, &mut ctx).unwrap();

        let result_canonical = {
            encoded
                .clone()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        };
        // Compare against the traditional array-based decompress path
        let expected =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();

        assert_arrays_eq!(result_canonical.into_array(), expected);
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_execute_f64(#[case] size: usize) {
        let values = PrimitiveArray::from_iter((0..size).map(|i| i as f64));
        let encoded = alp_encode(
            values.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let result_canonical = {
            let mut ctx = SESSION.create_execution_ctx();
            encoded
                .clone()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        };
        // Compare against the traditional array-based decompress path
        let expected =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();

        assert_arrays_eq!(result_canonical.into_array(), expected);
    }

    #[rstest]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_execute_with_patches(#[case] size: usize) {
        let values: Vec<f64> = (0..size)
            .map(|i| match i % 4 {
                0..=2 => 1.0,
                _ => PI,
            })
            .collect();

        let array = PrimitiveArray::from_iter(values);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().unwrap().array_len() > 0);

        let result_canonical = {
            let mut ctx = SESSION.create_execution_ctx();
            encoded
                .clone()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        };
        // Compare against the traditional array-based decompress path
        let expected =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();

        assert_arrays_eq!(result_canonical.into_array(), expected);
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_execute_with_validity(#[case] size: usize) {
        let values: Vec<Option<f32>> = (0..size)
            .map(|i| if i % 2 == 1 { None } else { Some(1.0) })
            .collect();

        let array = PrimitiveArray::from_option_iter(values);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let result_canonical = {
            let mut ctx = SESSION.create_execution_ctx();
            encoded
                .clone()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        };
        // Compare against the traditional array-based decompress path
        let expected =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();

        assert_arrays_eq!(result_canonical.into_array(), expected);
    }

    #[rstest]
    #[case(100)]
    #[case(1023)]
    #[case(1024)]
    #[case(1025)]
    #[case(2047)]
    #[case(2048)]
    #[case(2049)]
    fn test_execute_with_patches_and_validity(#[case] size: usize) {
        let values: Vec<Option<f64>> = (0..size)
            .map(|idx| match idx % 3 {
                0 => Some(1.0),
                1 => None,
                _ => Some(PI),
            })
            .collect();

        let array = PrimitiveArray::from_option_iter(values);
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(encoded.patches().unwrap().array_len() > 0);

        let result_canonical = {
            let mut ctx = SESSION.create_execution_ctx();
            encoded
                .clone()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        };
        // Compare against the traditional array-based decompress path
        let expected =
            decompress_into_array(encoded, &mut LEGACY_SESSION.create_execution_ctx()).unwrap();

        assert_arrays_eq!(result_canonical.into_array(), expected);
    }

    #[rstest]
    #[case(500, 100)]
    #[case(1000, 200)]
    #[case(2048, 512)]
    fn test_execute_sliced_vector(#[case] size: usize, #[case] slice_start: usize) {
        let values: Vec<Option<f64>> = (0..size)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else if i % 4 == 3 {
                    Some(PI)
                } else {
                    Some(1.0)
                }
            })
            .collect();

        let array = PrimitiveArray::from_option_iter(values.clone());
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let slice_end = size - slice_start;
        let slice_len = slice_end - slice_start;
        let sliced_encoded = encoded.slice(slice_start..slice_end).unwrap();

        let result_canonical = {
            let mut ctx = SESSION.create_execution_ctx();
            sliced_encoded.execute::<Canonical>(&mut ctx).unwrap()
        };
        let result_primitive = result_canonical.into_primitive();

        for idx in 0..slice_len {
            let expected_value = values[slice_start + idx];

            let result_valid = result_primitive
                .validity()
                .vortex_expect("result validity should be derivable")
                .is_valid(idx)
                .unwrap();
            assert_eq!(
                result_valid,
                expected_value.is_some(),
                "Validity mismatch at idx={idx}",
            );

            if let Some(expected_val) = expected_value {
                let result_val = result_primitive.as_slice::<f64>()[idx];
                assert_eq!(result_val, expected_val, "Value mismatch at idx={idx}",);
            }
        }
    }

    #[rstest]
    #[case(500, 100)]
    #[case(1000, 200)]
    #[case(2048, 512)]
    fn test_sliced_to_primitive(#[case] size: usize, #[case] slice_start: usize) {
        let values: Vec<Option<f64>> = (0..size)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else if i % 4 == 3 {
                    Some(PI)
                } else {
                    Some(1.0)
                }
            })
            .collect();

        let array = PrimitiveArray::from_option_iter(values.clone());
        let encoded = alp_encode(
            array.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();

        let slice_end = size - slice_start;
        let slice_len = slice_end - slice_start;
        let sliced_encoded = encoded.slice(slice_start..slice_end).unwrap();

        #[expect(deprecated)]
        let result_primitive = sliced_encoded.to_primitive();

        for idx in 0..slice_len {
            let expected_value = values[slice_start + idx];

            let result_valid = result_primitive
                .as_ref()
                .validity()
                .unwrap()
                .to_mask(
                    result_primitive.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )
                .unwrap()
                .value(idx);
            assert_eq!(
                result_valid,
                expected_value.is_some(),
                "Validity mismatch at idx={idx}",
            );

            if let Some(expected_val) = expected_value {
                let buf = result_primitive.to_buffer::<f64>();
                let result_val = buf.as_slice()[idx];
                assert_eq!(result_val, expected_val, "Value mismatch at idx={idx}",);
            }
        }
    }

    /// Regression test for issue #5948: execute_decompress drops patches when chunk_offsets is
    /// None.
    ///
    /// When patches exist but do NOT have chunk_offsets, the execute path incorrectly passes
    /// `None` to `decompress_unchunked_core` instead of the actual patches.
    ///
    /// This can happen after file IO serialization/deserialization where chunk_offsets may not
    /// be preserved, or when building ALPArrays manually without chunk_offsets.
    #[test]
    fn test_execute_decompress_with_patches_no_chunk_offsets_regression_5948() {
        // Create an array with values that will produce patches. PI doesn't encode cleanly.
        let values: Vec<f64> = vec![1.0, 2.0, PI, 4.0, 5.0];
        let original = PrimitiveArray::from_iter(values);

        // First encode normally to get a properly formed ALPArray with patches.
        let normally_encoded = alp_encode(
            original.as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        assert!(
            normally_encoded.patches().is_some(),
            "Test requires patches to be present"
        );

        let original_patches = normally_encoded.patches().unwrap();
        assert!(
            original_patches.chunk_offsets().is_some(),
            "Normal encoding should have chunk_offsets"
        );

        // Rebuild the patches WITHOUT chunk_offsets to simulate deserialized patches.
        let patches_without_chunk_offsets = Patches::new(
            original_patches.array_len(),
            original_patches.offset(),
            original_patches.indices().clone(),
            original_patches.values().clone(),
            None, // NO chunk_offsets - this triggers the bug!
        )
        .unwrap();

        // Build a new ALPArray with the same encoded data but patches without chunk_offsets.
        let alp_without_chunk_offsets = ALP::new(
            normally_encoded.encoded().clone(),
            normally_encoded.exponents(),
            Some(patches_without_chunk_offsets),
        );

        // The legacy decompress_into_array path should work correctly.
        let result_legacy = decompress_into_array(
            alp_without_chunk_offsets.clone(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        let legacy_slice = result_legacy.as_slice::<f64>();

        // Verify the legacy path produces correct values.
        assert!(
            (legacy_slice[2] - PI).abs() < 1e-10,
            "Legacy path should have PI at index 2, got {}",
            legacy_slice[2]
        );

        // The execute path has the bug - it drops patches when chunk_offsets is None.
        let result_execute = {
            let mut ctx = SESSION.create_execution_ctx();
            execute_decompress(alp_without_chunk_offsets, &mut ctx).unwrap()
        };
        let execute_slice = result_execute.as_slice::<f64>();

        // This assertion FAILS until the bug is fixed because execute_decompress drops patches.
        assert!(
            (execute_slice[2] - PI).abs() < 1e-10,
            "Execute path should have PI at index 2, but got {} (patches were dropped!)",
            execute_slice[2]
        );
    }
}

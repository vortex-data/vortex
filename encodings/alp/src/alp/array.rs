// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

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
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::require_child;
use vortex_array::require_patches;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ALPFloat;
use crate::alp::Exponents;
use crate::alp::decompress::execute_decompress;
use crate::alp::rules::PARENT_KERNELS;
use crate::alp::rules::RULES;

vtable!(ALP);

impl VTable for ALP {
    type Array = ALPArray;

    type Metadata = ProstMetadata<ALPMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &ALP
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ALPArray) -> usize {
        array.encoded().len()
    }

    fn dtype(array: &ALPArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ALPArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ALPArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.encoded().array_hash(state, precision);
        array.exponents.hash(state);
        array.patches().array_hash(state, precision);
    }

    fn array_eq(array: &ALPArray, other: &ALPArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.encoded().array_eq(other.encoded(), precision)
            && array.exponents == other.exponents
            && array.patches().array_eq(&other.patches(), precision)
    }

    fn nbuffers(_array: &ALPArray) -> usize {
        0
    }

    fn buffer(_array: &ALPArray, idx: usize) -> BufferHandle {
        vortex_panic!("ALPArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ALPArray, _idx: usize) -> Option<String> {
        None
    }

    fn slots(array: &ALPArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &ALPArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut ALPArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ALPArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );

        // If patch slots are being cleared, clear the metadata too
        if slots[PATCH_INDICES_SLOT].is_none() || slots[PATCH_VALUES_SLOT].is_none() {
            array.patch_offset = None;
            array.patch_offset_within_chunk = None;
        }

        array.slots = slots;
        Ok(())
    }

    fn metadata(array: &ALPArray) -> VortexResult<Self::Metadata> {
        let exponents = array.exponents();
        Ok(ProstMetadata(ALPMetadata {
            exp_e: exponents.e as u32,
            exp_f: exponents.f as u32,
            patches: array
                .patches()
                .map(|p| p.to_metadata(array.len(), array.dtype()))
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
            <ProstMetadata<ALPMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ALPArray> {
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

        ALPArray::try_new(
            encoded,
            Exponents {
                e: u8::try_from(metadata.exp_e)?,
                f: u8::try_from(metadata.exp_f)?,
            },
            patches,
        )
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let array = require_child!(array, array.encoded(), ENCODED_SLOT => Primitive);
        require_patches!(
            array,
            array.patches(),
            PATCH_INDICES_SLOT,
            PATCH_VALUES_SLOT,
            PATCH_CHUNK_OFFSETS_SLOT
        );

        Ok(ExecutionResult::done(
            execute_decompress(Arc::unwrap_or_clone(array).into_inner(), ctx)?.into_array(),
        ))
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

/// The ALP-encoded values array.
pub(super) const ENCODED_SLOT: usize = 0;
/// The indices of exception values that could not be ALP-encoded.
pub(super) const PATCH_INDICES_SLOT: usize = 1;
/// The exception values that could not be ALP-encoded.
pub(super) const PATCH_VALUES_SLOT: usize = 2;
/// Chunk offsets for the patch indices/values.
pub(super) const PATCH_CHUNK_OFFSETS_SLOT: usize = 3;
pub(super) const NUM_SLOTS: usize = 4;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = [
    "encoded",
    "patch_indices",
    "patch_values",
    "patch_chunk_offsets",
];

#[derive(Clone, Debug)]
pub struct ALPArray {
    slots: Vec<Option<ArrayRef>>,
    patch_offset: Option<usize>,
    patch_offset_within_chunk: Option<usize>,
    dtype: DType,
    exponents: Exponents,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ALP;

impl ALP {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.alp");
}

#[derive(Clone, prost::Message)]
pub struct ALPMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) exp_e: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) exp_f: u32,
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

impl ALPArray {
    fn validate(
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

impl ALPArray {
    /// Build a new `ALPArray` from components, panicking on validation failure.
    ///
    /// See [`ALPArray::try_new`] for reference on preconditions that must pass before
    /// calling this method.
    pub fn new(encoded: ArrayRef, exponents: Exponents, patches: Option<Patches>) -> Self {
        Self::try_new(encoded, exponents, patches).vortex_expect("ALPArray new")
    }

    /// Build a new `ALPArray` from components:
    ///
    /// * `encoded` contains the ALP-encoded ints. Any null values are replaced with placeholders
    /// * `exponents` are the ALP exponents, valid range depends on the data type
    /// * `patches` are any patch values that don't cleanly encode using the ALP conversion function
    ///
    /// This method validates the inputs and will return an error if any validation fails.
    ///
    /// # Validation
    ///
    /// * The `encoded` array must be either `i32` or `i64`
    ///     * If `i32`, any `patches` must have DType `f32` with same nullability
    ///     * If `i64`, then `patches`must have DType `f64` with same nullability
    /// * `exponents` must be in the valid range depending on if the ALPArray is of type `f32` or
    ///   `f64`.
    /// * `patches` must have an `array_len` equal to the length of `encoded`
    ///
    /// Any failure of these preconditions will result in an error being returned.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vortex_alp::{ALPArray, Exponents};
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    ///
    /// // Returns error because buffer has wrong PType.
    /// let result = ALPArray::try_new(
    ///     buffer![1i8].into_array(),
    ///     Exponents { e: 1, f: 1 },
    ///     None
    /// );
    /// assert!(result.is_err());
    ///
    /// // Returns error because Exponents are out of bounds for f32
    /// let result = ALPArray::try_new(
    ///     buffer![1i32, 2i32].into_array(),
    ///     Exponents { e: 100, f: 100 },
    ///     None
    /// );
    /// assert!(result.is_err());
    ///
    /// // Success!
    /// let value = ALPArray::try_new(
    ///     buffer![0i32].into_array(),
    ///     Exponents { e: 1, f: 1 },
    ///     None
    /// ).unwrap();
    ///
    /// assert_eq!(value.scalar_at(0).unwrap(), 0f32.into());
    /// ```
    pub fn try_new(
        encoded: ArrayRef,
        exponents: Exponents,
        patches: Option<Patches>,
    ) -> VortexResult<Self> {
        Self::validate(&encoded, exponents, patches.as_ref())?;

        let dtype = match encoded.dtype() {
            DType::Primitive(PType::I32, nullability) => DType::Primitive(PType::F32, *nullability),
            DType::Primitive(PType::I64, nullability) => DType::Primitive(PType::F64, *nullability),
            _ => unreachable!(),
        };

        let slots = Self::make_slots(&encoded, &patches);
        let (patch_offset, patch_offset_within_chunk) = match &patches {
            Some(p) => (Some(p.offset()), p.offset_within_chunk()),
            None => (None, None),
        };

        Ok(Self {
            dtype,
            slots,
            exponents,
            patch_offset,
            patch_offset_within_chunk,
            stats_set: Default::default(),
        })
    }

    /// Build a new `ALPArray` from components without validation.
    ///
    /// See [`ALPArray::try_new`] for information about the preconditions that should be checked
    /// **before** calling this method.
    pub(crate) unsafe fn new_unchecked(
        encoded: ArrayRef,
        exponents: Exponents,
        patches: Option<Patches>,
        dtype: DType,
    ) -> Self {
        let slots = Self::make_slots(&encoded, &patches);
        let (patch_offset, patch_offset_within_chunk) = match &patches {
            Some(p) => (Some(p.offset()), p.offset_within_chunk()),
            None => (None, None),
        };

        Self {
            dtype,
            slots,
            exponents,
            patch_offset,
            patch_offset_within_chunk,
            stats_set: Default::default(),
        }
    }

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

    pub fn ptype(&self) -> PType {
        self.dtype.as_ptype()
    }

    pub fn encoded(&self) -> &ArrayRef {
        self.slots[ENCODED_SLOT]
            .as_ref()
            .vortex_expect("ALPArray encoded slot")
    }

    #[inline]
    pub fn exponents(&self) -> Exponents {
        self.exponents
    }

    pub fn patches(&self) -> Option<Patches> {
        match (
            &self.slots[PATCH_INDICES_SLOT],
            &self.slots[PATCH_VALUES_SLOT],
        ) {
            (Some(indices), Some(values)) => {
                let patch_offset = self
                    .patch_offset
                    .vortex_expect("has patch slots but no patch_offset");
                Some(unsafe {
                    Patches::new_unchecked(
                        self.encoded().len(),
                        patch_offset,
                        indices.clone(),
                        values.clone(),
                        self.slots[PATCH_CHUNK_OFFSETS_SLOT].clone(),
                        self.patch_offset_within_chunk,
                    )
                })
            }
            _ => None,
        }
    }

    /// Consumes the array and returns its parts.
    #[inline]
    pub fn into_parts(mut self) -> (ArrayRef, Exponents, Option<Patches>, DType) {
        let patches = self.patches();
        let encoded = self.slots[ENCODED_SLOT]
            .take()
            .vortex_expect("ALPArray encoded slot");
        (encoded, self.exponents, patches, self.dtype)
    }
}

impl ValidityChild<ALP> for ALP {
    fn validity_child(array: &ALPArray) -> &ArrayRef {
        array.encoded()
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
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::session::ArraySession;
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
        let values = PrimitiveArray::from_iter((0..size).map(|i| i as f32));
        let encoded = alp_encode(&values, None).unwrap();

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
    fn test_execute_f64(#[case] size: usize) {
        let values = PrimitiveArray::from_iter((0..size).map(|i| i as f64));
        let encoded = alp_encode(&values, None).unwrap();

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
        let encoded = alp_encode(&array, None).unwrap();
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
        let encoded = alp_encode(&array, None).unwrap();

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
        let encoded = alp_encode(&array, None).unwrap();
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
        let encoded = alp_encode(&array, None).unwrap();

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

            let result_valid = result_primitive.validity().is_valid(idx).unwrap();
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
        let encoded = alp_encode(&array, None).unwrap();

        let slice_end = size - slice_start;
        let slice_len = slice_end - slice_start;
        let sliced_encoded = encoded.slice(slice_start..slice_end).unwrap();

        let result_primitive = sliced_encoded.to_primitive();

        for idx in 0..slice_len {
            let expected_value = values[slice_start + idx];

            let result_valid = result_primitive.validity_mask().unwrap().value(idx);
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
        let normally_encoded = alp_encode(&original, None).unwrap();
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
        let alp_without_chunk_offsets = ALPArray::new(
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

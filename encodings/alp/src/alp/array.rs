// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_array::Array;
use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DeserializeMetadata;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::kernel::BindCtx;
use vortex_array::kernel::KernelRef;
use vortex_array::kernel::kernel;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ArrayVTable;
use vortex_array::vtable::ArrayVTableExt;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::vtable::EncodeVTable;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_array::vtable::VisitorVTable;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ALPFloat;
use crate::alp::Exponents;
use crate::alp::alp_encode;
use crate::alp::decompress::decompress_into_array;
use crate::alp::decompress::decompress_into_vector;
use crate::match_each_alp_float_ptype;

vtable!(ALP);

impl VTable for ALPVTable {
    type Array = ALPArray;

    type Metadata = ProstMetadata<ALPMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.alp")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        ALPVTable.as_vtable()
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

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<ALPMetadata> as DeserializeMetadata>::deserialize(buffer)?,
        ))
    }

    fn build(
        &self,
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
                let indices = children.get(1, &p.indices_dtype(), p.len())?;
                let values = children.get(2, dtype, p.len())?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()
                    .map(|dtype| children.get(3, &dtype, usize::try_from(p.chunk_offsets_len())?))
                    .transpose()?;

                Ok::<_, VortexError>(Patches::new(
                    len,
                    p.offset(),
                    indices,
                    values,
                    chunk_offsets,
                ))
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

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // Children: encoded, patches (if present): indices, values, chunk_offsets (optional)
        let patches_info = array
            .patches
            .as_ref()
            .map(|p| (p.array_len(), p.offset(), p.chunk_offsets().is_some()));

        let expected_children = match &patches_info {
            Some((_, _, has_chunk_offsets)) => 1 + 2 + if *has_chunk_offsets { 1 } else { 0 },
            None => 1,
        };

        vortex_ensure!(
            children.len() == expected_children,
            "ALPArray expects {} children, got {}",
            expected_children,
            children.len()
        );

        let mut children_iter = children.into_iter();
        array.encoded = children_iter
            .next()
            .ok_or_else(|| vortex_err!("Expected encoded child"))?;

        if let Some((array_len, offset, _has_chunk_offsets)) = patches_info {
            let indices = children_iter
                .next()
                .ok_or_else(|| vortex_err!("Expected patch indices child"))?;
            let values = children_iter
                .next()
                .ok_or_else(|| vortex_err!("Expected patch values child"))?;
            let chunk_offsets = children_iter.next();

            array.patches = Some(Patches::new(
                array_len,
                offset,
                indices,
                values,
                chunk_offsets,
            ));
        }

        Ok(())
    }

    fn bind_kernel(array: &ALPArray, ctx: &mut BindCtx) -> VortexResult<KernelRef> {
        let encoded = array.encoded().bind_kernel(ctx)?;
        let patches_kernels = if let Some(patches) = array.patches() {
            Some((
                patches.indices().bind_kernel(ctx)?,
                patches.values().bind_kernel(ctx)?,
                patches
                    .chunk_offsets()
                    .as_ref()
                    .map(|co| co.bind_kernel(ctx))
                    .transpose()?,
            ))
        } else {
            None
        };

        let patches_offset = array.patches().map(|p| p.offset()).unwrap_or(0);
        let exponents = array.exponents();

        match_each_alp_float_ptype!(array.dtype().as_ptype(), |T| {
            Ok(kernel(move || {
                let encoded_vector = encoded.execute()?;
                let patches_vectors = match patches_kernels {
                    Some((idx_kernel, val_kernel, co_kernel)) => Some((
                        idx_kernel.execute()?,
                        val_kernel.execute()?,
                        co_kernel.map(|k| k.execute()).transpose()?,
                    )),
                    None => None,
                };

                decompress_into_vector::<T>(
                    encoded_vector,
                    exponents,
                    patches_vectors,
                    patches_offset,
                )
            }))
        })
    }
}

#[derive(Clone, Debug)]
pub struct ALPArray {
    encoded: ArrayRef,
    patches: Option<Patches>,
    dtype: DType,
    exponents: Exponents,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct ALPVTable;

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
        encoded: &dyn Array,
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
    fn validate_patches<T: ALPFloat>(patches: &Patches, encoded: &dyn Array) -> VortexResult<()> {
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
    /// assert_eq!(value.scalar_at(0), 0f32.into());
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

        Ok(Self {
            dtype,
            encoded,
            exponents,
            patches,
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
        Self {
            dtype,
            encoded,
            exponents,
            patches,
            stats_set: Default::default(),
        }
    }

    pub fn ptype(&self) -> PType {
        self.dtype.as_ptype()
    }

    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }

    #[inline]
    pub fn exponents(&self) -> Exponents {
        self.exponents
    }

    pub fn patches(&self) -> Option<&Patches> {
        self.patches.as_ref()
    }

    /// Consumes the array and returns its parts.
    #[inline]
    pub fn into_parts(self) -> (ArrayRef, Exponents, Option<Patches>, DType) {
        (self.encoded, self.exponents, self.patches, self.dtype)
    }
}

impl ValidityChild<ALPVTable> for ALPVTable {
    fn validity_child(array: &ALPArray) -> &dyn Array {
        array.encoded()
    }
}

impl BaseArrayVTable<ALPVTable> for ALPVTable {
    fn len(array: &ALPArray) -> usize {
        array.encoded.len()
    }

    fn dtype(array: &ALPArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ALPArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ALPArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.encoded.array_hash(state, precision);
        array.exponents.hash(state);
        array.patches.array_hash(state, precision);
    }

    fn array_eq(array: &ALPArray, other: &ALPArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.encoded.array_eq(&other.encoded, precision)
            && array.exponents == other.exponents
            && array.patches.array_eq(&other.patches, precision)
    }
}

impl CanonicalVTable<ALPVTable> for ALPVTable {
    fn canonicalize(array: &ALPArray) -> Canonical {
        Canonical::Primitive(decompress_into_array(array.clone()))
    }
}

impl EncodeVTable<ALPVTable> for ALPVTable {
    fn encode(
        _vtable: &ALPVTable,
        canonical: &Canonical,
        like: Option<&ALPArray>,
    ) -> VortexResult<Option<ALPArray>> {
        let parray = canonical.clone().into_primitive();
        let exponents = like.map(|a| a.exponents());
        let alp = alp_encode(&parray, exponents)?;

        Ok(Some(alp))
    }
}

impl VisitorVTable<ALPVTable> for ALPVTable {
    fn visit_buffers(_array: &ALPArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ALPArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", array.encoded());
        if let Some(patches) = array.patches() {
            visitor.visit_patches(patches);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VectorExecutor;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::vtable::ValidityHelper;
    use vortex_dtype::PTypeDowncast;
    use vortex_session::VortexSession;
    use vortex_vector::VectorOps;

    use super::*;

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

        let result_vector = encoded.to_array().execute_vector(&SESSION).unwrap();
        // Compare against the traditional array-based decompress path
        let expected = decompress_into_array(encoded);

        assert_eq!(result_vector.len(), size);

        let result_primitive = result_vector.into_primitive().into_f32();
        assert_eq!(result_primitive.as_ref(), expected.as_slice::<f32>());
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

        let result_vector = encoded.to_array().execute_vector(&SESSION).unwrap();
        // Compare against the traditional array-based decompress path
        let expected = decompress_into_array(encoded);

        assert_eq!(result_vector.len(), size);

        let result_primitive = result_vector.into_primitive().into_f64();
        assert_eq!(result_primitive.as_ref(), expected.as_slice::<f64>());
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

        let result_vector = encoded.to_array().execute_vector(&SESSION).unwrap();
        // Compare against the traditional array-based decompress path
        let expected = decompress_into_array(encoded);

        assert_eq!(result_vector.len(), size);

        let result_primitive = result_vector.into_primitive().into_f64();
        assert_eq!(result_primitive.as_ref(), expected.as_slice::<f64>());
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

        let result_vector = encoded.to_array().execute_vector(&SESSION).unwrap();
        // Compare against the traditional array-based decompress path
        let expected = decompress_into_array(encoded);

        assert_eq!(result_vector.len(), size);

        let result_primitive = result_vector.into_primitive().into_f32();
        assert_eq!(result_primitive.as_ref(), expected.as_slice::<f32>());

        // Test validity masks match
        for idx in 0..size {
            assert_eq!(
                result_primitive.validity().value(idx),
                expected.validity().is_valid(idx)
            );
        }
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

        let result_vector = encoded.to_array().execute_vector(&SESSION).unwrap();
        // Compare against the traditional array-based decompress path
        let expected = decompress_into_array(encoded);

        assert_eq!(result_vector.len(), size);

        let result_primitive = result_vector.into_primitive().into_f64();
        assert_eq!(result_primitive.as_ref(), expected.as_slice::<f64>());

        // Test validity masks match
        for idx in 0..size {
            assert_eq!(
                result_primitive.validity().value(idx),
                expected.validity().is_valid(idx)
            );
        }
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
        let sliced_encoded = encoded.slice(slice_start..slice_end);

        let result_vector = sliced_encoded.execute_vector_optimized(&SESSION).unwrap();
        let result_primitive = result_vector.into_primitive().into_f64();

        for idx in 0..slice_len {
            let expected_value = values[slice_start + idx];

            let result_valid = result_primitive.validity().value(idx);
            assert_eq!(
                result_valid,
                expected_value.is_some(),
                "Validity mismatch at idx={idx}",
            );

            if let Some(expected_val) = expected_value {
                let result_val = result_primitive.as_ref()[idx];
                assert_eq!(result_val, expected_val, "Value mismatch at idx={idx}",);
            }
        }
    }
}

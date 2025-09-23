// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_array::patches::Patches;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, VTable, ValidityChild, ValidityVTableFromChild,
};
use vortex_array::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, vtable};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};

use crate::ALPFloat;
use crate::alp::{Exponents, decompress};

vtable!(ALP);

impl VTable for ALPVTable {
    type Array = ALPArray;
    type Encoding = ALPEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.alp")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ALPEncoding.as_ref())
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

#[derive(Clone, Debug)]
pub struct ALPEncoding;

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
}

impl ValidityChild<ALPVTable> for ALPVTable {
    fn validity_child(array: &ALPArray) -> &dyn Array {
        array.encoded()
    }
}

impl ArrayVTable<ALPVTable> for ALPVTable {
    fn len(array: &ALPArray) -> usize {
        array.encoded.len()
    }

    fn dtype(array: &ALPArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ALPArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<ALPVTable> for ALPVTable {
    fn canonicalize(array: &ALPArray) -> Canonical {
        Canonical::Primitive(decompress(array))
    }
}

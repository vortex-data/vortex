// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::compute::mask;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{
    ArrayVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use crate::{Array, ArrayRef, EncodingId, EncodingRef, IntoArray, vtable};

vtable!(Masked);

mod canonical;
mod compute;
mod serde;

#[derive(Clone, Debug)]
pub struct MaskedEncoding;

impl VTable for MaskedVTable {
    type Array = MaskedArray;
    type Encoding = MaskedEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.masked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(MaskedEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct MaskedArray {
    child: ArrayRef,
    validity: Validity,
    dtype: DType,
    stats: ArrayStats,
}

impl ArrayVTable<MaskedVTable> for MaskedVTable {
    fn len(array: &MaskedArray) -> usize {
        array.child.len()
    }

    fn dtype(array: &MaskedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &MaskedArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }
}

impl ValidityHelper for MaskedArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl MaskedArray {
    pub fn try_new(child: ArrayRef, validity: Validity) -> VortexResult<Self> {
        if matches!(validity, Validity::NonNullable) {
            vortex_bail!("MaskedArray must have nullable validity, got {validity:?}")
        }

        if !child.all_valid() {
            vortex_bail!("MaskedArray children must not have nulls");
        }

        if let Some(validity_len) = validity.maybe_len()
            && validity_len != child.len()
        {
            vortex_bail!("Validity must be the same length as a MaskedArray's child");
        }

        // MaskedArray's nullability is determined solely by its validity, not the child's dtype
        // The child can have nullable dtype but must not have any actual null values
        let dtype = child.dtype().as_nullable();

        Ok(Self {
            child,
            validity,
            dtype,
            stats: ArrayStats::default(),
        })
    }

    fn masked_child(&self) -> VortexResult<ArrayRef> {
        // Invert the validity mask - we want to set values to null where validity is false
        let inverted_mask = !self.validity.to_mask(self.len());
        mask(&self.child, &inverted_mask)
    }
}

impl OperationsVTable<MaskedVTable> for MaskedVTable {
    fn slice(array: &MaskedArray, range: Range<usize>) -> ArrayRef {
        let child = array.child.slice(range.clone());
        let validity = array.validity.slice(range);

        MaskedArray {
            child,
            validity,
            dtype: array.dtype.clone(),
            stats: ArrayStats::default(),
        }
        .into_array()
    }

    fn scalar_at(array: &MaskedArray, index: usize) -> Scalar {
        // invalid indices are handled by the entrypoint function
        array.child.scalar_at(index).into_nullable()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use super::*;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;
    use crate::{Array, IntoArray, ToCanonical as _};

    #[rstest]
    #[case(Validity::AllValid, Nullability::Nullable)]
    #[case(Validity::from_iter([true, false, true]), Nullability::Nullable)]
    fn test_dtype_nullability(#[case] validity: Validity, #[case] expected: Nullability) {
        let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let array = MaskedArray::try_new(child, validity).unwrap();

        assert_eq!(
            array.dtype(),
            &DType::Primitive(vortex_dtype::PType::I32, expected)
        );
    }

    #[test]
    fn test_dtype_nullability_with_nullable_child() {
        // Child can have nullable dtype but no actual nulls
        // MaskedArray dtype should be determined by validity, not child's dtype
        let child = PrimitiveArray::new(vortex_buffer::buffer![1i32, 2, 3], Validity::AllValid)
            .into_array();

        // Child has nullable dtype
        assert!(child.dtype().is_nullable());
    }

    #[test]
    fn test_canonical_dtype_matches_array_dtype() {
        // The canonical form should have the same nullability as the array's dtype
        let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let array = MaskedArray::try_new(child, Validity::AllValid).unwrap();

        let canonical = array.to_canonical();
        assert_eq!(canonical.as_ref().dtype(), array.dtype());
    }

    #[test]
    fn test_masked_child_with_validity() {
        // When validity has nulls, masked_child should apply inverted mask
        let child = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array();
        let array =
            MaskedArray::try_new(child, Validity::from_iter([true, false, true, false, true]))
                .unwrap();

        let masked = array.masked_child().unwrap();
        let prim = masked.to_primitive();

        // Positions where validity is false should be null in masked_child
        assert_eq!(prim.valid_count(), 3);
        assert!(prim.is_valid(0));
        assert!(!prim.is_valid(1));
        assert!(prim.is_valid(2));
        assert!(!prim.is_valid(3));
        assert!(prim.is_valid(4));

        assert_eq!(
            array.as_ref().display_values().to_string(),
            masked.display_values().to_string()
        );
    }

    #[test]
    fn test_masked_child_all_valid() {
        // When validity is AllValid, masked_child should invert to AllInvalid
        let child = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
        let array = MaskedArray::try_new(child, Validity::AllValid).unwrap();

        let masked = array.masked_child().unwrap();
        assert_eq!(masked.len(), 3);
        assert_eq!(masked.valid_count(), 3);
        assert_eq!(
            array.as_ref().display_values().to_string(),
            masked.display_values().to_string()
        );
    }

    #[rstest]
    #[case(Validity::AllValid)]
    #[case(Validity::from_iter([true, true, true]))]
    #[case(Validity::from_iter([false, false, false]))]
    #[case(Validity::from_iter([true, false, true, false]))]
    fn test_masked_child_preserves_length(#[case] validity: Validity) {
        let len = match &validity {
            Validity::Array(arr) => arr.len(),
            _ => 3,
        };

        #[allow(clippy::cast_possible_truncation)]
        let child = PrimitiveArray::from_iter(0..len as i32).into_array();
        let array = MaskedArray::try_new(child, validity.clone()).unwrap();

        let masked = array.masked_child().unwrap();
        assert_eq!(masked.len(), len);
        assert_eq!(masked.validity_mask(), validity.to_mask(len));
        assert_eq!(
            array.as_ref().display_values().to_string(),
            masked.display_values().to_string()
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A test extension type representing unsigned integers divisible by a given divisor.
//!
//! This serves as the canonical-source reference implementation of [`RefinementVTable`]. The
//! source is a [`PrimitiveRefinedSource<u64>`], and the predicate checks that the integer is
//! divisible by the metadata-provided [`Divisor`].

use std::fmt;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::dtype::extension::ExtId;
use crate::dtype::extension::PrimitiveRefinedSource;
use crate::dtype::extension::RefinementVTable;

/// The divisor stored as extension metadata.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Divisor(pub u64);

impl fmt::Display for Divisor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "divisible by {}", self.0)
    }
}

/// Refinement type for unsigned integers that must be divisible by the metadata divisor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DivisibleInt;

impl RefinementVTable for DivisibleInt {
    type Source = PrimitiveRefinedSource<u64>;
    type Metadata = Divisor;
    type NativeValue<'a> = u64;

    fn id(&self) -> ExtId {
        ExtId::new("test.divisible_int")
    }

    fn refine_scalar(metadata: &Self::Metadata, source_value: u64) -> VortexResult<u64> {
        if !source_value.is_multiple_of(metadata.0) {
            vortex_bail!("{} is not divisible by {}", source_value, metadata.0);
        }
        Ok(source_value)
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.0.to_le_bytes().to_vec())
    }

    fn deserialize_metadata(&self, data: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_ensure!(data.len() == 8, "divisible int metadata must be 8 bytes");
        let bytes: [u8; 8] = data
            .try_into()
            .map_err(|_| vortex_error::vortex_err!("divisible int metadata must be 8 bytes"))?;
        let n = u64::from_le_bytes(bytes);
        vortex_ensure!(n > 0, "divisor must be greater than 0");
        Ok(Divisor(n))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::DivisibleInt;
    use super::Divisor;
    use crate::IntoArray;
    use crate::array::ArrayRef;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtId;
    use crate::dtype::extension::PrimitiveRefinedSource;
    use crate::dtype::extension::RefinementVTable;
    use crate::validity::Validity;

    #[test]
    fn metadata_roundtrip() -> VortexResult<()> {
        let vtable = DivisibleInt;
        let divisor = Divisor(42);

        let bytes = vtable.serialize_metadata(&divisor)?;
        let decoded = vtable.deserialize_metadata(&bytes)?;

        assert_eq!(decoded, divisor);
        Ok(())
    }

    #[test]
    fn rejects_zero_divisor() {
        let bytes = 0u64.to_le_bytes();
        assert!(DivisibleInt.deserialize_metadata(&bytes).is_err());
    }

    #[test]
    fn rejects_wrong_storage_dtype() {
        let divisor = Divisor(10);

        assert!(
            ExtDType::<DivisibleInt>::try_new(
                divisor,
                DType::Primitive(PType::I32, Nullability::NonNullable)
            )
            .is_err()
        );
        assert!(
            ExtDType::<DivisibleInt>::try_new(divisor, DType::Utf8(Nullability::NonNullable))
                .is_err()
        );
        assert!(
            ExtDType::<DivisibleInt>::try_new(
                divisor,
                DType::Primitive(PType::U64, Nullability::NonNullable)
            )
            .is_ok()
        );
    }

    #[test]
    fn default_validate_array_accepts_all_multiples() -> VortexResult<()> {
        // Every value is a multiple of 3.
        let arr: ArrayRef =
            PrimitiveArray::new(buffer![0u64, 3, 6, 9, 12], Validity::NonNullable).into_array();
        DivisibleInt::validate_array(&Divisor(3), &arr)?;
        Ok(())
    }

    #[test]
    fn default_validate_array_rejects_nonmultiple() {
        // 7 is not a multiple of 3.
        let arr: ArrayRef =
            PrimitiveArray::new(buffer![0u64, 3, 7, 9], Validity::NonNullable).into_array();
        assert!(DivisibleInt::validate_array(&Divisor(3), &arr).is_err());
    }

    // A toy refinement whose `validate_array` override always succeeds and bumps a counter.
    // Its `refine_scalar` always fails, so if the override is NOT taken, the default
    // scalar-iteration fallback would return `Err`. Counter plus `Ok` together prove the override
    // ran.
    static OVERRIDE_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct AlwaysAcceptedArray;

    impl RefinementVTable for AlwaysAcceptedArray {
        type Source = PrimitiveRefinedSource<u64>;
        type Metadata = Divisor;
        type NativeValue<'a> = u64;

        fn id(&self) -> ExtId {
            ExtId::new("test.always_accepted_array")
        }

        fn refine_scalar(_metadata: &Divisor, _source_value: u64) -> VortexResult<u64> {
            vortex_error::vortex_bail!("refine_scalar should not be reached when override runs")
        }

        fn validate_array(
            _metadata: &Self::Metadata,
            _source_array: &ArrayRef,
        ) -> VortexResult<()> {
            OVERRIDE_CALLS.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            DivisibleInt.serialize_metadata(metadata)
        }

        fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::Metadata> {
            DivisibleInt.deserialize_metadata(bytes)
        }
    }

    #[test]
    fn validate_array_override_is_invoked() -> VortexResult<()> {
        let before = OVERRIDE_CALLS.load(Ordering::SeqCst);
        // The content would fail `refine_scalar`, so only the override can produce `Ok` here.
        let arr: ArrayRef =
            PrimitiveArray::new(buffer![1u64, 2, 3, 4], Validity::NonNullable).into_array();
        AlwaysAcceptedArray::validate_array(&Divisor(1), &arr)?;
        let after = OVERRIDE_CALLS.load(Ordering::SeqCst);
        assert_eq!(
            after,
            before + 1,
            "override must have been invoked exactly once"
        );
        Ok(())
    }
}

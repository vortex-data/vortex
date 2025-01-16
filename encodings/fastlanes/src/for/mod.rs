use std::fmt::{Debug, Display};

pub use compress::*;
use serde::{Deserialize, Serialize};
use vortex_array::encoding::ids;
use vortex_array::stats::{StatisticsVTable, StatsSet};
use vortex_array::validate::ValidateVTable;
use vortex_array::validity::{ArrayValidity, LogicalValidity, ValidityVTable};
use vortex_array::variants::{PrimitiveArrayTrait, VariantsVTable};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{impl_encoding, ArrayDType, ArrayData, ArrayLen, Canonical, IntoCanonical};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_scalar::{PValue, Scalar};

mod compress;
mod compute;

impl_encoding!("fastlanes.for", ids::FL_FOR, FoR);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoRMetadata {
    reference: PValue,
    shift: u8,
}

impl Display for FoRMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl FoRArray {
    pub fn try_new(child: ArrayData, reference: Scalar, shift: u8) -> VortexResult<Self> {
        if reference.is_null() {
            vortex_bail!("Reference value cannot be null");
        }

        let reference = reference.cast(
            &reference
                .dtype()
                .with_nullability(child.dtype().nullability()),
        )?;

        let dtype = reference.dtype().clone();

        // Convert the reference into a PValue which is smaller to store.
        let reference = reference
            .as_primitive()
            .pvalue()
            .vortex_expect("Reference value is non-null");

        Self::try_from_parts(
            dtype,
            child.len(),
            FoRMetadata { reference, shift },
            [child].into(),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn encoded(&self) -> ArrayData {
        let dtype = if self.ptype().is_signed_int() {
            &DType::Primitive(self.ptype().to_unsigned(), self.dtype().nullability())
        } else {
            self.dtype()
        };
        self.as_ref()
            .child(0, dtype, self.len())
            .vortex_expect("FoRArray is missing encoded child array")
    }

    #[inline]
    pub fn reference_scalar(&self) -> Scalar {
        Scalar::primitive_value(
            self.metadata().reference,
            self.ptype(),
            self.dtype().nullability(),
        )
    }

    #[inline]
    pub fn shift(&self) -> u8 {
        self.metadata().shift
    }
}

impl ValidityVTable<FoRArray> for FoREncoding {
    fn is_valid(&self, array: &FoRArray, index: usize) -> bool {
        array.encoded().is_valid(index)
    }

    fn logical_validity(&self, array: &FoRArray) -> LogicalValidity {
        array.encoded().logical_validity()
    }
}

impl IntoCanonical for FoRArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        decompress(self).map(Canonical::Primitive)
    }
}

impl VisitorVTable<FoRArray> for FoREncoding {
    fn accept(&self, array: &FoRArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("encoded", &array.encoded())
    }
}

impl StatisticsVTable<FoRArray> for FoREncoding {}

impl ValidateVTable<FoRArray> for FoREncoding {}

impl VariantsVTable<FoRArray> for FoREncoding {
    fn as_primitive_array<'a>(&self, array: &'a FoRArray) -> Option<&'a dyn PrimitiveArrayTrait> {
        Some(array)
    }
}

impl PrimitiveArrayTrait for FoRArray {}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_scalar::PValue;

    use crate::FoRMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_for_metadata() {
        check_metadata(
            "for.metadata",
            FoRMetadata {
                reference: PValue::I64(i64::MAX),
                shift: u8::MAX,
            },
        );
    }
}

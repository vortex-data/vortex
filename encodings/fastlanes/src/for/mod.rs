use std::fmt::Debug;

pub use compress::*;
use serde::{Deserialize, Serialize};
use vortex_array::stats::StatsSet;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::visitor::ArrayVisitor;
use vortex_array::vtable::{
    CanonicalVTable, StatisticsVTable, ValidateVTable, ValidityVTable, VariantsVTable,
    VisitorVTable,
};
use vortex_array::{encoding_ids, impl_encoding, Array, Canonical, SerdeMetadata};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::{PValue, Scalar};

mod compress;
mod compute;

impl_encoding!(
    "fastlanes.for",
    encoding_ids::FL_FOR,
    FoR,
    SerdeMetadata<FoRMetadata>
);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct FoRMetadata {
    reference: PValue,
    // TODO(ngates): move shift into BitPackedArray and then ForMetadata is 64 bits of PValue.
    shift: u8,
}

impl FoRArray {
    pub fn try_new(child: Array, reference: Scalar, shift: u8) -> VortexResult<Self> {
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
            SerdeMetadata(FoRMetadata { reference, shift }),
            None,
            Some([child].into()),
            StatsSet::default(),
        )
    }

    #[inline]
    pub fn encoded(&self) -> Array {
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
    fn is_valid(&self, array: &FoRArray, index: usize) -> VortexResult<bool> {
        array.encoded().is_valid(index)
    }

    fn all_valid(&self, array: &FoRArray) -> VortexResult<bool> {
        array.encoded().all_valid()
    }

    fn validity_mask(&self, array: &FoRArray) -> VortexResult<Mask> {
        array.encoded().validity_mask()
    }
}

impl CanonicalVTable<FoRArray> for FoREncoding {
    fn into_canonical(&self, array: FoRArray) -> VortexResult<Canonical> {
        decompress(array).map(Canonical::Primitive)
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
    use vortex_array::SerdeMetadata;
    use vortex_scalar::PValue;

    use crate::FoRMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_for_metadata() {
        check_metadata(
            "for.metadata",
            SerdeMetadata(FoRMetadata {
                reference: PValue::I64(i64::MAX),
                shift: u8::MAX,
            }),
        );
    }
}

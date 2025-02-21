use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::{Scalar, ScalarValue};

use crate::encoding::encoding_ids;
use crate::stats::{Stat, StatsSet};
use crate::visitor::ArrayVisitor;
use crate::vtable::{StatisticsVTable, ValidateVTable, ValidityVTable, VisitorVTable};
use crate::{impl_encoding, EmptyMetadata};

mod canonical;
mod compute;
mod variants;

impl_encoding!(
    "vortex.constant",
    encoding_ids::CONSTANT,
    Constant,
    EmptyMetadata
);

impl ConstantArray {
    pub fn new<S>(scalar: S, length: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        let stats = StatsSet::constant(scalar.clone(), length);
        let (dtype, scalar_value) = scalar.into_parts();

        // Serialize the scalar_value into a FlatBuffer
        let value_buffer = scalar_value.to_flexbytes();

        Self::try_from_parts(
            dtype,
            length,
            EmptyMetadata,
            [value_buffer.into_inner()].into(),
            vec![].into(),
            stats,
        )
        .vortex_expect("Failed to create Constant array")
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> Scalar {
        let sv = ScalarValue::from_flexbytes(
            self.as_ref()
                .byte_buffer(0)
                .vortex_expect("Missing scalar value buffer"),
        )
        .vortex_expect("Failed to deserialize scalar value");
        Scalar::new(self.dtype().clone(), sv)
    }
}

impl ValidateVTable<ConstantArray> for ConstantEncoding {}

impl ValidityVTable<ConstantArray> for ConstantEncoding {
    fn is_valid(&self, array: &ConstantArray, _index: usize) -> VortexResult<bool> {
        Ok(!array.scalar().is_null())
    }

    fn all_valid(&self, array: &ConstantArray) -> VortexResult<bool> {
        Ok(!array.scalar().is_null())
    }

    fn all_invalid(&self, array: &ConstantArray) -> VortexResult<bool> {
        Ok(array.scalar().is_null())
    }

    fn validity_mask(&self, array: &ConstantArray) -> VortexResult<Mask> {
        Ok(match array.scalar().is_null() {
            true => Mask::AllFalse(array.len()),
            false => Mask::AllTrue(array.len()),
        })
    }
}

impl StatisticsVTable<ConstantArray> for ConstantEncoding {
    fn compute_statistics(&self, array: &ConstantArray, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::constant(array.scalar(), array.len()))
    }
}

impl VisitorVTable<ConstantArray> for ConstantEncoding {
    fn accept(&self, array: &ConstantArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_buffer(array.byte_buffer(0).vortex_expect("missing scalar buffer"))
    }
}

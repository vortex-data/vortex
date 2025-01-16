use std::fmt::Display;

use flexbuffers::Reader;
use serde::{Deserialize, Serialize};
use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_flatbuffers::{FlatBuffer, WriteFlatBuffer};
use vortex_scalar::{Scalar, ScalarValue};

use crate::encoding::ids;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::validate::ValidateVTable;
use crate::validity::{LogicalValidity, ValidityVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{impl_encoding, ArrayDType, ArrayLen};

mod canonical;
mod compute;
mod variants;

impl_encoding!("vortex.constant", ids::CONSTANT, Constant);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantMetadata;

impl ConstantArray {
    pub fn new<S>(scalar: S, length: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        let stats = StatsSet::constant(&scalar, length);
        let (dtype, scalar_value) = scalar.into_parts();

        // Serialize the scalar_value into a FlatBuffer
        let mut ser = flexbuffers::FlexbufferSerializer::new();
        scalar_value
            .serialize(&mut ser)
            .map_err(VortexError::FlexBuffersSerError)
            .vortex_expect("Failed to serialize ScalarValue");
        let value_buffer = FlatBuffer::copy_from(ser.view());

        Self::try_from_parts(dtype, length, value_buffer.into_inner(), [].into(), stats)
            .vortex_expect("Failed to create Constant array")
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> Scalar {
        let value = ScalarValue::deserialize(
            Reader::get_root(self.as_ref().metadata())
                .vortex_expect("Failed to get root of metadata"),
        )
        .map_err(VortexError::FlexBuffersDeError)
        .vortex_expect("Failed to deserialize ScalarValue");
        Scalar::new(self.dtype().clone(), value)
    }
}

impl ValidateVTable<ConstantArray> for ConstantEncoding {}

impl ValidityVTable<ConstantArray> for ConstantEncoding {
    fn is_valid(&self, array: &ConstantArray, _index: usize) -> bool {
        !array.scalar().is_null()
    }

    fn logical_validity(&self, array: &ConstantArray) -> LogicalValidity {
        match array.scalar().is_null() {
            true => LogicalValidity::AllInvalid(array.len()),
            false => LogicalValidity::AllValid(array.len()),
        }
    }
}

impl StatisticsVTable<ConstantArray> for ConstantEncoding {
    fn compute_statistics(&self, array: &ConstantArray, _stat: Stat) -> VortexResult<StatsSet> {
        Ok(StatsSet::constant(&array.scalar(), array.len()))
    }
}

impl VisitorVTable<ConstantArray> for ConstantEncoding {
    fn accept(&self, _array: &ConstantArray, _visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        Ok(())
    }
}

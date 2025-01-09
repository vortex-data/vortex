use std::fmt::Display;

use serde::{Deserialize, Serialize};
use vortex_avro::{FromAvro, ToAvro};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::encoding::ids;
use crate::stats::{Stat, StatisticsVTable, StatsSet};
use crate::validity::{LogicalValidity, ValidityVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::{impl_encoding, ArrayDType, ArrayLen, ArrayTrait};
mod canonical;
mod compute;
mod variants;

impl_encoding!("vortex.constant", ids::CONSTANT, Constant);

#[derive(Debug, Clone, Serialize, Deserialize, FromAvro, ToAvro)]
pub struct ConstantMetadata {
    pub(crate) scalar_value: ScalarValue,
}

impl Display for ConstantMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConstantMetadata {{ scalar_value: {} }}",
            self.scalar_value
        )
    }
}

impl ConstantArray {
    pub fn new<S>(scalar: S, length: usize) -> Self
    where
        S: Into<Scalar>,
    {
        let scalar = scalar.into();
        let stats = StatsSet::constant(&scalar, length);
        let (dtype, scalar_value) = scalar.into_parts();
        Self::try_from_parts(
            dtype,
            length,
            ConstantMetadata { scalar_value },
            [].into(),
            stats,
        )
        .vortex_expect("Failed to create Constant array")
    }

    /// Returns the [`Scalar`] value of this constant array.
    pub fn scalar(&self) -> Scalar {
        // NOTE(ngates): these clones are pretty cheap.
        Scalar::new(self.dtype().clone(), self.metadata().scalar_value.clone())
    }
}

impl ArrayTrait for ConstantArray {}

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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use vortex_avro::{from_avro_binary, to_avro_binary, AvroValue, FromAvro, ToAvro};
    use vortex_scalar::Scalar;

    use super::ConstantMetadata;

    #[test]
    #[allow(unused_variables)]
    fn test_avro() {
        let metadata: ConstantMetadata = ConstantMetadata {
            scalar_value: Scalar::primitive(42i32, vortex_dtype::Nullability::NonNullable)
                .into_value(),
        };

        println!("metadata(avro): {:?}", AvroValue::from(metadata.clone()));
        println!("write schema: {:?}", ConstantMetadata::write_schema());

        let metadata_bin = to_avro_binary(metadata).unwrap();
        let mut metadata_bin = Cursor::new(metadata_bin);
        let reverse: ConstantMetadata =
            from_avro_binary(&ConstantMetadata::read_schema(), &mut metadata_bin).unwrap();
        println!("done!");
    }
}

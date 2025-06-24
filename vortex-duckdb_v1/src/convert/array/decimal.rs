use duckdb::core::FlatVector;
use vortex::ArrayRef;
use vortex::arrays::{BooleanBuffer, DecimalArray};
use vortex::buffer::Buffer;
use vortex::dtype::{DType, DecimalDType};
use vortex::error::{VortexResult, vortex_bail, vortex_panic};
use vortex::scalar::{DecimalValueType, NativeDecimalType};
use vortex::validity::Validity;

use crate::convert::array::data_chunk_adaptor::SizedFlatVector;
use crate::{FromDuckDB, FromDuckDBType};

impl FromDuckDB<SizedFlatVector> for DecimalArray {
    fn from_duckdb(sized_vector: SizedFlatVector) -> VortexResult<ArrayRef> {
        let nullable = sized_vector.nullable;
        let vector = sized_vector.vector;

        let val = vector.validity_slice();

        // If validity buffer has a value this array must be nullable
        if val.is_some() {
            assert!(nullable)
        }

        let validity = if val.is_some() {
            // Use the validity slice
            let buf: BooleanBuffer = (0..sized_vector.len)
                .map(|i| !vector.row_is_null(i as u64))
                .collect();
            if buf.count_set_bits() == 0 {
                Validity::AllInvalid
            } else {
                Validity::from(buf)
            }
        } else if nullable {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };

        let dtype = DType::from_duckdb(vector.logical_type(), nullable.into())?;

        let Some(decimal_dtype) = dtype.as_decimal() else {
            vortex_panic!("converted decimal vector to non-decimal type")
        };

        let arr = match precision_to_duckdb_storage_size(decimal_dtype)? {
            DecimalValueType::I16 => {
                into_decimal::<i16>(vector, sized_vector.len, decimal_dtype, validity)
            }
            DecimalValueType::I32 => {
                into_decimal::<i32>(vector, sized_vector.len, decimal_dtype, validity)
            }
            DecimalValueType::I64 => {
                into_decimal::<i64>(vector, sized_vector.len, decimal_dtype, validity)
            }
            DecimalValueType::I128 => {
                into_decimal::<i128>(vector, sized_vector.len, decimal_dtype, validity)
            }
            ty => vortex_panic!("cannot handle type {:?}, should not be returned", ty),
        };

        Ok(arr.to_array())
    }
}

fn into_decimal<D: NativeDecimalType>(
    vector: FlatVector,
    len: usize,
    dtype: &DecimalDType,
    validity: Validity,
) -> DecimalArray {
    let buf: Buffer<D> = vector.as_slice_with_len(len).iter().cloned().collect();
    DecimalArray::new(buf, *dtype, validity)
}

/// Maps a decimal precision into the small type that can represent it.
/// see <https://duckdb.org/docs/stable/sql/data_types/numeric.html#fixed-point-decimals>
pub fn precision_to_duckdb_storage_size(
    decimal_dtype: &DecimalDType,
) -> VortexResult<DecimalValueType> {
    Ok(match decimal_dtype.precision() {
        1..=4 => DecimalValueType::I16,
        5..=9 => DecimalValueType::I32,
        10..=18 => DecimalValueType::I64,
        19..=38 => DecimalValueType::I128,
        decimal_dtype => vortex_bail!("cannot represent decimal in ducdkb {decimal_dtype}"),
    })
}

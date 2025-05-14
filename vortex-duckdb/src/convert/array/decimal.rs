use duckdb::core::FlatVector;
use duckdb::vtab::arrow::WritableVector;
use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::arrays::{BooleanBuffer, DecimalArray};
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexResult, vortex_bail, vortex_panic};
use vortex_scalar::{DecimalValueType, NativeDecimalType};

use crate::convert::array::data_chunk_adaptor::SizedFlatVector;
use crate::convert::array::validity::write_validity_from_mask;
use crate::{ConversionCache, FromDuckDB, FromDuckDBType, ToDuckDB};

impl ToDuckDB for DecimalArray {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        _cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        let mut vector = chunk.flat_vector();

        // Duckdb has an assumed storage width based on the decimal width (what we call precision)
        match (
            self.values_type(),
            precision_to_duckdb_storage_size(&self.decimal_dtype())?,
        ) {
            (DecimalValueType::I8, DecimalValueType::I16) => {
                convert_buffer_and_write_decimal_values::<i8, i16>(self, &mut vector)
            }
            (DecimalValueType::I16, DecimalValueType::I16) => {
                write_decimal_values::<i16>(self, &mut vector)
            }
            (DecimalValueType::I8, DecimalValueType::I32) => {
                convert_buffer_and_write_decimal_values::<i8, i32>(self, &mut vector)
            }
            (DecimalValueType::I16, DecimalValueType::I32) => {
                convert_buffer_and_write_decimal_values::<i16, i32>(self, &mut vector)
            }
            (DecimalValueType::I32, DecimalValueType::I32) => {
                write_decimal_values::<i32>(self, &mut vector)
            }
            (DecimalValueType::I8, DecimalValueType::I64) => {
                convert_buffer_and_write_decimal_values::<i8, i64>(self, &mut vector)
            }
            (DecimalValueType::I16, DecimalValueType::I64) => {
                convert_buffer_and_write_decimal_values::<i16, i64>(self, &mut vector)
            }
            (DecimalValueType::I32, DecimalValueType::I64) => {
                convert_buffer_and_write_decimal_values::<i32, i64>(self, &mut vector)
            }
            (DecimalValueType::I64, DecimalValueType::I64) => {
                write_decimal_values::<i64>(self, &mut vector)
            }
            (DecimalValueType::I8, DecimalValueType::I128) => {
                convert_buffer_and_write_decimal_values::<i8, i128>(self, &mut vector)
            }
            (DecimalValueType::I16, DecimalValueType::I128) => {
                convert_buffer_and_write_decimal_values::<i16, i128>(self, &mut vector)
            }
            (DecimalValueType::I32, DecimalValueType::I128) => {
                convert_buffer_and_write_decimal_values::<i32, i128>(self, &mut vector)
            }
            (DecimalValueType::I64, DecimalValueType::I128) => {
                convert_buffer_and_write_decimal_values::<i64, i128>(self, &mut vector)
            }
            (DecimalValueType::I128, DecimalValueType::I128) => {
                write_decimal_values::<i128>(self, &mut vector)
            }
            (from, to @ (DecimalValueType::I8 | DecimalValueType::I256)) => vortex_bail!(
                "cannot convert from ({:?}) to ({:?}) single the target decimal value type not supported by duckdb",
                from,
                to
            ),
            x => vortex_bail!(
                "cannot convert {:?} decimal to duckdb decimal, likely a downcast, which is not support yet",
                x
            ),
        };

        write_validity_from_mask(self.validity_mask()?, &mut vector);

        Ok(())
    }
}

// Writes a decimal array to a duckdb array, where the storage types differ.
fn convert_buffer_and_write_decimal_values<
    From: NativeDecimalType + AsPrimitive<To>,
    To: NativeDecimalType + 'static,
>(
    array: &DecimalArray,
    vector: &mut FlatVector,
) {
    let buf: Buffer<To> = array.buffer::<From>().iter().map(|v| v.as_()).collect();
    vector.copy(&buf);
}

// Writes a decimal array to a duckdb array, where the storage types are the same.
fn write_decimal_values<D: NativeDecimalType>(array: &DecimalArray, vector: &mut FlatVector) {
    vector.copy(&array.buffer::<D>());
}

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
/// see https://duckdb.org/docs/stable/sql/data_types/numeric.html#fixed-point-decimals
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

#[cfg(test)]
mod tests {
    use duckdb::core::DataChunkHandle;
    use vortex_array::ArrayRef;
    use vortex_array::arrays::{DecimalArray, DecimalVTable, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::convert::array::data_chunk_adaptor::SizedFlatVector;
    use crate::{ConversionCache, FromDuckDB, ToDuckDBType, to_duckdb_chunk};

    #[test]
    fn to_decimal_i16() {
        let array = DecimalArray::new(
            buffer![100i16, 200i16, 255i16],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        );
        let str = StructArray::from_fields(&[("a", array.to_array())]).unwrap();
        let mut chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        to_duckdb_chunk(&str, &mut chunk, &mut ConversionCache::default()).unwrap();

        assert_eq!(
            format!("{:?}", chunk),
            r#"Chunk - [1 Columns]
- FLAT DECIMAL(3,2): 3 = [ 1.00, 2.00, 2.55]
"#
        )
    }

    #[test]
    fn to_decimal_i32() {
        let array = DecimalArray::new(
            buffer![100i32, 200i32, 300i32],
            DecimalDType::new(5, 2),
            Validity::NonNullable,
        );
        let str = StructArray::from_fields(&[("a", array.to_array())]).unwrap();
        let mut chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        to_duckdb_chunk(&str, &mut chunk, &mut ConversionCache::default()).unwrap();
        assert_eq!(
            format!("{:?}", chunk),
            r#"Chunk - [1 Columns]
- FLAT DECIMAL(5,2): 3 = [ 1.00, 2.00, 3.00]
"#
        )
    }

    #[test]
    fn to_decimal_i8_p5() {
        let array = DecimalArray::new(
            buffer![100i8, 102i8, 109i8],
            DecimalDType::new(5, 2),
            Validity::NonNullable,
        );
        let str = StructArray::from_fields(&[("a", array.to_array())]).unwrap();
        let mut chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        to_duckdb_chunk(&str, &mut chunk, &mut ConversionCache::default()).unwrap();
        assert_eq!(
            format!("{:?}", chunk),
            r#"Chunk - [1 Columns]
- FLAT DECIMAL(5,2): 3 = [ 1.00, 1.02, 1.09]
"#
        )
    }

    #[test]
    fn to_decimal_i128() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128],
            DecimalDType::new(20, 2),
            Validity::AllValid,
        );
        let str = StructArray::from_fields(&[("a", array.to_array())]).unwrap();
        let mut chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        to_duckdb_chunk(&str, &mut chunk, &mut ConversionCache::default()).unwrap();
        assert_eq!(
            format!("{:?}", chunk),
            r#"Chunk - [1 Columns]
- FLAT DECIMAL(20,2): 3 = [ 1.00, 2.00, 3.00]
"#
        );

        let back = ArrayRef::from_duckdb(SizedFlatVector {
            vector: chunk.flat_vector(0),
            nullable: true,
            len: array.len(),
        })
        .unwrap();
        let back = back.as_::<DecimalVTable>();

        assert_eq!(back.dtype(), array.dtype());
        assert_eq!(back.validity(), array.validity());
    }
}

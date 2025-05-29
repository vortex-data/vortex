use std::marker::PhantomData;

use duckdb::vtab::arrow::WritableVector;
use num_traits::ToPrimitive;
use vortex::arrays::DecimalArray;
use vortex::buffer::Buffer;
use vortex::error::{VortexExpect, VortexResult};
use vortex::mask::Mask;
use vortex::scalar::{BigCast, NativeDecimalType, match_each_decimal_value_type};

use crate::exporter::FlatVectorExt;
use crate::{ColumnExporter, precision_to_duckdb_storage_size};

struct DecimalExporter<D: NativeDecimalType, N: NativeDecimalType> {
    values: Buffer<D>,
    validity: Mask,
    /// The DecimalType of the DuckDB column.
    dest_value_type: PhantomData<N>,
}

pub(crate) fn new_exporter(array: &DecimalArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let validity = array.validity_mask()?;
    let dest_values_type = precision_to_duckdb_storage_size(&array.decimal_dtype())?;

    match_each_decimal_value_type!(array.values_type(), |D| {
        match_each_decimal_value_type!(dest_values_type, |N| {
            Ok(Box::new(DecimalExporter {
                values: array.buffer::<D>(),
                validity,
                dest_value_type: PhantomData::<N>,
            }))
        })
    })
}

impl<D: NativeDecimalType, N: NativeDecimalType> ColumnExporter for DecimalExporter<D, N>
where
    D: ToPrimitive,
    N: BigCast,
{
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()> {
        let mut vector = vector.flat_vector();

        // Set validity if necessary.
        if vector.set_validity(&self.validity, offset, len) {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        // Copy the values from the Vortex array to the DuckDB vector.
        for (src, dst) in self.values[offset..offset + len]
            .iter()
            .zip(vector.as_mut_slice_with_len(len))
        {
            *dst = <N as BigCast>::from(*src).vortex_expect(
                "We know all decimals with this scale/precision fit into the target bit width",
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use duckdb::core::DataChunkHandle;
    use vortex::arrays::DecimalArray;
    use vortex::buffer::buffer;
    use vortex::dtype::DecimalDType;
    use vortex::validity::Validity;

    use super::*;
    use crate::ToDuckDBType;

    #[test]
    fn to_decimal_i16() {
        let array = DecimalArray::new(
            buffer![100i16, 200i16, 255i16],
            DecimalDType::new(3, 2),
            Validity::NonNullable,
        );

        let chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        chunk.set_len(array.len());

        new_exporter(&array)
            .unwrap()
            .export(0, 3, &mut chunk.flat_vector(0))
            .unwrap();

        assert_eq!(
            format!("{chunk:?}"),
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

        let chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        chunk.set_len(array.len());

        new_exporter(&array)
            .unwrap()
            .export(0, array.len(), &mut chunk.flat_vector(0))
            .unwrap();

        assert_eq!(
            format!("{chunk:?}"),
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

        let chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        chunk.set_len(array.len());

        new_exporter(&array)
            .unwrap()
            .export(0, array.len(), &mut chunk.flat_vector(0))
            .unwrap();

        assert_eq!(
            format!("{chunk:?}"),
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

        let chunk = DataChunkHandle::new(&[array.dtype().to_duckdb_type().unwrap()]);
        chunk.set_len(array.len());

        new_exporter(&array)
            .unwrap()
            .export(0, array.len(), &mut chunk.flat_vector(0))
            .unwrap();

        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- FLAT DECIMAL(20,2): 3 = [ 1.00, 2.00, 3.00]
"#
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use num_traits::ToPrimitive;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::decimal::DecimalDataParts;
use vortex::array::match_each_decimal_value_type;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::BigCast;
use vortex::dtype::DecimalDType;
use vortex::dtype::DecimalType;
use vortex::dtype::NativeDecimalType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::mask::Mask;

use crate::duckdb::VectorBuffer;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::validity;

struct DecimalExporter<D: NativeDecimalType, N: NativeDecimalType> {
    values: Buffer<D>,
    /// The DecimalType of the DuckDB column.
    dest_value_type: PhantomData<N>,
}

struct DecimalZeroCopyExporter<D: NativeDecimalType> {
    values: Buffer<D>,
    shared_buffer: VectorBuffer,
}

pub(crate) fn new_exporter(
    array: DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = array.len();
    let DecimalDataParts {
        validity,
        decimal_dtype,
        values_type,
        values,
    } = array.into_data_parts();
    let dest_values_type = precision_to_duckdb_storage_size(&decimal_dtype)?;

    if matches!(validity, Validity::AllInvalid) {
        return Ok(all_invalid::new_exporter());
    }
    let validity = validity.to_array(len).execute::<Mask>(ctx)?;

    let exporter = if values_type == dest_values_type {
        match_each_decimal_value_type!(values_type, |D| {
            let buffer = Buffer::<D>::from_byte_buffer(values.into_host_sync());
            Box::new(DecimalZeroCopyExporter {
                values: buffer.clone(),
                shared_buffer: VectorBuffer::new(buffer),
            }) as Box<dyn ColumnExporter>
        })
    } else {
        match_each_decimal_value_type!(values_type, |D| {
            match_each_decimal_value_type!(dest_values_type, |N| {
                Box::new(DecimalExporter {
                    values: Buffer::<D>::from_byte_buffer(values.into_host_sync()),
                    dest_value_type: PhantomData::<N>,
                }) as Box<dyn ColumnExporter>
            })
        })
    };

    Ok(validity::new_exporter(validity, exporter))
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
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Copy the values from the Vortex array to the DuckDB vector.
        for (src, dst) in self.values[offset..offset + len]
            .iter()
            .zip(unsafe { vector.as_slice_mut(len) })
        {
            *dst = <N as BigCast>::from(*src).vortex_expect(
                "We know all decimals with this scale/precision fit into the target bit width",
            );
        }

        Ok(())
    }
}

impl<D: NativeDecimalType> ColumnExporter for DecimalZeroCopyExporter<D> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        assert!(self.values.len() >= offset + len);

        let pos = unsafe { self.values.as_ptr().add(offset) };
        unsafe { vector.set_vector_buffer(&self.shared_buffer) };
        // While we are setting a *mut T this is an artifact of the C API, this is in fact const.
        unsafe { vector.set_data_ptr(pos as *mut D) };

        Ok(())
    }
}

/// Maps a decimal precision into the small type that can represent it.
/// see <https://duckdb.org/docs/stable/sql/data_types/numeric.html#fixed-point-decimals>
pub fn precision_to_duckdb_storage_size(decimal_dtype: &DecimalDType) -> VortexResult<DecimalType> {
    Ok(match decimal_dtype.precision() {
        1..=4 => DecimalType::I16,
        5..=9 => DecimalType::I32,
        10..=18 => DecimalType::I64,
        19..=38 => DecimalType::I128,
        decimal_dtype => vortex_bail!("cannot represent decimal in ducdkb {decimal_dtype}"),
    })
}

#[cfg(test)]
mod tests {
    use vortex::array::LEGACY_SESSION;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::DecimalArray;
    use vortex::dtype::DecimalDType;
    use vortex::error::VortexExpect;

    use super::*;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    pub(crate) fn new_zero_copy_exporter(
        array: &DecimalArray,
    ) -> VortexResult<Box<dyn ColumnExporter>> {
        let validity = array.as_ref().validity()?.to_mask(
            array.as_ref().len(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        let dest_values_type = precision_to_duckdb_storage_size(&array.decimal_dtype())?;

        assert_eq!(array.values_type(), dest_values_type);
        match_each_decimal_value_type!(array.values_type(), |D| {
            let buffer = array.buffer::<D>();
            Ok(validity::new_exporter(
                validity,
                Box::new(DecimalZeroCopyExporter {
                    values: buffer.clone(),
                    shared_buffer: VectorBuffer::new(buffer),
                }),
            ))
        })
    }

    #[test]
    fn test_decimal_zero_copy_exporter() {
        // Create a decimal array with precision=10, scale=2 (e.g., 123.45)
        let decimal_dtype = DecimalDType::new(10, 2);
        let arr = DecimalArray::from_option_iter(
            [Some(12345i64), Some(67890), Some(-12300)], // 123.45, 678.90, -123.00
            decimal_dtype,
        );

        // Create a DuckDB integer chunk since decimal will be stored as i32 for this precision
        let mut chunk = DataChunk::new([LogicalType::decimal_type(10, 2)
            .vortex_expect("LogicalTypeRef creation should succeed for test data")]);

        new_zero_copy_exporter(&arr)
            .unwrap()
            .export(
                0,
                3,
                chunk.get_vector_mut(0),
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap();
        chunk.set_len(3);

        // Verify the exported data matches expected format
        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT DECIMAL(10,2): 3 = [ 123.45, 678.90, -123.00]
"#
        );
    }

    #[test]
    fn test_decimal_zero_copy_exporter_subset() {
        // Create a smaller decimal array for simpler testing
        let decimal_dtype = DecimalDType::new(5, 1);
        let arr = DecimalArray::from_option_iter(
            [Some(100i32), Some(110), Some(120), Some(130), Some(140)],
            decimal_dtype,
        );

        let mut chunk = DataChunk::new([LogicalType::decimal_type(5, 1)
            .vortex_expect("LogicalTypeRef creation should succeed for test data")]);

        // Export first 3 elements
        new_zero_copy_exporter(&arr)
            .unwrap()
            .export(
                0,
                3,
                chunk.get_vector_mut(0),
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap();
        chunk.set_len(3);

        // Verify the exported data matches expected format
        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT DECIMAL(5,1): 3 = [ 10.0, 11.0, 12.0]
"#
        );
    }

    #[test]
    fn test_decimal_zero_copy_exporter_with_nulls() {
        // Create a decimal array with some null values
        let decimal_dtype = DecimalDType::new(8, 3);
        let arr =
            DecimalArray::from_option_iter([Some(123456i32), None, Some(789012i32)], decimal_dtype);

        let mut chunk = DataChunk::new([LogicalType::decimal_type(8, 3)
            .vortex_expect("LogicalTypeRef creation should succeed for test data")]);

        new_zero_copy_exporter(&arr)
            .unwrap()
            .export(
                0,
                3,
                chunk.get_vector_mut(0),
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap();
        chunk.set_len(3);

        // Verify the exported data matches expected format (NULL is represented as NULL)
        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT DECIMAL(8,3): 3 = [ 123.456, NULL, 789.012]
"#
        );
    }
}

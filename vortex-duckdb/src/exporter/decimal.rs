use std::marker::PhantomData;

use duckdb::vtab::arrow::WritableVector;
use num_traits::{NumCast, ToPrimitive};
use vortex_array::arrays::DecimalArray;
use vortex_buffer::Buffer;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_scalar::{NativeDecimalType, match_each_decimal_value_type};

use crate::exporter::FlatVectorExt;
use crate::{ColumnExporter, precision_to_duckdb_storage_size};

struct DecimalExporter<D: NativeDecimalType, N: NativeDecimalType> {
    values: Buffer<D>,
    validity: Mask,
    /// The DecimalType of the DuckDB column.
    dest_value_type: PhantomData<N>,
}

pub(crate) fn new_exporter(array: DecimalArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let validity = array.validity_mask()?;
    let dest_values_type = precision_to_duckdb_storage_size(&array.decimal_dtype())?;

    match_each_decimal_value_type!(array.values_type(), |$D| {
        match_each_decimal_value_type!(dest_values_type, |$N| {
            Ok(Box::new(DecimalExporter {
                values: array.buffer::<$D>(),
                validity,
                dest_value_type: PhantomData::<$N>,
            }))
        })
    })
}

impl<D: NativeDecimalType, N: NativeDecimalType> ColumnExporter for DecimalExporter<D, N>
where
    D: ToPrimitive,
    N: NumCast,
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
            *dst = <N as NumCast>::from(*src)
                .vortex_expect("Decimal value must fit into target precision, we checked");
        }

        Ok(())
    }
}

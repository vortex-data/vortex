// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use num_traits::ToPrimitive;
use vortex::arrays::DecimalArray;
use vortex::buffer::Buffer;
use vortex::dtype::DecimalDType;
use vortex::error::{VortexExpect, VortexResult, vortex_bail};
use vortex::mask::Mask;
use vortex::scalar::{BigCast, DecimalValueType, NativeDecimalType, match_each_decimal_value_type};

use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, VectorExt};

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

/// Maps a decimal precision into the small type that can represent it.
/// see <https://duckdb.org/docs/stable/sql/data_types/numeric.html#fixed-point-decimals>
fn precision_to_duckdb_storage_size(
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

impl<D: NativeDecimalType, N: NativeDecimalType> ColumnExporter for DecimalExporter<D, N>
where
    D: ToPrimitive,
    N: BigCast,
{
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Set validity if necessary.
        if vector.set_validity(&self.validity, offset, len) {
            // All values are null, so no point copying the data.
            return Ok(());
        }

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

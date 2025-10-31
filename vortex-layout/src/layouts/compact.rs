// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::{
    ExtensionArray, FixedSizeListArray, ListViewArray, PrimitiveArray, StructArray,
    narrowed_decimal,
};
use vortex_array::vtable::ValidityHelper;
use vortex_array::{Array, ArrayRef, Canonical, IntoArray, ToCanonical};
use vortex_decimal_byte_parts::DecimalBytePartsArray;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_pco::PcoArray;
use vortex_scalar::DecimalType;
use vortex_zstd::ZstdArray;

fn is_pco_number_type(ptype: PType) -> bool {
    matches!(
        ptype,
        PType::F16
            | PType::F32
            | PType::F64
            | PType::I16
            | PType::I32
            | PType::I64
            | PType::U16
            | PType::U32
            | PType::U64
    )
}

/// A simple compressor that uses the "compact" strategy:
/// - Pco for supported numeric types (16, 32, and 64-bit floats and ints)
/// - Zstd for everything else (primitive arrays only)
#[derive(Debug, Clone)]
pub struct CompactCompressor {
    pco_level: usize,
    zstd_level: i32,
    values_per_page: usize,
}

impl CompactCompressor {
    pub fn with_pco_level(mut self, level: usize) -> Self {
        self.pco_level = level;
        self
    }

    pub fn with_zstd_level(mut self, level: i32) -> Self {
        self.zstd_level = level;
        self
    }

    /// Sets the number of non-null primitive values to store per
    /// separately-decompressible page/frame.
    ///
    /// Fewer values per page can reduce the time to query a small slice of rows, but too
    /// few can increase compressed size and (de)compression time. The default is 0, which
    /// is used for maximally-large pages.
    pub fn with_values_per_page(mut self, values_per_page: usize) -> Self {
        self.values_per_page = values_per_page;
        self
    }

    pub fn compress(&self, array: &dyn Array) -> VortexResult<ArrayRef> {
        self.compress_canonical(array.to_canonical())
    }

    /// Compress a single array using the compact strategy
    pub fn compress_canonical(&self, canonical: Canonical) -> VortexResult<ArrayRef> {
        let uncompressed_nbytes = canonical.as_ref().nbytes();
        let compressed = match &canonical {
            // TODO compress BoolArrays
            Canonical::Primitive(primitive) => {
                // pco for applicable numbers, zstd for everything else
                let ptype = primitive.ptype();

                if is_pco_number_type(ptype) {
                    let pco_array =
                        PcoArray::from_primitive(primitive, self.pco_level, self.values_per_page)?;
                    pco_array.into_array()
                } else {
                    let zstd_array = ZstdArray::from_primitive(
                        primitive,
                        self.zstd_level,
                        self.values_per_page,
                    )?;
                    zstd_array.into_array()
                }
            }
            Canonical::Decimal(decimal) => {
                let decimal = narrowed_decimal(decimal.clone());
                let validity = decimal.validity();
                let int_values = match decimal.values_type() {
                    DecimalType::I8 => {
                        PrimitiveArray::new(decimal.buffer::<i8>(), validity.clone())
                    }
                    DecimalType::I16 => {
                        PrimitiveArray::new(decimal.buffer::<i16>(), validity.clone())
                    }
                    DecimalType::I32 => {
                        PrimitiveArray::new(decimal.buffer::<i32>(), validity.clone())
                    }
                    DecimalType::I64 => {
                        PrimitiveArray::new(decimal.buffer::<i64>(), validity.clone())
                    }
                    _ => {
                        // Vortex lacks support for i128 and i256.
                        return Ok(canonical.into_array());
                    }
                };
                let compressed = self.compress_canonical(Canonical::Primitive(int_values))?;
                DecimalBytePartsArray::try_new(compressed, decimal.decimal_dtype())?.to_array()
            }
            Canonical::VarBinView(vbv) => {
                // always zstd
                ZstdArray::from_var_bin_view(vbv, self.zstd_level, self.values_per_page)?
                    .into_array()
            }
            Canonical::Struct(struct_array) => {
                // recurse
                let fields = struct_array
                    .fields()
                    .iter()
                    .map(|field| self.compress(field))
                    .collect::<VortexResult<Vec<_>>>()?;

                StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity().clone(),
                )?
                .into_array()
            }
            Canonical::List(list_array) => {
                let compressed_elems = self.compress(list_array.elements())?;

                // Note that since the type of our offsets and sizes is not encoded in our `DType`,
                // we can narrow the widths.
                let compressed_offsets =
                    self.compress(&list_array.offsets().to_primitive().narrow()?.into_array())?;
                let compressed_sizes =
                    self.compress(&list_array.sizes().to_primitive().narrow()?.into_array())?;

                ListViewArray::try_new(
                    compressed_elems,
                    compressed_offsets,
                    compressed_sizes,
                    list_array.validity().clone(),
                )?
                .into_array()
            }
            Canonical::FixedSizeList(list_array) => {
                let compressed_elems = self.compress(list_array.elements())?;

                FixedSizeListArray::try_new(
                    compressed_elems,
                    list_array.list_size(),
                    list_array.validity().clone(),
                    list_array.len(),
                )?
                .into_array()
            }
            Canonical::Extension(ext_array) => {
                let compressed_storage = self.compress(ext_array.storage())?;

                ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage).into_array()
            }
            _ => return Ok(canonical.into_array()),
        };

        if compressed.nbytes() >= uncompressed_nbytes {
            return Ok(canonical.into_array());
        }
        Ok(compressed)
    }
}

impl Default for CompactCompressor {
    fn default() -> Self {
        Self {
            pco_level: pco::DEFAULT_COMPRESSION_LEVEL,
            zstd_level: 3,
            // This is probably high enough to not hurt performance or
            // compression. It also currently aligns with the default strategy's
            // number of rows per statistic, which allows efficient pushdown
            // (but nothing enforces this).
            values_per_page: 8192,
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, ToCanonical, assert_arrays_eq};
    use vortex_buffer::buffer;
    use vortex_dtype::FieldName;

    use super::*;

    #[test]
    fn test_compact_compressor_struct_with_mixed_types() {
        let compressor = CompactCompressor::default();

        // Create a struct array containing various types
        let columns = vec![
            // Pco types
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, 4.0, 5.0], Validity::NonNullable),
            PrimitiveArray::new(buffer![10i32, 20, 30, 40, 50], Validity::NonNullable),
            // Zstd types
            PrimitiveArray::new(buffer![11u8, 22, 33, 44, 55], Validity::NonNullable),
        ]
        .iter()
        .map(|a| a.clone().into_array())
        .collect::<Vec<_>>();
        let field_names: Vec<FieldName> =
            vec!["f64_field".into(), "i32_field".into(), "u8_field".into()];

        let n_rows = columns[0].len();
        let struct_array = StructArray::try_new(
            field_names.clone().into(),
            columns.clone(),
            n_rows,
            Validity::NonNullable,
        )
        .unwrap();

        // Compress the struct array
        let compressed = compressor.compress(struct_array.as_ref()).unwrap();

        // Verify we can decompress back to original
        let decompressed = compressed.to_canonical().into_array();
        assert_eq!(decompressed.len(), n_rows);
        let decompressed_struct = decompressed.to_struct();

        // Verify each field can be accessed and has correct data
        for (i, name) in decompressed_struct.names().iter().enumerate() {
            assert_eq!(name, field_names[i]);
            let decompressed_array = decompressed_struct.field_by_name(name).unwrap().clone();
            assert_eq!(decompressed_array.len(), n_rows);

            assert_arrays_eq!(decompressed_array.as_ref(), columns[i].as_ref());
        }
    }
}

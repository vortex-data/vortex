// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal array exporter for ClickHouse.
//!
//! This module exports Vortex decimal arrays to ClickHouse column buffers.
//! It handles different decimal precisions (Decimal32, Decimal64, Decimal128, Decimal256).

use std::any::Any;
use std::ffi::c_void;

use vortex::array::arrays::DecimalArray;
use vortex::array::{Array, ArrayRef, ToCanonical};
use vortex::dtype::{DecimalDType, DecimalType, Nullability};
use vortex::error::{VortexResult, vortex_bail};
use vortex::mask::Mask;

use super::{ColumnExporter, ExporterKind};

/// Map ClickHouse precision to the fixed-width storage type that ClickHouse expects.
///
/// ClickHouse always uses a fixed-width type for each decimal variant, regardless
/// of the actual magnitude of the stored values:
///   precision 1-9   -> Int32  (Decimal32)
///   precision 10-18 -> Int64  (Decimal64)
///   precision 19-38 -> Int128 (Decimal128)
///   precision 39-76 -> Int256 (Decimal256)
pub(crate) fn clickhouse_decimal_type(precision: u8) -> DecimalType {
    match precision {
        1..=9 => DecimalType::I32,
        10..=18 => DecimalType::I64,
        19..=38 => DecimalType::I128,
        39..=76 => DecimalType::I256,
        0 => unreachable!("precision must be greater than 0"),
        p => unreachable!("unsupported precision {p}"),
    }
}

/// Return the byte width of a `DecimalType` as exported to ClickHouse.
fn decimal_type_byte_width(dt: DecimalType) -> usize {
    match dt {
        DecimalType::I8 => 1,
        DecimalType::I16 => 2,
        DecimalType::I32 => 4,
        DecimalType::I64 => 8,
        DecimalType::I128 => 16,
        DecimalType::I256 => 32,
    }
}

/// Exporter for decimal arrays.
pub struct DecimalExporter {
    /// Cached canonical decimal array
    canonical: DecimalArray,
    /// Cached validity mask (None = non-nullable)
    validity: Option<Mask>,
    /// Current export position
    position: usize,
    /// Position at the start of last export (for validity export)
    last_export_start: usize,
    /// Number of rows exported in last export call
    last_export_count: usize,
    /// Total length of the array
    len: usize,
    /// Decimal dtype (precision and scale)
    decimal_dtype: DecimalDType,
    /// Vortex internal values type (may be smaller than ClickHouse expects)
    values_type: DecimalType,
    /// ClickHouse-expected export type (determined by precision ranges)
    export_type: DecimalType,
    /// Whether the array is nullable
    nullable: bool,
}

impl DecimalExporter {
    /// Create a new decimal exporter for the given array.
    pub fn new(array: ArrayRef) -> VortexResult<Self> {
        let len = array.len();
        let (decimal_dtype, nullable) = match array.dtype() {
            vortex::dtype::DType::Decimal(decimal_dtype, nullability) => {
                (*decimal_dtype, *nullability == Nullability::Nullable)
            }
            _ => vortex_bail!("DecimalExporter requires a decimal array"),
        };

        let canonical = array.to_decimal();
        let values_type = canonical.values_type();
        let validity = if nullable {
            Some(canonical.validity_mask()?)
        } else {
            None
        };

        let export_type = clickhouse_decimal_type(decimal_dtype.precision());

        Ok(Self {
            canonical,
            validity,
            position: 0,
            last_export_start: 0,
            last_export_count: 0,
            len,
            decimal_dtype,
            values_type,
            export_type,
            nullable,
        })
    }

    /// Get the decimal dtype
    pub fn decimal_dtype(&self) -> DecimalDType {
        self.decimal_dtype
    }

    /// Get the decimal values type
    pub fn values_type(&self) -> DecimalType {
        self.values_type
    }
}

impl ColumnExporter for DecimalExporter {
    fn kind(&self) -> ExporterKind {
        ExporterKind::Decimal
    }

    fn export(
        &mut self,
        column_ptr: *mut c_void,
        buffer_size_bytes: usize,
        max_rows: usize,
    ) -> VortexResult<usize> {
        if column_ptr.is_null() {
            vortex_bail!("column_ptr is null");
        }

        let remaining = self.len - self.position;
        let rows_to_export = remaining.min(max_rows);

        if rows_to_export == 0 {
            return Ok(0);
        }

        let elem_bytes = decimal_type_byte_width(self.export_type);
        let required_bytes = rows_to_export * elem_bytes;
        if buffer_size_bytes < required_bytes {
            vortex_bail!(
                "buffer too small: need {} bytes for {} Decimal({:?}) rows, got {}",
                required_bytes,
                rows_to_export,
                self.export_type,
                buffer_size_bytes
            );
        }

        // Record the start position for validity export
        self.last_export_start = self.position;
        self.last_export_count = rows_to_export;

        let start = self.position;
        let end = start + rows_to_export;

        // Export using the ClickHouse-expected type (export_type), reading from
        // the Vortex internal type (values_type) and widening if needed.
        macro_rules! export_widening {
            ($src_ty:ty, $dst_ty:ty) => {{
                let buffer = self.canonical.buffer::<$src_ty>();
                let slice = &buffer.as_slice()[start..end];
                let dst = column_ptr as *mut $dst_ty;
                for (i, &val) in slice.iter().enumerate() {
                    unsafe {
                        *dst.add(i) = val as $dst_ty;
                    }
                }
            }};
        }

        macro_rules! export_direct {
            ($ty:ty) => {{
                let buffer = self.canonical.buffer::<$ty>();
                let slice = &buffer.as_slice()[start..end];
                let dst = column_ptr as *mut $ty;
                unsafe {
                    std::ptr::copy_nonoverlapping(slice.as_ptr(), dst, rows_to_export);
                }
            }};
        }

        // Dispatch based on (values_type -> export_type)
        match (self.values_type, self.export_type) {
            // Same type: direct copy
            (DecimalType::I32, DecimalType::I32) => export_direct!(i32),
            (DecimalType::I64, DecimalType::I64) => export_direct!(i64),
            (DecimalType::I128, DecimalType::I128) => export_direct!(i128),
            (DecimalType::I256, DecimalType::I256) => {
                use vortex::dtype::i256;
                export_direct!(i256);
            }
            // Widen to I32
            (DecimalType::I8, DecimalType::I32) => export_widening!(i8, i32),
            (DecimalType::I16, DecimalType::I32) => export_widening!(i16, i32),
            // Widen to I64
            (DecimalType::I8, DecimalType::I64) => export_widening!(i8, i64),
            (DecimalType::I16, DecimalType::I64) => export_widening!(i16, i64),
            (DecimalType::I32, DecimalType::I64) => export_widening!(i32, i64),
            // Widen to I128
            (DecimalType::I8, DecimalType::I128) => export_widening!(i8, i128),
            (DecimalType::I16, DecimalType::I128) => export_widening!(i16, i128),
            (DecimalType::I32, DecimalType::I128) => export_widening!(i32, i128),
            (DecimalType::I64, DecimalType::I128) => export_widening!(i64, i128),
            // Widen to I256
            (DecimalType::I8, DecimalType::I256)
            | (DecimalType::I16, DecimalType::I256)
            | (DecimalType::I32, DecimalType::I256)
            | (DecimalType::I64, DecimalType::I256)
            | (DecimalType::I128, DecimalType::I256) => {
                use vortex::dtype::i256;
                match self.values_type {
                    DecimalType::I8 => {
                        let buffer = self.canonical.buffer::<i8>();
                        let slice = &buffer.as_slice()[start..end];
                        let dst = column_ptr as *mut i256;
                        for (i, &val) in slice.iter().enumerate() {
                            unsafe {
                                *dst.add(i) = i256::from_i128(val as i128);
                            }
                        }
                    }
                    DecimalType::I16 => {
                        let buffer = self.canonical.buffer::<i16>();
                        let slice = &buffer.as_slice()[start..end];
                        let dst = column_ptr as *mut i256;
                        for (i, &val) in slice.iter().enumerate() {
                            unsafe {
                                *dst.add(i) = i256::from_i128(val as i128);
                            }
                        }
                    }
                    DecimalType::I32 => {
                        let buffer = self.canonical.buffer::<i32>();
                        let slice = &buffer.as_slice()[start..end];
                        let dst = column_ptr as *mut i256;
                        for (i, &val) in slice.iter().enumerate() {
                            unsafe {
                                *dst.add(i) = i256::from_i128(val as i128);
                            }
                        }
                    }
                    DecimalType::I64 => {
                        let buffer = self.canonical.buffer::<i64>();
                        let slice = &buffer.as_slice()[start..end];
                        let dst = column_ptr as *mut i256;
                        for (i, &val) in slice.iter().enumerate() {
                            unsafe {
                                *dst.add(i) = i256::from_i128(val as i128);
                            }
                        }
                    }
                    DecimalType::I128 => {
                        let buffer = self.canonical.buffer::<i128>();
                        let slice = &buffer.as_slice()[start..end];
                        let dst = column_ptr as *mut i256;
                        for (i, &val) in slice.iter().enumerate() {
                            unsafe {
                                *dst.add(i) = i256::from_i128(val);
                            }
                        }
                    }
                    DecimalType::I256 => unreachable!(),
                }
            }
            // Unsupported: narrowing (should not happen)
            (src, dst) => vortex_bail!(
                "Unsupported decimal export: internal type {:?} to ClickHouse type {:?}",
                src,
                dst
            ),
        }

        self.position += rows_to_export;
        Ok(rows_to_export)
    }

    fn element_size_bytes(&self) -> usize {
        decimal_type_byte_width(self.export_type)
    }

    fn has_more(&self) -> bool {
        self.position < self.len
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_nullable(&self) -> bool {
        self.nullable
    }

    fn export_validity(&mut self, bitmap: *mut u8, max_rows: usize) -> VortexResult<usize> {
        super::export_validity_cached(
            bitmap,
            max_rows,
            self.validity.as_ref(),
            self.last_export_start,
            self.last_export_count,
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex::array::IntoArray;
    use vortex::array::arrays::DecimalArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::dtype::DecimalDType;

    #[test]
    fn test_decimal_exporter_i32() {
        let decimal_dtype = DecimalDType::new(9, 2);
        let values: Buffer<i32> = vec![12345i32, 67890, -12300].into();
        let array = DecimalArray::new(values, decimal_dtype, Validity::NonNullable).into_array();

        let mut exporter = DecimalExporter::new(array).expect("Failed to create exporter");

        assert!(exporter.has_more());
        assert_eq!(exporter.decimal_dtype().precision(), 9);
        assert_eq!(exporter.decimal_dtype().scale(), 2);
        assert_eq!(exporter.values_type(), DecimalType::I32);

        let mut output = vec![0i32; 3];
        let exported = exporter
            .export(
                output.as_mut_ptr() as *mut c_void,
                size_of_val(output.as_slice()),
                3,
            )
            .expect("Export failed");

        assert_eq!(exported, 3);
        assert_eq!(output, vec![12345, 67890, -12300]);
        assert!(!exporter.has_more());
    }

    #[test]
    fn test_decimal_exporter_i64() {
        let decimal_dtype = DecimalDType::new(18, 4);
        let values: Buffer<i64> = vec![1234567890i64, -9876543210].into();
        let array = DecimalArray::new(values, decimal_dtype, Validity::NonNullable).into_array();

        let mut exporter = DecimalExporter::new(array).expect("Failed to create exporter");

        assert_eq!(exporter.values_type(), DecimalType::I64);

        let mut output = vec![0i64; 2];
        let exported = exporter
            .export(
                output.as_mut_ptr() as *mut c_void,
                size_of_val(output.as_slice()),
                2,
            )
            .expect("Export failed");

        assert_eq!(exported, 2);
        assert_eq!(output, vec![1234567890, -9876543210]);
    }

    #[test]
    fn test_decimal_exporter_i128() {
        let decimal_dtype = DecimalDType::new(38, 10);
        let values: Buffer<i128> = vec![12345678901234567890i128, -9876543210987654321].into();
        let array = DecimalArray::new(values, decimal_dtype, Validity::NonNullable).into_array();

        let mut exporter = DecimalExporter::new(array).expect("Failed to create exporter");

        assert_eq!(exporter.values_type(), DecimalType::I128);

        let mut output = vec![0i128; 2];
        let exported = exporter
            .export(
                output.as_mut_ptr() as *mut c_void,
                size_of_val(output.as_slice()),
                2,
            )
            .expect("Export failed");

        assert_eq!(exported, 2);
        assert_eq!(output, vec![12345678901234567890i128, -9876543210987654321]);
    }

    #[test]
    fn test_decimal_exporter_partial() {
        let decimal_dtype = DecimalDType::new(9, 2);
        let values: Buffer<i32> = vec![100, 200, 300, 400, 500].into();
        let array = DecimalArray::new(values, decimal_dtype, Validity::NonNullable).into_array();

        let mut exporter = DecimalExporter::new(array).expect("Failed to create exporter");

        // Export first 2 rows
        let mut output1 = vec![0i32; 2];
        let exported1 = exporter
            .export(
                output1.as_mut_ptr() as *mut c_void,
                size_of_val(output1.as_slice()),
                2,
            )
            .expect("Export failed");

        assert_eq!(exported1, 2);
        assert_eq!(output1, vec![100, 200]);
        assert!(exporter.has_more());

        // Export next 3 rows
        let mut output2 = vec![0i32; 3];
        let exported2 = exporter
            .export(
                output2.as_mut_ptr() as *mut c_void,
                size_of_val(output2.as_slice()),
                5,
            ) // request more than available
            .expect("Export failed");

        assert_eq!(exported2, 3);
        assert_eq!(output2, vec![300, 400, 500]);
        assert!(!exporter.has_more());
    }

    #[test]
    fn test_decimal_exporter_nullable() {
        let decimal_dtype = DecimalDType::new(9, 2);
        let array =
            DecimalArray::from_option_iter(vec![Some(100i32), None, Some(300)], decimal_dtype)
                .into_array();

        let mut exporter = DecimalExporter::new(array).expect("Failed to create exporter");

        assert!(exporter.is_nullable());

        // Export data
        let mut data = vec![0i32; 3];
        let exported = exporter
            .export(
                data.as_mut_ptr() as *mut c_void,
                size_of_val(data.as_slice()),
                3,
            )
            .expect("Export failed");

        assert_eq!(exported, 3);

        // Export validity
        let mut validity = vec![0u8; 1];
        let validity_rows = exporter
            .export_validity(validity.as_mut_ptr(), 3)
            .expect("Export validity failed");

        assert_eq!(validity_rows, 3);
        // Validity bitmap: bit 0 = valid, bit 1 = invalid, bit 2 = valid
        // So we expect binary: 101 = 5
        assert_eq!(validity[0], 0b101);
    }
}

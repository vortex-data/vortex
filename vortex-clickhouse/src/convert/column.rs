// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Column data conversion between Vortex Array and ClickHouse columns.
//!
//! This module handles the actual data conversion between Vortex's Array types
//! and ClickHouse's column data structures. It provides functions for both
//! reading (Vortex -> ClickHouse) and writing (ClickHouse -> Vortex) directions.

use std::ffi::c_void;

use vortex::array::arrays::{BoolArray, PrimitiveArray, StructArray, VarBinViewArray};
use vortex::array::validity::Validity;
use vortex::array::{Array, ArrayRef, IntoArray, ToCanonical};
use vortex::buffer::Buffer;
use vortex::dtype::{DType, FieldNames, PType};
use vortex::error::{VortexResult, vortex_bail, vortex_err};

use super::dtype::clickhouse_type_to_vortex;
use crate::exporter::{ColumnExporter, PrimitiveExporter};

/// Convert a ClickHouse column to a Vortex Array.
///
/// This is used when writing data from ClickHouse to Vortex format.
/// The column data is read through the FFI boundary and converted to
/// the appropriate Vortex array type.
///
/// # Arguments
/// * `column_ptr` - Pointer to the ClickHouse column data
/// * `num_rows` - Number of rows to read
/// * `ch_type` - ClickHouse type string (e.g., "Int32", "String", "Array(UInt64)")
///
/// # Returns
/// A Vortex ArrayRef containing the converted data.
pub fn clickhouse_column_to_vortex(
    column_ptr: *const c_void,
    num_rows: usize,
    ch_type: &str,
) -> VortexResult<ArrayRef> {
    if column_ptr.is_null() {
        vortex_bail!("column_ptr is null");
    }

    if num_rows == 0 {
        // Return empty array of the appropriate type
        let dtype = clickhouse_type_to_vortex(ch_type)?;
        return create_empty_array(&dtype);
    }

    let dtype = clickhouse_type_to_vortex(ch_type)?;

    match &dtype {
        DType::Primitive(ptype, nullability) => {
            convert_primitive_column(column_ptr, num_rows, *ptype, *nullability)
        }
        DType::Utf8(nullability) => convert_string_column(column_ptr, num_rows, *nullability),
        DType::Bool(nullability) => convert_bool_column(column_ptr, num_rows, *nullability),
        _ => {
            vortex_bail!(
                "Unsupported ClickHouse type for column conversion: {}",
                ch_type
            )
        }
    }
}

/// Convert a Vortex Array to ClickHouse column data.
///
/// This is used when reading Vortex data into ClickHouse.
/// The Vortex array is converted and written to the provided column buffer.
///
/// # Arguments
/// * `array` - The Vortex array to convert
/// * `column_ptr` - Pointer to the ClickHouse column buffer
///
/// # Returns
/// Ok(()) on success, or an error if conversion fails.
pub fn vortex_to_clickhouse_column(array: &ArrayRef, column_ptr: *mut c_void) -> VortexResult<()> {
    if column_ptr.is_null() {
        vortex_bail!("column_ptr is null");
    }

    match array.dtype() {
        DType::Primitive(_, _) => {
            let mut exporter = PrimitiveExporter::new(array.clone())?;
            let buffer_size_bytes = exporter.element_size_bytes() * array.len();
            exporter.export(column_ptr, buffer_size_bytes, array.len())?;
            Ok(())
        }
        DType::Utf8(_) | DType::Binary(_) => {
            // String export requires callback-based approach
            // This is handled differently in the C++ layer
            vortex_bail!("String columns should use export_with_callback")
        }
        _ => {
            vortex_bail!(
                "Unsupported Vortex type for column conversion: {:?}",
                array.dtype()
            )
        }
    }
}

/// Create an empty array of the given dtype.
fn create_empty_array(dtype: &DType) -> VortexResult<ArrayRef> {
    match dtype {
        DType::Primitive(ptype, nullability) => {
            let validity = if nullability.is_nullable() {
                Validity::AllValid
            } else {
                Validity::NonNullable
            };
            match ptype {
                PType::I8 => {
                    Ok(PrimitiveArray::new(Buffer::<i8>::from(vec![]), validity).into_array())
                }
                PType::I16 => {
                    Ok(PrimitiveArray::new(Buffer::<i16>::from(vec![]), validity).into_array())
                }
                PType::I32 => {
                    Ok(PrimitiveArray::new(Buffer::<i32>::from(vec![]), validity).into_array())
                }
                PType::I64 => {
                    Ok(PrimitiveArray::new(Buffer::<i64>::from(vec![]), validity).into_array())
                }
                PType::U8 => {
                    Ok(PrimitiveArray::new(Buffer::<u8>::from(vec![]), validity).into_array())
                }
                PType::U16 => {
                    Ok(PrimitiveArray::new(Buffer::<u16>::from(vec![]), validity).into_array())
                }
                PType::U32 => {
                    Ok(PrimitiveArray::new(Buffer::<u32>::from(vec![]), validity).into_array())
                }
                PType::U64 => {
                    Ok(PrimitiveArray::new(Buffer::<u64>::from(vec![]), validity).into_array())
                }
                PType::F32 => {
                    Ok(PrimitiveArray::new(Buffer::<f32>::from(vec![]), validity).into_array())
                }
                PType::F64 => {
                    Ok(PrimitiveArray::new(Buffer::<f64>::from(vec![]), validity).into_array())
                }
                PType::F16 => vortex_bail!("F16 type not supported"),
            }
        }
        DType::Utf8(_) => Ok(VarBinViewArray::from_iter_str(Vec::<&str>::new()).into_array()),
        DType::Bool(_) => Ok(BoolArray::from_iter(Vec::<bool>::new()).into_array()),
        _ => vortex_bail!("Unsupported dtype for empty array: {:?}", dtype),
    }
}

/// Convert a primitive column from ClickHouse to Vortex.
fn convert_primitive_column(
    column_ptr: *const c_void,
    num_rows: usize,
    ptype: PType,
    nullability: vortex::dtype::Nullability,
) -> VortexResult<ArrayRef> {
    let validity = if nullability.is_nullable() {
        // Note: For nullable columns, ClickHouse's null bitmap is handled at the C++ FFI layer
        // via vx_ch_array_builder_append_batch_nullable_* functions
        Validity::AllValid
    } else {
        Validity::NonNullable
    };

    macro_rules! convert_primitive {
        ($rust_ty:ty) => {{
            let src = column_ptr as *const $rust_ty;
            let mut data = Vec::with_capacity(num_rows);
            unsafe {
                data.set_len(num_rows);
                std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), num_rows);
            }
            let buffer: Buffer<$rust_ty> = data.into();
            Ok(PrimitiveArray::new(buffer, validity).into_array())
        }};
    }

    match ptype {
        PType::I8 => convert_primitive!(i8),
        PType::I16 => convert_primitive!(i16),
        PType::I32 => convert_primitive!(i32),
        PType::I64 => convert_primitive!(i64),
        PType::U8 => convert_primitive!(u8),
        PType::U16 => convert_primitive!(u16),
        PType::U32 => convert_primitive!(u32),
        PType::U64 => convert_primitive!(u64),
        PType::F32 => convert_primitive!(f32),
        PType::F64 => convert_primitive!(f64),
        PType::F16 => vortex_bail!("F16 type not supported"),
    }
}

/// Convert a string column from ClickHouse to Vortex.
///
/// This expects the column data to be laid out as:
/// - An array of (offset, length) pairs followed by the actual string data
///
/// For simplicity, we use a callback-based approach where ClickHouse provides
/// individual strings.
fn convert_string_column(
    _column_ptr: *const c_void,
    _num_rows: usize,
    _nullability: vortex::dtype::Nullability,
) -> VortexResult<ArrayRef> {
    // String conversion requires special handling due to ClickHouse's string layout
    // This is typically done through the callback-based approach in the C++ layer
    vortex_bail!("String column conversion requires callback-based approach")
}

/// Convert a bool column from ClickHouse to Vortex.
fn convert_bool_column(
    column_ptr: *const c_void,
    num_rows: usize,
    _nullability: vortex::dtype::Nullability,
) -> VortexResult<ArrayRef> {
    // ClickHouse stores bools as UInt8 (0 or 1)
    let src = column_ptr as *const u8;
    let mut data = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let val = unsafe { *src.add(i) };
        data.push(val != 0);
    }

    Ok(BoolArray::from_iter(data).into_array())
}

/// Builder for constructing Vortex arrays from ClickHouse data incrementally.
///
/// This is useful when receiving data row by row or in batches from ClickHouse.
pub struct VortexColumnBuilder {
    dtype: DType,
    inner: ColumnBuilderInner,
}

enum ColumnBuilderInner {
    Bool(Vec<Option<bool>>),
    I8(Vec<i8>, Vec<bool>),
    I16(Vec<i16>, Vec<bool>),
    I32(Vec<i32>, Vec<bool>),
    I64(Vec<i64>, Vec<bool>),
    U8(Vec<u8>, Vec<bool>),
    U16(Vec<u16>, Vec<bool>),
    U32(Vec<u32>, Vec<bool>),
    U64(Vec<u64>, Vec<bool>),
    F32(Vec<f32>, Vec<bool>),
    F64(Vec<f64>, Vec<bool>),
    String(Vec<Option<String>>),
}

impl VortexColumnBuilder {
    /// Create a new column builder for the given ClickHouse type.
    pub fn new(ch_type: &str, capacity: usize) -> VortexResult<Self> {
        let dtype = clickhouse_type_to_vortex(ch_type)?;

        let inner = match &dtype {
            DType::Bool(_) => ColumnBuilderInner::Bool(Vec::with_capacity(capacity)),
            DType::Primitive(ptype, _) => match ptype {
                PType::I8 => ColumnBuilderInner::I8(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::I16 => ColumnBuilderInner::I16(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::I32 => ColumnBuilderInner::I32(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::I64 => ColumnBuilderInner::I64(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::U8 => ColumnBuilderInner::U8(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::U16 => ColumnBuilderInner::U16(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::U32 => ColumnBuilderInner::U32(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::U64 => ColumnBuilderInner::U64(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::F32 => ColumnBuilderInner::F32(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::F64 => ColumnBuilderInner::F64(
                    Vec::with_capacity(capacity),
                    Vec::with_capacity(capacity),
                ),
                PType::F16 => vortex_bail!("F16 type not supported"),
            },
            DType::Utf8(_) => ColumnBuilderInner::String(Vec::with_capacity(capacity)),
            _ => vortex_bail!("Unsupported type for column builder: {:?}", dtype),
        };

        Ok(Self { dtype, inner })
    }

    /// Append a null value.
    pub fn append_null(&mut self) {
        match &mut self.inner {
            ColumnBuilderInner::Bool(v) => v.push(None),
            ColumnBuilderInner::I8(values, validity) => {
                values.push(0);
                validity.push(false);
            }
            ColumnBuilderInner::I16(values, validity) => {
                values.push(0);
                validity.push(false);
            }
            ColumnBuilderInner::I32(values, validity) => {
                values.push(0);
                validity.push(false);
            }
            ColumnBuilderInner::I64(values, validity) => {
                values.push(0);
                validity.push(false);
            }
            ColumnBuilderInner::U8(values, validity) => {
                values.push(0);
                validity.push(false);
            }
            ColumnBuilderInner::U16(values, validity) => {
                values.push(0);
                validity.push(false);
            }
            ColumnBuilderInner::U32(values, validity) => {
                values.push(0);
                validity.push(false);
            }
            ColumnBuilderInner::U64(values, validity) => {
                values.push(0);
                validity.push(false);
            }
            ColumnBuilderInner::F32(values, validity) => {
                values.push(0.0);
                validity.push(false);
            }
            ColumnBuilderInner::F64(values, validity) => {
                values.push(0.0);
                validity.push(false);
            }
            ColumnBuilderInner::String(v) => v.push(None),
        }
    }

    /// Append an i8 value.
    pub fn append_i8(&mut self, value: i8) {
        if let ColumnBuilderInner::I8(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append an i16 value.
    pub fn append_i16(&mut self, value: i16) {
        if let ColumnBuilderInner::I16(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append an i32 value.
    pub fn append_i32(&mut self, value: i32) {
        if let ColumnBuilderInner::I32(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append an i64 value.
    pub fn append_i64(&mut self, value: i64) {
        if let ColumnBuilderInner::I64(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append a u8 value.
    pub fn append_u8(&mut self, value: u8) {
        if let ColumnBuilderInner::U8(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append a u16 value.
    pub fn append_u16(&mut self, value: u16) {
        if let ColumnBuilderInner::U16(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append a u32 value.
    pub fn append_u32(&mut self, value: u32) {
        if let ColumnBuilderInner::U32(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append a u64 value.
    pub fn append_u64(&mut self, value: u64) {
        if let ColumnBuilderInner::U64(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append an f32 value.
    pub fn append_f32(&mut self, value: f32) {
        if let ColumnBuilderInner::F32(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append an f64 value.
    pub fn append_f64(&mut self, value: f64) {
        if let ColumnBuilderInner::F64(values, validity) = &mut self.inner {
            values.push(value);
            validity.push(true);
        }
    }

    /// Append a string value.
    pub fn append_string(&mut self, value: &str) {
        if let ColumnBuilderInner::String(v) = &mut self.inner {
            v.push(Some(value.to_string()));
        }
    }

    /// Finish building and return the Vortex array.
    pub fn finish(self) -> VortexResult<ArrayRef> {
        let is_nullable = self.dtype.is_nullable();

        match self.inner {
            ColumnBuilderInner::Bool(values) => Ok(BoolArray::from_iter(values).into_array()),
            ColumnBuilderInner::I8(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<i8> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::I16(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<i16> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::I32(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<i32> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::I64(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<i64> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::U8(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<u8> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::U16(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<u16> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::U32(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<u32> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::U64(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<u64> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::F32(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<f32> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::F64(values, validity) => {
                let validity = make_validity(is_nullable, &validity);
                let buffer: Buffer<f64> = values.into();
                Ok(PrimitiveArray::new(buffer, validity).into_array())
            }
            ColumnBuilderInner::String(values) => {
                Ok(VarBinViewArray::from_iter_nullable_str(values).into_array())
            }
        }
    }
}

fn make_validity(is_nullable: bool, validity: &[bool]) -> Validity {
    if is_nullable && validity.iter().any(|&v| !v) {
        Validity::from_iter(validity.iter().copied())
    } else {
        Validity::NonNullable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_column_conversion() {
        // Create test data
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];

        // Convert from ClickHouse to Vortex
        let array = convert_primitive_column(
            data.as_ptr() as *const c_void,
            data.len(),
            PType::I32,
            vortex::dtype::Nullability::NonNullable,
        )
        .expect("Conversion failed");

        assert_eq!(array.len(), 5);

        // Verify values
        let primitive = array.to_primitive();
        let values = primitive.as_slice::<i32>();
        assert_eq!(values, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_column_builder() {
        let mut builder = VortexColumnBuilder::new("Int64", 5).unwrap();

        builder.append_i64(10);
        builder.append_i64(20);
        builder.append_null();
        builder.append_i64(40);
        builder.append_i64(50);

        let array = builder.finish().unwrap();
        assert_eq!(array.len(), 5);
    }

    #[test]
    fn test_string_builder() {
        let mut builder = VortexColumnBuilder::new("String", 3).unwrap();

        builder.append_string("hello");
        builder.append_null();
        builder.append_string("world");

        let array = builder.finish().unwrap();
        assert_eq!(array.len(), 3);

        // Verify via scalar
        let scalar = array.scalar_at(0).unwrap();
        assert!(!scalar.is_null());

        let scalar = array.scalar_at(1).unwrap();
        assert!(scalar.is_null());
    }

    #[test]
    fn test_bool_column_conversion() {
        let data: Vec<u8> = vec![1, 0, 1, 1, 0];

        let array = convert_bool_column(
            data.as_ptr() as *const c_void,
            data.len(),
            vortex::dtype::Nullability::NonNullable,
        )
        .expect("Conversion failed");

        assert_eq!(array.len(), 5);

        // Verify values via scalar
        for (i, expected) in [true, false, true, true, false].iter().enumerate() {
            let scalar = array.scalar_at(i).unwrap();
            assert_eq!(scalar.as_bool().value().unwrap(), *expected);
        }
    }
}

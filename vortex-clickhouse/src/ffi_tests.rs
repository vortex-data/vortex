// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI and integration tests for vortex-clickhouse.
//!
//! These tests verify that the C ABI interface works correctly and that
//! memory management follows the "caller owns" principle.
//!
//! # Test Categories
//!
//! 1. **FFI Boundary Tests**: Verify C ABI function signatures and behavior
//! 2. **Memory Safety Tests**: Ensure proper allocation/deallocation patterns
//! 3. **End-to-End Tests**: Complete read/write workflows through FFI

#[cfg(test)]
mod ffi_tests {
    use std::ffi::CString;
    use std::io::Write;
    use std::ptr;
    use std::sync::Arc;

    use tempfile::NamedTempFile;
    use vortex::array::IntoArray;
    use vortex::array::arrays::{PrimitiveArray, StructArray, VarBinViewArray};
    use vortex::array::stream::ArrayStreamExt;
    use vortex::array::validity::Validity;
    use vortex::buffer::{Buffer, ByteBufferMut};
    use vortex::dtype::FieldNames;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::runtime::BlockingRuntime;

    use crate::scan::{vortex_scanner_free, vortex_scanner_new, vortex_scanner_num_columns};
    use crate::{RUNTIME, SESSION};

    // =========================================================================
    // Test Fixtures and Helpers
    // =========================================================================

    /// Create a test Vortex file with sample data and return the path.
    /// Uses a sync-friendly approach by writing to a temp file.
    fn create_test_vortex_file() -> NamedTempFile {
        // Create sample data
        let id_buffer: Buffer<i64> = vec![1i64, 2, 3, 4, 5].into();
        let id_array = PrimitiveArray::new(id_buffer, Validity::NonNullable).into_array();

        let value_buffer: Buffer<f64> = vec![1.1, 2.2, 3.3, 4.4, 5.5].into();
        let value_array = PrimitiveArray::new(value_buffer, Validity::NonNullable).into_array();

        let name_array =
            VarBinViewArray::from_iter_str(vec!["alice", "bob", "carol", "dave", "eve"])
                .into_array();

        let field_names: Vec<Arc<str>> =
            vec![Arc::from("id"), Arc::from("value"), Arc::from("name")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![id_array, value_array, name_array],
            5,
            Validity::NonNullable,
        )
        .expect("Failed to create struct array");

        // Write to in-memory buffer first
        let mut buf = ByteBufferMut::empty();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write test file");
        });

        // Write buffer to temp file
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(buf.as_ref())
            .expect("Failed to write to temp file");
        temp_file.flush().expect("Failed to flush temp file");

        temp_file
    }

    /// Create a test file with nullable columns.
    fn create_nullable_test_file() -> NamedTempFile {
        // Create data with nulls
        let values: Vec<Option<i64>> = vec![Some(1), None, Some(3), None, Some(5)];
        let id_array = PrimitiveArray::from_option_iter(values).into_array();

        let str_values: Vec<Option<&str>> = vec![Some("a"), None, Some("c"), Some("d"), None];
        let name_array = VarBinViewArray::from_iter_nullable_str(str_values).into_array();

        let field_names: Vec<Arc<str>> = vec![Arc::from("id"), Arc::from("name")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![id_array, name_array],
            5,
            Validity::NonNullable,
        )
        .expect("Failed to create struct array");

        let mut buf = ByteBufferMut::empty();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write test file");
        });

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(buf.as_ref())
            .expect("Failed to write to temp file");
        temp_file.flush().expect("Failed to flush temp file");

        temp_file
    }

    /// Create a large test file for performance testing.
    fn create_large_test_file(num_rows: usize) -> NamedTempFile {
        let ids: Vec<i64> = (0..num_rows as i64).collect();
        let id_buffer: Buffer<i64> = ids.into();
        let id_array = PrimitiveArray::new(id_buffer, Validity::NonNullable).into_array();

        let values: Vec<f64> = (0..num_rows).map(|i| i as f64 * 1.5).collect();
        let value_buffer: Buffer<f64> = values.into();
        let value_array = PrimitiveArray::new(value_buffer, Validity::NonNullable).into_array();

        let field_names: Vec<Arc<str>> = vec![Arc::from("id"), Arc::from("value")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![id_array, value_array],
            num_rows,
            Validity::NonNullable,
        )
        .expect("Failed to create struct array");

        let mut buf = ByteBufferMut::empty();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write test file");
        });

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(buf.as_ref())
            .expect("Failed to write to temp file");
        temp_file.flush().expect("Failed to flush temp file");

        temp_file
    }

    // =========================================================================
    // Scanner FFI Tests
    // =========================================================================

    #[test]
    fn test_scanner_new_valid_file() {
        let temp_file = create_test_vortex_file();
        let path = temp_file.path().to_string_lossy().to_string();

        let c_path = CString::new(path).expect("Failed to create CString");
        let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };

        assert!(
            !scanner.is_null(),
            "Scanner should not be null for valid file"
        );

        // Verify column count
        let num_cols = unsafe { vortex_scanner_num_columns(scanner) };
        assert_eq!(num_cols, 3, "Should have 3 columns");

        // Clean up
        unsafe { vortex_scanner_free(scanner) };
    }

    #[test]
    fn test_scanner_new_invalid_path() {
        let c_path =
            CString::new("/nonexistent/path/file.vortex").expect("Failed to create CString");
        let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };

        assert!(scanner.is_null(), "Scanner should be null for invalid path");
    }

    #[test]
    fn test_scanner_new_empty_path() {
        let c_path = CString::new("").expect("Failed to create CString");
        let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };

        assert!(scanner.is_null(), "Scanner should be null for empty path");
    }

    #[test]
    fn test_scanner_free_null() {
        // Should not crash when freeing null pointer
        unsafe { vortex_scanner_free(ptr::null_mut()) };
    }

    #[test]
    fn test_scanner_num_columns_null() {
        let num_cols = unsafe { vortex_scanner_num_columns(ptr::null()) };
        assert_eq!(num_cols, 0, "Should return 0 for null scanner");
    }

    // =========================================================================
    // Memory Management Tests
    // =========================================================================

    #[test]
    fn test_scanner_memory_lifecycle() {
        // Create multiple files
        for _ in 0..5 {
            let temp_file = create_test_vortex_file();
            let path = temp_file.path().to_string_lossy().to_string();
            let c_path = CString::new(path).expect("Failed to create CString");

            let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };
            assert!(!scanner.is_null());

            // Use scanner
            let _ = unsafe { vortex_scanner_num_columns(scanner) };

            // Free scanner
            unsafe { vortex_scanner_free(scanner) };
        }
        // No memory leaks should occur (would be caught by ASAN/MSAN in CI)
    }

    #[test]
    fn test_scanner_stress_allocation() {
        let temp_file = create_test_vortex_file();
        let path = temp_file.path().to_string_lossy().to_string();

        for _ in 0..100 {
            let c_path = CString::new(path.clone()).expect("Failed to create CString");
            let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };
            assert!(!scanner.is_null());
            unsafe { vortex_scanner_free(scanner) };
        }
    }

    // =========================================================================
    // Nullable Data Tests
    // =========================================================================

    #[test]
    fn test_scanner_with_nullable_data() {
        let temp_file = create_nullable_test_file();
        let path = temp_file.path().to_string_lossy().to_string();

        let c_path = CString::new(path).expect("Failed to create CString");
        let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };

        assert!(!scanner.is_null(), "Scanner should handle nullable data");

        let num_cols = unsafe { vortex_scanner_num_columns(scanner) };
        assert_eq!(num_cols, 2);

        unsafe { vortex_scanner_free(scanner) };
    }

    // =========================================================================
    // Large Data Tests
    // =========================================================================

    #[test]
    fn test_scanner_large_file() {
        let temp_file = create_large_test_file(100_000);
        let path = temp_file.path().to_string_lossy().to_string();

        let c_path = CString::new(path).expect("Failed to create CString");
        let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };

        assert!(!scanner.is_null());

        let num_cols = unsafe { vortex_scanner_num_columns(scanner) };
        assert_eq!(num_cols, 2);

        unsafe { vortex_scanner_free(scanner) };
    }

    // =========================================================================
    // Schema Access Tests
    // =========================================================================

    #[test]
    fn test_scanner_schema_struct() {
        let temp_file = create_test_vortex_file();
        let path = temp_file.path().to_string_lossy().to_string();

        let c_path = CString::new(path).expect("Failed to create CString");
        let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };

        assert!(!scanner.is_null());

        // Schema should be a struct with 3 fields
        let num_cols = unsafe { vortex_scanner_num_columns(scanner) };
        assert_eq!(num_cols, 3);

        unsafe { vortex_scanner_free(scanner) };
    }
}

// =========================================================================
// Exporter FFI Tests
// =========================================================================

#[cfg(test)]
mod exporter_ffi_tests {
    use std::sync::Arc;

    use vortex::array::IntoArray;
    use vortex::array::arrays::{PrimitiveArray, StructArray, VarBinViewArray};
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::dtype::FieldNames;

    use crate::exporter::{
        ColumnExporter, PrimitiveExporter, StructExporter, VarBinViewExporter, new_exporter,
    };

    #[test]
    fn test_exporter_factory_primitive_i32() {
        let buffer: Buffer<i32> = vec![1i32, 2, 3, 4, 5].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let exporter = new_exporter(array).expect("Should create exporter");
        assert!(exporter.has_more());
    }

    #[test]
    fn test_exporter_factory_primitive_i64() {
        let buffer: Buffer<i64> = vec![1i64, 2, 3, 4, 5].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let exporter = new_exporter(array).expect("Should create exporter");
        assert!(exporter.has_more());
    }

    #[test]
    fn test_exporter_factory_primitive_f64() {
        let buffer: Buffer<f64> = vec![1.1f64, 2.2, 3.3].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let exporter = new_exporter(array).expect("Should create exporter");
        assert!(exporter.has_more());
    }

    #[test]
    fn test_exporter_factory_string() {
        let array = VarBinViewArray::from_iter_str(vec!["hello", "world", "test"]).into_array();

        let exporter = new_exporter(array).expect("Should create exporter");
        assert!(exporter.has_more());
    }

    #[test]
    fn test_exporter_factory_struct() {
        let id_buffer: Buffer<i64> = vec![1i64, 2, 3].into();
        let id_array = PrimitiveArray::new(id_buffer, Validity::NonNullable).into_array();

        let name_array = VarBinViewArray::from_iter_str(vec!["a", "b", "c"]).into_array();

        let field_names: Vec<Arc<str>> = vec![Arc::from("id"), Arc::from("name")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![id_array, name_array],
            3,
            Validity::NonNullable,
        )
        .expect("Failed to create struct");

        let exporter = new_exporter(struct_array.into_array()).expect("Should create exporter");
        assert!(exporter.has_more());
    }

    // -------------------------------------------------------------------------
    // Primitive Exporter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_primitive_export_all_at_once() {
        let buffer: Buffer<i32> = vec![10i32, 20, 30, 40, 50].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut exporter = PrimitiveExporter::new(array).expect("Failed to create exporter");

        let mut output = vec![0i32; 5];
        let exported = exporter
            .export(
                output.as_mut_ptr() as *mut std::ffi::c_void,
                size_of_val(output.as_slice()),
                5,
            )
            .expect("Export failed");

        assert_eq!(exported, 5);
        assert_eq!(output, vec![10, 20, 30, 40, 50]);
        assert!(!exporter.has_more());
    }

    #[test]
    fn test_primitive_export_in_chunks() {
        let buffer: Buffer<i64> = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut exporter = PrimitiveExporter::new(array).expect("Failed to create exporter");

        // Export in chunks of 3
        let mut total_exported = 0;
        let mut all_values = Vec::new();

        while exporter.has_more() {
            let mut chunk = vec![0i64; 3];
            let exported = exporter
                .export(
                    chunk.as_mut_ptr() as *mut std::ffi::c_void,
                    size_of_val(chunk.as_slice()),
                    3,
                )
                .expect("Export failed");
            total_exported += exported;
            all_values.extend_from_slice(&chunk[..exported]);
        }

        assert_eq!(total_exported, 10);
        assert_eq!(all_values, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_primitive_export_f32() {
        let buffer: Buffer<f32> = vec![1.5f32, 2.5, 3.5].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut exporter = PrimitiveExporter::new(array).expect("Failed to create exporter");

        let mut output = vec![0.0f32; 3];
        let exported = exporter
            .export(
                output.as_mut_ptr() as *mut std::ffi::c_void,
                size_of_val(output.as_slice()),
                3,
            )
            .expect("Export failed");

        assert_eq!(exported, 3);
        assert_eq!(output, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_primitive_export_empty() {
        let buffer: Buffer<i32> = vec![].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let exporter = PrimitiveExporter::new(array).expect("Failed to create exporter");
        assert!(!exporter.has_more());
    }

    #[test]
    fn test_primitive_export_null_ptr_error() {
        let buffer: Buffer<i32> = vec![1i32, 2, 3].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut exporter = PrimitiveExporter::new(array).expect("Failed to create exporter");

        let result = exporter.export(std::ptr::null_mut(), 0, 3);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Struct Exporter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_struct_exporter_field_count() {
        let id_buffer: Buffer<i64> = vec![1i64, 2, 3].into();
        let id_array = PrimitiveArray::new(id_buffer, Validity::NonNullable).into_array();

        let value_buffer: Buffer<f64> = vec![1.1, 2.2, 3.3].into();
        let value_array = PrimitiveArray::new(value_buffer, Validity::NonNullable).into_array();

        let name_array = VarBinViewArray::from_iter_str(vec!["a", "b", "c"]).into_array();

        let field_names: Vec<Arc<str>> =
            vec![Arc::from("id"), Arc::from("value"), Arc::from("name")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![id_array, value_array, name_array],
            3,
            Validity::NonNullable,
        )
        .expect("Failed to create struct");

        let exporter = StructExporter::new(struct_array.into_array())
            .expect("Failed to create struct exporter");

        assert_eq!(exporter.num_fields(), 3);
    }

    // -------------------------------------------------------------------------
    // VarBinView Exporter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_varbinview_exporter_creation() {
        let array = VarBinViewArray::from_iter_str(vec!["hello", "world"]).into_array();
        let exporter = VarBinViewExporter::new(array).expect("Should create exporter");
        assert!(exporter.has_more());
    }

    #[test]
    fn test_varbinview_exporter_empty() {
        let array = VarBinViewArray::from_iter_str(Vec::<&str>::new()).into_array();
        let exporter = VarBinViewExporter::new(array).expect("Should create exporter");
        assert!(!exporter.has_more());
    }
}

// =========================================================================
// Column Conversion Tests
// =========================================================================

#[cfg(test)]
mod column_conversion_tests {
    use std::ffi::c_void;

    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::array::{Array, IntoArray, ToCanonical};
    use vortex::buffer::Buffer;

    use crate::convert::column::{
        VortexColumnBuilder, clickhouse_column_to_vortex, vortex_to_clickhouse_column,
    };

    // -------------------------------------------------------------------------
    // ClickHouse -> Vortex Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clickhouse_to_vortex_i32() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let array =
            clickhouse_column_to_vortex(data.as_ptr() as *const c_void, data.len(), "Int32")
                .expect("Conversion failed");

        assert_eq!(array.len(), 5);
        let primitive = array.to_primitive();
        let values = primitive.as_slice::<i32>();
        assert_eq!(values, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_clickhouse_to_vortex_i64() {
        let data: Vec<i64> = vec![100, 200, 300];
        let array =
            clickhouse_column_to_vortex(data.as_ptr() as *const c_void, data.len(), "Int64")
                .expect("Conversion failed");

        assert_eq!(array.len(), 3);
        let primitive = array.to_primitive();
        let values = primitive.as_slice::<i64>();
        assert_eq!(values, &[100, 200, 300]);
    }

    #[test]
    fn test_clickhouse_to_vortex_f64() {
        let data: Vec<f64> = vec![1.1, 2.2, 3.3];
        let array =
            clickhouse_column_to_vortex(data.as_ptr() as *const c_void, data.len(), "Float64")
                .expect("Conversion failed");

        assert_eq!(array.len(), 3);
        let primitive = array.to_primitive();
        let values = primitive.as_slice::<f64>();
        assert_eq!(values, &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_clickhouse_to_vortex_bool() {
        let data: Vec<u8> = vec![1, 0, 1, 1, 0];
        let array = clickhouse_column_to_vortex(data.as_ptr() as *const c_void, data.len(), "Bool")
            .expect("Conversion failed");

        assert_eq!(array.len(), 5);
        for (i, expected) in [true, false, true, true, false].iter().enumerate() {
            let scalar = array.scalar_at(i).unwrap();
            assert_eq!(scalar.as_bool().value().unwrap(), *expected);
        }
    }

    #[test]
    fn test_clickhouse_to_vortex_empty() {
        let data: Vec<i32> = vec![];
        let array = clickhouse_column_to_vortex(data.as_ptr() as *const c_void, 0, "Int32")
            .expect("Conversion failed");

        assert_eq!(array.len(), 0);
    }

    #[test]
    fn test_clickhouse_to_vortex_null_ptr_error() {
        let result = clickhouse_column_to_vortex(std::ptr::null(), 5, "Int32");
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Vortex -> ClickHouse Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vortex_to_clickhouse_i32() {
        let buffer: Buffer<i32> = vec![10, 20, 30].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut output = vec![0i32; 3];
        vortex_to_clickhouse_column(&array, output.as_mut_ptr() as *mut c_void)
            .expect("Conversion failed");

        assert_eq!(output, vec![10, 20, 30]);
    }

    #[test]
    fn test_vortex_to_clickhouse_f64() {
        let buffer: Buffer<f64> = vec![1.5, 2.5, 3.5].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut output = vec![0.0f64; 3];
        vortex_to_clickhouse_column(&array, output.as_mut_ptr() as *mut c_void)
            .expect("Conversion failed");

        assert_eq!(output, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_vortex_to_clickhouse_null_ptr_error() {
        let buffer: Buffer<i32> = vec![1, 2, 3].into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let result = vortex_to_clickhouse_column(&array, std::ptr::null_mut());
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Column Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_i64() {
        let mut builder = VortexColumnBuilder::new("Nullable(Int64)", 5).unwrap();

        builder.append_i64(10);
        builder.append_i64(20);
        builder.append_null();
        builder.append_i64(40);
        builder.append_i64(50);

        let array = builder.finish().unwrap();
        assert_eq!(array.len(), 5);

        // Check non-null values
        let scalar = array.scalar_at(0).unwrap();
        assert!(!scalar.is_null());

        let scalar = array.scalar_at(2).unwrap();
        assert!(scalar.is_null());
    }

    #[test]
    fn test_builder_f64() {
        let mut builder = VortexColumnBuilder::new("Float64", 3).unwrap();

        builder.append_f64(1.1);
        builder.append_f64(2.2);
        builder.append_f64(3.3);

        let array = builder.finish().unwrap();
        assert_eq!(array.len(), 3);
    }

    #[test]
    fn test_builder_string() {
        let mut builder = VortexColumnBuilder::new("String", 4).unwrap();

        builder.append_string("hello");
        builder.append_null();
        builder.append_string("world");
        builder.append_string("!");

        let array = builder.finish().unwrap();
        assert_eq!(array.len(), 4);

        let scalar = array.scalar_at(0).unwrap();
        assert!(!scalar.is_null());

        let scalar = array.scalar_at(1).unwrap();
        assert!(scalar.is_null());
    }

    #[test]
    fn test_builder_nullable_int() {
        let mut builder = VortexColumnBuilder::new("Nullable(Int32)", 3).unwrap();

        // Nullable(Int32) should be I32 with nullable flag
        builder.append_null();

        let array = builder.finish().unwrap();
        assert_eq!(array.len(), 1);
    }

    #[test]
    fn test_builder_empty() {
        let builder = VortexColumnBuilder::new("Int64", 0).unwrap();
        let array = builder.finish().unwrap();
        assert_eq!(array.len(), 0);
    }
}

// =========================================================================
// End-to-End Integration Tests
// =========================================================================

#[cfg(test)]
mod e2e_integration_tests {
    use std::ffi::CString;
    use std::io::Write;
    use std::sync::Arc;

    use tempfile::NamedTempFile;
    use vortex::array::IntoArray;
    use vortex::array::arrays::{PrimitiveArray, StructArray};
    use vortex::array::stream::ArrayStreamExt;
    use vortex::array::validity::Validity;
    use vortex::buffer::{Buffer, ByteBufferMut};
    use vortex::dtype::{DType, FieldNames};
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::runtime::BlockingRuntime;

    use crate::convert::dtype::vortex_to_clickhouse_type;
    use crate::exporter::{ColumnExporter, PrimitiveExporter};
    use crate::scan::{
        VortexScanner, vortex_scanner_free, vortex_scanner_new, vortex_scanner_num_columns,
    };
    use crate::{RUNTIME, SESSION};

    fn create_simple_test_file() -> NamedTempFile {
        let original_data: Vec<i64> = vec![100, 200, 300, 400, 500];
        let id_buffer: Buffer<i64> = original_data.clone().into();
        let id_array = PrimitiveArray::new(id_buffer, Validity::NonNullable).into_array();

        let field_names: Vec<Arc<str>> = vec![Arc::from("id")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![id_array],
            5,
            Validity::NonNullable,
        )
        .expect("Failed to create struct");

        let mut buf = ByteBufferMut::empty();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write");
        });

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(buf.as_ref())
            .expect("Failed to write to temp file");
        temp_file.flush().expect("Failed to flush temp file");

        temp_file
    }

    /// Test complete read workflow: Create file -> Open scanner -> Export data
    #[test]
    fn test_complete_read_workflow() {
        let temp_file = create_simple_test_file();
        let path_str = temp_file.path().to_string_lossy().to_string();

        // Open scanner via FFI
        let c_path = CString::new(path_str.clone()).expect("CString failed");
        let scanner = unsafe { vortex_scanner_new(c_path.as_ptr()) };
        assert!(!scanner.is_null());

        // Verify schema
        let num_cols = unsafe { vortex_scanner_num_columns(scanner) };
        assert_eq!(num_cols, 1);

        // Read back via Rust API
        let scanner_obj = VortexScanner::new(&path_str).expect("Failed to create scanner");
        let schema = scanner_obj.schema();
        assert!(matches!(schema, DType::Struct(..)));

        // Clean up
        unsafe { vortex_scanner_free(scanner) };
    }

    /// Test complete write workflow: Build columns -> Write file -> Read back
    #[test]
    fn test_complete_write_workflow() {
        // Simulate ClickHouse column data
        let id_data: Vec<i64> = vec![1, 2, 3, 4, 5];
        let value_data: Vec<f64> = vec![1.1, 2.2, 3.3, 4.4, 5.5];

        // Convert to Vortex arrays
        let id_array = {
            let buffer: Buffer<i64> = id_data.clone().into();
            PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
        };
        let value_array = {
            let buffer: Buffer<f64> = value_data.clone().into();
            PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
        };

        // Create struct and write
        let field_names: Vec<Arc<str>> = vec![Arc::from("id"), Arc::from("value")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![id_array, value_array],
            5,
            Validity::NonNullable,
        )
        .expect("Failed to create struct");

        let mut buf = ByteBufferMut::empty();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write");
        });

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(buf.as_ref())
            .expect("Failed to write to temp file");
        temp_file.flush().expect("Failed to flush temp file");

        // Read back and verify
        let path_str = temp_file.path().to_string_lossy().to_string();
        let scanner = VortexScanner::new(&path_str).expect("Failed to create scanner");
        assert!(matches!(scanner.schema(), DType::Struct(..)));
    }

    /// Test type conversion roundtrip with actual file I/O
    #[test]
    fn test_type_conversion_with_file_io() {
        // Test various primitive types
        let test_cases: Vec<(&str, Box<dyn Fn() -> vortex::array::ArrayRef>)> = vec![
            (
                "Int8",
                Box::new(|| {
                    let buffer: Buffer<i8> = vec![1i8, -1, 127, -128].into();
                    PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
                }),
            ),
            (
                "Int16",
                Box::new(|| {
                    let buffer: Buffer<i16> = vec![1i16, -1, 32767].into();
                    PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
                }),
            ),
            (
                "Int32",
                Box::new(|| {
                    let buffer: Buffer<i32> = vec![1i32, -1, 2147483647].into();
                    PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
                }),
            ),
            (
                "Int64",
                Box::new(|| {
                    let buffer: Buffer<i64> = vec![1i64, -1, 9223372036854775807].into();
                    PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
                }),
            ),
            (
                "UInt8",
                Box::new(|| {
                    let buffer: Buffer<u8> = vec![0u8, 1, 255].into();
                    PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
                }),
            ),
            (
                "UInt64",
                Box::new(|| {
                    let buffer: Buffer<u64> = vec![0u64, 1, 18446744073709551615].into();
                    PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
                }),
            ),
            (
                "Float32",
                Box::new(|| {
                    let buffer: Buffer<f32> = vec![0.0f32, 1.5, -3.14].into();
                    PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
                }),
            ),
            (
                "Float64",
                Box::new(|| {
                    let buffer: Buffer<f64> = vec![0.0f64, 1.5, -3.14159265359].into();
                    PrimitiveArray::new(buffer, Validity::NonNullable).into_array()
                }),
            ),
        ];

        for (type_name, create_array) in test_cases {
            let array = create_array();
            let original_len = array.len();

            // Wrap in struct
            let field_names: Vec<Arc<str>> = vec![Arc::from("col")];
            let struct_array = StructArray::try_new(
                FieldNames::from(field_names),
                vec![array],
                original_len,
                Validity::NonNullable,
            )
            .unwrap();

            // Write to buffer
            let mut buf = ByteBufferMut::empty();
            (*RUNTIME).block_on(async {
                SESSION
                    .write_options()
                    .write(&mut buf, struct_array.to_array_stream())
                    .await
                    .expect(&format!("Failed to write {}", type_name));
            });

            // Write buffer to temp file
            let mut temp_file = NamedTempFile::new()
                .expect(&format!("Failed to create temp file for {}", type_name));
            temp_file
                .write_all(buf.as_ref())
                .expect(&format!("Failed to write to temp file for {}", type_name));
            temp_file.flush().expect("Failed to flush");

            // Read back
            let path_str = temp_file.path().to_string_lossy().to_string();
            let scanner = VortexScanner::new(&path_str)
                .expect(&format!("Failed to create scanner for {}", type_name));

            let schema = scanner.schema();
            if let DType::Struct(fields, _) = schema {
                let field_dtype = fields.field_by_index(0).unwrap();

                // Verify type mapping
                let ch_type = vortex_to_clickhouse_type(&field_dtype)
                    .expect(&format!("Failed to convert dtype for {}", type_name));

                // Should match or be nullable version
                assert!(
                    ch_type == type_name || ch_type == format!("Nullable({})", type_name),
                    "Type mismatch for {}: expected {} or Nullable({}), got {}",
                    type_name,
                    type_name,
                    type_name,
                    ch_type
                );
            }
        }
    }

    /// Test exporter data integrity
    #[test]
    fn test_exporter_data_integrity() {
        let original: Vec<i64> = (0..1000).collect();
        let buffer: Buffer<i64> = original.clone().into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut exporter = PrimitiveExporter::new(array).expect("Failed to create exporter");

        let mut exported: Vec<i64> = Vec::with_capacity(1000);
        while exporter.has_more() {
            let mut chunk = vec![0i64; 100];
            let count = exporter
                .export(
                    chunk.as_mut_ptr() as *mut std::ffi::c_void,
                    size_of_val(chunk.as_slice()),
                    100,
                )
                .expect("Export failed");
            exported.extend_from_slice(&chunk[..count]);
        }

        assert_eq!(original, exported, "Data mismatch after export");
    }
}

// =========================================================================
// Error Handling Tests
// =========================================================================

#[cfg(test)]
mod error_handling_tests {
    use crate::convert::column::VortexColumnBuilder;
    use crate::convert::dtype::clickhouse_type_to_vortex;
    use crate::scan::VortexScanner;

    #[test]
    fn test_scanner_invalid_path() {
        let result = VortexScanner::new("/this/path/does/not/exist.vortex");
        assert!(result.is_err());
    }

    #[test]
    fn test_scanner_invalid_glob() {
        // Invalid glob pattern
        let result = VortexScanner::new("/path/[invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_type_conversion_supported_special_types() {
        // These ClickHouse types are all supported and should parse successfully
        assert!(clickhouse_type_to_vortex("IPv4").is_ok());
        assert!(clickhouse_type_to_vortex("IPv6").is_ok());
        assert!(clickhouse_type_to_vortex("Enum8('a'=1)").is_ok());
        assert!(clickhouse_type_to_vortex("Map(String, Int32)").is_ok());
    }

    #[test]
    fn test_type_conversion_unsupported() {
        // Genuinely unsupported ClickHouse types
        let unsupported = vec![
            "Nothing",
            "SimpleAggregateFunction(sum, Int64)",
            "AggregateFunction(uniq, String)",
        ];

        for type_str in unsupported {
            let result = clickhouse_type_to_vortex(type_str);
            assert!(
                result.is_err(),
                "Expected '{}' to be unsupported, but it succeeded",
                type_str,
            );
        }
    }

    #[test]
    fn test_builder_unsupported_type() {
        // Complex types not yet supported in builder
        let result = VortexColumnBuilder::new("Array(Int32)", 10);
        assert!(result.is_err());

        let result = VortexColumnBuilder::new("Tuple(Int32, String)", 10);
        assert!(result.is_err());
    }
}

// =========================================================================
// Writer FFI Tests
// =========================================================================

#[cfg(test)]
mod writer_ffi_tests {
    use std::ffi::{CString, c_void};
    use std::ptr;

    use tempfile::NamedTempFile;

    use crate::copy::{
        vortex_writer_add_column, vortex_writer_begin_batch, vortex_writer_end_batch,
        vortex_writer_finalize, vortex_writer_free, vortex_writer_new, vortex_writer_num_columns,
        vortex_writer_total_rows, vortex_writer_write_batch, vortex_writer_write_column,
        vortex_writer_write_column_nullable, vortex_writer_write_string_column,
        vortex_writer_write_string_column_nullable,
    };
    use crate::scan::{vortex_scanner_free, vortex_scanner_new, vortex_scanner_num_columns};

    // -------------------------------------------------------------------------
    // Null Pointer Safety
    // -------------------------------------------------------------------------

    #[test]
    fn test_writer_ffi_null_path() {
        let writer = unsafe { vortex_writer_new(ptr::null()) };
        assert!(writer.is_null());
    }

    #[test]
    fn test_writer_ffi_free_null() {
        // Should not panic when freeing null pointer
        unsafe { vortex_writer_free(ptr::null_mut()) };
    }

    #[test]
    fn test_writer_ffi_add_column_null_writer() {
        let name = CString::new("id").unwrap();
        let ch_type = CString::new("Int64").unwrap();
        let result = unsafe {
            vortex_writer_add_column(ptr::null_mut(), name.as_ptr(), ch_type.as_ptr(), 0)
        };
        assert!(result < 0);
    }

    #[test]
    fn test_writer_ffi_begin_batch_null_writer() {
        let result = unsafe { vortex_writer_begin_batch(ptr::null_mut(), 10) };
        assert!(result < 0);
    }

    #[test]
    fn test_writer_ffi_write_column_null_writer() {
        let data: Vec<i64> = vec![1, 2, 3];
        let result = unsafe {
            vortex_writer_write_column(ptr::null_mut(), 0, data.as_ptr() as *const c_void, 3)
        };
        assert!(result < 0);
    }

    #[test]
    fn test_writer_ffi_write_column_null_data() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path).unwrap();

        unsafe {
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            let name = CString::new("id").unwrap();
            let ch_type = CString::new("Int64").unwrap();
            vortex_writer_add_column(writer, name.as_ptr(), ch_type.as_ptr(), 0);

            vortex_writer_begin_batch(writer, 3);
            let result = vortex_writer_write_column(writer, 0, ptr::null(), 3);
            assert!(result < 0);

            vortex_writer_free(writer);
        }
    }

    #[test]
    fn test_writer_ffi_finalize_null_writer() {
        let result = unsafe { vortex_writer_finalize(ptr::null_mut()) };
        assert!(result < 0);
    }

    #[test]
    fn test_writer_ffi_num_columns_null() {
        let num_cols = unsafe { vortex_writer_num_columns(ptr::null()) };
        assert_eq!(num_cols, 0);
    }

    #[test]
    fn test_writer_ffi_total_rows_null() {
        let rows = unsafe { vortex_writer_total_rows(ptr::null()) };
        assert_eq!(rows, 0);
    }

    // -------------------------------------------------------------------------
    // Write Workflow via FFI
    // -------------------------------------------------------------------------

    #[test]
    fn test_writer_ffi_primitive_write_workflow() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path.clone()).unwrap();

        unsafe {
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            // Add columns
            let id_name = CString::new("id").unwrap();
            let id_type = CString::new("Int64").unwrap();
            assert_eq!(
                vortex_writer_add_column(writer, id_name.as_ptr(), id_type.as_ptr(), 0),
                0
            );

            let val_name = CString::new("value").unwrap();
            let val_type = CString::new("Float64").unwrap();
            assert_eq!(
                vortex_writer_add_column(writer, val_name.as_ptr(), val_type.as_ptr(), 0),
                0
            );

            assert_eq!(vortex_writer_num_columns(writer), 2);

            // Write batch
            assert_eq!(vortex_writer_begin_batch(writer, 3), 0);

            let ids: Vec<i64> = vec![1, 2, 3];
            assert_eq!(
                vortex_writer_write_column(writer, 0, ids.as_ptr() as *const c_void, 3),
                0
            );

            let values: Vec<f64> = vec![1.1, 2.2, 3.3];
            assert_eq!(
                vortex_writer_write_column(writer, 1, values.as_ptr() as *const c_void, 3),
                0
            );

            assert_eq!(vortex_writer_end_batch(writer), 0);
            assert_eq!(vortex_writer_total_rows(writer), 3);

            assert_eq!(vortex_writer_finalize(writer), 0);
            vortex_writer_free(writer);
        }

        assert!(std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_writer_ffi_string_column() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path.clone()).unwrap();

        unsafe {
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            let name = CString::new("name").unwrap();
            let ch_type = CString::new("String").unwrap();
            assert_eq!(
                vortex_writer_add_column(writer, name.as_ptr(), ch_type.as_ptr(), 0),
                0
            );

            assert_eq!(vortex_writer_begin_batch(writer, 3), 0);

            let strings = b"AliceBobCharlie";
            let offsets: Vec<u64> = vec![0, 5, 8, 15];
            assert_eq!(
                vortex_writer_write_string_column(writer, 0, strings.as_ptr(), offsets.as_ptr(), 3,),
                0
            );

            assert_eq!(vortex_writer_end_batch(writer), 0);
            assert_eq!(vortex_writer_finalize(writer), 0);
            vortex_writer_free(writer);
        }

        assert!(std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_writer_ffi_nullable_column() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path.clone()).unwrap();

        unsafe {
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            let name = CString::new("value").unwrap();
            let ch_type = CString::new("Int64").unwrap();
            assert_eq!(
                vortex_writer_add_column(writer, name.as_ptr(), ch_type.as_ptr(), 1),
                0
            );

            assert_eq!(vortex_writer_begin_batch(writer, 4), 0);

            let data: Vec<i64> = vec![10, 0, 30, 0];
            // ClickHouse null_map convention: 0 = valid, 1 = null
            let null_map: Vec<u8> = vec![0, 1, 0, 1];
            assert_eq!(
                vortex_writer_write_column_nullable(
                    writer,
                    0,
                    data.as_ptr() as *const c_void,
                    null_map.as_ptr(),
                    4,
                ),
                0
            );

            assert_eq!(vortex_writer_end_batch(writer), 0);
            assert_eq!(vortex_writer_total_rows(writer), 4);
            assert_eq!(vortex_writer_finalize(writer), 0);
            vortex_writer_free(writer);
        }

        assert!(std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_writer_ffi_nullable_string_column() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path.clone()).unwrap();

        unsafe {
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            let name = CString::new("name").unwrap();
            let ch_type = CString::new("String").unwrap();
            assert_eq!(
                vortex_writer_add_column(writer, name.as_ptr(), ch_type.as_ptr(), 1),
                0
            );

            assert_eq!(vortex_writer_begin_batch(writer, 3), 0);

            let strings = b"AliceCharlie";
            let offsets: Vec<u64> = vec![0, 5, 5, 12]; // "Alice", "", "Charlie"
            let null_map: Vec<u8> = vec![0, 1, 0]; // second is null
            assert_eq!(
                vortex_writer_write_string_column_nullable(
                    writer,
                    0,
                    strings.as_ptr(),
                    offsets.as_ptr(),
                    null_map.as_ptr(),
                    3,
                ),
                0
            );

            assert_eq!(vortex_writer_end_batch(writer), 0);
            assert_eq!(vortex_writer_finalize(writer), 0);
            vortex_writer_free(writer);
        }

        assert!(std::path::Path::new(&path).exists());
    }

    #[test]
    fn test_writer_ffi_write_batch_simplified() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path.clone()).unwrap();

        unsafe {
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            let id_name = CString::new("id").unwrap();
            let id_type = CString::new("Int64").unwrap();
            vortex_writer_add_column(writer, id_name.as_ptr(), id_type.as_ptr(), 0);

            let val_name = CString::new("value").unwrap();
            let val_type = CString::new("Float64").unwrap();
            vortex_writer_add_column(writer, val_name.as_ptr(), val_type.as_ptr(), 0);

            let ids: Vec<i64> = vec![1, 2, 3];
            let values: Vec<f64> = vec![1.1, 2.2, 3.3];
            let column_ptrs: Vec<*const c_void> = vec![
                ids.as_ptr() as *const c_void,
                values.as_ptr() as *const c_void,
            ];

            assert_eq!(
                vortex_writer_write_batch(writer, column_ptrs.as_ptr(), 2, 3),
                0
            );
            assert_eq!(vortex_writer_total_rows(writer), 3);

            assert_eq!(vortex_writer_finalize(writer), 0);
            vortex_writer_free(writer);
        }

        assert!(std::path::Path::new(&path).exists());
    }

    // -------------------------------------------------------------------------
    // Write-then-Read Roundtrip via FFI
    // -------------------------------------------------------------------------

    #[test]
    fn test_writer_ffi_write_then_read_roundtrip() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path).unwrap();

        // Write via FFI
        unsafe {
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            let id_name = CString::new("id").unwrap();
            let id_type = CString::new("Int64").unwrap();
            vortex_writer_add_column(writer, id_name.as_ptr(), id_type.as_ptr(), 0);

            let data: Vec<i64> = vec![100, 200, 300, 400, 500];
            let column_ptrs: Vec<*const c_void> = vec![data.as_ptr() as *const c_void];
            assert_eq!(
                vortex_writer_write_batch(writer, column_ptrs.as_ptr(), 1, 5),
                0
            );

            assert_eq!(vortex_writer_finalize(writer), 0);
            vortex_writer_free(writer);
        }

        // Read back via FFI scanner
        unsafe {
            let scanner = vortex_scanner_new(c_path.as_ptr());
            assert!(
                !scanner.is_null(),
                "Scanner should open file written by writer"
            );

            let num_cols = vortex_scanner_num_columns(scanner);
            assert_eq!(num_cols, 1);

            vortex_scanner_free(scanner);
        }
    }

    #[test]
    fn test_writer_ffi_multiple_batches() {
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path.clone()).unwrap();

        unsafe {
            let writer = vortex_writer_new(c_path.as_ptr());
            assert!(!writer.is_null());

            let name = CString::new("id").unwrap();
            let ch_type = CString::new("Int64").unwrap();
            vortex_writer_add_column(writer, name.as_ptr(), ch_type.as_ptr(), 0);

            // Batch 1
            let data1: Vec<i64> = vec![1, 2, 3];
            let ptrs1: Vec<*const c_void> = vec![data1.as_ptr() as *const c_void];
            assert_eq!(vortex_writer_write_batch(writer, ptrs1.as_ptr(), 1, 3), 0);

            // Batch 2
            let data2: Vec<i64> = vec![4, 5];
            let ptrs2: Vec<*const c_void> = vec![data2.as_ptr() as *const c_void];
            assert_eq!(vortex_writer_write_batch(writer, ptrs2.as_ptr(), 1, 2), 0);

            assert_eq!(vortex_writer_total_rows(writer), 5);
            assert_eq!(vortex_writer_finalize(writer), 0);
            vortex_writer_free(writer);
        }

        assert!(std::path::Path::new(&path).exists());
    }
}

// =========================================================================
// Nullable Export Validity FFI Tests
// =========================================================================

#[cfg(test)]
mod nullable_export_validity_tests {
    use std::ffi::CString;
    use std::io::Write;
    use std::sync::Arc;

    use tempfile::NamedTempFile;
    use vortex::array::IntoArray;
    use vortex::array::arrays::{PrimitiveArray, StructArray, VarBinViewArray};
    use vortex::array::validity::Validity;
    use vortex::buffer::ByteBufferMut;
    use vortex::dtype::FieldNames;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::runtime::BlockingRuntime;

    use crate::scan::{
        vortex_exporter_export, vortex_exporter_export_strings, vortex_exporter_export_validity,
        vortex_exporter_free, vortex_exporter_get_field, vortex_exporter_is_nullable,
        vortex_exporter_len, vortex_exporter_num_fields, vortex_exporter_string_data_size,
        vortex_scanner_free, vortex_scanner_has_more, vortex_scanner_new,
        vortex_scanner_read_batch,
    };
    use crate::{RUNTIME, SESSION};

    /// Create a test file with a nullable Int64 column and a nullable String column.
    fn create_nullable_file() -> NamedTempFile {
        let values: Vec<Option<i64>> = vec![Some(10), None, Some(30), None, Some(50)];
        let int_array = PrimitiveArray::from_option_iter(values).into_array();

        let str_values: Vec<Option<&str>> = vec![Some("a"), None, Some("c"), Some("d"), None];
        let str_array = VarBinViewArray::from_iter_nullable_str(str_values).into_array();

        let field_names: Vec<Arc<str>> = vec![Arc::from("id"), Arc::from("name")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![int_array, str_array],
            5,
            Validity::NonNullable,
        )
        .expect("Failed to create struct array");

        let mut buf = ByteBufferMut::empty();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write test file");
        });

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(buf.as_ref())
            .expect("Failed to write to temp file");
        temp_file.flush().expect("Failed to flush temp file");

        temp_file
    }

    #[test]
    fn test_export_validity_nullable_primitive() {
        let temp_file = create_nullable_file();
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path).unwrap();

        unsafe {
            let scanner = vortex_scanner_new(c_path.as_ptr());
            assert!(!scanner.is_null());

            // Read a batch
            assert_eq!(vortex_scanner_has_more(scanner), 1);
            let batch = vortex_scanner_read_batch(scanner);
            assert!(!batch.is_null());

            // The batch is a struct exporter; get the first field (nullable Int64)
            let num_fields = vortex_exporter_num_fields(batch);
            assert!(num_fields >= 2);

            let int_exporter = vortex_exporter_get_field(batch, 0);
            assert!(!int_exporter.is_null());

            // Verify it is nullable
            let is_nullable = vortex_exporter_is_nullable(int_exporter);
            assert_eq!(is_nullable, 1, "Int64 column should be nullable");

            let len = vortex_exporter_len(int_exporter);
            assert_eq!(len, 5);

            // Export data first (required before export_validity)
            let mut data_buf = vec![0i64; 5];
            let exported = vortex_exporter_export(
                int_exporter,
                data_buf.as_mut_ptr() as *mut std::ffi::c_void,
                size_of_val(data_buf.as_slice()),
                5,
            );
            assert_eq!(exported, 5);

            // Export validity bitmap
            // The Vortex bitmap uses 1-bit per row packed into bytes.
            let bitmap_size = 5_usize.div_ceil(8);
            let mut validity_bitmap = vec![0u8; bitmap_size];
            let validity_rows =
                vortex_exporter_export_validity(int_exporter, validity_bitmap.as_mut_ptr(), 5);
            assert_eq!(validity_rows, 5);

            // Non-null values at index 0, 2, 4; null at index 1, 3
            // In a packed bitmap (LSB first): bit 0=1, bit 1=0, bit 2=1, bit 3=0, bit 4=1
            // = 0b00010101 = 0x15
            assert_eq!(
                validity_bitmap[0], 0b00010101,
                "Validity bitmap should mark indices 1 and 3 as null"
            );

            vortex_exporter_free(int_exporter);
            vortex_exporter_free(batch);
            vortex_scanner_free(scanner);
        }
    }

    #[test]
    fn test_export_validity_nullable_string() {
        let temp_file = create_nullable_file();
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path).unwrap();

        unsafe {
            let scanner = vortex_scanner_new(c_path.as_ptr());
            assert!(!scanner.is_null());

            let batch = vortex_scanner_read_batch(scanner);
            assert!(!batch.is_null());

            // Get the second field (nullable String)
            let str_exporter = vortex_exporter_get_field(batch, 1);
            assert!(!str_exporter.is_null());

            let is_nullable = vortex_exporter_is_nullable(str_exporter);
            assert_eq!(is_nullable, 1, "String column should be nullable");

            let len = vortex_exporter_len(str_exporter);
            assert_eq!(len, 5);

            // For string exporters, we must export strings first to set the
            // last_export_start/last_export_count used by export_validity.
            let mut total_bytes: usize = 0;
            let mut num_rows: usize = 0;
            let size_result = vortex_exporter_string_data_size(
                str_exporter,
                &raw mut total_bytes,
                &raw mut num_rows,
            );
            assert_eq!(size_result, 0);
            assert_eq!(num_rows, 5);

            let mut data_buf = vec![0u8; total_bytes.max(1)];
            let mut lengths_buf = vec![0u32; 5];
            let mut offsets_buf = vec![0u64; 5];
            let exported = vortex_exporter_export_strings(
                str_exporter,
                data_buf.as_mut_ptr(),
                lengths_buf.as_mut_ptr(),
                offsets_buf.as_mut_ptr(),
                5,
            );
            assert_eq!(exported, 5);

            // Now export validity bitmap
            let bitmap_size = 5_usize.div_ceil(8);
            let mut validity_bitmap = vec![0u8; bitmap_size];
            let validity_rows =
                vortex_exporter_export_validity(str_exporter, validity_bitmap.as_mut_ptr(), 5);
            assert_eq!(validity_rows, 5);

            // Non-null at 0, 2, 3; null at 1, 4
            // Packed LSB first: bit 0=1, 1=0, 2=1, 3=1, 4=0 = 0b00001101 = 0x0D
            assert_eq!(
                validity_bitmap[0], 0b00001101,
                "Validity bitmap should mark indices 1 and 4 as null"
            );

            vortex_exporter_free(str_exporter);
            vortex_exporter_free(batch);
            vortex_scanner_free(scanner);
        }
    }

    #[test]
    fn test_export_validity_null_handle() {
        let result = unsafe {
            vortex_exporter_export_validity(std::ptr::null_mut(), std::ptr::null_mut(), 5)
        };
        assert!(result < 0, "Should return error for null handle");
    }

    #[test]
    fn test_export_validity_null_bitmap() {
        let temp_file = create_nullable_file();
        let path = temp_file.path().to_string_lossy().to_string();
        let c_path = CString::new(path).unwrap();

        unsafe {
            let scanner = vortex_scanner_new(c_path.as_ptr());
            let batch = vortex_scanner_read_batch(scanner);
            let int_exporter = vortex_exporter_get_field(batch, 0);

            // Pass null bitmap pointer
            let result = vortex_exporter_export_validity(int_exporter, std::ptr::null_mut(), 5);
            assert!(result < 0, "Should return error for null bitmap pointer");

            vortex_exporter_free(int_exporter);
            vortex_exporter_free(batch);
            vortex_scanner_free(scanner);
        }
    }
}

// =========================================================================
// Performance / Stress Tests
// =========================================================================

#[cfg(test)]
mod performance_tests {
    use std::io::Write;
    use std::sync::Arc;

    use tempfile::NamedTempFile;
    use vortex::array::IntoArray;
    use vortex::array::arrays::{PrimitiveArray, StructArray};
    use vortex::array::stream::ArrayStreamExt;
    use vortex::array::validity::Validity;
    use vortex::buffer::{Buffer, ByteBufferMut};
    use vortex::dtype::FieldNames;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::runtime::BlockingRuntime;

    use crate::exporter::{ColumnExporter, PrimitiveExporter};
    use crate::scan::VortexScanner;
    use crate::{RUNTIME, SESSION};

    #[test]
    fn test_large_column_export() {
        let num_rows = 1_000_000;
        let data: Vec<i64> = (0..num_rows as i64).collect();
        let buffer: Buffer<i64> = data.into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let mut exporter = PrimitiveExporter::new(array).expect("Failed to create exporter");

        let mut total = 0usize;
        let chunk_size = 65536;
        let mut output = vec![0i64; chunk_size];

        while exporter.has_more() {
            let exported = exporter
                .export(
                    output.as_mut_ptr() as *mut std::ffi::c_void,
                    size_of_val(output.as_slice()),
                    chunk_size,
                )
                .expect("Export failed");
            total += exported;
        }

        assert_eq!(total, num_rows);
    }

    #[test]
    fn test_scanner_large_file() {
        let num_rows = 500_000;
        let data: Vec<i64> = (0..num_rows as i64).collect();
        let buffer: Buffer<i64> = data.into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let field_names: Vec<Arc<str>> = vec![Arc::from("id")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![array],
            num_rows,
            Validity::NonNullable,
        )
        .unwrap();

        let mut buf = ByteBufferMut::empty();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write");
        });

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(buf.as_ref())
            .expect("Failed to write to temp file");
        temp_file.flush().expect("Failed to flush");

        let path_str = temp_file.path().to_string_lossy().to_string();

        // Time scanner creation
        let start = std::time::Instant::now();
        let scanner = VortexScanner::new(&path_str).expect("Failed to create scanner");
        let elapsed = start.elapsed();

        // Should be fast (< 1 second for just opening)
        assert!(
            elapsed.as_secs() < 2,
            "Scanner creation too slow: {:?}",
            elapsed
        );

        assert_eq!(scanner.file_paths().len(), 1);
    }

    #[test]
    fn test_repeated_scanner_creation() {
        let data: Vec<i64> = vec![1, 2, 3, 4, 5];
        let buffer: Buffer<i64> = data.into();
        let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

        let field_names: Vec<Arc<str>> = vec![Arc::from("id")];
        let struct_array = StructArray::try_new(
            FieldNames::from(field_names),
            vec![array],
            5,
            Validity::NonNullable,
        )
        .unwrap();

        let mut buf = ByteBufferMut::empty();
        (*RUNTIME).block_on(async {
            SESSION
                .write_options()
                .write(&mut buf, struct_array.to_array_stream())
                .await
                .expect("Failed to write");
        });

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(buf.as_ref())
            .expect("Failed to write to temp file");
        temp_file.flush().expect("Failed to flush");

        let path_str = temp_file.path().to_string_lossy().to_string();

        // Create many scanners repeatedly
        for _ in 0..100 {
            let scanner = VortexScanner::new(&path_str).expect("Failed to create scanner");
            assert_eq!(scanner.file_paths().len(), 1);
        }
    }
}

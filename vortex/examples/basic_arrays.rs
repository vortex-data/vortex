//! Basic Array Creation and Operations
//!
//! This example demonstrates how to create different types of arrays in Vortex
//! and perform basic operations on them.
//!
//! Run with: cargo run --example basic_arrays

use vortex::arrays::{
    BoolArray, ChunkedArray, ListArray, PrimitiveArray, StructArray, VarBinArray,
};
use vortex::buffer::buffer;
use vortex::dtype::{DType, Nullability, PType, StructDType};
use vortex::validity::Validity;
use vortex::{Array, ArrayLen, IntoArray};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vortex Basic Arrays Example ===\n");

    // 1. Create primitive numeric arrays
    println!("1. Primitive Arrays:");
    create_primitive_arrays()?;

    // 2. Create string arrays
    println!("\n2. String Arrays:");
    create_string_arrays()?;

    // 3. Create boolean arrays
    println!("\n3. Boolean Arrays:");
    create_boolean_arrays()?;

    // 4. Create struct arrays (columnar records)
    println!("\n4. Struct Arrays (Columnar Records):");
    create_struct_arrays()?;

    // 5. Create nested arrays (lists)
    println!("\n5. Nested Arrays (Lists):");
    create_list_arrays()?;

    // 6. Create chunked arrays (partitioned data)
    println!("\n6. Chunked Arrays (Partitioned Data):");
    create_chunked_arrays()?;

    // 7. Working with nullable data
    println!("\n7. Nullable Arrays:");
    create_nullable_arrays()?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

fn create_primitive_arrays() -> Result<(), Box<dyn std::error::Error>> {
    // Create integer arrays using the buffer! macro
    let int_array: PrimitiveArray = PrimitiveArray::from(vec![1u32, 2, 3, 4, 5]);
    println!("  Integer array length: {}", int_array.len());
    println!("  DType: {}", int_array.dtype());

    // Create float arrays
    let float_array: PrimitiveArray = PrimitiveArray::from(vec![1.5f64, 2.7, 3.14, 4.0]);
    println!("  Float array length: {}", float_array.len());

    // Create large arrays efficiently
    let large_array: PrimitiveArray = (0..1000).map(|i| i as i64).collect();
    println!(
        "  Large array: {} elements, size in memory: ~{} bytes",
        large_array.len(),
        large_array.len() * 8
    );

    Ok(())
}

fn create_string_arrays() -> Result<(), Box<dyn std::error::Error>> {
    // Create a string array from a vector
    let strings = vec!["hello", "world", "vortex", "arrays"];
    let string_array =
        VarBinArray::from_iter(strings.iter(), DType::Utf8(Nullability::NonNullable));
    println!("  String array: {:?}", string_array);
    println!("  Length: {}", string_array.len());

    // Create binary data array
    let binary_data = vec![vec![1u8, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
    let binary_array =
        VarBinArray::from_iter(binary_data.iter(), DType::Binary(Nullability::NonNullable));
    println!("  Binary array: {:?}", binary_array);

    Ok(())
}

fn create_boolean_arrays() -> Result<(), Box<dyn std::error::Error>> {
    // Create a boolean array
    let bools = vec![true, false, true, true, false];
    let bool_array = BoolArray::from(bools);
    println!("  Boolean array: {:?}", bool_array);
    println!("  Length: {}", bool_array.len());

    Ok(())
}

fn create_struct_arrays() -> Result<(), Box<dyn std::error::Error>> {
    // Create a struct array (like a table with named columns)
    let names = VarBinArray::from_iter(
        ["Alice", "Bob", "Charlie"].iter(),
        DType::Utf8(Nullability::NonNullable),
    );
    let ages: PrimitiveArray = PrimitiveArray::from(vec![25u32, 30, 35]);
    let scores: PrimitiveArray = PrimitiveArray::from(vec![95.5f64, 87.3, 92.1]);

    let struct_array = StructArray::try_new(
        ["name", "age", "score"].into(),
        vec![names.into_array(), ages.into_array(), scores.into_array()],
        3,
        Validity::NonNullable,
    )?;

    println!("  Struct array (3 records with 3 fields):");
    println!("    DType: {}", struct_array.dtype());
    println!("    Fields: {:?}", struct_array.names());
    println!("    Length: {}", struct_array.len());

    // Access individual columns
    let name_column = struct_array.field_by_name("name")?;
    println!("    Name column: {:?}", name_column);

    Ok(())
}

fn create_list_arrays() -> Result<(), Box<dyn std::error::Error>> {
    // Create a list array (array of arrays)
    // Each element is a list of integers

    // Concatenate the lists into a flat array
    let flat_values: PrimitiveArray = PrimitiveArray::from(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9]);

    // Define offsets: where each list starts
    let offsets: PrimitiveArray = PrimitiveArray::from(vec![0i32, 3, 5, 9]);

    let list_array = ListArray::try_new(
        flat_values.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
    )?;

    println!("  List array (3 lists):");
    println!("    DType: {}", list_array.dtype());
    println!("    Length: {} lists", list_array.len());
    println!("    Total elements: {}", list_array.elements().len());

    Ok(())
}

fn create_chunked_arrays() -> Result<(), Box<dyn std::error::Error>> {
    // Chunked arrays are useful for partitioning data or streaming processing
    let chunk1: PrimitiveArray = PrimitiveArray::from(vec![1u64, 2, 3]);
    let chunk2: PrimitiveArray = PrimitiveArray::from(vec![4u64, 5, 6]);
    let chunk3: PrimitiveArray = PrimitiveArray::from(vec![7u64, 8, 9, 10]);

    let chunked = ChunkedArray::try_new(
        vec![
            chunk1.into_array(),
            chunk2.into_array(),
            chunk3.into_array(),
        ],
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )?;

    println!("  Chunked array:");
    println!("    Total length: {}", chunked.len());
    println!("    Number of chunks: {}", chunked.nchunks());
    println!("    DType: {}", chunked.dtype());

    // Access individual chunks
    for (i, chunk) in chunked.chunks().enumerate() {
        println!("    Chunk {}: length = {}", i, chunk.len());
    }

    Ok(())
}

fn create_nullable_arrays() -> Result<(), Box<dyn std::error::Error>> {
    // Create arrays with null values
    let values = vec![1u32, 2, 3, 4, 5];

    // Create a validity mask: true = valid, false = null
    let validity = Validity::from(vec![true, false, true, false, true]);

    let nullable_array = PrimitiveArray::from_vec(values, validity);

    println!("  Nullable array:");
    println!("    Values: {:?}", nullable_array);
    println!("    Length: {}", nullable_array.len());
    println!("    DType: {}", nullable_array.dtype());
    println!("    Note: Indices 1 and 3 are null");

    Ok(())
}

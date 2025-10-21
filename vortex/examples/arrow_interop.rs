//! Arrow Interoperability Example
//!
//! This example demonstrates how to convert between Vortex and Apache Arrow formats,
//! enabling integration with the broader Arrow ecosystem.
//!
//! Run with: cargo run --example arrow_interop

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use vortex::arrays::{PrimitiveArray, StructArray, VarBinArray};
use vortex::dtype::{DType, Nullability};
use vortex::validity::Validity;
use vortex::{Array, ArrayLen, IntoArray, IntoCanonical};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vortex <-> Arrow Interoperability Example ===\n");

    println!("This example demonstrates bidirectional conversion between");
    println!("Vortex and Apache Arrow formats.\n");

    // 1. Create Vortex data and convert to Arrow
    println!("1. Vortex -> Arrow Conversion:");
    vortex_to_arrow_conversion()?;

    // 2. Create Arrow data and convert to Vortex
    println!("\n2. Arrow -> Vortex Conversion:");
    arrow_to_vortex_conversion()?;

    // 3. Round-trip conversion
    println!("\n3. Round-trip Conversion (Vortex -> Arrow -> Vortex):");
    round_trip_conversion()?;

    // 4. Working with Arrow RecordBatch
    println!("\n4. Arrow RecordBatch Integration:");
    record_batch_integration()?;

    println!("\n=== All interoperability examples completed! ===");
    Ok(())
}

fn vortex_to_arrow_conversion() -> Result<(), Box<dyn std::error::Error>> {
    // Create Vortex arrays
    let names = VarBinArray::from_iter(
        ["Alice", "Bob", "Charlie"].iter(),
        DType::Utf8(Nullability::NonNullable),
    );

    let ages: PrimitiveArray = PrimitiveArray::from(vec![30i32, 25, 35]);

    let struct_array = StructArray::try_new(
        ["name", "age"].into(),
        vec![names.into_array(), ages.into_array()],
        3,
        Validity::NonNullable,
    )?;

    println!("   Created Vortex struct array:");
    println!("     Length: {}", struct_array.len());
    println!("     Fields: {:?}", struct_array.names());

    // Convert to Arrow
    let arrow_array = struct_array.into_array().into_canonical()?.into_arrow()?;

    println!("\n   Converted to Arrow:");
    println!("     Arrow type: {:?}", arrow_array.data_type());
    println!("     Arrow length: {}", arrow_array.len());

    Ok(())
}

fn arrow_to_vortex_conversion() -> Result<(), Box<dyn std::error::Error>> {
    // Create Arrow arrays
    let arrow_names = StringArray::from(vec!["Diana", "Eve", "Frank"]);
    let arrow_ages = Int32Array::from(vec![28, 32, 41]);

    println!("   Created Arrow arrays:");
    println!("     Names: {} strings", arrow_names.len());
    println!("     Ages: {} integers", arrow_ages.len());

    // Convert to Vortex
    let vortex_names = vortex::ArrayRef::from_arrow(&arrow_names, false);
    let vortex_ages = vortex::ArrayRef::from_arrow(&arrow_ages, false);

    println!("\n   Converted to Vortex:");
    println!("     Names DType: {}", vortex_names.dtype());
    println!("     Ages DType: {}", vortex_ages.dtype());
    println!("     Names encoding: {}", vortex_names.encoding().id());
    println!("     Ages encoding: {}", vortex_ages.encoding().id());

    Ok(())
}

fn round_trip_conversion() -> Result<(), Box<dyn std::error::Error>> {
    // Start with Vortex
    let original_data: PrimitiveArray = PrimitiveArray::from(vec![1i32, 2, 3, 4, 5]);

    println!("   Original Vortex array:");
    println!("     Length: {}", original_data.len());
    println!("     DType: {}", original_data.dtype());

    // Convert to Arrow
    let arrow_array = original_data
        .clone()
        .into_array()
        .into_canonical()?
        .into_arrow()?;

    println!("\n   After Vortex -> Arrow:");
    println!("     Arrow type: {:?}", arrow_array.data_type());
    println!("     Arrow length: {}", arrow_array.len());

    // Convert back to Vortex
    let vortex_restored = vortex::ArrayRef::from_arrow(arrow_array.as_ref(), false);

    println!("\n   After Arrow -> Vortex:");
    println!("     Length: {}", vortex_restored.len());
    println!("     DType: {}", vortex_restored.dtype());

    // Verify data integrity
    let original_canonical = original_data.into_array().into_canonical()?;
    let original_primitive = original_canonical
        .into_primitive()
        .ok_or("Expected primitive")?;

    let restored_canonical = vortex_restored.into_canonical()?;
    let restored_primitive = restored_canonical
        .into_primitive()
        .ok_or("Expected primitive")?;

    let mut matches = true;
    for i in 0..original_primitive.len() {
        let original_val = original_primitive.get_as::<i32>(i).ok_or("Invalid value")?;
        let restored_val = restored_primitive.get_as::<i32>(i).ok_or("Invalid value")?;
        if original_val != restored_val {
            matches = false;
            break;
        }
    }

    println!(
        "     Data integrity check: {}",
        if matches { "PASSED ✓" } else { "FAILED ✗" }
    );

    Ok(())
}

fn record_batch_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Create an Arrow RecordBatch
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Int32, false),
    ]));

    let id_array = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
    let name_array = Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "Diana"]));
    let score_array = Arc::new(Int32Array::from(vec![95, 87, 92, 88]));

    let record_batch =
        RecordBatch::try_new(schema.clone(), vec![id_array, name_array, score_array])?;

    println!("   Created Arrow RecordBatch:");
    println!("     Schema: {:?}", record_batch.schema());
    println!("     Num rows: {}", record_batch.num_rows());
    println!("     Num columns: {}", record_batch.num_columns());

    // Convert RecordBatch to Vortex StructArray
    let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    let vortex_columns: Vec<vortex::ArrayRef> = record_batch
        .columns()
        .iter()
        .map(|col| vortex::ArrayRef::from_arrow(col.as_ref(), false))
        .collect();

    let vortex_struct = StructArray::try_new(
        column_names.into(),
        vortex_columns,
        record_batch.num_rows(),
        Validity::NonNullable,
    )?;

    println!("\n   Converted to Vortex StructArray:");
    println!("     Length: {}", vortex_struct.len());
    println!("     Fields: {:?}", vortex_struct.names());
    println!("     DType: {}", vortex_struct.dtype());

    // Can now use Vortex operations on this data
    let id_column = vortex_struct.field_by_name("id")?;
    println!("\n   Accessing 'id' column:");
    println!("     Length: {}", id_column.len());
    println!("     Encoding: {}", id_column.encoding().id());

    Ok(())
}

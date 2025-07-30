// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use tokio::fs::File;
use vortex::IntoArray;
use vortex::arrays::{PrimitiveArray, StructArray};
use vortex::buffer::{Buffer, buffer};
use vortex::dtype::FieldNames;
use vortex::error::VortexExpect;
use vortex::file::VortexWriteOptions;
use vortex::validity::Validity;

#[cxx::bridge(namespace = "vortex::ffi::testing")]
mod ffi {
    extern "Rust" {
        fn generate_test_vortex_file(output_path: &str) -> Result<()>;
        fn generate_test_vortex_file_1m(output_path: &str) -> Result<()>;
    }
}

fn write_array_to_file(
    array: StructArray,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let mut file = File::create(&output_path)
            .await
            .vortex_expect("Failed to create test file");

        VortexWriteOptions::default()
            .write(&mut file, array.to_array_stream())
            .await
            .vortex_expect("Failed to write test data to file");
    });
    Ok(())
}

fn generate_test_vortex_file(
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Create a simple test array
    let test_array = PrimitiveArray::new(
        buffer![10i32, 20i32, 30i32, 40i32, 50i32],
        Validity::NonNullable,
    )
    .into_array();

    let struct_array = StructArray::try_new(
        FieldNames::from_iter(vec!["a".to_string(), "b".to_string()]),
        vec![test_array.clone(), test_array.clone()],
        5,
        Validity::NonNullable,
    )
    .vortex_expect("Failed to create test array");
    write_array_to_file(struct_array, output_path)?;
    Ok(())
}

// Generate 1M rows of test data simply for testing multi-threaded read behavior of the threadsafe cloneable reader.
fn generate_test_vortex_file_1m(
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    const NUM_ROWS: usize = 1024 * 1024;
    let mut id_data = Vec::with_capacity(NUM_ROWS);
    let mut value_data = Vec::with_capacity(NUM_ROWS);

    // Create sequential data for validation
    for i in 0..NUM_ROWS {
        id_data.push(i as i64);
        value_data.push(i32::try_from(i * 2)); // Simple pattern for validation
    }

    let id_array =
        PrimitiveArray::new(Buffer::copy_from(&id_data), Validity::NonNullable).into_array();

    let value_array =
        PrimitiveArray::new(Buffer::copy_from(&value_data), Validity::NonNullable).into_array();

    let struct_array = StructArray::try_new(
        FieldNames::from_iter(vec!["id".to_string(), "value".to_string()]),
        vec![id_array, value_array],
        NUM_ROWS,
        Validity::NonNullable,
    )
    .vortex_expect("Failed to create test array");

    write_array_to_file(struct_array, output_path)?;
    Ok(())
}

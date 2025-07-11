// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::path::Path;

use vortex_array::arrays::StructArray;
use vortex_dtype::FieldNames;

fn main() {
    let mut _builder = cxx_build::bridge("src/lib.rs");

    // Generate a simple test Vortex file for testing
    generate_test_vortex_file();

    println!("cargo:rerun-if-changed=src/");
}

fn generate_test_vortex_file() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        use tokio::fs::File;
        use vortex_array::IntoArray;
        use vortex_array::arrays::PrimitiveArray;
        use vortex_array::validity::Validity;
        use vortex_buffer::buffer;
        use vortex_file::VortexWriteOptions;

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
        .unwrap();

        // Write directly to file in the build directory
        let out_dir = env::var("OUT_DIR").unwrap();
        let build_dir = Path::new(&out_dir).join("../../../build");
        std::fs::create_dir_all(&build_dir).unwrap();

        let test_file_path = build_dir.join("test_data.vortex");
        let mut file = File::create(&test_file_path).await.unwrap();

        VortexWriteOptions::default()
            .write(&mut file, struct_array.to_array_stream())
            .await
            .unwrap();

        println!(
            "Generated test Vortex file at: {}",
            test_file_path.display()
        );
    });
}

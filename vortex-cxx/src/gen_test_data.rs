// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod gen_test_data {
    #[cxx::bridge(namespace = "vortex::ffi")]
    mod ffi {
        extern "Rust" {
            fn generate_test_vortex_file(output_path: &str) -> Result<()>;
        }
    }
    #[cfg(feature = "gen_test_data")]
    fn generate_test_vortex_file(
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            use tokio::fs::File;
            use vortex::IntoArray;
            use vortex::arrays::{PrimitiveArray, StructArray};
            use vortex::buffer::buffer;
            use vortex::dtype::FieldNames;
            use vortex::error::VortexExpect;
            use vortex::file::VortexWriteOptions;
            use vortex::validity::Validity;
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

            let mut file = File::create(&output_path)
                .await
                .vortex_expect("Failed to create test file");

            VortexWriteOptions::default()
                .write(&mut file, struct_array.to_array_stream())
                .await
                .vortex_expect("Failed to write test data to file");
        });
        Ok(())
    }
}

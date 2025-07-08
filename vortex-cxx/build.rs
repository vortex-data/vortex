use std::path::Path;
use std::{env, fs};

fn main() {
    // Collect all .cpp files in src/
    let cpp_files = fs::read_dir("src")
        .unwrap_or_else(|_| panic!("Could not read src directory"))
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("cpp"));

    let mut builder = cxx_build::bridge("src/lib.rs");
    // Build with all .cpp files
    cpp_files
        .fold(&mut builder, |builder, cpp_file| builder.file(cpp_file))
        .include("include")
        .std("c++17")
        .compile("vortex-cxx");

    // Generate a simple test Vortex file for testing
    generate_test_vortex_file();

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=include/vortex.hpp");
}

fn generate_test_vortex_file() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        use std::fs::File;
        use std::io::Write;

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

        // Write to a buffer first
        let mut buffer = vortex_buffer::ByteBufferMut::empty();
        VortexWriteOptions::default()
            .write(&mut buffer, test_array.to_array_stream())
            .await
            .unwrap();

        // Write the buffer to a file in the build directory
        let out_dir = env::var("OUT_DIR").unwrap();
        let build_dir = Path::new(&out_dir).join("../../../build");
        std::fs::create_dir_all(&build_dir).unwrap();

        let test_file_path = build_dir.join("test_data.vortex");
        let mut file = File::create(&test_file_path).unwrap();
        file.write_all(&buffer.freeze()).unwrap();

        println!(
            "Generated test Vortex file at: {}",
            test_file_path.display()
        );
    });
}

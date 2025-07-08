use std::env;
use std::path::Path;

fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("src/array.cpp")
        .file("src/dtype.cpp")
        .file("src/utils.cpp")
        .include("include")
        .std("c++17")
        .compile("vortex-cxx");

    // Generate a simple test Vortex file for testing
    generate_test_vortex_file();

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/array.cpp");
    println!("cargo:rerun-if-changed=src/dtype.cpp");
    println!("cargo:rerun-if-changed=src/utils.cpp");
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

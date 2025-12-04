// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test binary for testing the `update_s3_object` function using the AWS CLI.
//!
//! Usage:
//!   cargo run -p vortex-wasm --bin test_s3_update -- --profile <PROFILE_NAME>
//!   cargo run -p vortex-wasm --bin test_s3_update -- --upload --profile <PROFILE_NAME>

#![allow(clippy::expect_used, clippy::exit)]

use std::env;
use std::fs;
use std::process::Command;
use std::sync::Arc;

use vortex::VortexSessionDefault;
use vortex::array::builders::builder_with_capacity;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability::NonNullable;
use vortex::dtype::PType;
use vortex::io::session::RuntimeSessionExt;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;
use vortex_wasm::website::update_s3::update_s3_object;

const BUCKET: &str = "vortex-benchmark-results-database";
const KEY: &str = "test/random_access.vortex";

fn main() {
    let session = VortexSession::default().with_tokio();
    let args: Vec<String> = env::args().collect();

    // Parse --profile argument.
    let profile = args
        .iter()
        .position(|a| a == "--profile")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);

    if profile.is_none() {
        eprintln!("Warning: No --profile specified. AWS CLI will use default credentials.");
        eprintln!("Usage: test_s3_update [--upload] --profile <PROFILE_NAME>");
    }

    // Check for --upload flag.
    if args.iter().any(|a| a == "--upload") {
        println!("Uploading random_access.vortex to S3...");
        let local_path = "/Users/connor/spiral/vortex-data/vortex/vortex-wasm/random_access.vortex";
        let file_bytes = fs::read(local_path).expect("Failed to read local file");
        let size = file_bytes.len();

        let mut cmd = Command::new("aws");
        cmd.args(["s3", "cp", local_path, &format!("s3://{}/{}", BUCKET, KEY)]);
        if let Some(p) = profile {
            cmd.args(["--profile", p]);
        }

        let status = cmd.status().expect("Failed to run aws CLI");

        if !status.success() {
            eprintln!("Failed to upload to S3");
            std::process::exit(1);
        }

        println!("Uploaded {} bytes to s3://{}/{}", size, BUCKET, KEY);
    }

    // Single update test.
    println!("\nTesting update_s3_object...");

    let result = update_s3_object(&session, BUCKET, KEY, profile, |existing_array| {
        let existing_len = existing_array.len();
        println!("  Existing array has {} entries", existing_len);

        // Create a new entry to append.
        let new_entry = create_test_entry();

        // Build a new array with existing data + new entry.
        let dtype = existing_array.dtype().clone();
        let mut builder = builder_with_capacity(&dtype, existing_len + 1);
        builder.extend_from_array(&existing_array);
        builder.append_scalar(&new_entry)?;

        let result = builder.finish();
        println!("  New array has {} entries", result.len());

        Ok(result)
    });

    match result {
        Ok(()) => {
            println!("update_s3_object succeeded!");
        }
        Err(e) => {
            println!("update_s3_object failed: {}", e);
        }
    }

    println!("Done!");
}

/// Creates a test entry matching the BenchmarkEntry schema.
fn create_test_entry() -> Scalar {
    let u8_dtype = DType::Primitive(PType::U8, NonNullable);

    // Build the dtype to match the schema:
    // {commit_id=fixed_size_list(u8)[20], group_name=u32, chart_name=u32, series_name=u32, value=u64}
    let dtype = DType::Struct(
        vortex::dtype::StructFields::new(
            FieldNames::from([
                "commit_id",
                "group_name",
                "chart_name",
                "series_name",
                "value",
            ]),
            vec![
                DType::FixedSizeList(Arc::new(u8_dtype.clone()), 20, NonNullable),
                DType::Primitive(PType::U32, NonNullable),
                DType::Primitive(PType::U32, NonNullable),
                DType::Primitive(PType::U32, NonNullable),
                DType::Primitive(PType::U64, NonNullable),
            ],
        ),
        NonNullable,
    );

    // Create a test commit_id (20 bytes of 'x').
    let commit_id_bytes: Vec<Scalar> = b"xxxxxxxxxxxxxxxxxxxx"
        .iter()
        .map(|&b| Scalar::primitive(b, NonNullable))
        .collect();
    let commit_id_scalar = Scalar::fixed_size_list(u8_dtype, commit_id_bytes, NonNullable);

    Scalar::struct_(
        dtype,
        vec![
            commit_id_scalar,
            Scalar::primitive(2u32, NonNullable), // group_name: random-access
            Scalar::primitive(2u32, NonNullable), // chart_name: random-access
            Scalar::primitive(3u32, NonNullable), // series_name: vortex-nvme
            Scalar::primitive(999999u64, NonNullable), // value: test value
        ],
    )
}

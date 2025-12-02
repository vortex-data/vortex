// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test binary for uploading a Vortex file to S3 and testing the `update_s3_object` function.

#![allow(clippy::expect_used)]

use std::env;
use std::fs;

use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use bench_vortex::website::update_s3::update_s3_object;
use vortex::VortexSessionDefault;
use vortex::array::Array;
use vortex::array::builders::builder_with_capacity;
use vortex::array::stream::ArrayStreamExt;
use vortex::dtype::FieldNames;
use vortex::file::OpenOptionsSessionExt;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;

const BUCKET: &str = "vortex-benchmark-results-database";
const KEY: &str = "test/random_access.vortex";

fn main() {
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(async_main());
}

async fn async_main() {
    let session = VortexSession::default();

    // Load AWS config with SSO profile.
    let config = aws_config::defaults(BehaviorVersion::latest())
        .profile_name("PowerUserAccess-375504701696")
        .load()
        .await;
    let client = Client::new(&config);

    // Check for --upload flag.
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--upload") {
        println!("Uploading random_access.vortex to S3...");
        let local_path = "/Users/connor/spiral/vortex-data/vortex/vortex-wasm/random_access.vortex";
        let file_bytes = fs::read(local_path).expect("Failed to read local file");
        let size = file_bytes.len();

        client
            .put_object()
            .bucket(BUCKET)
            .key(KEY)
            .body(ByteStream::from(file_bytes))
            .send()
            .await
            .expect("Failed to upload to S3");

        println!("Uploaded {} bytes to s3://{}/{}", size, BUCKET, KEY);
    }

    // Check for --concurrent flag to test atomicity with multiple concurrent updates.
    let concurrent = args.iter().any(|a| a == "--concurrent");
    let num_concurrent: usize = args
        .iter()
        .position(|a| a == "--concurrent")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    if concurrent {
        println!(
            "\nTesting concurrent updates with {} tasks...",
            num_concurrent
        );

        // Get initial count.
        let initial_count = get_entry_count(&client, &session).await;
        println!("Initial entry count: {}", initial_count);

        // Spawn concurrent update tasks.
        let mut handles = Vec::new();
        for i in 0..num_concurrent {
            let client = client.clone();
            let session = session.clone();
            handles.push(tokio::spawn(async move {
                let result = update_s3_object(
                    &client,
                    &session,
                    BUCKET,
                    KEY,
                    |existing_array| async move {
                        let existing_len = existing_array.len();

                        // Create a new entry to append.
                        let new_entry = create_test_entry();

                        // Build a new array with existing data + new entry.
                        let dtype = existing_array.dtype().clone();
                        let mut builder = builder_with_capacity(&dtype, existing_len + 1);
                        builder.extend_from_array(&existing_array);
                        builder.append_scalar(&new_entry)?;

                        Ok(builder.finish())
                    },
                )
                .await;

                match result {
                    Ok(()) => {
                        println!("  Task {} succeeded", i);
                        true
                    }
                    Err(e) => {
                        println!("  Task {} failed: {}", i, e);
                        false
                    }
                }
            }));
        }

        // Wait for all tasks.
        let mut successes = 0;
        let mut failures = 0;
        for handle in handles {
            if handle.await.unwrap_or(false) {
                successes += 1;
            } else {
                failures += 1;
            }
        }

        // Verify final count.
        let final_count = get_entry_count(&client, &session).await;
        println!("\nResults:");
        println!("  Successes: {}", successes);
        println!("  Failures: {}", failures);
        println!("  Initial count: {}", initial_count);
        println!("  Final count: {}", final_count);
        println!("  Expected count: {}", initial_count + successes);

        if final_count == initial_count + successes {
            println!(
                "\n✓ Atomicity verified! All {} successful updates were applied.",
                successes
            );
        } else {
            println!("\n✗ Atomicity FAILED! Count mismatch.");
        }
    } else {
        // Single update test.
        println!("\nTesting update_s3_object...");

        let result = update_s3_object(
            &client,
            &session,
            BUCKET,
            KEY,
            |existing_array| async move {
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
            },
        )
        .await;

        match result {
            Ok(()) => {
                println!("update_s3_object succeeded!");
                let count = get_entry_count(&client, &session).await;
                println!("Verified: updated file has {} entries", count);
            }
            Err(e) => {
                println!("update_s3_object failed: {}", e);
            }
        }
    }

    println!("Done!");
}

async fn get_entry_count(client: &Client, session: &VortexSession) -> usize {
    let get_result = client
        .get_object()
        .bucket(BUCKET)
        .key(KEY)
        .send()
        .await
        .expect("Failed to download file");

    let bytes = get_result
        .body
        .collect()
        .await
        .expect("Failed to read body")
        .into_bytes();

    let file = session
        .open_options()
        .open_buffer(bytes)
        .expect("Failed to open buffer");

    let array = file
        .scan()
        .expect("Failed to scan")
        .into_array_stream()
        .expect("Failed to get stream")
        .read_all()
        .await
        .expect("Failed to read all");

    array.len()
}

/// Creates a test entry matching the BenchmarkEntry schema.
fn create_test_entry() -> Scalar {
    use std::sync::Arc;

    use vortex::dtype::DType;
    use vortex::dtype::Nullability::NonNullable;
    use vortex::dtype::PType;

    let u8_dtype = DType::Primitive(PType::U8, NonNullable);

    // Build the dtype to match the schema:
    // {commit_id=fixed_size_list(u8)[20], benchmark_group=u32, chart_name=u32, series_name=u32, value=u64}
    let dtype = DType::Struct(
        vortex::dtype::StructFields::new(
            FieldNames::from([
                "commit_id",
                "benchmark_group",
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
            Scalar::primitive(2u32, NonNullable), // benchmark_group: random-access
            Scalar::primitive(2u32, NonNullable), // chart_name: random-access
            Scalar::primitive(3u32, NonNullable), // series_name: vortex-nvme
            Scalar::primitive(999999u64, NonNullable), // value: test value
        ],
    )
}

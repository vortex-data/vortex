// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary to migrate a JSON array of [`CommitInfo`] objects to a Vortex file.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p vortex-wasm --bin migrate_commits -- <input.json> <output.vortex>
//! ```

// TODO(connor): We don't use the `TemporalArray` right now because it doesn't have easy interop yet
// for the chrono `DateTime` type, and bringing in arrow for just this is too heavyweight.

#![allow(clippy::expect_used, clippy::panic)]

use std::env;
use std::fs;

use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::compressor::CompactCompressor;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::dtype::Nullability;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::session::VortexSession;
use vortex_wasm::website::commit::CommitInfo;

#[tokio::main]
async fn main() {
    let session = VortexSession::default();

    let args: Vec<String> = env::args().collect();
    let input_path = args
        .get(1)
        .expect("Usage: migrate_commits <input_json> <output_file>");
    let output_path = args
        .get(2)
        .map(String::as_str)
        .expect("Usage: migrate_commits <input_json> <output_file>");

    let contents = fs::read_to_string(input_path).expect("Failed to read file");
    let commits: Vec<CommitInfo> = serde_json::from_str(&contents).expect("Failed to parse JSON");

    let num_commits = commits.len();
    println!("Parsed {num_commits} commits from JSON");

    // Extract fields into columnar vectors.
    let mut timestamps: Vec<i64> = Vec::with_capacity(num_commits);
    let mut author_names: Vec<&str> = Vec::with_capacity(num_commits);
    let mut author_emails: Vec<&str> = Vec::with_capacity(num_commits);
    let mut messages: Vec<&str> = Vec::with_capacity(num_commits);
    let mut commit_id_bytes: Vec<u8> = Vec::with_capacity(num_commits * 20);

    for commit in &commits {
        timestamps.push(commit.timestamp());
        author_names.push(commit.author().name());
        author_emails.push(commit.author().email());
        messages.push(commit.message());
        commit_id_bytes.extend_from_slice(&commit.commit_id().0);
    }

    // Build Vortex arrays.

    // Timestamp array.
    let timestamp_array = PrimitiveArray::new(Buffer::from(timestamps), Validity::NonNullable);

    // Author struct array (nested).
    let author_name_array = VarBinArray::from_iter(
        author_names.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );
    let author_email_array = VarBinArray::from_iter(
        author_emails.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );
    let author_array = StructArray::try_new(
        FieldNames::from(["name", "email"]),
        vec![
            author_name_array.into_array(),
            author_email_array.into_array(),
        ],
        num_commits,
        Validity::NonNullable,
    )
    .expect("Failed to create author struct array");

    // Message array.
    let message_array = VarBinArray::from_iter(
        messages.iter().map(|s| Some(*s)),
        DType::Utf8(Nullability::NonNullable),
    );

    // Commit ID array (FixedSizeList<u8, 20>).
    let commit_id_elements =
        PrimitiveArray::new(Buffer::from(commit_id_bytes), Validity::NonNullable);
    let commit_id_array = FixedSizeListArray::try_new(
        commit_id_elements.into_array(),
        20,
        Validity::NonNullable,
        num_commits,
    )
    .expect("Failed to create commit_id array");

    // Outer struct array with field order: timestamp, author, message, commit_id.
    let struct_array = StructArray::try_new(
        FieldNames::from(["timestamp", "author", "message", "commit_id"]),
        vec![
            timestamp_array.into_array(),
            author_array.into_array(),
            message_array.into_array(),
            commit_id_array.into_array(),
        ],
        num_commits,
        Validity::NonNullable,
    )
    .expect("Failed to create struct array");

    println!("Schema: {}", struct_array.dtype());

    // Write to Vortex file with compression.
    let file = tokio::fs::File::create(output_path)
        .await
        .expect("Failed to create output file");

    session
        .write_options()
        .with_strategy(
            WriteStrategyBuilder::new()
                .with_compressor(CompactCompressor::default())
                .build(),
        )
        .write(file, struct_array.to_array_stream())
        .await
        .expect("Failed to write Vortex file");

    println!("Wrote {num_commits} commits to {output_path}");
}

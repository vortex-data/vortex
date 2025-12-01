// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Utilities for the Vortex benchmark website.

use std::path::Path;

use vortex_array::builders::builder_with_capacity;
use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_metrics::VortexMetrics;
use vortex_scalar::Scalar;
use vortex_session::VortexSession;

/// Reads a Vortex file and appends a single scalar entry, writing the result back.
///
/// This function:
/// 1. Reads the existing Vortex file (using the scalar's dtype)
/// 2. Appends the new scalar to the existing data using a builder
/// 3. Writes the combined data back to the output path
///
/// The input and output paths can be the same to overwrite the existing file.
///
/// # Arguments
///
/// * `input_path` - Path to the existing Vortex file to read.
/// * `output_path` - Path to write the updated Vortex file (can be same as input).
/// * `new_entry` - The scalar to append. Its dtype is used for reading/writing the file.
///
/// # Returns
///
/// The total number of entries in the resulting file.
pub fn append_entry(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    new_entry: Scalar,
) -> VortexResult<usize> {
    let runtime = CurrentThreadRuntime::new();

    let session = VortexSession::empty()
        .with::<VortexMetrics>()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
        .with_handle(runtime.handle());

    vortex_file::register_default_encodings(&session);

    runtime.block_on(naive_append_entry_async(
        &session,
        input_path.as_ref(),
        output_path.as_ref(),
        new_entry,
    ))
}

/// SUPER NAIVE append to a Vortex file.
async fn naive_append_entry_async(
    session: &VortexSession,
    input_path: &Path,
    output_path: &Path,
    new_entry: Scalar,
) -> VortexResult<usize> {
    let dtype = new_entry.dtype().clone();

    // Read the existing file.
    let file = session
        .open_options()
        .with_dtype(dtype.clone())
        .open(input_path)
        .await?;

    // Read all existing data.
    let existing_array = file.scan()?.into_array_stream()?.read_all().await?;
    let existing_len = existing_array.len();

    // Create a builder and extend with existing data, then append the new entry.
    let total_capacity = existing_len + 1;
    let mut builder = builder_with_capacity(&dtype, total_capacity);

    // Add existing data.
    builder.extend_from_array(&existing_array);

    // Append the new entry.
    builder.append_scalar(&new_entry)?;

    let combined = builder.finish();
    let total_len = combined.len();

    // Write to output file.
    let file = async_fs::File::create(output_path).await?;

    let mut writer = session.write_options().writer(file, dtype);

    writer.push(combined).await?;
    writer.finish().await?;

    Ok(total_len)
}

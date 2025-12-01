// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Update operations for Vortex files.
//!
//! This module provides functions to read a Vortex file, apply a transformation, and write the
//! result back to a file. The update operation uses atomic file replacement for safety.

use std::future::Future;
use std::path::Path;

use vortex_array::ArrayRef;
use vortex_array::expr::session::ExprSession;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::session::LayoutSession;
use vortex_metrics::VortexMetrics;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::WriteOptionsSessionExt;
use crate::WriteSummary;
use crate::register_default_encodings;

/// Updates a Vortex file by reading it, applying a transformation, and writing the result.
///
/// This is a blocking convenience wrapper around [`update_file_async`]. It creates a new session
/// with default encodings and a current-thread runtime.
///
/// # Arguments
///
/// * `input_path` - Path to the existing Vortex file to read.
/// * `output_path` - Path to write the updated Vortex file. Can be the same as input.
/// * `update_fn` - An async function that takes the file's array data and returns the updated
///   array. The returned array must have the same dtype as the input.
///
/// # Returns
///
/// A [`WriteSummary`] containing information about the written file.
///
/// # Errors
///
/// Returns an error if:
/// - The input file cannot be read.
/// - The update function returns an error.
/// - The update function returns an array with a different dtype.
/// - The output file cannot be written.
///
/// # Atomic Write Guarantee
///
/// The write operation uses a temporary file and atomic rename to ensure that the output file is
/// never left in a corrupted state, even if the process crashes during the write.
pub fn update_file<F, Fut>(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    update_fn: F,
) -> VortexResult<WriteSummary>
where
    F: FnOnce(ArrayRef) -> Fut,
    Fut: Future<Output = VortexResult<ArrayRef>>,
{
    let runtime = CurrentThreadRuntime::new();

    let session = VortexSession::empty()
        .with::<VortexMetrics>()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ExprSession>()
        .with::<RuntimeSession>()
        .with_handle(runtime.handle());

    register_default_encodings(&session);

    runtime.block_on(update_file_async(
        &session,
        input_path.as_ref(),
        output_path.as_ref(),
        update_fn,
    ))
}

/// Updates a Vortex file asynchronously by reading it, applying a transformation, and writing the
/// result.
///
/// This function:
/// 1. Reads the existing Vortex file into memory.
/// 2. Calls the update function with the array data.
/// 3. Validates the returned array has the same dtype.
/// 4. Writes the updated data to a temporary file.
/// 5. Atomically renames the temporary file to the output path.
///
/// # Arguments
///
/// * `session` - The Vortex session to use for reading and writing.
/// * `input_path` - Path to the existing Vortex file to read.
/// * `output_path` - Path to write the updated Vortex file. Can be the same as input.
/// * `update_fn` - An async function that takes the file's array data and returns the updated
///   array. The returned array must have the same dtype as the input.
///
/// # Returns
///
/// A [`WriteSummary`] containing information about the written file.
///
/// # Errors
///
/// Returns an error if:
/// - The input file cannot be read.
/// - The update function returns an error.
/// - The update function returns an array with a different dtype.
/// - The output file cannot be written.
pub async fn update_file_async<F, Fut>(
    session: &VortexSession,
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    update_fn: F,
) -> VortexResult<WriteSummary>
where
    F: FnOnce(ArrayRef) -> Fut,
    Fut: Future<Output = VortexResult<ArrayRef>>,
{
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    // Read the existing file.
    let file = session.open_options().open(input_path).await?;
    let original_dtype = file.dtype().clone();

    // Read all existing data into memory.
    let existing_array = file.scan()?.into_array_stream()?.read_all().await?;

    // Apply the user's update function.
    let updated_array = update_fn(existing_array).await?;

    // Validate that the dtype matches.
    if updated_array.dtype() != &original_dtype {
        vortex_bail!(
            "Update function changed dtype from {} to {}. \
             The updated array must have the same dtype as the input file.",
            original_dtype,
            updated_array.dtype()
        );
    }

    // Generate a temporary file path in the same directory as output.
    // This ensures the rename will be atomic (same filesystem).
    let temp_path = generate_temp_path(output_path);

    // Write to the temporary file.
    let temp_file = async_fs::File::create(&temp_path).await?;
    let mut writer = session.write_options().writer(temp_file, original_dtype);
    writer.push(updated_array).await?;
    let summary = writer.finish().await?;

    // Atomically rename the temp file to the output path.
    async_fs::rename(&temp_path, output_path).await?;

    Ok(summary)
}

/// Generates a temporary file path in the same directory as the target path.
fn generate_temp_path(target: &Path) -> std::path::PathBuf {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let file_name = target
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "file".into());

    let temp_name = format!(".{}.{}.tmp", file_name, uuid::Uuid::new_v4());
    parent.join(temp_name)
}

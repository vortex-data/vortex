// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Functions for reading benchmark data from S3.

use aws_sdk_s3::Client;
use vortex::array::Array;
use vortex::array::ToCanonical;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::session::VortexSession;

use super::entry::BenchmarkEntry;
use super::entry::CommitId;
use super::entry::NameId;

/// Reads benchmark entries from an S3 object containing a Vortex file.
///
/// This function downloads the Vortex file from S3, parses the columnar struct array, and converts
/// it to a vector of row-wise [`BenchmarkEntry`] structs.
///
/// # Arguments
///
/// * `client` - The AWS S3 client to use for operations.
/// * `session` - The Vortex session for reading files.
/// * `bucket` - The S3 bucket name.
/// * `key` - The S3 object key.
///
/// # Errors
///
/// Returns an error if:
/// - The S3 object does not exist or cannot be downloaded.
/// - The file is not a valid Vortex file.
/// - The schema does not match the expected [`BenchmarkEntry`] schema.
pub async fn read_benchmark_entries(
    client: &Client,
    session: &VortexSession,
    bucket: &str,
    key: &str,
) -> VortexResult<Vec<BenchmarkEntry>> {
    // Download the file from S3.
    let get_result = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| vortex_err!("Failed to download S3 object: {}", e))?;

    let bytes = get_result
        .body
        .collect()
        .await
        .map_err(|e| vortex_err!("Failed to read S3 object body: {}", e))?
        .into_bytes();

    // Parse as Vortex file and read all data.
    let file = session.open_options().open_buffer(bytes)?;
    let array = file.scan()?.into_array_stream()?.read_all().await?;

    // Convert the array to benchmark entries.
    array_to_benchmark_entries(&array)
}

/// Converts a Vortex array (expected to be a struct array) into a vector of [`BenchmarkEntry`].
///
/// The array must have the following schema:
/// - `commit_id`: FixedSizeList<u8, 20>
/// - `benchmark_group`: u32
/// - `chart_name`: u32
/// - `series_name`: u32
/// - `value`: u64
pub fn array_to_benchmark_entries(array: &dyn Array) -> VortexResult<Vec<BenchmarkEntry>> {
    // Convert to canonical struct array.
    let struct_array: StructArray = array.to_struct();

    let len = struct_array.len();
    let mut entries = Vec::with_capacity(len);

    // Extract each field.
    let commit_id_field = struct_array.field_by_name("commit_id")?;
    let benchmark_group_field = struct_array.field_by_name("benchmark_group")?;
    let chart_name_field = struct_array.field_by_name("chart_name")?;
    let series_name_field = struct_array.field_by_name("series_name")?;
    let value_field = struct_array.field_by_name("value")?;

    // Convert commit_id to canonical fixed-size list and get the underlying bytes.
    let commit_id_fsl: FixedSizeListArray = commit_id_field.to_fixed_size_list();
    if commit_id_fsl.list_size() != 20 {
        vortex_bail!(
            "Expected commit_id to have list_size 20, got {}",
            commit_id_fsl.list_size()
        );
    }

    // Get the elements as a primitive array of u8.
    let commit_id_elements: PrimitiveArray = commit_id_fsl.elements().to_primitive();
    let commit_id_bytes: &[u8] = commit_id_elements.as_slice();

    // Convert primitive fields.
    let benchmark_group_prim: PrimitiveArray = benchmark_group_field.to_primitive();
    let benchmark_groups: &[u32] = benchmark_group_prim.as_slice();

    let chart_name_prim: PrimitiveArray = chart_name_field.to_primitive();
    let chart_names: &[u32] = chart_name_prim.as_slice();

    let series_name_prim: PrimitiveArray = series_name_field.to_primitive();
    let series_names: &[u32] = series_name_prim.as_slice();

    let value_prim: PrimitiveArray = value_field.to_primitive();
    let values: &[u64] = value_prim.as_slice();

    // Build the entries.
    for i in 0..len {
        // Extract the 20-byte commit_id for this row.
        let start = i * 20;
        let end = start + 20;
        let mut commit_id_arr = [0u8; 20];
        commit_id_arr.copy_from_slice(&commit_id_bytes[start..end]);

        entries.push(BenchmarkEntry {
            commit_id: CommitId(commit_id_arr),
            benchmark_group: NameId(benchmark_groups[i]),
            chart_name: NameId(chart_names[i]),
            series_name: NameId(series_names[i]),
            value: values[i],
        });
    }

    Ok(entries)
}

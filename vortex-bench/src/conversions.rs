// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use futures::StreamExt;
use futures::TryStreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetRecordBatchStream;
use sysinfo::System;
use tokio::fs::File;
use tokio::fs::OpenOptions;
use tokio::fs::create_dir_all;
use tokio::io::AsyncWriteExt;
use tracing::Instrument;
use tracing::info;
use tracing::trace;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::List;
use vortex::array::arrays::ListView;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::list::ListArrayExt;
use vortex::array::arrays::listview::recursive_list_from_list_view;
use vortex::array::arrow::FromArrowArray;
use vortex::array::builders::builder_with_capacity;
use vortex::array::extension::EmptyMetadata;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::extension::ExtDType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_tensor::vector::Vector;

use crate::CompactionStrategy;
use crate::Format;
use crate::SESSION;
use crate::utils::file::idempotent_async;

/// Memory budget per concurrent conversion stream in GB. This is somewhat arbitary.
const MEMORY_PER_STREAM_GB: u64 = 4;

/// Minimum number of concurrent conversion streams.
const MIN_CONCURRENCY: u64 = 1;

/// Maximum number of concurrent conversion streams. This is somewhat arbitary.
const MAX_CONCURRENCY: u64 = 16;

/// Returns the available system memory in bytes.
fn available_memory_bytes() -> u64 {
    System::new_all().available_memory()
}

/// Calculate appropriate concurrency based on available memory.
fn calculate_concurrency() -> usize {
    let available_gb = available_memory_bytes() / (1024 * 1024 * 1024);
    let concurrency = (available_gb / MEMORY_PER_STREAM_GB).clamp(MIN_CONCURRENCY, MAX_CONCURRENCY);

    info!(
        "Available memory: {}GB, maximum concurrency is: {}",
        available_gb, concurrency
    );

    concurrency as usize
}

/// Read a Parquet file and return it as a Vortex [`ChunkedArray`].
///
/// Note: This loads the entire file into memory. For large files, use the streaming conversion like
/// in [`parquet_to_vortex_stream`] instead.
pub async fn parquet_to_vortex_chunks(parquet_path: PathBuf) -> anyhow::Result<ChunkedArray> {
    let file = File::open(parquet_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let reader = builder.build()?;

    let chunks: Vec<ArrayRef> = parquet_to_vortex_stream(reader)
        .map(|r| r.map_err(anyhow::Error::from))
        .try_collect()
        .await?;

    Ok(ChunkedArray::from_iter(chunks))
}

/// Create a streaming Vortex array from a Parquet reader.
///
/// Streams record batches and converts them to Vortex arrays on-the-fly, avoiding loading the
/// entire file into memory.
pub fn parquet_to_vortex_stream(
    reader: ParquetRecordBatchStream<File>,
) -> impl futures::Stream<Item = VortexResult<ArrayRef>> {
    reader.map(move |result| {
        result.map_err(|e| vortex_err!(External: e)).and_then(|rb| {
            let chunk = ArrayRef::from_arrow(rb, false)?;
            let mut builder = builder_with_capacity(chunk.dtype(), chunk.len());

            // Canonicalize the chunk.
            chunk.append_to_builder(
                builder.as_mut(),
                &mut VortexSession::default().create_execution_ctx(),
            )?;

            Ok(builder.finish())
        })
    })
}

/// Convert a single Parquet file to Vortex format using streaming.
///
/// Streams data directly from Parquet to Vortex without loading the entire file into memory.
pub async fn convert_parquet_file_to_vortex(
    parquet_path: &Path,
    output_path: &Path,
    compaction: CompactionStrategy,
) -> anyhow::Result<()> {
    let file = File::open(parquet_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let dtype = DType::from_arrow(builder.schema().as_ref());

    let stream = parquet_to_vortex_stream(builder.build()?);

    let mut output_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(output_path)
        .await?;

    compaction
        .apply_options(SESSION.write_options())
        .write(
            &mut output_file,
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stream)),
        )
        .await?;

    Ok(())
}

/// Convert all Parquet files in a directory to Vortex format.
///
/// This function reads Parquet files from `{input_path}/parquet/` and writes Vortex files to
/// `{input_path}/vortex-file-compressed/` (for Default compaction) or
/// `{input_path}/vortex-compact/` (for Compact compaction).
///
/// The conversion is idempotent: existing Vortex files will not be regenerated.
pub async fn convert_parquet_directory_to_vortex(
    input_path: &Path,
    compaction: CompactionStrategy,
) -> anyhow::Result<()> {
    let (format, dir_name) = match compaction {
        CompactionStrategy::Compact => (Format::VortexCompact, Format::VortexCompact.name()),
        CompactionStrategy::Default => (Format::OnDiskVortex, Format::OnDiskVortex.name()),
    };

    let vortex_dir = input_path.join(dir_name);
    let parquet_path = input_path.join(Format::Parquet.name());
    create_dir_all(&vortex_dir).await?;

    let parquet_inputs = fs::read_dir(&parquet_path)?.collect::<std::io::Result<Vec<_>>>()?;
    trace!(
        "Found {} parquet files in {}",
        parquet_inputs.len(),
        parquet_path.to_str().unwrap()
    );

    let iter = parquet_inputs
        .iter()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "parquet"));

    let concurrency = calculate_concurrency();
    futures::stream::iter(iter)
        .map(|dir_entry| {
            let filename = {
                let mut temp = dir_entry.path();
                temp.set_extension("");
                temp.file_name().unwrap().to_str().unwrap().to_string()
            };
            let parquet_file_path = parquet_path.join(format!("{filename}.parquet"));
            let output_path = vortex_dir.join(format!("{filename}.{}", format.ext()));

            tokio::spawn(
                async move {
                    idempotent_async(output_path.as_path(), move |vtx_file| async move {
                        info!(
                            "Processing file '{filename}' with {:?} strategy",
                            compaction
                        );
                        convert_parquet_file_to_vortex(&parquet_file_path, &vtx_file, compaction)
                            .await
                    })
                    .await
                    .expect("Failed to write Vortex file")
                }
                .in_current_span(),
            )
        })
        .buffer_unordered(concurrency)
        .try_collect::<Vec<_>>()
        .await?;

    Ok(())
}

/// Convert a Parquet file to Vortex format with the specified compaction strategy.
///
/// Uses `idempotent_async` to skip conversion if the output file already exists.
pub async fn write_parquet_as_vortex(
    parquet_path: PathBuf,
    vortex_path: &str,
    compaction: CompactionStrategy,
) -> anyhow::Result<PathBuf> {
    idempotent_async(vortex_path, |output_fname| async move {
        let mut output_file = File::create(&output_fname).await?;
        let data = parquet_to_vortex_chunks(parquet_path).await?;
        let write_options = compaction.apply_options(SESSION.write_options());
        write_options
            .write(&mut output_file, data.into_array().to_array_stream())
            .await?;
        output_file.flush().await?;
        Ok(())
    })
    .await
}

/// Rewrap a list-of-float column as a [`vortex_tensor::vector::Vector`] extension array.
///
/// Parquet has no fixed-size list logical type, so an embedding column ingested via
/// [`parquet_to_vortex_chunks`] arrives as `List<f32>` (or `List<f64>` / `List<f16>`) even
/// when every row has the same length. This helper validates that every list in `input`
/// has the same length `D` and reconstructs the column as
/// `Extension<Vector>(FixedSizeList<T, D>)` — the shape expected by the vector-search
/// scalar functions in `vortex-tensor`.
///
/// The input may be either a single [`List`] array or a [`Chunked`] array of lists (the
/// common case after `parquet_to_vortex_chunks`). Chunked inputs are converted chunk-by-chunk
/// and reassembled as a [`ChunkedArray`] of `Extension<Vector>`.
///
/// # Errors
///
/// Returns an error if:
/// - `input` is not a `List` or `Chunked` array.
/// - The element type is not a non-nullable float primitive (`f16`, `f32`, or `f64`).
/// - Any row has a different length than the first row.
/// - The list validity is nullable (vector elements cannot be null at the row level).
/// - The input has zero rows (the dimension cannot be inferred from empty input).
pub fn list_to_vector_ext(input: ArrayRef) -> VortexResult<ArrayRef> {
    if let Some(chunked) = input.as_opt::<Chunked>() {
        let converted: Vec<ArrayRef> = chunked
            .iter_chunks()
            .map(|chunk| list_to_vector_ext(chunk.clone()))
            .collect::<VortexResult<_>>()?;
        if converted.is_empty() {
            vortex_bail!("list_to_vector_ext: chunked input has no chunks");
        }
        return Ok(ChunkedArray::from_iter(converted).into_array());
    }

    // `parquet_to_vortex_chunks` produces `ListView` arrays for list columns by default;
    // materialize them into a flat `List` representation before we validate offsets.
    if input.as_opt::<ListView>().is_some() {
        let flat = recursive_list_from_list_view(input)?;
        return list_to_vector_ext(flat);
    }

    let Some(list) = input.as_opt::<List>() else {
        vortex_bail!(
            "list_to_vector_ext expects a List array, got dtype {}",
            input.dtype()
        );
    };

    if !matches!(
        list.list_validity(),
        Validity::NonNullable | Validity::AllValid
    ) {
        vortex_bail!(
            "list_to_vector_ext: list rows must be non-nullable for Vector extension wrapping"
        );
    }

    let element_dtype = list.element_dtype().clone();
    let DType::Primitive(ptype, elem_nullability) = &element_dtype else {
        vortex_bail!(
            "list_to_vector_ext: element dtype must be a primitive float, got {}",
            element_dtype
        );
    };
    if !ptype.is_float() {
        vortex_bail!(
            "list_to_vector_ext: element type must be float (f16/f32/f64), got {}",
            ptype
        );
    }
    if elem_nullability.is_nullable() {
        vortex_bail!(
            "list_to_vector_ext: element type must be non-nullable, got nullable {}",
            ptype
        );
    }

    let num_rows = input.len();
    if num_rows == 0 {
        vortex_bail!("list_to_vector_ext: cannot infer vector dimension from empty input");
    }

    // Walk the offsets array once, reusing the previous iteration's `end` as the
    // next iteration's `start`. Each `offset_at` call goes through
    // `ListArrayExt::offset_at`, which has a fast path when the offsets child is a
    // `Primitive` array (direct slice index). That's the common case after
    // `parquet_to_vortex_chunks`, so for a 100K-row column we do ~100K primitive
    // slice indexes rather than 200K. The loop body is O(1) either way.
    let mut prev_end = list.offset_at(0)?;
    let first_end = list.offset_at(1)?;
    let dim = first_end.checked_sub(prev_end).ok_or_else(|| {
        vortex_err!("list_to_vector_ext: offsets are not monotonically increasing")
    })?;
    if dim == 0 {
        vortex_bail!("list_to_vector_ext: first row has zero elements");
    }
    prev_end = first_end;

    for i in 1..num_rows {
        let end = list.offset_at(i + 1)?;
        let row_len = end.checked_sub(prev_end).ok_or_else(|| {
            vortex_err!("list_to_vector_ext: offsets are not monotonically increasing")
        })?;
        if row_len != dim {
            vortex_bail!(
                "list_to_vector_ext: row {} has length {} but expected {}",
                i,
                row_len,
                dim
            );
        }
        prev_end = end;
    }

    let elements = list.sliced_elements()?;
    let expected_elements = num_rows
        .checked_mul(dim)
        .ok_or_else(|| vortex_err!("list_to_vector_ext: num_rows * dim overflows usize"))?;
    if elements.len() != expected_elements {
        vortex_bail!(
            "list_to_vector_ext: elements buffer has length {} but expected {}",
            elements.len(),
            expected_elements
        );
    }

    let dim_u32 = u32::try_from(dim)
        .map_err(|_| vortex_err!("list_to_vector_ext: dimension {dim} does not fit in u32"))?;

    let fsl = FixedSizeListArray::try_new(elements, dim_u32, Validity::NonNullable, num_rows)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::Extension;
    use vortex::array::arrays::List;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::extension::ExtensionArrayExt;
    use vortex::array::validity::Validity;
    use vortex::buffer::BufferMut;
    use vortex::dtype::DType;

    use super::list_to_vector_ext;

    fn list_f32(rows: &[&[f32]]) -> vortex::array::ArrayRef {
        let mut elements = BufferMut::<f32>::with_capacity(rows.iter().map(|r| r.len()).sum());
        let mut offsets = BufferMut::<i32>::with_capacity(rows.len() + 1);
        offsets.push(0);
        for row in rows {
            for &v in row.iter() {
                elements.push(v);
            }
            offsets.push(i32::try_from(elements.len()).unwrap());
        }

        let elements_array =
            PrimitiveArray::new::<f32>(elements.freeze(), Validity::NonNullable).into_array();
        let offsets_array =
            PrimitiveArray::new::<i32>(offsets.freeze(), Validity::NonNullable).into_array();
        vortex::array::Array::<List>::new(elements_array, offsets_array, Validity::NonNullable)
            .into_array()
    }

    #[test]
    fn uniform_list_becomes_vector_extension() {
        let list = list_f32(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        let wrapped = list_to_vector_ext(list).unwrap();
        assert_eq!(wrapped.len(), 3);
        let ext = wrapped.as_opt::<Extension>().expect("returns Extension");
        assert!(matches!(
            ext.storage_array().dtype(),
            DType::FixedSizeList(_, 3, _)
        ));
    }

    #[test]
    fn mismatched_row_length_is_rejected() {
        let list = list_f32(&[&[1.0, 2.0, 3.0], &[4.0, 5.0]]);
        let err = list_to_vector_ext(list).unwrap_err().to_string();
        assert!(
            err.contains("row 1 has length 2 but expected 3"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn non_list_input_is_rejected() {
        let primitive = PrimitiveArray::new::<f32>(
            BufferMut::<f32>::from_iter([1.0f32, 2.0, 3.0]).freeze(),
            Validity::NonNullable,
        )
        .into_array();
        let err = list_to_vector_ext(primitive).unwrap_err().to_string();
        assert!(
            err.contains("expects a List array"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_input_is_rejected() {
        let list = list_f32(&[]);
        let err = list_to_vector_ext(list).unwrap_err().to_string();
        assert!(
            err.contains("cannot infer vector dimension from empty input"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn non_float_element_type_is_rejected() {
        // Build a List<i32>.
        let elements = PrimitiveArray::new::<i32>(
            BufferMut::<i32>::from_iter([1i32, 2, 3, 4]).freeze(),
            Validity::NonNullable,
        )
        .into_array();
        let offsets = PrimitiveArray::new::<i32>(
            BufferMut::<i32>::from_iter([0i32, 2, 4]).freeze(),
            Validity::NonNullable,
        )
        .into_array();
        let list = vortex::array::Array::<List>::new(elements, offsets, Validity::NonNullable)
            .into_array();

        let err = list_to_vector_ext(list).unwrap_err().to_string();
        assert!(
            err.contains("element type must be float"),
            "unexpected error: {err}",
        );
    }
}

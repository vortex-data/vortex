// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Merging chunked Parquet Variant arrays into one Parquet Variant array.
//!
//! Variant values have no builder, so concatenating Variant chunks keeps every chunk's storage
//! separate: a `Chunked` array of Parquet Variant chunks canonicalizes to a Variant whose core
//! storage is still one Parquet Variant per chunk. Compression then sees each chunk's `metadata`
//! and `value` binaries on their own, paying dictionary and symbol table overhead per chunk.
//!
//! Rewriting the `Chunked` parent into a single Parquet Variant whose children are chunked lets
//! those children canonicalize, and compress, as one array each.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantArraySlotsExt;

/// Rewrites a `Chunked` array of Parquet Variant chunks with matching storage shapes into one
/// Parquet Variant array with chunked `metadata`, `value`, `typed_value` and validity children.
pub(crate) fn chunked_parquet_variant_reduce_parent(
    _child: &ArrayRef,
    parent: &ArrayRef,
    _child_idx: usize,
) -> VortexResult<Option<ArrayRef>> {
    let Some(chunked) = parent.as_opt::<Chunked>() else {
        return Ok(None);
    };
    if chunked.nchunks() < 2 {
        return Ok(None);
    }

    let mut chunks = Vec::with_capacity(chunked.nchunks());
    for chunk in chunked.iter_chunks() {
        let Some(chunk) = chunk.as_opt::<ParquetVariant>() else {
            return Ok(None);
        };
        chunks.push(chunk);
    }

    let first = &chunks[0];
    let value_dtype = first.value().map(|value| value.dtype().clone());
    let typed_value_dtype = first.typed_value().map(|typed| typed.dtype().clone());
    let same_shape = chunks.iter().all(|chunk| {
        chunk.metadata().dtype() == first.metadata().dtype()
            && chunk.value().map(|value| value.dtype()) == value_dtype.as_ref()
            && chunk.typed_value().map(|typed| typed.dtype()) == typed_value_dtype.as_ref()
    });
    if !same_shape {
        return Ok(None);
    }

    let metadata = ChunkedArray::try_new(
        chunks.iter().map(|chunk| chunk.metadata().clone()),
        first.metadata().dtype().clone(),
    )?
    .into_array();
    let value = value_dtype
        .map(|dtype| {
            ChunkedArray::try_new(
                chunks.iter().filter_map(|chunk| chunk.value().cloned()),
                dtype,
            )
        })
        .transpose()?
        .map(IntoArray::into_array);
    let typed_value = typed_value_dtype
        .map(|dtype| {
            ChunkedArray::try_new(
                chunks
                    .iter()
                    .filter_map(|chunk| chunk.typed_value().cloned()),
                dtype,
            )
        })
        .transpose()?
        .map(IntoArray::into_array);

    let validities: Vec<Validity> = chunks
        .iter()
        .map(|chunk| chunk.parquet_variant_validity())
        .collect();
    let validity = match parent.dtype().nullability() {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable
            if validities
                .iter()
                .all(|validity| matches!(validity, Validity::NonNullable | Validity::AllValid)) =>
        {
            Validity::AllValid
        }
        Nullability::Nullable => Validity::Array(
            ChunkedArray::try_new(
                validities
                    .iter()
                    .zip(&chunks)
                    .map(|(validity, chunk)| validity.to_array(chunk.len())),
                DType::Bool(Nullability::NonNullable),
            )?
            .into_array(),
        ),
    };

    Ok(Some(
        ParquetVariant::try_new(validity, metadata, value, typed_value)?.into_array(),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::StringArray;
    use parquet_variant_compute::json_to_variant;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_arrow::ArrowSessionExt;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ParquetVariant;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn json_variant(values: Vec<Option<&str>>) -> VortexResult<ArrayRef> {
        let json: ArrowArrayRef = Arc::new(StringArray::from(values));
        ParquetVariant::from_arrow_variant(&json_to_variant(&json)?, &SESSION.arrow())
    }

    #[test]
    fn chunked_parquet_variant_merges_into_one_parquet_variant() -> VortexResult<()> {
        let chunks = [
            json_variant(vec![Some(r#"{"a": 1}"#), None])?,
            json_variant(vec![Some(r#"{"a": 2}"#), None])?,
        ];
        let dtype = chunks[0].dtype().clone();
        let chunked = ChunkedArray::try_new(chunks.clone(), dtype)?.into_array();

        let merged = chunked.optimize_ctx(&SESSION)?;
        assert!(
            merged.is::<ParquetVariant>(),
            "got {}",
            merged.encoding_id()
        );

        let mut ctx = SESSION.create_execution_ctx();
        let expected = chunks
            .iter()
            .flat_map(|chunk| (0..chunk.len()).map(move |row| (chunk, row)));
        for (idx, (chunk, row)) in expected.enumerate() {
            assert_eq!(
                merged.execute_scalar(idx, &mut ctx)?,
                chunk.execute_scalar(row, &mut ctx)?
            );
        }
        Ok(())
    }
}

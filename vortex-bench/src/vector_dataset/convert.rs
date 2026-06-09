// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// TODO(connor): Should we re-export this through `conversions.rs`?

use vortex::array::ArrayRef;
use vortex::array::EmptyMetadata;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::List;
use vortex::array::arrays::ListView;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::array::arrays::list::ListArrayExt;
use vortex::array::arrays::listview::recursive_list_from_list_view;
use vortex::array::validity::Validity;
use vortex::dtype::DType;
use vortex::dtype::extension::ExtDType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex_tensor::vector::Vector;

use crate::SESSION;

/// Rewrap a list-of-float column as a [`vortex_tensor::vector::Vector`] extension array.
///
/// Parquet has no fixed-size list logical type, so an embedding column ingested via
/// `parquet_to_vortex_chunks` arrives as `List<f32>` (or `List<f64>`) even when every row has the
/// same length.
///
/// This helper validates that every list in `input` has the same length `D` and reconstructs the
/// column as an `Extension<Vector>(FixedSizeList<T, D>)` array, which is the type expected by the
/// vector search scalar functions in `vortex-tensor`.
///
/// The input may be either a single [`ListView`] array or a [`Chunked`] array of lists (the common
/// case after `parquet_to_vortex_chunks`). Chunked inputs are converted chunk-by-chunk and
/// reassembled as a [`ChunkedArray`] of `Extension<Vector>`. We also convert [`ListView`] to
/// [`List`] so that we know all elements are contiguous (this might be slow).
///
/// # Errors
///
/// Returns an error if:
/// - `input` is not a `ListView`, `List`, or `Chunked` array.
/// - The element type is not a float primitive (`f16`, `f32`, or `f64`).
/// - A nullable element dtype (`List<f32?>`) is accepted as long as the runtime validity is
///   `NonNullable` or `AllValid` since parquet has no non-nullable-element list logical type, so
///   arrow-rs always marks list-of-float element fields as nullable on read regardless of whether
///   any element is actually missing. In that case the elements are rewrapped as non-nullable
///   before being embedded in the FSL.
/// - The element dtype is nullable *and* any element is actually null (i.e., `Validity::AllInvalid`
///   or any `Validity::Array` mask). Vector extension elements must be non-null, and that is
///   verified on construction.
/// - Any row has a different length than the first row.
/// - The list validity is nullable (vector elements cannot be null at the row level).
/// - The input has zero rows (the dimension cannot be inferred from empty input).
pub fn list_to_vector_ext(input: ArrayRef) -> VortexResult<ArrayRef> {
    if let Some(chunked) = input.as_opt::<Chunked>() {
        let converted: Vec<ArrayRef> = chunked
            .iter_chunks()
            .map(|chunk| list_to_vector_ext(chunk.clone()))
            .collect::<VortexResult<_>>()?;

        let Some(first) = converted.first() else {
            vortex_bail!("list_to_vector_ext: chunked input has no chunks");
        };

        let dtype = first.dtype().clone();
        return Ok(ChunkedArray::try_new(converted, dtype)?.into_array());
    }

    // `parquet_to_vortex_chunks` produces `ListView` arrays for list columns by default;
    // materialize them into a flat `List` representation before we validate offsets.
    if input.as_opt::<ListView>().is_some() {
        let flat = recursive_list_from_list_view(input, &mut SESSION.create_execution_ctx())?;
        return list_to_vector_ext(flat);
    }

    let Some(list) = input.as_opt::<List>() else {
        vortex_bail!(
            "list_to_vector_ext: expected a List array, got dtype {}",
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

    // Extract the flat elements buffer up front: the nullable-handling branch below
    // needs to inspect runtime validity before we can decide whether to rewrap it.
    let raw_elements = list.sliced_elements()?;

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

        let row_len = end
            .checked_sub(prev_end)
            .vortex_expect("list offsets must be monotonically increasing");
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

    let expected_elements = num_rows
        .checked_mul(dim)
        .ok_or_else(|| vortex_err!("list_to_vector_ext: num_rows * dim overflows usize"))?;
    if raw_elements.len() != expected_elements {
        vortex_bail!(
            "list_to_vector_ext: elements buffer has length {} but expected {}",
            raw_elements.len(),
            expected_elements
        );
    }

    // Parquet has no non-nullable-element list logical type, so arrow-rs marks every
    // `List<float>`'s element field as nullable on read regardless of what the writer intended.
    // That propagates through `DType::from_arrow`, so every real embedding parquet file arrives
    // shaped as `List<f32?>` even when every value is present. A nullable element dtype is
    // losslessly convertible to a non-nullable FSL as long as the runtime validity is
    // `NonNullable`/`AllValid`; we must only reject when a real null is present.
    let elements = if elem_nullability.is_nullable() {
        let primitive = raw_elements.as_opt::<Primitive>().ok_or_else(|| {
            vortex_err!(
                "list_to_vector_ext: expected nullable-float elements to downcast to \
                 Primitive, got dtype {}",
                raw_elements.dtype()
            )
        })?;
        match primitive.validity()? {
            Validity::NonNullable | Validity::AllValid => {
                // `to_host_sync` is a no-op for host-resident buffers, so this is a
                // metadata change (rebuilding the array with a non-nullable dtype),
                // not a data copy.
                let byte_buffer = primitive.buffer_handle().to_host_sync();
                PrimitiveArray::from_byte_buffer(byte_buffer, *ptype, Validity::NonNullable)
                    .into_array()
            }
            Validity::AllInvalid => {
                vortex_bail!(
                    "list_to_vector_ext: list has nullable element dtype with all-invalid \
                     elements; Vector extension elements must be non-null"
                );
            }
            Validity::Array(_) => {
                vortex_bail!(
                    "list_to_vector_ext: list has nullable element dtype with one or more \
                     actual null elements; Vector extension elements must be non-null"
                );
            }
        }
    } else {
        raw_elements
    };

    let dim_u32 = u32::try_from(dim)
        .map_err(|_| vortex_err!("list_to_vector_ext: dimension {dim} does not fit in u32"))?;

    // Finally, construct the `FixedSizeListArray` and wrap it in a Vector array.
    let fsl = FixedSizeListArray::try_new(elements, dim_u32, Validity::NonNullable, num_rows)?;
    let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
    Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
}

#[cfg(test)]
mod tests {
    use vortex::array::Array;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::Extension;
    use vortex::array::arrays::List;
    use vortex::array::arrays::ListViewArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::extension::ExtensionArrayExt;
    use vortex::array::validity::Validity;
    use vortex::buffer::BufferMut;
    use vortex::dtype::DType;

    use super::list_to_vector_ext;

    /// Build a `List<f32>` whose elements carry the given [`Validity`]. Passing
    /// `Validity::NonNullable` produces a `List<f32>`; any other variant produces
    /// a `List<f32?>`, matching the shape `parquet_to_vortex_chunks` produces for
    /// embedding columns after arrow-rs' canonicalization.
    fn list_f32_with_element_validity(
        values: &[f32],
        dim: usize,
        element_validity: Validity,
    ) -> ArrayRef {
        assert_eq!(
            values.len() % dim,
            0,
            "values.len() must be a multiple of dim"
        );
        let num_rows = values.len() / dim;
        let elements = PrimitiveArray::new::<f32>(
            BufferMut::<f32>::from_iter(values.iter().copied()).freeze(),
            element_validity,
        )
        .into_array();
        let mut offsets_buf = BufferMut::<i32>::with_capacity(num_rows + 1);
        for i in 0..=num_rows {
            offsets_buf.push(i32::try_from(i * dim).unwrap());
        }
        let offsets =
            PrimitiveArray::new::<i32>(offsets_buf.freeze(), Validity::NonNullable).into_array();
        Array::<List>::new(elements, offsets, Validity::NonNullable).into_array()
    }

    fn list_f32(rows: &[&[f32]]) -> ArrayRef {
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
        Array::<List>::new(elements_array, offsets_array, Validity::NonNullable).into_array()
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
            err.contains("expected a List array"),
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

    /// Build a `ListView<f32>` whose every row is a length-`dim` slice of the flattened
    /// `values` buffer. This shape matches what `parquet_to_vortex_chunks` produces for
    /// embedding columns after arrow-rs' canonicalization, and exercises the
    /// `list_to_vector_ext` fast-path that collapses `ListView` → `List` before
    /// validating offsets.
    fn list_view_f32(dim: usize, rows: &[&[f32]]) -> ArrayRef {
        let mut values = BufferMut::<f32>::with_capacity(rows.len() * dim);
        for row in rows {
            assert_eq!(row.len(), dim);
            for &v in row.iter() {
                values.push(v);
            }
        }
        let elements =
            PrimitiveArray::new::<f32>(values.freeze(), Validity::NonNullable).into_array();

        let dim_i32 = i32::try_from(dim).unwrap();
        let num_rows = rows.len();

        let mut offsets_buf = BufferMut::<i32>::with_capacity(num_rows);
        for i in 0..num_rows {
            offsets_buf.push(i32::try_from(i).unwrap() * dim_i32);
        }
        let offsets =
            PrimitiveArray::new::<i32>(offsets_buf.freeze(), Validity::NonNullable).into_array();

        let mut sizes_buf = BufferMut::<i32>::with_capacity(num_rows);
        for _ in 0..num_rows {
            sizes_buf.push(dim_i32);
        }
        let sizes =
            PrimitiveArray::new::<i32>(sizes_buf.freeze(), Validity::NonNullable).into_array();

        ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)
            .unwrap()
            .into_array()
    }

    #[test]
    fn list_view_input_is_rewrapped_as_vector_extension() {
        // Simulates the post-parquet-ingest shape: the `emb` column arrives as a
        // ListView, not a List. `list_to_vector_ext` must materialize it via
        // `recursive_list_from_list_view` and then validate offsets on the flattened
        // `List` form.
        let list_view = list_view_f32(3, &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let wrapped = list_to_vector_ext(list_view).unwrap();
        assert_eq!(wrapped.len(), 2);
        let ext = wrapped.as_opt::<Extension>().expect("returns Extension");
        assert!(matches!(
            ext.storage_array().dtype(),
            DType::FixedSizeList(_, 3, _)
        ));
    }

    #[test]
    fn all_invalid_list_validity_is_rejected() {
        // A list with `Validity::AllInvalid` means every row is null. The Vector
        // extension type requires non-nullable elements at the FSL level, so we
        // must reject this input rather than silently dropping the validity mask.
        let elements = PrimitiveArray::new::<f32>(
            BufferMut::<f32>::from_iter([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).freeze(),
            Validity::NonNullable,
        )
        .into_array();
        let offsets = PrimitiveArray::new::<i32>(
            BufferMut::<i32>::from_iter([0i32, 3, 6]).freeze(),
            Validity::NonNullable,
        )
        .into_array();
        let list = Array::<List>::new(elements, offsets, Validity::AllInvalid).into_array();

        let err = list_to_vector_ext(list).unwrap_err().to_string();
        assert!(
            err.contains("list rows must be non-nullable"),
            "unexpected error: {err}"
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
        let list = Array::<List>::new(elements, offsets, Validity::NonNullable).into_array();

        let err = list_to_vector_ext(list).unwrap_err().to_string();
        assert!(
            err.contains("element type must be float"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn nullable_elements_with_real_nulls_are_rejected() {
        // A `List<f32?>` whose elements carry a real `Validity::Array` mask with
        // at least one `false` bit has one or more actually-missing values. The
        // rejection here is about runtime nulls, not dtype metadata: a nullable
        // element dtype with all-valid runtime validity is accepted (see
        // `nullable_element_dtype_with_all_valid_elements_is_accepted`), because
        // parquet-ingested embeddings always arrive shaped that way even when
        // every value is present. A real null, on the other hand, cannot be
        // represented in the Vector extension FSL and must be rejected rather
        // than silently dropped.
        let element_validity = Validity::Array(
            BoolArray::from_iter([true, true, false, true, true, true]).into_array(),
        );
        let list =
            list_f32_with_element_validity(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, element_validity);

        let err = list_to_vector_ext(list).unwrap_err().to_string();
        assert!(
            err.contains("one or more actual null elements"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nullable_element_dtype_with_all_valid_elements_is_accepted() {
        // This is the regression test for the Cohere parquet case: every real
        // VectorDBBench parquet file arrives as `List<f32?>` with
        // `Validity::AllValid` elements because parquet has no non-nullable
        // list-element logical type and arrow-rs propagates the nullable bit
        // through `DType::from_arrow`. `list_to_vector_ext` must accept this
        // shape by rewrapping the elements as non-nullable before building the
        // FSL, rather than rejecting outright on the dtype metadata.
        let list =
            list_f32_with_element_validity(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, Validity::AllValid);

        let wrapped = list_to_vector_ext(list).unwrap();
        assert_eq!(wrapped.len(), 2);
        let ext = wrapped.as_opt::<Extension>().expect("returns Extension");
        assert!(matches!(
            ext.storage_array().dtype(),
            DType::FixedSizeList(_, 3, _)
        ));
    }

    #[test]
    fn nullable_element_dtype_with_all_invalid_elements_is_rejected() {
        // A `List<f32?>` whose elements are `Validity::AllInvalid` means every
        // value is missing. Rewrapping as non-nullable would silently drop the
        // validity and produce bogus vectors, so this must be rejected.
        let list = list_f32_with_element_validity(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3,
            Validity::AllInvalid,
        );

        let err = list_to_vector_ext(list).unwrap_err().to_string();
        assert!(
            err.contains("all-invalid elements"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn chunked_input_with_mixed_dimensions_returns_error() {
        let dim_three = list_f32(&[&[1.0, 2.0, 3.0]]);
        let dim_two = list_f32(&[&[4.0, 5.0]]);
        let chunked =
            ChunkedArray::try_new(vec![dim_three.clone(), dim_two], dim_three.dtype().clone())
                .unwrap()
                .into_array();

        let err = list_to_vector_ext(chunked).unwrap_err().to_string();
        assert!(err.contains("Mismatched types"), "unexpected error: {err}");
    }
}

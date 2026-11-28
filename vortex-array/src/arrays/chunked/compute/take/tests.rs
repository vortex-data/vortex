// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DictArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::assert_arrays_eq;
use crate::compute::conformance::take::test_take_conformance;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::scalar::Scalar;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn contiguous_pieces(starts: &[u64], lengths: &[u64]) -> VortexResult<ArrayRef> {
    let len = usize::try_from(lengths.iter().sum::<u64>())?;
    Ok(PiecewiseSequenceArray::try_new(
        starts.iter().copied().collect::<Buffer<u64>>().into_array(),
        lengths
            .iter()
            .copied()
            .collect::<Buffer<u64>>()
            .into_array(),
        ConstantArray::new(1u64, starts.len()).into_array(),
        len,
    )?
    .into_array())
}

fn draw_validity(tc: &TestCase, len: usize, nullable: bool) -> Validity {
    if !nullable {
        return Validity::NonNullable;
    }
    match tc.draw(gs::integers::<u8>().max_value(3)) {
        0 => Validity::AllValid,
        1 => Validity::AllInvalid,
        _ => Validity::from_iter(tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len))),
    }
}

fn draw_chunk_of_kind(tc: &TestCase, kind: u8, dtype: &DType, len: usize) -> ArrayRef {
    let nullable = dtype.is_nullable();
    // Occasionally emit a ConstantArray chunk (for nullable or non-nullable chunks).
    if len > 0 && tc.draw(gs::integers::<u8>().max_value(5)) == 0 {
        let scalar = if nullable && tc.draw(gs::booleans()) {
            Scalar::null(dtype.clone())
        } else {
            Scalar::default_value(dtype)
        };
        return ConstantArray::new(scalar, len).into_array();
    }

    // Occasionally slice a slightly larger chunk to test non-zero buffer offsets.
    let pad = if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
        tc.draw(gs::integers::<usize>().min_value(1).max_value(3))
    } else {
        0
    };
    let full_len = len + pad;

    let full = match kind {
        0 => {
            let vals: Vec<i32> = tc.draw(
                gs::vecs(gs::integers::<i32>().min_value(-200).max_value(200))
                    .min_size(full_len)
                    .max_size(full_len),
            );
            let validity = draw_validity(tc, full_len, nullable);
            PrimitiveArray::new(Buffer::from(vals), validity).into_array()
        }
        1 => {
            let list_size = 2usize;
            let elem_len = full_len * list_size;
            let vals: Vec<i32> = tc.draw(
                gs::vecs(gs::integers::<i32>().min_value(-200).max_value(200))
                    .min_size(elem_len)
                    .max_size(elem_len),
            );
            let validity = draw_validity(tc, full_len, nullable);
            let fsl = FixedSizeListArray::try_new(
                PrimitiveArray::from_iter(vals).into_array(),
                2,
                validity,
                full_len,
            )
            .vortex_expect("valid fsl")
            .into_array();
            // Occasionally wrap in a nested ChunkedArray so the chunk is not directly FSL-encoded.
            if pad == 0 && tc.draw(gs::integers::<u8>().max_value(4)) == 0 {
                ChunkedArray::try_new(vec![fsl], dtype.clone())
                    .vortex_expect("nested chunked")
                    .into_array()
            } else {
                fsl
            }
        }
        2 => {
            let elem_len = full_len * 2;
            let tokens_vals: Vec<i32> = tc.draw(
                gs::vecs(gs::integers::<i32>().min_value(-200).max_value(200))
                    .min_size(elem_len)
                    .max_size(elem_len),
            );
            let tokens = FixedSizeListArray::try_new(
                PrimitiveArray::from_iter(tokens_vals).into_array(),
                2,
                Validity::NonNullable,
                full_len,
            )
            .vortex_expect("valid tokens")
            .into_array();
            let ids: Vec<i32> = tc.draw(
                gs::vecs(gs::integers::<i32>().min_value(-200).max_value(200))
                    .min_size(full_len)
                    .max_size(full_len),
            );
            let validity = draw_validity(tc, full_len, nullable);
            StructArray::try_new(
                ["tokens", "id"].into(),
                vec![tokens, PrimitiveArray::from_iter(ids).into_array()],
                full_len,
                validity,
            )
            .vortex_expect("valid struct")
            .into_array()
        }
        _ => {
            // Non-FSL struct chunk (exercises the `take_chunked_struct -> Ok(None)` fallback).
            let a_vals: Vec<i32> = tc.draw(
                gs::vecs(gs::integers::<i32>().min_value(-200).max_value(200))
                    .min_size(full_len)
                    .max_size(full_len),
            );
            let b_vals: Vec<i32> = tc.draw(
                gs::vecs(gs::integers::<i32>().min_value(-200).max_value(200))
                    .min_size(full_len)
                    .max_size(full_len),
            );
            let validity = draw_validity(tc, full_len, nullable);
            StructArray::try_new(
                ["a", "b"].into(),
                vec![
                    PrimitiveArray::from_iter(a_vals).into_array(),
                    PrimitiveArray::from_iter(b_vals).into_array(),
                ],
                full_len,
                validity,
            )
            .vortex_expect("valid flat struct")
            .into_array()
        }
    };

    if pad > 0 {
        let start = tc.draw(gs::integers::<usize>().max_value(pad));
        full.slice(start..start + len).vortex_expect("slice chunk")
    } else {
        full
    }
}

fn build_indices_array(ptype: PType, values: &[usize], validity: Validity) -> ArrayRef {
    match ptype {
        PType::U8 => PrimitiveArray::new(
            values
                .iter()
                .map(|&v| u8::try_from(v).vortex_expect("u8"))
                .collect::<Buffer<u8>>(),
            validity,
        )
        .into_array(),
        PType::U16 => PrimitiveArray::new(
            values
                .iter()
                .map(|&v| u16::try_from(v).vortex_expect("u16"))
                .collect::<Buffer<u16>>(),
            validity,
        )
        .into_array(),
        PType::U32 => PrimitiveArray::new(
            values
                .iter()
                .map(|&v| u32::try_from(v).vortex_expect("u32"))
                .collect::<Buffer<u32>>(),
            validity,
        )
        .into_array(),
        PType::U64 => PrimitiveArray::new(
            values.iter().map(|&v| v as u64).collect::<Buffer<u64>>(),
            validity,
        )
        .into_array(),
        PType::I8 => PrimitiveArray::new(
            values
                .iter()
                .map(|&v| i8::try_from(v).vortex_expect("i8"))
                .collect::<Buffer<i8>>(),
            validity,
        )
        .into_array(),
        PType::I16 => PrimitiveArray::new(
            values
                .iter()
                .map(|&v| i16::try_from(v).vortex_expect("i16"))
                .collect::<Buffer<i16>>(),
            validity,
        )
        .into_array(),
        PType::I32 => PrimitiveArray::new(
            values
                .iter()
                .map(|&v| i32::try_from(v).vortex_expect("i32"))
                .collect::<Buffer<i32>>(),
            validity,
        )
        .into_array(),
        PType::I64 => PrimitiveArray::new(
            values
                .iter()
                .map(|&v| i64::try_from(v).vortex_expect("i64"))
                .collect::<Buffer<i64>>(),
            validity,
        )
        .into_array(),
        other => unreachable!("unsupported index ptype {other}"),
    }
}

/// Differential model comparing `ChunkedArray::take` against taking from the recursively
/// canonicalized flat array across 0..=6 chunks (including empty, single-element, sliced,
/// nullable, constant, FSL, Struct-of-FSL, and non-FSL Struct chunks) and all 8 integer index
/// types plus `PiecewiseSequenceArray` indices.
#[hegel::test(test_cases = 140)]
fn hegel_take_chunked_matches_canonical(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let kind = tc.draw(gs::integers::<u8>().max_value(3));
    let nullable = tc.draw(gs::booleans());
    let nullability = Nullability::from(nullable);
    let dtype = match kind {
        0 => DType::Primitive(PType::I32, nullability),
        1 => DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            nullability,
        ),
        2 => DType::Struct(
            StructFields::from_iter([
                (
                    "tokens",
                    DType::FixedSizeList(
                        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                        2,
                        Nullability::NonNullable,
                    ),
                ),
                ("id", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ]),
            nullability,
        ),
        _ => DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ]),
            nullability,
        ),
    };

    let nchunks = tc.draw(gs::integers::<usize>().max_value(6));
    let mut chunks = Vec::with_capacity(nchunks);
    for _ in 0..nchunks {
        let len = match tc.draw(gs::integers::<u8>().max_value(4)) {
            0 => 0,
            1 => 1,
            _ => tc.draw(gs::integers::<usize>().min_value(2).max_value(12)),
        };
        chunks.push(draw_chunk_of_kind(&tc, kind, &dtype, len));
    }

    let chunked = ChunkedArray::try_new(chunks, dtype)
        .vortex_expect("valid chunked")
        .into_array();
    let total_len = chunked.len();
    // Recursively canonicalize so nested FSL/Struct children are also flattened (independent of
    // `Chunked::take`).
    let flat = chunked
        .clone()
        .execute::<RecursiveCanonical>(&mut ctx)
        .vortex_expect("recursively canonicalize chunked")
        .0
        .into_array();

    // Choose between primitive indices (sorted, shuffled, duplicates, empty, nullable) and
    // piecewise sequence indices (contiguous, adjacent-mergeable, out-of-order, strided).
    let indices = if total_len > 0 && tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
        let num_pieces = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let mut starts = Vec::with_capacity(num_pieces);
        let mut lengths = Vec::with_capacity(num_pieces);
        let mut mults = Vec::with_capacity(num_pieces);
        let non_unit = tc.draw(gs::integers::<u8>().max_value(3)) == 0;
        for _ in 0..num_pieces {
            let s = tc.draw(gs::integers::<usize>().max_value(total_len - 1));
            let m = if non_unit {
                tc.draw(gs::integers::<usize>().min_value(1).max_value(3))
            } else {
                1
            };
            let max_l = ((total_len - 1 - s) / m + 1).min(8);
            let l = tc.draw(gs::integers::<usize>().max_value(max_l));
            starts.push(s as u64);
            lengths.push(l as u64);
            mults.push(m as u64);
        }
        let pw_len = usize::try_from(lengths.iter().sum::<u64>()).vortex_expect("usize");
        let mult_arr = if mults.iter().all(|&m| m == 1) {
            ConstantArray::new(1u64, starts.len()).into_array()
        } else {
            Buffer::from(mults).into_array()
        };
        PiecewiseSequenceArray::try_new(
            Buffer::from(starts).into_array(),
            Buffer::from(lengths).into_array(),
            mult_arr,
            pw_len,
        )
        .vortex_expect("valid piecewise")
        .into_array()
    } else {
        let take_len = tc.draw(gs::integers::<usize>().max_value(24));
        let indices_nullable = tc.draw(gs::booleans());
        let validity = if total_len == 0 {
            if indices_nullable {
                Validity::AllInvalid
            } else {
                Validity::NonNullable
            }
        } else {
            draw_validity(&tc, take_len, indices_nullable)
        };
        let actual_take_len = if total_len == 0 && !indices_nullable {
            0
        } else {
            take_len
        };
        let validity = if actual_take_len == 0 {
            Validity::from(Nullability::from(indices_nullable))
        } else {
            validity
        };

        let mut vals: Vec<usize> = if total_len == 0 {
            vec![0; actual_take_len]
        } else {
            (0..actual_take_len)
                .map(|_| tc.draw(gs::integers::<usize>().max_value(total_len - 1)))
                .collect()
        };
        // Bias half the cases toward sorted indices to exercise the sorted fast path.
        if tc.draw(gs::booleans()) {
            vals.sort_unstable();
        }
        let max_v = vals.iter().copied().max().unwrap_or(0);
        let mut ptypes = vec![PType::U32, PType::U64, PType::I32, PType::I64];
        if max_v <= usize::from(u16::MAX) {
            ptypes.push(PType::U16);
        }
        if max_v <= usize::from(u8::MAX) {
            ptypes.push(PType::U8);
        }
        if max_v <= usize::try_from(i16::MAX).vortex_expect("i16") {
            ptypes.push(PType::I16);
        }
        if max_v <= usize::try_from(i8::MAX).vortex_expect("i8") {
            ptypes.push(PType::I8);
        }
        let ptype = ptypes[tc.draw(gs::integers::<usize>().max_value(ptypes.len() - 1))];
        build_indices_array(ptype, &vals, validity)
    };

    let actual = chunked.take(indices.clone()).vortex_expect("chunked take");
    let expected = flat.take(indices).vortex_expect("flat take");
    assert_arrays_eq!(actual, expected, &mut ctx);
}

/// Exercises both the direct-append path (`MIN_AVERAGE_DIRECT_RUN = 32`) and the sparse-long-run
/// path (`MIN_AVERAGE_EXECUTED_RUN = 256`, `SPARSE_CHUNK_RATIO = 8`, chunk len >= 2048) with
/// out-of-order and boundary-crossing piecewise runs.
#[hegel::test(test_cases = 25)]
fn hegel_take_piecewise_long_runs(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let nullable = tc.draw(gs::booleans());
    let sparse_mode = tc.draw(gs::booleans());
    let nchunks = if sparse_mode {
        tc.draw(gs::integers::<usize>().min_value(2).max_value(3))
    } else {
        tc.draw(gs::integers::<usize>().min_value(3).max_value(5))
    };
    let mut chunks = Vec::with_capacity(nchunks);
    let mut start = 0i32;
    for _ in 0..nchunks {
        let len = if sparse_mode {
            // >= 256 * 8 = 2048 so sparse_long_runs (`MIN_AVERAGE_EXECUTED_RUN`) is reachable.
            tc.draw(gs::integers::<i32>().min_value(2200).max_value(2600))
        } else {
            tc.draw(gs::integers::<i32>().min_value(300).max_value(512))
        };
        let end = start + len;
        let chunk = if nullable {
            PrimitiveArray::from_option_iter((start..end).map(|r| (r % 5 != 2).then_some(r)))
                .into_array()
        } else {
            PrimitiveArray::from_iter(start..end).into_array()
        };
        chunks.push(chunk);
        start = end;
    }
    let total = usize::try_from(start).vortex_expect("positive");
    let dtype = DType::Primitive(PType::I32, Nullability::from(nullable));
    let chunked = ChunkedArray::try_new(chunks, dtype)
        .vortex_expect("chunked")
        .into_array();
    let flat = chunked
        .clone()
        .execute::<RecursiveCanonical>(&mut ctx)
        .vortex_expect("flat")
        .0
        .into_array();

    let (nruns, min_run, max_run) = if sparse_mode {
        (
            tc.draw(gs::integers::<usize>().min_value(1).max_value(2)),
            256usize,
            280usize,
        )
    } else {
        (
            tc.draw(gs::integers::<usize>().min_value(2).max_value(5)),
            100usize,
            400usize.min(total),
        )
    };
    let mut starts = Vec::with_capacity(nruns);
    let mut lengths = Vec::with_capacity(nruns);
    for _ in 0..nruns {
        let l = tc.draw(
            gs::integers::<usize>()
                .min_value(min_run)
                .max_value(max_run),
        );
        let s = tc.draw(gs::integers::<usize>().max_value(total - l));
        starts.push(s as u64);
        lengths.push(l as u64);
    }

    let indices = contiguous_pieces(&starts, &lengths).vortex_expect("pieces");
    let actual = chunked.take(indices.clone()).vortex_expect("chunked take");
    let expected = flat.take(indices).vortex_expect("flat take");
    assert_arrays_eq!(actual, expected, &mut ctx);
}

// --- Conformance entry points, internal-invariant tests, and OOB guards ---

fn chunked_struct_of_fsl() -> VortexResult<(ChunkedArray, StructArray)> {
    let lengths = [3usize, 2, 4];
    let mut chunks = Vec::new();
    let mut start = 0usize;

    for length in lengths {
        let rows = start..start + length;
        let tokens = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter((rows.start * 2..rows.end * 2).map(|v| v as u64))
                .into_array(),
            2,
            Validity::NonNullable,
            length,
        )?;
        let ids = PrimitiveArray::from_iter(rows.clone().map(|v| v as u64));
        let validity = Validity::from_iter(rows.map(|row| row % 4 != 1));
        chunks.push(
            StructArray::try_new(
                ["tokens", "id"].into(),
                vec![tokens.into_array(), ids.into_array()],
                length,
                validity,
            )?
            .into_array(),
        );
        start += length;
    }

    let dtype = chunks[0].dtype().clone();
    let chunked = ChunkedArray::try_new(chunks, dtype)?;

    let tokens = FixedSizeListArray::try_new(
        PrimitiveArray::from_iter((0..start * 2).map(|v| v as u64)).into_array(),
        2,
        Validity::NonNullable,
        start,
    )?;
    let flat = StructArray::try_new(
        ["tokens", "id"].into(),
        vec![
            tokens.into_array(),
            PrimitiveArray::from_iter((0..start).map(|v| v as u64)).into_array(),
        ],
        start,
        Validity::from_iter((0..start).map(|row| row % 4 != 1)),
    )?;
    Ok((chunked, flat))
}

#[test]
fn test_touched_chunks_piecewise_without_expanding() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let lengths = [2usize, 0, 3, 1, 2];
    let mut chunks = Vec::new();
    let mut start = 0u64;

    for length in lengths {
        let end = start + length as u64;
        chunks.push(
            FixedSizeListArray::try_new(
                PrimitiveArray::from_iter(start * 2..end * 2).into_array(),
                2,
                Validity::NonNullable,
                length,
            )?
            .into_array(),
        );
        start = end;
    }

    let dtype = chunks[0].dtype().clone();
    let chunked = ChunkedArray::try_new(chunks, dtype)?;
    let flat = FixedSizeListArray::try_new(
        PrimitiveArray::from_iter(0..start * 2).into_array(),
        2,
        Validity::NonNullable,
        usize::try_from(start)?,
    )?
    .into_array();

    for (indices, expected_chunks) in [
        (contiguous_pieces(&[1], &[3])?, 2usize),
        (contiguous_pieces(&[6, 1, 6], &[2, 2, 1])?, 3usize),
        (contiguous_pieces(&[2], &[2])?, 1usize),
    ] {
        let (touched, remapped) = super::touched_chunks(chunked.as_view(), &indices, &mut ctx)?
            .expect("sparse piecewise indices should select a subset of chunks");
        assert_eq!(touched.len(), expected_chunks);
        assert!(remapped.is::<super::PiecewiseSequence>());

        let result = chunked.clone().into_array().take(indices.clone())?;
        assert_arrays_eq!(result, flat.take(indices)?, &mut ctx);
    }

    let out_of_bounds = contiguous_pieces(&[6], &[3])?;
    assert!(super::touched_chunks(chunked.as_view(), &out_of_bounds, &mut ctx).is_err());
    Ok(())
}

/// A take from struct chunks with dictionary-encoded fixed-size-list fields does not swizzle
/// them. The swizzle would decode every dictionary code of the chunks that the take touches.
#[test]
fn test_take_chunked_struct_of_dict_fsl() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let struct_of_dict_fsl = |values: Vec<u64>, codes: Vec<u32>| -> VortexResult<ArrayRef> {
        let values_len = values.len() / 2;
        let values = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(values).into_array(),
            2,
            Validity::NonNullable,
            values_len,
        )?;
        let length = codes.len();
        let tokens = DictArray::try_new(
            PrimitiveArray::from_iter(codes).into_array(),
            values.into_array(),
        )?;
        Ok(StructArray::try_new(
            ["tokens"].into(),
            vec![tokens.into_array()],
            length,
            Validity::NonNullable,
        )?
        .into_array())
    };

    let s0 = struct_of_dict_fsl(vec![0, 1, 2, 3], vec![1, 0, 1, 1])?;
    let s1 = struct_of_dict_fsl(vec![10, 11, 12, 13, 14, 15], vec![2, 0, 1])?;
    let s2 = struct_of_dict_fsl(vec![20, 21], vec![0, 0])?;
    let dtype = s0.dtype().clone();
    let chunked = ChunkedArray::try_new(vec![s0, s1, s2], dtype)?;
    let flat = chunked
        .clone()
        .into_array()
        .execute::<RecursiveCanonical>(&mut ctx)?
        .0
        .into_array();

    for indices in [
        buffer![0u64, 4].into_array(),
        buffer![6u64, 1, 5].into_array(),
        buffer![8u64, 0, 4].into_array(),
    ] {
        let swizzled = super::take_chunked_struct(chunked.as_view(), &indices, &mut ctx)?;
        assert!(swizzled.is_none());

        let result = chunked.clone().into_array().take(indices.clone())?;
        assert_arrays_eq!(result, flat.take(indices)?, &mut ctx);
    }

    Ok(())
}

#[test]
fn test_take_nullable_empty_field_struct() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let struct_array =
        StructArray::try_new(FieldNames::default(), vec![], 100, Validity::NonNullable)?;
    let arr = ChunkedArray::from_iter(vec![
        struct_array.clone().into_array(),
        struct_array.into_array(),
    ]);

    let result =
        arr.take(PrimitiveArray::from_option_iter(vec![Some(0i32), None, Some(101)]).into_array())?;
    let expect = StructArray::try_new(
        FieldNames::default(),
        vec![],
        3,
        Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array()),
    )?;
    assert_arrays_eq!(result, expect, &mut ctx);
    Ok(())
}

#[test]
fn test_take_out_of_bounds() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let arr = ChunkedArray::try_new(
        vec![
            buffer![0i32, 1, 2, 3, 4].into_array(),
            buffer![5i32, 6, 7, 8, 9].into_array(),
            buffer![10i32, 11, 12, 13, 14].into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;

    // Sorted, unsorted, and piecewise out-of-bounds indices must all return an error.
    for indices in [
        buffer![0u64, 100].into_array(),
        buffer![100u64, 0].into_array(),
        contiguous_pieces(&[12], &[5])?,
    ] {
        let result = arr
            .take(indices)
            .and_then(|taken| taken.execute::<Canonical>(&mut ctx));
        assert!(result.is_err(), "expected out-of-bounds error");
    }

    // Exercise the primitive-index bounds check in `take_chunked_fsl -> touched_chunks`,
    // including empty and untouched chunks.
    let mut fsl_chunks = Vec::new();
    let mut start = 0u64;
    for length in [2usize, 0, 3, 1, 2] {
        let end = start + length as u64;
        fsl_chunks.push(
            FixedSizeListArray::try_new(
                PrimitiveArray::from_iter(start * 2..end * 2).into_array(),
                2,
                Validity::NonNullable,
                length,
            )?
            .into_array(),
        );
        start = end;
    }
    let fsl_dtype = fsl_chunks[0].dtype().clone();
    let fsl_chunked = ChunkedArray::try_new(fsl_chunks, fsl_dtype)?;
    let fsl_result = fsl_chunked
        .take(buffer![1u64, 8].into_array())
        .and_then(|taken| taken.execute::<Canonical>(&mut ctx));
    assert!(fsl_result.is_err(), "expected FSL out-of-bounds error");
    Ok(())
}

#[test]
fn test_take_shuffled_large() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let nchunks: i32 = 100;
    let chunk_len: i32 = 1_000;
    let total = nchunks * chunk_len;

    let chunks: Vec<_> = (0..nchunks)
        .map(|c| {
            let start = c * chunk_len;
            PrimitiveArray::from_iter(start..start + chunk_len).into_array()
        })
        .collect();
    let dtype = chunks[0].dtype().clone();
    let arr = ChunkedArray::try_new(chunks, dtype)?;

    // Fisher-Yates shuffle with a fixed seed for determinism.
    let mut indices: Vec<u64> = (0..u64::try_from(total)?).collect();
    let mut seed: u64 = 0xdeadbeef;
    for i in (1..indices.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (seed >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }

    let expected_vals: Vec<i32> = indices
        .iter()
        .map(|&idx| i32::try_from(idx))
        .collect::<Result<_, _>>()?;
    let expected = PrimitiveArray::from_iter(expected_vals);
    let indices_arr = PrimitiveArray::new(Buffer::from(indices), Validity::NonNullable);
    let result = arr.take(indices_arr.into_array())?;
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_take_fsl_over_chunked_elements_conformance() -> VortexResult<()> {
    let elements = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_iter(0i32..8).into_array(),
            PrimitiveArray::from_iter(8i32..13).into_array(),
            PrimitiveArray::from_iter(13i32..18).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )?
    .into_array();
    let fsl = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 6)?;
    test_take_conformance(&fsl.into_array(), &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_take_chunked_struct_conformance() -> VortexResult<()> {
    let (chunked, _) = chunked_struct_of_fsl()?;
    test_take_conformance(&chunked.into_array(), &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_take_chunked_fsl_conformance() -> VortexResult<()> {
    let c0 = FixedSizeListArray::try_new(
        PrimitiveArray::from_iter(0i32..6).into_array(),
        2,
        Validity::NonNullable,
        3,
    )?;
    let c1 = FixedSizeListArray::try_new(
        PrimitiveArray::from_iter(6i32..10).into_array(),
        2,
        Validity::NonNullable,
        2,
    )?;
    let dtype = c0.dtype().clone();
    let arr = ChunkedArray::try_new(vec![c0.into_array(), c1.into_array()], dtype)?;
    test_take_conformance(&arr.into_array(), &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_take_chunked_conformance() -> VortexResult<()> {
    let a = buffer![1i32, 2, 3].into_array();
    let b = buffer![4i32, 5].into_array();
    let arr = ChunkedArray::try_new(
        vec![a, b],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )?;
    test_take_conformance(&arr.into_array(), &mut SESSION.create_execution_ctx());

    let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]);
    let b = PrimitiveArray::from_option_iter([Some(4i32), Some(5)]);
    let dtype = a.dtype().clone();
    let arr = ChunkedArray::try_new(vec![a.into_array(), b.into_array()], dtype)?;
    test_take_conformance(&arr.into_array(), &mut SESSION.create_execution_ctx());

    let chunk = buffer![10i32, 20, 30, 40, 50].into_array();
    let arr = ChunkedArray::try_new(
        vec![chunk.clone(), chunk.clone(), chunk.clone()],
        chunk.dtype().clone(),
    )?;
    test_take_conformance(&arr.into_array(), &mut SESSION.create_execution_ctx());
    Ok(())
}

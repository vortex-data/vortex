// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::compute::conformance::hegel::EncodingCheckOptions;
use vortex_array::compute::conformance::hegel::check_encoding_roundtrip_and_ops;
use vortex_array::compute::conformance::hegel::draw_bool_pattern;
use vortex_array::compute::conformance::hegel::draw_validity_for;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::Nullability::Nullable;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::patches::Patches;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_buffer::buffer_mut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::Sparse;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    crate::initialize(&session);
    session
});

struct SparseShape {
    len: usize,
    positions: Vec<usize>,
    nullability: Nullability,
    fill_is_null: bool,
    patch_validity: Validity,
    patch_mask: Mask,
}

impl SparseShape {
    fn full_valid_bits(&self) -> Vec<bool> {
        let mut valid = vec![!self.fill_is_null; self.len];
        for (k, &pos) in self.positions.iter().enumerate() {
            valid[pos] = self.patch_mask.value(k);
        }
        valid
    }

    fn full_validity(&self) -> Validity {
        if self.nullability == NonNullable {
            Validity::NonNullable
        } else {
            Validity::from_iter(self.full_valid_bits())
        }
    }

    fn scatter<T: Clone>(&self, fill: T, patches: &[T]) -> Vec<T> {
        let mut out = vec![fill; self.len];
        for (k, &pos) in self.positions.iter().enumerate() {
            out[pos] = patches[k].clone();
        }
        out
    }
}

fn draw_patch_indices(tc: &TestCase, len: usize, offset: usize) -> (Vec<usize>, ArrayRef) {
    let mut positions = Vec::new();
    match tc.draw(gs::integers::<u8>().max_value(3)) {
        0 => positions.push(tc.draw(gs::integers::<usize>().max_value(len - 1))),
        1 => {
            for i in 0..len {
                if tc.draw(gs::integers::<u8>().max_value(4)) == 0 {
                    positions.push(i);
                }
            }
        }
        2 => {
            for i in 0..len {
                if tc.draw(gs::booleans()) {
                    positions.push(i);
                }
            }
        }
        _ => positions.extend(0..len),
    }
    if positions.is_empty() {
        positions.push(tc.draw(gs::integers::<usize>().max_value(len - 1)));
    }

    let max_idx = positions.last().copied().unwrap_or(0) + offset;
    let mut widths = vec![PType::U64, PType::U32];
    if max_idx <= u16::MAX as usize {
        widths.push(PType::U16);
    }
    if max_idx <= u8::MAX as usize {
        widths.push(PType::U8);
    }
    let idx_array = match tc.draw(gs::sampled_from(&widths)) {
        PType::U8 => PrimitiveArray::from_iter(
            positions
                .iter()
                .map(|&i| u8::try_from(i + offset).vortex_expect("fits u8")),
        )
        .into_array(),
        PType::U16 => PrimitiveArray::from_iter(
            positions
                .iter()
                .map(|&i| u16::try_from(i + offset).vortex_expect("fits u16")),
        )
        .into_array(),
        PType::U32 => PrimitiveArray::from_iter(
            positions
                .iter()
                .map(|&i| u32::try_from(i + offset).vortex_expect("fits u32")),
        )
        .into_array(),
        _ => PrimitiveArray::from_iter(positions.iter().map(|&i| (i + offset) as u64)).into_array(),
    };
    (positions, idx_array)
}

fn draw_sparse_strings(tc: &TestCase, shape: &SparseShape) -> (ArrayRef, Scalar, ArrayRef) {
    const SAMPLES: [&str; 5] = [
        "",
        "short",
        "hello",
        "a_long_non_inlinable_string_payload",
        "utf8_你好_long_payload_value",
    ];
    let is_binary = tc.draw(gs::booleans());
    let use_varbin = tc.draw(gs::booleans());
    let fill_str = tc.draw(gs::sampled_from(&SAMPLES));
    let patch_strs: Vec<&str> = (0..shape.positions.len())
        .map(|_| tc.draw(gs::sampled_from(&SAMPLES)))
        .collect();
    let full_strs = shape.scatter(fill_str, &patch_strs);
    let full_valid = shape.full_valid_bits();

    if is_binary {
        let dtype = DType::Binary(shape.nullability);
        let fill_scalar = if shape.fill_is_null {
            Scalar::null(dtype.clone())
        } else {
            Scalar::binary(
                ByteBuffer::from(fill_str.as_bytes().to_vec()),
                shape.nullability,
            )
        };
        let patches = if use_varbin {
            VarBinArray::from_iter(
                patch_strs
                    .iter()
                    .enumerate()
                    .map(|(i, val)| shape.patch_mask.value(i).then_some(val.as_bytes())),
                dtype.clone(),
            )
            .into_array()
        } else {
            VarBinViewArray::from_iter(
                patch_strs
                    .iter()
                    .enumerate()
                    .map(|(i, val)| shape.patch_mask.value(i).then_some(val.as_bytes())),
                dtype.clone(),
            )
            .into_array()
        };
        let canonical = VarBinViewArray::from_iter(
            full_strs
                .iter()
                .enumerate()
                .map(|(i, val)| full_valid[i].then_some(val.as_bytes())),
            dtype,
        )
        .into_array();
        (patches, fill_scalar, canonical)
    } else {
        let dtype = DType::Utf8(shape.nullability);
        let fill_scalar = if shape.fill_is_null {
            Scalar::null(dtype.clone())
        } else {
            Scalar::utf8(fill_str, shape.nullability)
        };
        let patches = if use_varbin {
            VarBinArray::from_iter(
                patch_strs
                    .iter()
                    .enumerate()
                    .map(|(i, val)| shape.patch_mask.value(i).then_some(*val)),
                dtype.clone(),
            )
            .into_array()
        } else {
            VarBinViewArray::from_iter(
                patch_strs
                    .iter()
                    .enumerate()
                    .map(|(i, val)| shape.patch_mask.value(i).then_some(*val)),
                dtype.clone(),
            )
            .into_array()
        };
        let canonical = VarBinViewArray::from_iter(
            full_strs
                .iter()
                .enumerate()
                .map(|(i, val)| full_valid[i].then_some(*val)),
            dtype,
        )
        .into_array();
        (patches, fill_scalar, canonical)
    }
}

fn draw_sparse_struct(
    tc: &TestCase,
    shape: &SparseShape,
) -> VortexResult<(ArrayRef, Scalar, ArrayRef)> {
    let num_patches = shape.positions.len();
    let fields = StructFields::new(
        FieldNames::from(["a", "b"]),
        vec![
            DType::Primitive(PType::I32, Nullable),
            DType::Primitive(PType::I32, NonNullable),
        ],
    );
    let struct_dtype = DType::Struct(fields.clone(), shape.nullability);
    let fill_a = tc
        .draw(gs::booleans())
        .then(|| tc.draw(gs::integers::<i32>()));
    let fill_b = tc.draw(gs::integers::<i32>());
    let fill_scalar = if shape.fill_is_null {
        Scalar::null(struct_dtype)
    } else {
        Scalar::struct_(
            struct_dtype,
            vec![
                match fill_a {
                    Some(val) => Scalar::primitive(val, Nullable),
                    None => Scalar::null(DType::Primitive(PType::I32, Nullable)),
                },
                Scalar::primitive(fill_b, NonNullable),
            ],
        )
    };
    let patch_a: Vec<Option<i32>> = (0..num_patches)
        .map(|_| {
            tc.draw(gs::booleans())
                .then(|| tc.draw(gs::integers::<i32>()))
        })
        .collect();
    let patch_b = tc.draw(
        gs::vecs(gs::integers::<i32>())
            .min_size(num_patches)
            .max_size(num_patches),
    );
    let patch_struct = StructArray::try_new_with_dtype(
        vec![
            PrimitiveArray::from_option_iter(patch_a.iter().copied()).into_array(),
            PrimitiveArray::new(Buffer::from(patch_b.clone()), Validity::NonNullable).into_array(),
        ],
        fields.clone(),
        num_patches,
        shape.patch_validity.clone(),
    )?
    .into_array();
    let canonical = StructArray::try_new_with_dtype(
        vec![
            PrimitiveArray::from_option_iter(shape.scatter(fill_a, &patch_a)).into_array(),
            PrimitiveArray::new(
                Buffer::from(shape.scatter(fill_b, &patch_b)),
                Validity::NonNullable,
            )
            .into_array(),
        ],
        fields,
        shape.len,
        shape.full_validity(),
    )?
    .into_array();
    Ok((patch_struct, fill_scalar, canonical))
}

fn draw_sparse_fsl(tc: &TestCase, s: &SparseShape) -> VortexResult<(ArrayRef, Scalar, ArrayRef)> {
    let n = s.positions.len();
    let list_size = tc.draw(gs::integers::<u32>().min_value(1).max_value(3));
    let sz = list_size as usize;
    let elem_dtype = Arc::new(DType::Primitive(PType::I32, NonNullable));
    let fill_elems = tc.draw(gs::vecs(gs::integers::<i32>()).min_size(sz).max_size(sz));
    let fill_scalar = if s.fill_is_null {
        Scalar::null(DType::FixedSizeList(elem_dtype, list_size, Nullable))
    } else {
        Scalar::fixed_size_list(
            elem_dtype,
            fill_elems
                .iter()
                .map(|&v| Scalar::primitive(v, NonNullable))
                .collect(),
            s.nullability,
        )
    };
    let patch_elems = tc.draw(
        gs::vecs(gs::integers::<i32>())
            .min_size(n * sz)
            .max_size(n * sz),
    );
    let patch_fsl = FixedSizeListArray::try_new(
        PrimitiveArray::new(Buffer::from(patch_elems.clone()), Validity::NonNullable).into_array(),
        list_size,
        s.patch_validity.clone(),
        n,
    )?
    .into_array();
    let mut full_elems = Vec::with_capacity(s.len * sz);
    for _ in 0..s.len {
        full_elems.extend_from_slice(&fill_elems);
    }
    for (k, &pos) in s.positions.iter().enumerate() {
        full_elems[pos * sz..(pos + 1) * sz].copy_from_slice(&patch_elems[k * sz..(k + 1) * sz]);
    }
    let canonical = FixedSizeListArray::try_new(
        PrimitiveArray::new(Buffer::from(full_elems), Validity::NonNullable).into_array(),
        list_size,
        s.full_validity(),
        s.len,
    )?
    .into_array();
    Ok((patch_fsl, fill_scalar, canonical))
}

fn build_listview(
    lists: &[Vec<i32>],
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut flat = Vec::new();
    let mut offsets = Vec::with_capacity(lists.len());
    let mut sizes = Vec::with_capacity(lists.len());
    for item in lists {
        offsets.push(u32::try_from(flat.len())?);
        sizes.push(u32::try_from(item.len())?);
        flat.extend_from_slice(item);
    }
    Ok(ListViewArray::new(
        PrimitiveArray::new(Buffer::from(flat), Validity::NonNullable).into_array(),
        PrimitiveArray::new(Buffer::from(offsets), Validity::NonNullable).into_array(),
        PrimitiveArray::new(Buffer::from(sizes), Validity::NonNullable).into_array(),
        validity,
        ctx,
    )
    .into_array())
}

fn draw_sparse_listview(
    tc: &TestCase,
    s: &SparseShape,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, Scalar, ArrayRef)> {
    let elem_dtype = Arc::new(DType::Primitive(PType::I32, NonNullable));
    let fill_list = tc.draw(gs::vecs(gs::integers::<i32>()).max_size(3));
    let fill_scalar = if s.fill_is_null {
        Scalar::null(DType::List(elem_dtype, Nullable))
    } else {
        Scalar::list(
            elem_dtype,
            fill_list
                .iter()
                .map(|&v| Scalar::primitive(v, NonNullable))
                .collect(),
            s.nullability,
        )
    };
    let patch_lists: Vec<Vec<i32>> = (0..s.positions.len())
        .map(|_| tc.draw(gs::vecs(gs::integers::<i32>()).max_size(3)))
        .collect();
    // Occasionally slice the patch ListView so non-zero source offsets are exercised too.
    let patch_lv = if tc.draw(gs::booleans()) {
        let mut padded = vec![vec![999i32, 998]];
        padded.extend(patch_lists.iter().cloned());
        let padded_validity = if s.nullability == NonNullable {
            Validity::NonNullable
        } else {
            Validity::from_iter(
                std::iter::once(true).chain((0..s.positions.len()).map(|i| s.patch_mask.value(i))),
            )
        };
        build_listview(&padded, padded_validity, ctx)?
            .slice(1..1 + s.positions.len())?
            .execute::<ListViewArray>(ctx)?
            .into_array()
    } else {
        build_listview(&patch_lists, s.patch_validity.clone(), ctx)?
    };
    let canonical = build_listview(&s.scatter(fill_list, &patch_lists), s.full_validity(), ctx)?;
    Ok((patch_lv, fill_scalar, canonical))
}

fn draw_sparse_case(tc: &TestCase, ctx: &mut ExecutionCtx) -> VortexResult<(ArrayRef, ArrayRef)> {
    let len = tc.draw(gs::integers::<usize>().min_value(1).max_value(48));
    let offset = if tc.draw(gs::integers::<u8>().max_value(2)) == 0 {
        tc.draw(gs::integers::<usize>().min_value(1).max_value(16))
    } else {
        0
    };
    let nullability = if tc.draw(gs::booleans()) {
        Nullable
    } else {
        NonNullable
    };
    let fill_is_null = nullability == Nullable && tc.draw(gs::booleans());
    let (positions, indices_arr) = draw_patch_indices(tc, len, offset);
    let patch_validity = draw_validity_for(tc, nullability, positions.len());
    let patch_mask = patch_validity.execute_mask(positions.len(), ctx)?;
    let s = SparseShape {
        len,
        positions,
        nullability,
        fill_is_null,
        patch_validity,
        patch_mask,
    };
    let n = s.positions.len();

    let (patch_values, fill_scalar, canonical) = match tc.draw(gs::integers::<u8>().max_value(7)) {
        0 => {
            let fill = tc.draw(gs::integers::<i32>());
            let mut vals = tc.draw(gs::vecs(gs::integers::<i32>()).min_size(n).max_size(n));
            if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                vals.sort_unstable();
            }
            let fill_scalar = if s.fill_is_null {
                Scalar::null(DType::Primitive(PType::I32, Nullable))
            } else {
                Scalar::primitive(fill, s.nullability)
            };
            let full = s.scatter(fill, &vals);
            (
                PrimitiveArray::new(Buffer::from(vals), s.patch_validity.clone()).into_array(),
                fill_scalar,
                PrimitiveArray::new(Buffer::from(full), s.full_validity()).into_array(),
            )
        }
        1 => {
            const SPECIALS: [f64; 6] = [0.0, -0.0, 1.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
            let draw_f = |tc: &TestCase| {
                if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                    tc.draw(gs::sampled_from(&SPECIALS))
                } else {
                    tc.draw(gs::floats::<f64>())
                }
            };
            let fill = draw_f(tc);
            let vals: Vec<f64> = (0..n).map(|_| draw_f(tc)).collect();
            let fill_scalar = if s.fill_is_null {
                Scalar::null(DType::Primitive(PType::F64, Nullable))
            } else {
                Scalar::primitive(fill, s.nullability)
            };
            let full = s.scatter(fill, &vals);
            (
                PrimitiveArray::new(Buffer::from(vals), s.patch_validity.clone()).into_array(),
                fill_scalar,
                PrimitiveArray::new(Buffer::from(full), s.full_validity()).into_array(),
            )
        }
        2 => {
            let fill = tc.draw(gs::booleans());
            let vals = draw_bool_pattern(tc, n);
            let fill_scalar = if s.fill_is_null {
                Scalar::null(DType::Bool(Nullable))
            } else {
                Scalar::bool(fill, s.nullability)
            };
            let full = s.scatter(fill, &vals);
            (
                BoolArray::new(BitBuffer::from_iter(vals), s.patch_validity.clone()).into_array(),
                fill_scalar,
                BoolArray::new(BitBuffer::from_iter(full), s.full_validity()).into_array(),
            )
        }
        3 => draw_sparse_strings(tc, &s),
        4 => {
            let dec = DecimalDType::new(9, 2);
            let fill = tc.draw(gs::integers::<i32>().min_value(-999_999).max_value(999_999));
            let vals = tc.draw(
                gs::vecs(gs::integers::<i32>().min_value(-999_999).max_value(999_999))
                    .min_size(n)
                    .max_size(n),
            );
            let fill_scalar = if s.fill_is_null {
                Scalar::null(DType::Decimal(dec, Nullable))
            } else {
                Scalar::decimal(DecimalValue::I32(fill), dec, s.nullability)
            };
            let full = s.scatter(fill, &vals);
            (
                DecimalArray::new(Buffer::from(vals), dec, s.patch_validity.clone(), ctx)
                    .into_array(),
                fill_scalar,
                DecimalArray::new(Buffer::from(full), dec, s.full_validity(), ctx).into_array(),
            )
        }
        5 => draw_sparse_struct(tc, &s)?,
        6 => draw_sparse_fsl(tc, &s)?,
        _ => draw_sparse_listview(tc, &s, ctx)?,
    };

    let patches = Patches::new(len, offset, indices_arr, patch_values, None, ctx)?;
    let sparse = Sparse::try_new_from_patches(patches, fill_scalar)?.into_array();
    Ok((sparse, canonical))
}

#[hegel::test(test_cases = 64, print_blob = true)]
fn test_sparse_hegel_encoding_harness(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let (encoded, canonical) = draw_sparse_case(&tc, &mut ctx).vortex_expect("draw sparse case");
    check_encoding_roundtrip_and_ops(
        &tc,
        &encoded,
        &canonical,
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
    .vortex_expect("Sparse harness check");
}

// Regression test for https://github.com/vortex-data/vortex/issues/12035.
#[rstest]
#[case::valid_fill(true)]
#[case::null_fill(false)]
fn test_sparse_struct_no_fields(#[case] fill_valid: bool) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let struct_fields = StructFields::empty();
    let struct_dtype = DType::Struct(struct_fields.clone(), Nullable);

    let indices = buffer![0u64, 1, 7].into_array();
    let patch_values = StructArray::try_new_with_dtype(
        Vec::new(),
        struct_fields.clone(),
        3,
        Validity::from_iter([true, false, true]),
    )?
    .into_array();

    let fill_scalar = if fill_valid {
        Scalar::struct_(struct_dtype, Vec::new())
    } else {
        Scalar::null(struct_dtype)
    };
    let len = 10;
    let sparse_struct = Sparse::try_new(indices, patch_values, len, fill_scalar, &mut ctx)?;

    let valid = (0..len).map(|i| match i {
        0 | 7 => true,
        1 => false,
        _ => fill_valid,
    });
    let expected = StructArray::try_new_with_dtype(
        Vec::new(),
        struct_fields,
        len,
        Validity::from_iter(valid),
    )?;

    assert_arrays_eq!(sparse_struct, expected, &mut ctx);
    Ok(())
}

/// Each gap between patches is appended as a single constant array, so the fill value's
/// elements are stored once per gap however many rows the gap covers.
#[test]
fn test_sparse_list_fill_stores_one_copy_per_gap() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let lists = unsafe {
        ListViewArray::new_unchecked(
            buffer![1i32, 2].into_array(),
            buffer![0u32, 1].into_array(),
            buffer![1u32, 1].into_array(),
            Validity::AllValid,
        )
        .with_zero_copy_to_list(true)
    }
    .into_array();

    let indices = buffer![10u32, 20].into_array();
    let fill = vec![7i32, 8, 9];
    let sparse = Sparse::try_new(
        indices,
        lists,
        10_000,
        Scalar::from(Some(fill.clone())),
        &mut ctx,
    )?
    .into_array();

    let actual = sparse.execute::<ListViewArray>(&mut ctx)?;
    assert_eq!(actual.len(), 10_000);
    assert_eq!(
        actual.elements().len(),
        2 + 3 * fill.len(),
        "the fill value should be stored once per gap, not once per row",
    );

    let fill_elements = PrimitiveArray::from_iter(fill);
    for index in [0, 9, 11, 19, 21, 9_999] {
        assert_arrays_eq!(
            actual
                .list_elements_at(index, &mut ctx)
                .vortex_expect("fill list"),
            fill_elements,
            &mut ctx
        );
    }
    assert_arrays_eq!(
        actual
            .list_elements_at(10, &mut ctx)
            .vortex_expect("patch list"),
        PrimitiveArray::from_iter([1i32]),
        &mut ctx
    );
    assert_arrays_eq!(
        actual
            .list_elements_at(20, &mut ctx)
            .vortex_expect("patch list"),
        PrimitiveArray::from_iter([2i32]),
        &mut ctx
    );

    Ok(())
}

/// Nested builders chunk a child on the boundaries it is appended on, so the number of appends
/// canonicalization makes is visible in the elements child.
#[test]
fn test_sparse_list_bounds_element_chunks() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    const PATCHES: usize = 100;
    let patches_u32 = u32::try_from(PATCHES).vortex_expect("fits in u32");
    let patches_i32 = i32::try_from(PATCHES).vortex_expect("fits in i32");

    let patch_values = ListViewArray::new(
        PrimitiveArray::from_iter(0..patches_i32).into_array(),
        PrimitiveArray::from_iter(0..patches_u32).into_array(),
        PrimitiveArray::from_iter(std::iter::repeat_n(1u32, PATCHES)).into_array(),
        Validity::AllValid,
        &mut ctx,
    )
    .into_array();

    let indices = PrimitiveArray::from_iter(0..patches_u32).into_array();
    let fill = Scalar::from(Some(vec![-1i32]));
    let sparse = Sparse::try_new(indices, patch_values, 2 * PATCHES, fill, &mut ctx)?.into_array();

    let actual = sparse.execute::<ListViewArray>(&mut ctx)?;
    let chunks = actual
        .elements()
        .as_opt::<Chunked>()
        .map_or(1, |array| array.nchunks());
    assert!(
        chunks <= PATCHES + 1,
        "fill rows must not each add an element chunk"
    );

    let expected_lists = (0..patches_i32)
        .map(|i| Some(vec![i]))
        .chain(std::iter::repeat_n(Some(vec![-1i32]), PATCHES));
    let expected =
        ListArray::from_iter_opt_slow::<u32, _, _>(expected_lists, Arc::new(PType::I32.into()))?;
    assert_arrays_eq!(actual, expected, &mut ctx);

    Ok(())
}

/// Patches on consecutive rows are appended as one slice of the patch array, covering null
/// patches inside a run, overlapping out-of-order views, gaps, and trailing gaps.
#[test]
fn test_sparse_list_appends_consecutive_patches_as_one_run() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let patches = unsafe {
        ListViewArray::new_unchecked(
            buffer![10i32, 20, 30].into_array(),
            buffer![1u32, 0, 0, 0].into_array(),
            buffer![2u32, 1, 3, 2].into_array(),
            Validity::from_iter([true, false, true, true]),
        )
    };
    assert!(!patches.is_zero_copy_to_list());

    let indices = buffer![0u8, 1, 2, 5].into_array();
    let fill = Scalar::from(Some(vec![7i32, 8]));
    let sparse = Sparse::try_new(indices, patches.into_array(), 7, fill, &mut ctx)?.into_array();

    let actual = sparse.execute::<ListViewArray>(&mut ctx)?;
    assert_eq!(actual.elements().len(), 7 + 2 * 2);

    let expected = ListViewArray::new(
        buffer![20i32, 30, 10, 20, 30, 7, 8, 7, 8, 10, 20, 7, 8].into_array(),
        buffer![0u8, 2, 2, 5, 7, 9, 11].into_array(),
        buffer![2u8, 0, 3, 2, 2, 2, 2].into_array(),
        Validity::from_iter([true, false, true, true, true, true, true]),
        &mut ctx,
    );
    assert_arrays_eq!(actual, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_sparse_list_grows_offset_type() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let elements = buffer![1i32, 2, 1, 2].into_array();
    let offsets = buffer![0u8, 1, 2, 3, 4].into_array();
    let lists = ListArray::try_new(elements, offsets, Validity::AllValid, &mut ctx)?.into_array();

    let indices = buffer![0u8, 1u8, 2u8, 3u8].into_array();
    let fill_value = Scalar::from(Some(vec![42i32; 252]));
    let sparse = Sparse::try_new(indices, lists, 5, fill_value, &mut ctx)?.into_array();

    let actual = sparse.execute::<Canonical>(&mut ctx)?.into_array();
    let mut expected_elements = buffer_mut![1, 2, 1, 2];
    expected_elements.extend(buffer![42i32; 252]);
    let expected = ListArray::try_new(
        expected_elements.freeze().into_array(),
        buffer![0u32, 1, 2, 3, 4, 256].into_array(),
        Validity::AllValid,
        &mut ctx,
    )?
    .into_array();

    let actual_listview = actual.clone().execute::<ListViewArray>(&mut ctx)?;
    assert_eq!(
        actual_listview.offsets().dtype(),
        &DType::Primitive(PType::U32, NonNullable)
    );
    assert_arrays_eq!(&actual, &expected, &mut ctx);

    let arrow_dtype = SESSION.arrow().to_arrow_field("", expected.dtype())?;
    let actual = actual.execute_arrow(Some(&arrow_dtype), &mut ctx)?;
    let expected = expected.execute_arrow(Some(&arrow_dtype), &mut ctx)?;
    assert_eq!(actual.data_type(), expected.data_type());
    Ok(())
}

/// Appending a sparse string array to either variable-binary builder must match its
/// canonical decode across long, short (inlinable), and null fill values and non-zero offsets.
#[rstest]
#[case::long_fill(Some("a fill value that is too long to inline"))]
#[case::short_fill(Some("123"))]
#[case::null_fill(None)]
fn test_sparse_append_to_string_builders(#[case] fill: Option<&str>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let long = "a patch value that is far too long to inline";
    let strings = <VarBinViewArray as FromIterator<_>>::from_iter([Some(long), None, Some("tiny")])
        .into_array();
    let fill_scalar = match fill {
        Some(value) => Scalar::from(value.to_owned()).into_nullable(),
        None => Scalar::null(DType::Utf8(Nullable)),
    };
    let array = Sparse::try_new(
        buffer![1u16, 4, 8].into_array(),
        strings,
        10,
        fill_scalar,
        &mut ctx,
    )?
    .into_array();

    let expected_values = [
        fill,
        Some(long),
        fill,
        fill,
        None,
        fill,
        fill,
        fill,
        Some("tiny"),
        fill,
    ];
    for (candidate, range) in [(array.clone(), 0..10), (array.slice(1..9)?, 1..9)] {
        let expected =
            VarBinViewArray::from_iter_nullable_str(expected_values[range].iter().copied());
        assert_arrays_eq!(candidate, expected, &mut ctx);

        let mut view_builder = VarBinViewBuilder::with_capacity_in(
            candidate.dtype().clone(),
            4,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        candidate.append_to_builder(&mut view_builder, &mut ctx)?;
        assert_arrays_eq!(view_builder.finish_into_varbinview(), expected, &mut ctx);

        let mut varbin_builder = VarBinBuilder::<i32>::new_in(
            candidate.dtype().clone(),
            vortex_buffer::BufferAllocatorRef::static_ref(),
        );
        candidate.append_to_builder(&mut varbin_builder, &mut ctx)?;
        assert_arrays_eq!(varbin_builder.finish_into_varbin(), expected, &mut ctx);
    }
    Ok(())
}

/// A fill value short enough to inline lives in its views; the canonical decode must not
/// retain a data buffer nothing references.
#[test]
fn test_sparse_short_fill_pushes_no_buffer() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let strings = VarBinViewArray::from_iter_str(["hello", "goodbye"]);
    assert!(strings.data_buffers().is_empty());

    let array = Sparse::try_new(
        buffer![1u16, 3].into_array(),
        strings.into_array(),
        6,
        Scalar::from("123".to_owned()),
        &mut ctx,
    )?;

    let actual = array
        .as_array()
        .clone()
        .execute::<VarBinViewArray>(&mut ctx)?;
    assert!(
        actual.data_buffers().is_empty(),
        "an inlinable fill value must not push an unreferenced buffer"
    );

    let expected = VarBinViewArray::from_iter_str(["123", "hello", "123", "goodbye", "123", "123"]);
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

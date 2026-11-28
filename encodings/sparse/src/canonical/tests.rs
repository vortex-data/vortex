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

#[rstest]
#[case(Some(true))]
#[case(Some(false))]
#[case(None)]
fn test_sparse_bool(#[case] fill_value: Option<bool>) {
    let mut ctx = SESSION.create_execution_ctx();
    let indices = buffer![0u64, 1, 7].into_array();
    let values = BoolArray::from_iter([Some(true), None, Some(false)]).into_array();
    let sparse_bools =
        Sparse::try_new(indices, values, 10, Scalar::from(fill_value), &mut ctx).unwrap();

    let expected = BoolArray::from_iter([
        Some(true),
        None,
        fill_value,
        fill_value,
        fill_value,
        fill_value,
        fill_value,
        Some(false),
        fill_value,
        fill_value,
    ]);

    assert_arrays_eq!(sparse_bools, expected, &mut ctx);
}

#[rstest]
#[case(Some(0i32))]
#[case(Some(-1i32))]
#[case(None)]
fn test_sparse_primitive(#[case] fill_value: Option<i32>) {
    let mut ctx = SESSION.create_execution_ctx();
    let indices = buffer![0u64, 1, 7].into_array();
    let values = PrimitiveArray::from_option_iter([Some(0i32), None, Some(1)]).into_array();
    let sparse_ints =
        Sparse::try_new(indices, values, 10, Scalar::from(fill_value), &mut ctx).unwrap();
    assert_eq!(*sparse_ints.dtype(), DType::Primitive(PType::I32, Nullable));

    let expected = PrimitiveArray::from_option_iter([
        Some(0i32),
        None,
        fill_value,
        fill_value,
        fill_value,
        fill_value,
        fill_value,
        Some(1),
        fill_value,
        fill_value,
    ]);

    assert_arrays_eq!(&sparse_ints, &expected, &mut ctx);
}

#[test]
fn test_sparse_struct_valid_fill() {
    let mut ctx = SESSION.create_execution_ctx();
    let field_names = FieldNames::from_iter(["a", "b"]);
    let field_types = vec![
        DType::Primitive(PType::I32, Nullable),
        DType::Primitive(PType::I32, Nullable),
    ];
    let struct_fields = StructFields::new(field_names, field_types);
    let struct_dtype = DType::Struct(struct_fields.clone(), Nullable);

    let indices = buffer![0u64, 1, 7, 8].into_array();
    let patch_values_a =
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(20), Some(30)]).into_array();
    let patch_values_b =
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(3)]).into_array();
    let patch_values = StructArray::try_new_with_dtype(
        vec![patch_values_a, patch_values_b],
        struct_fields.clone(),
        4,
        Validity::Array(
            BoolArray::from_indices(4, vec![0, 1, 2], Validity::NonNullable).into_array(),
        ),
    )
    .unwrap()
    .into_array();

    let fill_scalar = Scalar::struct_(
        struct_dtype,
        vec![Scalar::from(Some(-10i32)), Scalar::from(Some(-1i32))],
    );
    let len = 10;
    let sparse_struct = Sparse::try_new(indices, patch_values, len, fill_scalar, &mut ctx).unwrap();

    let expected_a = PrimitiveArray::from_option_iter((0..len).map(|i| {
        if i == 0 {
            Some(10)
        } else if i == 1 {
            None
        } else if i == 7 {
            Some(20)
        } else {
            Some(-10)
        }
    }));
    let expected_b = PrimitiveArray::from_option_iter((0..len).map(|i| {
        if i == 0 {
            Some(1i32)
        } else if i == 1 {
            Some(2)
        } else if i == 7 {
            None
        } else {
            Some(-1)
        }
    }));

    let expected = StructArray::try_new_with_dtype(
        vec![expected_a.into_array(), expected_b.into_array()],
        struct_fields,
        len,
        // NB: patch indices: [0, 1, 7, 8]; patch validity: [Valid, Valid, Valid, Invalid]; ergo 8 is Invalid.
        Validity::from_mask(Mask::from_excluded_indices(10, vec![8]), Nullable),
    )
    .unwrap()
    .into_array();

    assert_arrays_eq!(sparse_struct, expected, &mut ctx);
}

#[test]
fn test_sparse_struct_non_nullable() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let patch_values = StructArray::from_fields(&[("a", buffer![10i32].into_array())])?;
    let fill_scalar = Scalar::default_value(patch_values.dtype());
    let sparse_struct = Sparse::try_new(
        buffer![0u64].into_array(),
        patch_values.into_array(),
        3,
        fill_scalar,
        &mut ctx,
    )?;

    let expected = StructArray::from_fields(&[("a", buffer![10i32, 0, 0].into_array())])?;
    assert_arrays_eq!(sparse_struct, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_sparse_struct_invalid_fill() {
    let mut ctx = SESSION.create_execution_ctx();
    let field_names = FieldNames::from_iter(["a", "b"]);
    let field_types = vec![
        DType::Primitive(PType::I32, Nullable),
        DType::Primitive(PType::I32, Nullable),
    ];
    let struct_fields = StructFields::new(field_names, field_types);
    let struct_dtype = DType::Struct(struct_fields.clone(), Nullable);

    let indices = buffer![0u64, 1, 7, 8].into_array();
    let patch_values_a =
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(20), Some(30)]).into_array();
    let patch_values_b =
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(3)]).into_array();
    let patch_values = StructArray::try_new_with_dtype(
        vec![patch_values_a, patch_values_b],
        struct_fields.clone(),
        4,
        Validity::Array(
            BoolArray::from_indices(4, vec![0, 1, 2], Validity::NonNullable).into_array(),
        ),
    )
    .unwrap()
    .into_array();

    let fill_scalar = Scalar::null(struct_dtype);
    let len = 10;
    let sparse_struct = Sparse::try_new(indices, patch_values, len, fill_scalar, &mut ctx).unwrap();

    let expected_a = PrimitiveArray::from_option_iter((0..len).map(|i| {
        if i == 0 {
            Some(10)
        } else if i == 1 {
            None
        } else if i == 7 {
            Some(20)
        } else {
            Some(-10)
        }
    }));
    let expected_b = PrimitiveArray::from_option_iter((0..len).map(|i| {
        if i == 0 {
            Some(1i32)
        } else if i == 1 {
            Some(2)
        } else if i == 7 {
            None
        } else {
            Some(-1)
        }
    }));

    let expected = StructArray::try_new_with_dtype(
        vec![expected_a.into_array(), expected_b.into_array()],
        struct_fields,
        len,
        // NB: patch indices: [0, 1, 7, 8]; patch validity: [Valid, Valid, Valid, Invalid]; ergo 0, 1, 7 are valid.
        Validity::from_mask(Mask::from_indices(10, vec![0, 1, 7]), Nullable),
    )
    .unwrap()
    .into_array();

    assert_arrays_eq!(sparse_struct, expected, &mut ctx);
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

#[test]
fn test_sparse_decimal() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let indices = buffer![0u32, 1, 7, 8].into_array();
    let decimal_dtype = DecimalDType::new(3, 2);
    let patch_values = DecimalArray::new(
        buffer![100i16, 200, 300, 4000],
        decimal_dtype,
        Validity::from_iter([true, true, true, false]),
        &mut ctx,
    )
    .into_array();
    let fill_scalar = Scalar::decimal(DecimalValue::I32(123), decimal_dtype, Nullable);
    let sparse = Sparse::try_new(indices, patch_values, 10, fill_scalar, &mut ctx)?;
    let expected = DecimalArray::new(
        buffer![100i16, 200, 123, 123, 123, 123, 123, 300, 0, 123],
        decimal_dtype,
        Validity::from_mask(Mask::from_excluded_indices(10, vec![8]), Nullable),
        &mut ctx,
    );
    assert_arrays_eq!(sparse, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_sparse_utf8_varbinview_non_null_fill() {
    let mut ctx = SESSION.create_execution_ctx();
    let strings = <VarBinViewArray as FromIterator<_>>::from_iter([
        Some("hello"),
        Some("goodbye"),
        Some("hello"),
        None,
        Some("bonjour"),
        Some("你好"),
        None,
    ])
    .into_array();

    let array = Sparse::try_new(
        buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
        strings,
        12,
        Scalar::from(Some("123".to_owned())),
        &mut ctx,
    )
    .unwrap();

    let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
        Some("hello"),
        Some("123"),
        Some("123"),
        Some("goodbye"),
        Some("hello"),
        None,
        Some("123"),
        Some("bonjour"),
        Some("123"),
        Some("你好"),
        None,
        Some("123"),
    ])
    .into_array();

    assert_arrays_eq!(array, expected, &mut ctx);
}

#[test]
fn test_sparse_utf8_varbinview_null_fill() {
    let mut ctx = SESSION.create_execution_ctx();
    let strings = <VarBinViewArray as FromIterator<_>>::from_iter([
        Some("hello"),
        Some("goodbye"),
        Some("hello"),
        None,
        Some("bonjour"),
        Some("你好"),
        None,
    ])
    .into_array();

    let array = Sparse::try_new(
        buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
        strings,
        12,
        Scalar::null(DType::Utf8(Nullable)),
        &mut ctx,
    )
    .unwrap();

    let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
        Some("hello"),
        None,
        None,
        Some("goodbye"),
        Some("hello"),
        None,
        None,
        Some("bonjour"),
        None,
        Some("你好"),
        None,
        None,
    ])
    .into_array();

    assert_arrays_eq!(array, expected, &mut ctx);
}

#[test]
fn test_sparse_utf8_varbinview_non_nullable() {
    let mut ctx = SESSION.create_execution_ctx();
    let strings = VarBinViewArray::from_iter_str(["hello", "goodbye", "hello", "bonjour", "你好"])
        .into_array();

    let array = Sparse::try_new(
        buffer![0u16, 3, 4, 5, 8].into_array(),
        strings,
        9,
        Scalar::from("123".to_owned()),
        &mut ctx,
    )
    .unwrap();

    let expected = VarBinViewArray::from_iter_str([
        "hello", "123", "123", "goodbye", "hello", "bonjour", "123", "123", "你好",
    ])
    .into_array();

    assert_arrays_eq!(array, expected, &mut ctx);
}

#[test]
fn test_sparse_utf8_varbin_null_fill() {
    let mut ctx = SESSION.create_execution_ctx();
    let strings = <VarBinArray as FromIterator<_>>::from_iter([
        Some("hello"),
        Some("goodbye"),
        Some("hello"),
        None,
        Some("bonjour"),
        Some("你好"),
        None,
    ])
    .into_array();

    let array = Sparse::try_new(
        buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
        strings,
        12,
        Scalar::null(DType::Utf8(Nullable)),
        &mut ctx,
    )
    .unwrap();

    let expected = <VarBinViewArray as FromIterator<_>>::from_iter([
        Some("hello"),
        None,
        None,
        Some("goodbye"),
        Some("hello"),
        None,
        None,
        Some("bonjour"),
        None,
        Some("你好"),
        None,
        None,
    ])
    .into_array();

    assert_arrays_eq!(array, expected, &mut ctx);
}

#[test]
fn test_sparse_binary_varbinview_non_null_fill() {
    let mut ctx = SESSION.create_execution_ctx();
    let binaries = VarBinViewArray::from_iter_nullable_bin([
        Some(b"hello" as &[u8]),
        Some(b"goodbye"),
        Some(b"hello"),
        None,
        Some(b"\x00"),
        Some(b"\xE4\xBD\xA0\xE5\xA5\xBD"),
        None,
    ])
    .into_array();

    let array = Sparse::try_new(
        buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
        binaries,
        12,
        Scalar::from(Some(ByteBuffer::from(b"123".to_vec()))),
        &mut ctx,
    )
    .unwrap();

    let expected = VarBinViewArray::from_iter_nullable_bin([
        Some(b"hello" as &[u8]),
        Some(b"123"),
        Some(b"123"),
        Some(b"goodbye"),
        Some(b"hello"),
        None,
        Some(b"123"),
        Some(b"\x00"),
        Some(b"123"),
        Some(b"\xE4\xBD\xA0\xE5\xA5\xBD"),
        None,
        Some(b"123"),
    ])
    .into_array();

    assert_arrays_eq!(array, expected, &mut ctx);
}

#[test]
fn test_sparse_list_null_fill() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Use ListViewArray consistently
    let elements = buffer![1i32, 2, 1, 2].into_array();
    // Create ListView with offsets and sizes
    // List 0: [1] at offset 0, size 1
    // List 1: [2] at offset 1, size 1
    // List 2: [1] at offset 2, size 1
    // List 3: [2] at offset 3, size 1
    let offsets = buffer![0u32, 1, 2, 3].into_array();
    let sizes = buffer![1u32, 1, 1, 1].into_array();
    let lists = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
    let fill_value = Scalar::null(lists.dtype().clone());
    let sparse = Sparse::try_new(indices, lists, 6, fill_value, &mut ctx)?.into_array();

    let expected = ListArray::from_iter_opt_slow::<u32, _, _>(
        [
            Some(vec![1i32]),
            None,
            None,
            Some(vec![2]),
            Some(vec![1]),
            Some(vec![2]),
        ],
        Arc::new(PType::I32.into()),
    )?;
    assert_arrays_eq!(sparse, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_sparse_list_null_fill_sliced_sparse_values() {
    let mut ctx = SESSION.create_execution_ctx();
    // Create ListViewArray with 8 elements forming 8 single-element lists
    let elements = buffer![1i32, 2, 1, 2, 1, 2, 1, 2].into_array();
    let offsets = buffer![0u32, 1, 2, 3, 4, 5, 6, 7].into_array();
    let sizes = buffer![1u32, 1, 1, 1, 1, 1, 1, 1].into_array();
    let lists = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    // Slice to get lists 2..6, which are: [1], [2], [1], [2]
    let lists = lists.slice(2..6).unwrap();

    let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
    let fill_value = Scalar::null(lists.dtype().clone());
    let sparse = Sparse::try_new(indices, lists, 6, fill_value, &mut ctx)
        .unwrap()
        .into_array();

    let expected = ListArray::from_iter_opt_slow::<u32, _, _>(
        [
            Some(vec![1i32]),
            None,
            None,
            Some(vec![2]),
            Some(vec![1]),
            Some(vec![2]),
        ],
        Arc::new(PType::I32.into()),
    )
    .unwrap();
    assert_arrays_eq!(sparse, expected, &mut ctx);
}

#[test]
fn test_sparse_list_non_null_fill() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Create ListViewArray with 4 single-element lists
    let elements = buffer![1i32, 2, 1, 2].into_array();
    let offsets = buffer![0u32, 1, 2, 3].into_array();
    let sizes = buffer![1u32, 1, 1, 1].into_array();
    let lists = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    let indices = buffer![0u8, 3u8, 4u8, 5u8].into_array();
    let fill_value = Scalar::from(Some(vec![5i32, 6, 7, 8]));
    let sparse = Sparse::try_new(indices, lists, 6, fill_value, &mut ctx)?.into_array();

    let actual = sparse.execute::<Canonical>(&mut ctx)?.into_array();
    let result_listview = actual.execute::<ListViewArray>(&mut ctx)?;

    // Check the structure
    assert_eq!(result_listview.len(), 6);

    // Verify sizes: positions 0,3,4,5 have sparse data, positions 1,2 have fill values
    assert_eq!(result_listview.size_at(0, &mut ctx), 1); // [1] from sparse
    assert_eq!(result_listview.size_at(1, &mut ctx), 4); // [5,6,7,8] fill value
    assert_eq!(result_listview.size_at(2, &mut ctx), 4); // [5,6,7,8] fill value
    assert_eq!(result_listview.size_at(3, &mut ctx), 1); // [2] from sparse
    assert_eq!(result_listview.size_at(4, &mut ctx), 1); // [1] from sparse
    assert_eq!(result_listview.size_at(5, &mut ctx), 1); // [2] from sparse

    // Verify actual values
    let elements_array = result_listview
        .elements()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let elements_slice = elements_array.as_slice::<i32>();

    // List 0: [1]
    let list0_offset = result_listview.offset_at(0, &mut ctx) as usize;
    assert_eq!(elements_slice[list0_offset], 1);

    // List 1: [5,6,7,8]
    let list1_offset = result_listview.offset_at(1, &mut ctx) as usize;
    let list1_size = result_listview.size_at(1, &mut ctx) as usize;
    assert_eq!(
        &elements_slice[list1_offset..list1_offset + list1_size],
        &[5, 6, 7, 8]
    );

    // List 2: [5,6,7,8]
    let list2_offset = result_listview.offset_at(2, &mut ctx) as usize;
    let list2_size = result_listview.size_at(2, &mut ctx) as usize;
    assert_eq!(
        &elements_slice[list2_offset..list2_offset + list2_size],
        &[5, 6, 7, 8]
    );

    // List 3: [2]
    let list3_offset = result_listview.offset_at(3, &mut ctx) as usize;
    assert_eq!(elements_slice[list3_offset], 2);

    // List 4: [1]
    let list4_offset = result_listview.offset_at(4, &mut ctx) as usize;
    assert_eq!(elements_slice[list4_offset], 1);

    // List 5: [2]
    let list5_offset = result_listview.offset_at(5, &mut ctx) as usize;
    assert_eq!(elements_slice[list5_offset], 2);
    Ok(())
}

/// Each gap between patches is appended as a single constant array, so the fill value's
/// elements are stored once per gap however many rows the gap covers.
#[test]
fn test_sparse_list_fill_stores_one_copy_per_gap() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    // Two single-element patch lists: [1] and [2].
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

    // Patches at 10 and 20 of 10,000 rows, so the fill covers three gaps.
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
/// canonicalization makes is visible in the elements child. Patches go in one at a time, so
/// they cost a chunk each; a gap covers all its rows with a single append, so it costs one
/// chunk however many rows it fills.
#[test]
fn test_sparse_list_bounds_element_chunks() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    const PATCHES: usize = 100;
    let patches_u32 = u32::try_from(PATCHES).vortex_expect("fits in u32");
    let patches_i32 = i32::try_from(PATCHES).vortex_expect("fits in i32");

    // `PATCHES` single-element lists, patched onto rows 0..PATCHES of a 2 * PATCHES-row array,
    // so there is exactly one patch run followed by exactly one gap.
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

/// Patches on consecutive rows are appended as one slice of the patch array, so this covers the
/// run arithmetic together with everything that has to survive it: a null patch inside a run,
/// patch views that overlap and are out of order, runs separated by gaps, and a trailing gap.
#[test]
fn test_sparse_list_appends_consecutive_patches_as_one_run() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    // Overlapping, out-of-order patch views over three elements:
    // - patch 0: [20, 30]
    // - patch 1: [10]      (null, so its elements are never read)
    // - patch 2: [10, 20, 30]
    // - patch 3: [10, 20]
    let patches = unsafe {
        ListViewArray::new_unchecked(
            buffer![10i32, 20, 30].into_array(),
            buffer![1u32, 0, 0, 0].into_array(),
            buffer![2u32, 1, 3, 2].into_array(),
            Validity::from_iter([true, false, true, true]),
        )
    };
    assert!(!patches.is_zero_copy_to_list());

    // Rows 0..3 are one run of patches, row 5 is another, and rows 3-4 and 6 are gaps.
    let indices = buffer![0u8, 1, 2, 5].into_array();
    let fill = Scalar::from(Some(vec![7i32, 8]));
    let sparse = Sparse::try_new(indices, patches.into_array(), 7, fill, &mut ctx)?.into_array();

    let actual = sparse.execute::<ListViewArray>(&mut ctx)?;

    // The seven elements the patches reference, plus one copy of the two-element fill for each
    // of the two gaps. The run of patches on rows 0..3 has no gap inside it to pay for.
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
fn test_sparse_binary_varbin_null_fill() {
    let mut ctx = SESSION.create_execution_ctx();
    let strings = <VarBinArray as FromIterator<_>>::from_iter([
        Some(b"hello" as &[u8]),
        Some(b"goodbye"),
        Some(b"hello"),
        None,
        Some(b"\x00"),
        Some(b"\xE4\xBD\xA0\xE5\xA5\xBD"),
        None,
    ])
    .into_array();

    let array = Sparse::try_new(
        buffer![0u16, 3, 4, 5, 7, 9, 10].into_array(),
        strings,
        12,
        Scalar::null(DType::Binary(Nullable)),
        &mut ctx,
    )
    .unwrap();

    let expected = VarBinViewArray::from_iter_nullable_bin([
        Some(b"hello" as &[u8]),
        None,
        None,
        Some(b"goodbye"),
        Some(b"hello"),
        None,
        None,
        Some(b"\x00"),
        None,
        Some(b"\xE4\xBD\xA0\xE5\xA5\xBD"),
        None,
        None,
    ])
    .into_array();

    assert_arrays_eq!(array, expected, &mut ctx);
}

#[test]
fn test_sparse_fixed_size_list_null_fill() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Create a FixedSizeListArray with 3 lists of size 3.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let fsl = FixedSizeListArray::try_new(elements, 3, Validity::AllValid, 3)?.into_array();

    let indices = buffer![0u8, 2u8, 3u8].into_array();
    let fill_value = Scalar::null(DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::I32, NonNullable)),
        3,
        Nullable,
    ));
    let sparse = Sparse::try_new(indices, fsl, 5, fill_value, &mut ctx)?.into_array();

    // Expected: [1,2,3], null, [4,5,6], [7,8,9], null.
    let expected_elements = buffer![1i32, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9, 0, 0, 0].into_array();
    let expected = FixedSizeListArray::try_new(
        expected_elements,
        3,
        Validity::Array(BoolArray::from_iter([true, false, true, true, false]).into_array()),
        5,
    )?
    .into_array();

    assert_arrays_eq!(sparse, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_sparse_fixed_size_list_non_null_fill() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    // Non-nullable values to match the non-nullable fill below.
    let fsl = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 3)?.into_array();

    let indices = buffer![0u8, 2u8, 4u8].into_array();
    let fill_value = Scalar::fixed_size_list(
        Arc::new(DType::Primitive(PType::I32, NonNullable)),
        vec![
            Scalar::primitive(99i32, NonNullable),
            Scalar::primitive(88i32, NonNullable),
        ],
        NonNullable,
    );
    let sparse = Sparse::try_new(indices, fsl, 6, fill_value, &mut ctx)?.into_array();

    // Expected: [1,2], [99,88], [3,4], [99,88], [5,6], [99,88].
    let expected_elements = buffer![1i32, 2, 99, 88, 3, 4, 99, 88, 5, 6, 99, 88].into_array();
    let expected =
        FixedSizeListArray::try_new(expected_elements, 2, Validity::NonNullable, 6)?.into_array();

    assert_arrays_eq!(sparse, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_sparse_fixed_size_list_with_validity() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Create FSL values with some nulls.
    let elements = buffer![10i32, 20, 30, 40, 50, 60].into_array();
    let fsl = FixedSizeListArray::try_new(
        elements,
        2,
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        3,
    )?
    .into_array();

    let indices = buffer![1u16, 3u16, 4u16].into_array();
    let fill_value = Scalar::fixed_size_list(
        Arc::new(DType::Primitive(PType::I32, NonNullable)),
        vec![
            Scalar::primitive(7i32, NonNullable),
            Scalar::primitive(8i32, NonNullable),
        ],
        Nullable,
    );
    let sparse = Sparse::try_new(indices, fsl, 6, fill_value, &mut ctx)?.into_array();

    // Expected validity: [true, true, true, false, true, true].
    // Expected elements: [7,8], [10,20], [7,8], [30,40], [50,60], [7,8].
    let expected_elements = buffer![7i32, 8, 10, 20, 7, 8, 30, 40, 50, 60, 7, 8].into_array();
    let expected = FixedSizeListArray::try_new(
        expected_elements,
        2,
        Validity::Array(BoolArray::from_iter([true, true, true, false, true, true]).into_array()),
        6,
    )?
    .into_array();

    assert_arrays_eq!(sparse, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_sparse_fixed_size_list_truly_sparse() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Test with a truly sparse array where most values are the fill value.
    // This demonstrates the compression benefit of sparse encoding.

    // Create patch values: only 3 distinct lists out of 100 total positions.
    let elements = buffer![10i32, 11, 20, 21, 30, 31].into_array();
    // Non-nullable values to match the non-nullable fill below.
    let fsl = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 3)?.into_array();

    // Patches at positions 5, 50, and 95 out of 100.
    let indices = buffer![5u32, 50, 95].into_array();

    // Fill value [99, 99] will appear 97 times but stored only once.
    let fill_value = Scalar::fixed_size_list(
        Arc::new(DType::Primitive(PType::I32, NonNullable)),
        vec![
            Scalar::primitive(99i32, NonNullable),
            Scalar::primitive(99i32, NonNullable),
        ],
        NonNullable,
    );

    let sparse = Sparse::try_new(indices, fsl, 100, fill_value, &mut ctx)?.into_array();

    // Build expected: 97 copies of [99,99] with patches at positions 5, 50, 95.
    let mut expected_elements_vec = Vec::with_capacity(200);
    // Positions 0-4: fill values
    for _ in 0..5 {
        expected_elements_vec.extend([99i32, 99]);
    }
    // Position 5: first patch [10, 11]
    expected_elements_vec.extend([10, 11]);
    // Positions 6-49: fill values
    for _ in 6..50 {
        expected_elements_vec.extend([99, 99]);
    }
    // Position 50: second patch [20, 21]
    expected_elements_vec.extend([20, 21]);
    // Positions 51-94: fill values
    for _ in 51..95 {
        expected_elements_vec.extend([99, 99]);
    }
    // Position 95: third patch [30, 31]
    expected_elements_vec.extend([30, 31]);
    // Positions 96-99: fill values
    for _ in 96..100 {
        expected_elements_vec.extend([99, 99]);
    }
    let expected_elements = PrimitiveArray::from_iter(expected_elements_vec).into_array();
    let expected =
        FixedSizeListArray::try_new(expected_elements, 2, Validity::NonNullable, 100)?.into_array();

    assert_arrays_eq!(sparse, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_sparse_fixed_size_list_single_element() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Test with a single element FSL array.
    let elements = buffer![42i32, 43].into_array();
    // Non-nullable values to match the non-nullable fill below.
    let fsl = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 1)?.into_array();

    let indices = buffer![0u32].into_array();
    let fill_value = Scalar::fixed_size_list(
        Arc::new(DType::Primitive(PType::I32, NonNullable)),
        vec![
            Scalar::primitive(1i32, NonNullable),
            Scalar::primitive(2i32, NonNullable),
        ],
        NonNullable,
    );
    let sparse = Sparse::try_new(indices, fsl, 1, fill_value, &mut ctx)?.into_array();

    // Expected: just [42, 43].
    let expected_elements = buffer![42i32, 43].into_array();
    let expected =
        FixedSizeListArray::try_new(expected_elements, 2, Validity::NonNullable, 1)?.into_array();

    assert_arrays_eq!(sparse, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_sparse_list_grows_offset_type() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let elements = buffer![1i32, 2, 1, 2].into_array();
    let offsets = buffer![0u8, 1, 2, 3, 4].into_array();
    let lists = ListArray::try_new(elements, offsets, Validity::AllValid, &mut ctx)?.into_array();

    let indices = buffer![0u8, 1u8, 2u8, 3u8].into_array();
    let fill_value = Scalar::from(Some(vec![42i32; 252])); // 252 + 4 elements = 256 > u8::MAX
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

    // The canonical offsets outgrow the source's u8 offsets; u32 is the smallest offset type
    // a list builder can produce (see `OffsetBuilderPType`).
    let actual_listview = actual.clone().execute::<ListViewArray>(&mut ctx)?;
    assert_eq!(
        actual_listview.offsets().dtype(),
        &DType::Primitive(PType::U32, NonNullable)
    );
    assert_arrays_eq!(&actual, &expected, &mut ctx);

    // Note that the preferred arrow list representation is `List` (not `ListView`).
    let arrow_dtype = SESSION.arrow().to_arrow_field("", expected.dtype())?;
    let actual = actual.execute_arrow(Some(&arrow_dtype), &mut ctx)?;
    let expected = expected.execute_arrow(Some(&arrow_dtype), &mut ctx)?;

    assert_eq!(actual.data_type(), expected.data_type());
    Ok(())
}

#[test]
fn test_sparse_listview_null_fill_with_gaps() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // This test specifically catches the bug where the old implementation
    // incorrectly tracked `last_valid_offset` as the START of the last list
    // instead of properly handling ListView's offset/size pairs.

    // Create a ListViewArray with non-trivial offsets and sizes.
    // Elements: [10, 11, 12, 20, 21, 30, 31, 32, 33]
    // List 0: elements[0..3] = [10, 11, 12]
    // List 1: elements[3..5] = [20, 21]
    // List 2: elements[5..9] = [30, 31, 32, 33]
    let elements = buffer![10i32, 11, 12, 20, 21, 30, 31, 32, 33].into_array();
    let offsets = buffer![0u32, 3, 5].into_array();
    let sizes = buffer![3u32, 2, 4].into_array();

    let list_view = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
            .with_zero_copy_to_list(true)
    };

    let list_dtype = list_view.dtype().clone();

    // Create sparse array with indices [1, 4, 7] and length 10
    // This means we have:
    // - Index 0: null
    // - Index 1: List 0 [10, 11, 12]
    // - Index 2-3: null
    // - Index 4: List 1 [20, 21]
    // - Index 5-6: null
    // - Index 7: List 2 [30, 31, 32, 33]
    // - Index 8-9: null
    let indices = buffer![1u8, 4, 7].into_array();
    let sparse = Sparse::try_new(
        indices,
        list_view.into_array(),
        10,
        Scalar::null(list_dtype),
        &mut ctx,
    )?;

    let expected = ListArray::from_iter_opt_slow::<u32, _, _>(
        [
            None,
            Some(vec![10i32, 11, 12]),
            None,
            None,
            Some(vec![20, 21]),
            None,
            None,
            Some(vec![30, 31, 32, 33]),
            None,
            None,
        ],
        Arc::new(PType::I32.into()),
    )?;
    assert_arrays_eq!(sparse, expected, &mut ctx);

    Ok(())
}

#[test]
fn test_sparse_listview_sliced_values_null_fill() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // This test uses sliced ListView values to ensure proper handling
    // of non-zero starting offsets in the source data.

    // Create a larger ListViewArray and then slice it
    // Original elements: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    let elements = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();

    // Create 5 lists with different offsets and sizes
    // List 0: [0, 1] at offset 0
    // List 1: [2, 3, 4] at offset 2
    // List 2: [5] at offset 5
    // List 3: [6, 7] at offset 6
    // List 4: [8, 9] at offset 8
    let offsets = buffer![0u32, 2, 5, 6, 8].into_array();
    let sizes = buffer![2u32, 3, 1, 2, 2].into_array();

    let full_listview = unsafe {
        ListViewArray::new_unchecked(elements, offsets, sizes, Validity::AllValid)
            .with_zero_copy_to_list(true)
    }
    .into_array();

    // Slice to get lists 1, 2, 3 (indices 1..4)
    // This gives us lists with elements:
    // - Index 0: [2, 3, 4] (original list 1)
    // - Index 1: [5] (original list 2)
    // - Index 2: [6, 7] (original list 3)
    let sliced = full_listview.slice(1..4)?;

    // Create sparse array with indices [0, 1] and length 5
    // Expected result:
    // - Index 0: [2, 3, 4] (from sliced[0])
    // - Index 1: [5] (from sliced[1])
    // - Index 2: null
    // - Index 3: null
    // - Index 4: null
    let indices = buffer![0u8, 1].into_array();
    // Extract only the values we need from the sliced array
    let values = sliced.slice(0..2)?;
    let sparse = Sparse::try_new(
        indices,
        values,
        5,
        Scalar::null(sliced.dtype().clone()),
        &mut ctx,
    )?;

    let expected = ListArray::from_iter_opt_slow::<u32, _, _>(
        [Some(vec![2i32, 3, 4]), Some(vec![5]), None, None, None],
        Arc::new(PType::I32.into()),
    )?;
    assert_arrays_eq!(sparse, expected, &mut ctx);

    Ok(())
}

/// Appending a sparse string array to either variable-binary builder must match its
/// canonical decode exactly — across long, short (inlinable), and null fill values, null
/// patch values, and a sliced array whose patch indices carry an offset.
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

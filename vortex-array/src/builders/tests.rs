// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use allocator_api2::alloc::AllocError;
use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Global;
use hegel::TestCase;
use hegel::generators as gs;
use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::RecursiveCanonical;
use crate::VortexSessionExecute;
use crate::array::IntoArray;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::extension::ExtensionArraySlotsExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::assert_arrays_eq;
use crate::builders::ArrayBuilder;
use crate::builders::ListBuilder;
use crate::builders::ListViewBuilder;
use crate::builders::VarBinViewBuilder;
use crate::builders::builder_with_capacity_in;
use crate::builders::dict::DictConstraints;
use crate::builders::dict::UNCONSTRAINED;
use crate::builders::dict::dict_encode;
use crate::builders::dict::dict_encode_by_frequency;
use crate::builders::dict::dict_encode_by_frequency_with_constraints;
use crate::builders::dict::dict_encoder_in;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::OffsetBuilderPType;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::half::f16;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

// --- Hegel differential state-machine models for core, VarBinView, List/ListView, and Dict builders ---

const ALL_PTYPES: [PType; 11] = [
    PType::I8,
    PType::I16,
    PType::I32,
    PType::I64,
    PType::U8,
    PType::U16,
    PType::U32,
    PType::U64,
    PType::F16,
    PType::F32,
    PType::F64,
];

const ALL_DECIMAL_DTYPES: [(u8, i8); 6] = [(2, 1), (4, 2), (9, 2), (18, 4), (38, 4), (76, 6)];

fn draw_scalar_for_dtype(tc: &TestCase, dtype: &DType) -> Scalar {
    if dtype.is_nullable()
        && !matches!(dtype, DType::Null)
        && tc.draw(gs::integers::<u8>().max_value(3)) == 0
    {
        return Scalar::null(dtype.clone());
    }
    match dtype {
        DType::Null => Scalar::null(DType::Null),
        DType::Bool(n) => Scalar::bool(tc.draw(gs::booleans()), *n),
        DType::Primitive(ptype, nullability) => {
            let val = tc.draw(gs::integers::<i8>().min_value(-120).max_value(120));
            let abs = val.unsigned_abs();
            match ptype {
                PType::I8 => Scalar::primitive(val, *nullability),
                PType::I16 => Scalar::primitive(i16::from(val), *nullability),
                PType::I32 => Scalar::primitive(i32::from(val), *nullability),
                PType::I64 => Scalar::primitive(i64::from(val), *nullability),
                PType::U8 => Scalar::primitive(abs, *nullability),
                PType::U16 => Scalar::primitive(u16::from(abs), *nullability),
                PType::U32 => Scalar::primitive(u32::from(abs), *nullability),
                PType::U64 => Scalar::primitive(u64::from(abs), *nullability),
                PType::F16 => Scalar::primitive(f16::from_f32(f32::from(val) * 0.5), *nullability),
                PType::F32 => Scalar::primitive(f32::from(val) * 0.5, *nullability),
                PType::F64 => Scalar::primitive(f64::from(val) * 0.5, *nullability),
            }
        }
        DType::Decimal(dec_dtype, n) => {
            let bound = if dec_dtype.precision() <= 2 { 99 } else { 5000 };
            let val = tc.draw(gs::integers::<i64>().min_value(-bound).max_value(bound));
            Scalar::decimal(DecimalValue::I128(i128::from(val)), *dec_dtype, *n)
        }
        DType::Utf8(n) => {
            // Mix empty, short inline (<=12B), and long out-of-line (>12B) strings.
            let len = match tc.draw(gs::integers::<u8>().max_value(3)) {
                0 => 0,
                1 => tc.draw(gs::integers::<usize>().min_value(1).max_value(12)),
                _ => tc.draw(gs::integers::<usize>().min_value(13).max_value(28)),
            };
            let ch = char::from(b'a' + tc.draw(gs::integers::<u8>().max_value(25)));
            let s: String = std::iter::repeat_n(ch, len).collect();
            Scalar::utf8(s, *n)
        }
        DType::Binary(n) => {
            let len = match tc.draw(gs::integers::<u8>().max_value(3)) {
                0 => 0,
                1 => tc.draw(gs::integers::<usize>().min_value(1).max_value(12)),
                _ => tc.draw(gs::integers::<usize>().min_value(13).max_value(28)),
            };
            let byte = tc.draw(gs::integers::<u8>());
            Scalar::binary(vec![byte; len], *n)
        }
        DType::Struct(fields, n) => {
            let children: Vec<_> = fields
                .fields()
                .map(|f| draw_scalar_for_dtype(tc, &f))
                .collect();
            Scalar::struct_(DType::Struct(fields.clone(), *n), children)
        }
        DType::List(elem, n) => {
            let len = tc.draw(gs::integers::<usize>().max_value(4));
            let elems = (0..len).map(|_| draw_scalar_for_dtype(tc, elem)).collect();
            Scalar::list(Arc::clone(elem), elems, *n)
        }
        DType::FixedSizeList(elem, size, n) => {
            let elems = (0..*size)
                .map(|_| draw_scalar_for_dtype(tc, elem))
                .collect();
            Scalar::fixed_size_list(Arc::clone(elem), elems, *n)
        }
        DType::Extension(ext) => {
            let storage = draw_scalar_for_dtype(tc, ext.storage_dtype());
            Scalar::extension_ref(ext.clone(), storage)
        }
        other => unreachable!("unsupported test dtype {other}"),
    }
}

fn expected_zero_scalar(dtype: &DType) -> Scalar {
    match dtype {
        DType::Null => Scalar::null(DType::Null),
        DType::Struct(fields, _) => Scalar::struct_(
            dtype.clone(),
            fields.fields().map(|f| expected_zero_scalar(&f)),
        ),
        DType::FixedSizeList(elem, size, n) => Scalar::fixed_size_list(
            Arc::clone(elem),
            vec![expected_zero_scalar(elem); *size as usize],
            *n,
        ),
        _ => Scalar::zero_value(dtype),
    }
}

fn build_array_from_scalars(dtype: &DType, scalars: &[Scalar]) -> ArrayRef {
    let mut builder =
        builder_with_capacity_in(dtype, scalars.len(), BufferAllocatorRef::static_ref());
    for scalar in scalars {
        builder.append_scalar(scalar).vortex_expect("append scalar");
    }
    builder.finish()
}

fn run_builder_state_machine(tc: &TestCase, dtype: DType, ctx: &mut ExecutionCtx) {
    let init_cap = tc.draw(gs::integers::<usize>().max_value(4));
    let mut b_finish = builder_with_capacity_in(&dtype, init_cap, BufferAllocatorRef::static_ref());
    let mut b_canon = builder_with_capacity_in(&dtype, init_cap, BufferAllocatorRef::static_ref());

    // Run 1 or 2 batches to verify builder reset & reuse after finish().
    let batches = tc.draw(gs::integers::<usize>().min_value(1).max_value(2));
    for _ in 0..batches {
        let mut model: Vec<Scalar> = Vec::new();
        let ops = tc.draw(gs::integers::<usize>().max_value(12));
        for _ in 0..ops {
            let op = tc.draw(gs::integers::<u8>().max_value(5));
            match op {
                0 => {
                    let scalar = draw_scalar_for_dtype(tc, &dtype);
                    b_finish
                        .append_scalar(&scalar)
                        .vortex_expect("append_scalar");
                    b_canon
                        .append_scalar(&scalar)
                        .vortex_expect("append_scalar");
                    model.push(scalar);
                }
                1 => {
                    let count = tc.draw(gs::integers::<usize>().max_value(4));
                    b_finish.append_zeros(count);
                    b_canon.append_zeros(count);
                    model.extend(std::iter::repeat_n(expected_zero_scalar(&dtype), count));
                }
                2 => {
                    let count = tc.draw(gs::integers::<usize>().max_value(4));
                    b_finish.append_defaults(count);
                    b_canon.append_defaults(count);
                    let expected = if dtype.is_nullable() {
                        Scalar::null(dtype.clone())
                    } else {
                        expected_zero_scalar(&dtype)
                    };
                    model.extend(std::iter::repeat_n(expected, count));
                }
                3 if dtype.is_nullable() => {
                    let count = tc.draw(gs::integers::<usize>().max_value(4));
                    if count == 1 && tc.draw(gs::booleans()) {
                        b_finish.append_null();
                        b_canon.append_null();
                    } else {
                        b_finish.append_nulls(count);
                        b_canon.append_nulls(count);
                    }
                    model.extend(std::iter::repeat_n(Scalar::null(dtype.clone()), count));
                }
                4 => {
                    let extra = tc.draw(gs::integers::<usize>().max_value(8));
                    b_finish.reserve_exact(extra);
                    b_canon.reserve_exact(extra);
                }
                _ => {
                    // Build a source array (plain, sliced, ConstantArray, or ChunkedArray).
                    let chunk_len = tc.draw(gs::integers::<usize>().max_value(5));
                    let (chunk_arr, chunk_scalars) =
                        match tc.draw(gs::integers::<u8>().max_value(3)) {
                            0 if chunk_len > 0 => {
                                let fill = draw_scalar_for_dtype(tc, &dtype);
                                (
                                    ConstantArray::new(fill.clone(), chunk_len).into_array(),
                                    vec![fill; chunk_len],
                                )
                            }
                            1 if chunk_len >= 2 => {
                                let all: Vec<Scalar> = (0..chunk_len + 2)
                                    .map(|_| draw_scalar_for_dtype(tc, &dtype))
                                    .collect();
                                let full = build_array_from_scalars(&dtype, &all);
                                let start = tc.draw(gs::integers::<usize>().max_value(2));
                                (
                                    full.slice(start..start + chunk_len).vortex_expect("slice"),
                                    all[start..start + chunk_len].to_vec(),
                                )
                            }
                            2 if chunk_len >= 2 => {
                                let mid = chunk_len / 2;
                                let s1: Vec<Scalar> = (0..mid)
                                    .map(|_| draw_scalar_for_dtype(tc, &dtype))
                                    .collect();
                                let s2: Vec<Scalar> = (mid..chunk_len)
                                    .map(|_| draw_scalar_for_dtype(tc, &dtype))
                                    .collect();
                                let chunked = ChunkedArray::try_new(
                                    vec![
                                        build_array_from_scalars(&dtype, &s1),
                                        build_array_from_scalars(&dtype, &s2),
                                    ],
                                    dtype.clone(),
                                )
                                .vortex_expect("chunked")
                                .into_array();
                                let mut all = s1;
                                all.extend(s2);
                                (chunked, all)
                            }
                            _ => {
                                let scalars: Vec<Scalar> = (0..chunk_len)
                                    .map(|_| draw_scalar_for_dtype(tc, &dtype))
                                    .collect();
                                (build_array_from_scalars(&dtype, &scalars), scalars)
                            }
                        };
                    chunk_arr
                        .append_to_builder(b_finish.as_mut(), ctx)
                        .vortex_expect("extend finish");
                    chunk_arr
                        .append_to_builder(b_canon.as_mut(), ctx)
                        .vortex_expect("extend canon");
                    model.extend(chunk_scalars);
                }
            }
            assert_eq!(b_finish.len(), model.len());
            assert_eq!(b_canon.len(), model.len());
            assert_eq!(b_finish.is_empty(), model.is_empty());
        }

        let arr_finish = b_finish.finish();
        let arr_canon = b_canon.finish_into_canonical(ctx).into_array();
        assert_eq!(arr_finish.dtype(), &dtype);
        assert_eq!(arr_canon.dtype(), &dtype);
        assert_eq!(arr_finish.len(), model.len());
        assert_arrays_eq!(&arr_finish, &arr_canon, ctx);
        for (i, expected) in model.iter().enumerate() {
            assert_eq!(
                arr_finish.execute_scalar(i, ctx).vortex_expect("scalar"),
                *expected,
                "scalar mismatch at index {i} for dtype {dtype}"
            );
        }
    }
}

/// Stratified state-machine model over `Primitive` (all 11 `PType`s) and `Bool` builders.
#[hegel::test(test_cases = 120)]
fn hegel_primitive_builders_state_machine(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let nullable = Nullability::from(tc.draw(gs::booleans()));
    let idx = tc.draw(gs::integers::<usize>().max_value(ALL_PTYPES.len()));
    let dtype = if idx == ALL_PTYPES.len() {
        DType::Bool(nullable)
    } else {
        DType::Primitive(ALL_PTYPES[idx], nullable)
    };
    run_builder_state_machine(&tc, dtype, &mut ctx);
}

/// Stratified state-machine model over all 6 `Decimal` storage widths (`i8`..`i256`), `Utf8`,
/// `Binary`, `Null`, and `Extension` builders.
#[hegel::test(test_cases = 100)]
fn hegel_decimal_and_scalar_builders_state_machine(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let nullable = Nullability::from(tc.draw(gs::booleans()));
    let dtype = match tc.draw(gs::integers::<u8>().max_value(9)) {
        0..=5 => {
            let (precision, scale) =
                ALL_DECIMAL_DTYPES[tc.draw(gs::integers::<usize>().max_value(5))];
            DType::Decimal(DecimalDType::new(precision, scale), nullable)
        }
        6 => DType::Utf8(nullable),
        7 => DType::Binary(nullable),
        8 => DType::Null,
        _ => DType::Extension(Timestamp::new(TimeUnit::Milliseconds, nullable).erased()),
    };
    run_builder_state_machine(&tc, dtype, &mut ctx);
}

/// Stratified state-machine model over nested builders: flat `Struct`, nested `Struct`-in-`Struct`,
/// `Struct` with a `Null` field, `List`, degenerate `size = 0` `FixedSizeList`, and `size = 1..=3`
/// `FixedSizeList`.
#[hegel::test(test_cases = 100)]
fn hegel_nested_builders_state_machine(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let nullable = Nullability::from(tc.draw(gs::booleans()));
    let child_null = Nullability::from(tc.draw(gs::booleans()));
    let dtype = match tc.draw(gs::integers::<u8>().max_value(5)) {
        0 => DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, child_null)),
                ("b", DType::Utf8(child_null)),
            ]),
            nullable,
        ),
        1 => DType::Struct(
            StructFields::from_iter([
                ("flag", DType::Bool(child_null)),
                (
                    "inner",
                    DType::Struct(
                        StructFields::from_iter([(
                            "val",
                            DType::Primitive(PType::F64, child_null),
                        )]),
                        child_null,
                    ),
                ),
            ]),
            nullable,
        ),
        2 => DType::Struct(
            StructFields::from_iter([
                ("a", DType::Null),
                ("b", DType::Primitive(PType::I32, child_null)),
            ]),
            nullable,
        ),
        3 => DType::List(Arc::new(DType::Primitive(PType::I32, child_null)), nullable),
        4 => DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, child_null)),
            0,
            nullable,
        ),
        _ => {
            let size = tc.draw(gs::integers::<u32>().min_value(1).max_value(3));
            DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::I32, child_null)),
                size,
                nullable,
            )
        }
    };
    run_builder_state_machine(&tc, dtype, &mut ctx);
}

/// Differential model for `VarBinViewBuilder` across normal, deduplicating, and compacting modes,
/// mixing inline (`<=12B`) and out-of-line (`>12B`) values, `append_buffer_with_lengths`,
/// `append_buffer_with_offsets`, sliced `VarBinViewArray` / `VarBinArray` extends, and reuse.
/// Compares against `VarBinArray::from_iter` (independent of `VarBinViewBuilder`).
#[hegel::test(test_cases = 60)]
fn hegel_varbinview_builder_modes_and_bulk_ops(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let is_utf8 = tc.draw(gs::booleans());
    let nullable = tc.draw(gs::booleans());
    let dtype = if is_utf8 {
        DType::Utf8(Nullability::from(nullable))
    } else {
        DType::Binary(Nullability::from(nullable))
    };

    let mut builder = match tc.draw(gs::integers::<u8>().max_value(2)) {
        0 => VarBinViewBuilder::with_capacity_in(
            dtype.clone(),
            0,
            BufferAllocatorRef::statically_allocated(),
        ),
        1 => VarBinViewBuilder::with_buffer_deduplication_in(
            dtype.clone(),
            0,
            BufferAllocatorRef::statically_allocated(),
        ),
        _ => {
            let threshold = match tc.draw(gs::integers::<u8>().max_value(2)) {
                0 => 0.0,
                1 => 0.5,
                _ => 1.0,
            };
            VarBinViewBuilder::with_compaction_in(
                dtype.clone(),
                0,
                threshold,
                BufferAllocatorRef::statically_allocated(),
            )
        }
    };

    let draw_bytes = |tc: &TestCase| -> Vec<u8> {
        let len = match tc.draw(gs::integers::<u8>().max_value(3)) {
            0 => 0,
            1 => tc.draw(gs::integers::<usize>().min_value(1).max_value(12)),
            _ => tc.draw(gs::integers::<usize>().min_value(13).max_value(40)),
        };
        let byte = if is_utf8 {
            b'a' + tc.draw(gs::integers::<u8>().max_value(25))
        } else {
            tc.draw(gs::integers::<u8>())
        };
        vec![byte; len]
    };

    for _ in 0..tc.draw(gs::integers::<usize>().min_value(1).max_value(2)) {
        let mut model: Vec<Option<Vec<u8>>> = Vec::new();
        for _ in 0..tc.draw(gs::integers::<usize>().max_value(10)) {
            match tc.draw(gs::integers::<u8>().max_value(5)) {
                0 => {
                    let bytes = draw_bytes(&tc);
                    builder.append_value(&bytes);
                    model.push(Some(bytes));
                }
                1 if nullable => {
                    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
                    builder.append_nulls(count);
                    model.extend(std::iter::repeat_n(None, count));
                }
                2 => {
                    let count = tc.draw(gs::integers::<usize>().max_value(3));
                    builder.append_zeros(count);
                    model.extend(std::iter::repeat_n(Some(Vec::new()), count));
                }
                3 => {
                    // append_buffer_with_lengths or append_buffer_with_offsets (with prefix pad).
                    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
                    let pad = tc.draw(gs::integers::<usize>().max_value(4));
                    let mut heap = vec![b'x'; pad];
                    let mut lengths = Vec::with_capacity(count);
                    let mut offsets = Vec::with_capacity(count + 1);
                    offsets.push(u32::try_from(pad).vortex_expect("u32"));
                    let mut valid = Vec::with_capacity(count);
                    for _ in 0..count {
                        let is_valid = !nullable || tc.draw(gs::booleans());
                        let bytes = draw_bytes(&tc);
                        heap.extend_from_slice(&bytes);
                        lengths.push(u32::try_from(bytes.len()).vortex_expect("u32"));
                        offsets.push(u32::try_from(heap.len()).vortex_expect("u32"));
                        valid.push(is_valid);
                        model.push(is_valid.then_some(bytes));
                    }
                    let mask = Mask::from_iter(valid);
                    if pad == 0 && tc.draw(gs::booleans()) {
                        builder.append_buffer_with_lengths(
                            ByteBuffer::copy_from(heap),
                            &lengths,
                            &mask,
                        );
                    } else {
                        builder.append_buffer_with_offsets(
                            ByteBuffer::copy_from(heap),
                            &offsets,
                            &mask,
                        );
                    }
                }
                _ => {
                    // Extend from a VarBinViewArray or VarBinArray (optionally sliced).
                    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
                    let items: Vec<Option<Vec<u8>>> = (0..count)
                        .map(|_| {
                            if nullable && tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                                None
                            } else {
                                Some(draw_bytes(&tc))
                            }
                        })
                        .collect();
                    let src = if tc.draw(gs::booleans()) {
                        VarBinArray::from_iter(items.clone(), dtype.clone()).into_array()
                    } else {
                        VarBinViewArray::from_iter(items.clone(), dtype.clone()).into_array()
                    };
                    let start = tc.draw(gs::integers::<usize>().max_value(count));
                    let end = tc.draw(gs::integers::<usize>().min_value(start).max_value(count));
                    let sliced = src.slice(start..end).vortex_expect("slice");
                    sliced
                        .append_to_builder(&mut builder, &mut ctx)
                        .vortex_expect("extend varbinview");
                    model.extend_from_slice(&items[start..end]);
                }
            }
        }

        let actual = builder.finish_into_varbinview();
        // Compare against `VarBinArray` (independent of `VarBinViewBuilder`).
        let expected = VarBinArray::from_iter(model, dtype.clone());
        assert_arrays_eq!(actual, expected, &mut ctx);
    }
}

/// Builds a reference `ListArray` directly from flat buffers and offsets without calling
/// `ListBuilder`, avoiding circularity in list-builder differential tests.
fn list_array_from_model_with_offset_ptype(
    model: &[Option<Vec<i32>>],
    nullable: bool,
    offset_ptype: PType,
    ctx: &mut ExecutionCtx,
) -> ListArray {
    let mut flat_elems: Vec<i32> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(model.len() + 1);
    offsets.push(0);
    for entry in model {
        if let Some(vals) = entry {
            flat_elems.extend_from_slice(vals);
        }
        offsets.push(flat_elems.len());
    }
    let offsets_arr = match offset_ptype {
        PType::U32 => offsets
            .into_iter()
            .map(|v| u32::try_from(v).vortex_expect("u32 offset"))
            .collect::<Buffer<u32>>()
            .into_array(),
        PType::U64 => offsets
            .into_iter()
            .map(|v| v as u64)
            .collect::<Buffer<u64>>()
            .into_array(),
        PType::I32 => offsets
            .into_iter()
            .map(|v| i32::try_from(v).vortex_expect("i32 offset"))
            .collect::<Buffer<i32>>()
            .into_array(),
        PType::I64 => offsets
            .into_iter()
            .map(|v| i64::try_from(v).vortex_expect("i64 offset"))
            .collect::<Buffer<i64>>()
            .into_array(),
        other => unreachable!("unsupported list offset ptype {other}"),
    };
    let validity = if nullable {
        Validity::from_iter(model.iter().map(Option::is_some))
    } else {
        Validity::NonNullable
    };
    ListArray::new(
        Buffer::from(flat_elems).into_array(),
        offsets_arr,
        validity,
        ctx,
    )
}

fn list_array_from_model(
    model: &[Option<Vec<i32>>],
    nullable: bool,
    ctx: &mut ExecutionCtx,
) -> ListArray {
    list_array_from_model_with_offset_ptype(model, nullable, PType::U32, ctx)
}

fn draw_list_source_chunk(
    tc: &TestCase,
    nullable: bool,
    ctx: &mut ExecutionCtx,
) -> (ArrayRef, Vec<Option<Vec<i32>>>) {
    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let lists: Vec<Option<Vec<i32>>> =
        (0..count)
            .map(|_| {
                if nullable && tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                    None
                } else {
                    Some(tc.draw(
                        gs::vecs(gs::integers::<i32>().min_value(-50).max_value(50)).max_size(3),
                    ))
                }
            })
            .collect();
    let offset_ptypes = [PType::U32, PType::U64, PType::I32, PType::I64];
    let offset_ptype = offset_ptypes[tc.draw(gs::integers::<usize>().max_value(3))];
    let list_arr =
        list_array_from_model_with_offset_ptype(&lists, nullable, offset_ptype, ctx).into_array();
    let src = if tc.draw(gs::booleans()) {
        list_arr
            .execute::<ListViewArray>(ctx)
            .vortex_expect("listview")
            .into_array()
    } else {
        list_arr
    };
    let start = tc.draw(gs::integers::<usize>().max_value(count));
    let end = tc.draw(gs::integers::<usize>().min_value(start).max_value(count));
    (
        src.slice(start..end).vortex_expect("slice"),
        lists[start..end].to_vec(),
    )
}

fn run_list_builder_model<O: OffsetBuilderPType>(tc: &TestCase, ctx: &mut ExecutionCtx) {
    let nullable = tc.draw(gs::booleans());
    let elem_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
    let mut builder = ListBuilder::<O>::with_capacity_in(
        Arc::clone(&elem_dtype),
        Nullability::from(nullable),
        tc.draw(gs::integers::<usize>().max_value(3)),
        BufferAllocatorRef::static_ref(),
    );

    // Consecutive empty finishes must each preserve a single leading zero offset.
    for _ in 0..2 {
        let empty = builder.finish_into_list();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.elements().len(), 0);
        assert_arrays_eq!(
            empty.offsets(),
            PrimitiveArray::from_iter([O::default()]),
            ctx
        );
    }

    // Run 2 batches and keep batch 1 alive across batch 2 to verify reuse does not mutate batch 1.
    let mut prev_batch: Option<(ListArray, ListArray)> = None;
    for _ in 0..2 {
        let mut model: Vec<Option<Vec<i32>>> = Vec::new();
        for _ in 0..tc.draw(gs::integers::<usize>().max_value(8)) {
            match tc.draw(gs::integers::<u8>().max_value(4)) {
                0 => {
                    let vals: Vec<i32> = tc.draw(
                        gs::vecs(gs::integers::<i32>().min_value(-50).max_value(50)).max_size(4),
                    );
                    let scalar = Scalar::list(
                        Arc::clone(&elem_dtype),
                        vals.iter().copied().map(Scalar::from).collect(),
                        Nullability::from(nullable),
                    );
                    builder
                        .append_value(scalar.as_list())
                        .vortex_expect("append_value");
                    model.push(Some(vals));
                }
                1 => {
                    let vals: Vec<i32> = tc.draw(
                        gs::vecs(gs::integers::<i32>().min_value(-50).max_value(50)).max_size(4),
                    );
                    builder
                        .append_array_as_list(&Buffer::from(vals.clone()).into_array(), ctx)
                        .vortex_expect("append_array_as_list");
                    model.push(Some(vals));
                }
                2 if nullable => {
                    let count = tc.draw(gs::integers::<usize>().max_value(4));
                    builder.append_nulls(count);
                    model.extend(std::iter::repeat_n(None, count));
                }
                3 => {
                    let count = tc.draw(gs::integers::<usize>().max_value(4));
                    builder.append_zeros(count);
                    model.extend(std::iter::repeat_n(Some(Vec::new()), count));
                }
                _ => {
                    let (chunk, chunk_model) = draw_list_source_chunk(tc, nullable, ctx);
                    chunk
                        .append_to_builder(&mut builder, ctx)
                        .vortex_expect("extend list");
                    model.extend(chunk_model);
                }
            }
        }

        let actual = builder.finish_into_list();
        let expected_elem_len: usize = model.iter().flatten().map(Vec::len).sum();
        assert_eq!(
            actual.elements().len(),
            expected_elem_len,
            "list builder must not retain stale elements across batches"
        );
        let first_offset = usize::try_from(
            &actual
                .offsets()
                .execute_scalar(0, ctx)
                .vortex_expect("offset 0"),
        )
        .vortex_expect("usize offset");
        assert_eq!(first_offset, 0, "list builder offsets must start at 0");
        let expected = list_array_from_model(&model, nullable, ctx);
        assert_arrays_eq!(&actual, &expected, ctx);
        if let Some((batch1, batch1_expected)) = prev_batch.take() {
            assert_arrays_eq!(batch1, batch1_expected, ctx);
        } else {
            prev_batch = Some((actual, expected));
        }
    }
}

fn run_listview_builder_model<O: OffsetBuilderPType, S: OffsetBuilderPType>(
    tc: &TestCase,
    ctx: &mut ExecutionCtx,
) {
    let nullable = tc.draw(gs::booleans());
    let elem_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
    let mut builder = ListViewBuilder::<O, S>::with_capacity_in(
        Arc::clone(&elem_dtype),
        Nullability::from(nullable),
        tc.draw(gs::integers::<usize>().max_value(3)),
        BufferAllocatorRef::static_ref(),
    );

    // Run 2 batches and keep batch 1 alive across batch 2 to verify reuse & immutability.
    let mut prev_batch: Option<(ListViewArray, ListArray)> = None;
    for _ in 0..2 {
        let mut model: Vec<Option<Vec<i32>>> = Vec::new();
        for _ in 0..tc.draw(gs::integers::<usize>().max_value(8)) {
            match tc.draw(gs::integers::<u8>().max_value(4)) {
                0 => {
                    let vals: Vec<i32> = tc.draw(
                        gs::vecs(gs::integers::<i32>().min_value(-50).max_value(50)).max_size(4),
                    );
                    let scalar = Scalar::list(
                        Arc::clone(&elem_dtype),
                        vals.iter().copied().map(Scalar::from).collect(),
                        Nullability::from(nullable),
                    );
                    builder
                        .append_value(scalar.as_list())
                        .vortex_expect("append_value");
                    model.push(Some(vals));
                }
                1 => {
                    let vals: Vec<i32> = tc.draw(
                        gs::vecs(gs::integers::<i32>().min_value(-50).max_value(50)).max_size(4),
                    );
                    builder
                        .append_array_as_list(&Buffer::from(vals.clone()).into_array(), ctx)
                        .vortex_expect("append_array_as_list");
                    model.push(Some(vals));
                }
                2 if nullable => {
                    let count = tc.draw(gs::integers::<usize>().max_value(4));
                    builder.append_nulls(count);
                    model.extend(std::iter::repeat_n(None, count));
                }
                3 => {
                    let count = tc.draw(gs::integers::<usize>().max_value(4));
                    builder.append_zeros(count);
                    model.extend(std::iter::repeat_n(Some(Vec::new()), count));
                }
                _ => {
                    let (chunk, chunk_model) = draw_list_source_chunk(tc, nullable, ctx);
                    chunk
                        .append_to_builder(&mut builder, ctx)
                        .vortex_expect("extend listview");
                    model.extend(chunk_model);
                }
            }
        }

        let actual = builder.finish_into_listview();
        let expected_elem_len: usize = model.iter().flatten().map(Vec::len).sum();
        assert_eq!(
            actual.elements().len(),
            expected_elem_len,
            "listview builder must not retain stale elements across batches"
        );
        if !actual.is_empty() {
            assert_eq!(
                actual.offset_at(0, ctx),
                0,
                "listview builder offsets must start at 0"
            );
        }
        // Contiguous appends must preserve `is_zero_copy_to_list()`, with no gaps or overlaps
        // (`offset[i] + size[i] == offset[i + 1]`).
        assert!(
            actual.is_zero_copy_to_list(),
            "contiguous appends must preserve is_zero_copy_to_list"
        );
        let elem_len = actual.elements().len();
        for i in 0..actual.len() {
            let offset = actual.offset_at(i, ctx);
            let size = actual.size_at(i, ctx);
            assert!(
                offset + size <= elem_len,
                "ZCTL view {i} exceeds elements len: {offset} + {size} > {elem_len}"
            );
            if i + 1 < actual.len() {
                let next_offset = actual.offset_at(i + 1, ctx);
                assert_eq!(
                    offset + size,
                    next_offset,
                    "ZCTL invariant violated at {i}: offset={offset}, size={size}, next_offset={next_offset}"
                );
            }
        }
        let expected = list_array_from_model(&model, nullable, ctx);
        assert_arrays_eq!(&actual, &expected, ctx);
        if let Some((batch1, batch1_expected)) = prev_batch.take() {
            assert_arrays_eq!(batch1, batch1_expected, ctx);
        } else {
            prev_batch = Some((actual, expected));
        }
    }
}

/// Differential model for `ListBuilder<O>` (`u32`, `u64`, `i32`, `i64`) and
/// `ListViewBuilder<O, S>` (`(u32, u32)`, `(u64, u32)`, `(i64, i32)`) covering dynamic
/// `append_to_builder` dispatch, offset rebasing, bulk empty/null appends, leading-zero empty
/// finishes, and multi-batch reuse immutability.
#[hegel::test(test_cases = 140)]
fn hegel_list_and_listview_builder_offset_types(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    match tc.draw(gs::integers::<u8>().max_value(6)) {
        0 => run_list_builder_model::<u32>(&tc, &mut ctx),
        1 => run_list_builder_model::<u64>(&tc, &mut ctx),
        2 => run_list_builder_model::<i32>(&tc, &mut ctx),
        3 => run_list_builder_model::<i64>(&tc, &mut ctx),
        4 => run_listview_builder_model::<u32, u32>(&tc, &mut ctx),
        5 => run_listview_builder_model::<u64, u32>(&tc, &mut ctx),
        _ => run_listview_builder_model::<i64, i32>(&tc, &mut ctx),
    }
}

fn distinct_scalars(array: &ArrayRef, ctx: &mut ExecutionCtx) -> Vec<Scalar> {
    let mut seen: Vec<Scalar> = Vec::new();
    for i in 0..array.len() {
        let scalar = array.execute_scalar(i, ctx).vortex_expect("scalar");
        if !seen.contains(&scalar) {
            seen.push(scalar);
        }
    }
    seen
}

fn expected_dict_bytes(dtype: &DType, distinct: &[Scalar]) -> usize {
    match dtype {
        DType::Primitive(ptype, _) => distinct.len() * ptype.byte_width(),
        DType::Decimal(dec_dtype, _) => {
            distinct.len() * dec_dtype.canonical_values_type().byte_width()
        }
        DType::Utf8(_) | DType::Binary(_) => distinct
            .iter()
            .map(|scalar| {
                let payload_len = if scalar.is_null() {
                    0
                } else if matches!(dtype, DType::Utf8(_)) {
                    scalar.as_utf8().value().map_or(0, |v| v.len())
                } else {
                    scalar.as_binary().value().map_or(0, |v| v.len())
                };
                size_of::<BinaryView>()
                    + if payload_len > BinaryView::MAX_INLINED_SIZE {
                        payload_len
                    } else {
                        0
                    }
            })
            .sum(),
        other => unreachable!("unexpected dict dtype {other}"),
    }
}

/// Differential model for primitive, decimal, and bytes (`Utf8`/`Binary`) dictionary builders:
/// checks `dict_encode`, `dict_encode_by_frequency` (including non-increasing frequency ordering),
/// `max_len` and `max_bytes` constraints, and streaming multi-chunk `dict_encoder_in` + `reset()`
/// clearing stale entries.
#[hegel::test(test_cases = 100)]
fn hegel_dict_builders_roundtrip_and_frequency(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let nullable = tc.draw(gs::booleans());
    let len = tc.draw(gs::integers::<usize>().max_value(32));
    let kind = tc.draw(gs::integers::<u8>().max_value(2));

    // Fix the concrete dtype once so `chunk2` always matches `arr.dtype()`.
    let decimal_precision = ALL_DECIMAL_DTYPES
        [tc.draw(gs::integers::<usize>().max_value(ALL_DECIMAL_DTYPES.len() - 1))]
    .0;
    let bytes_dtype = if tc.draw(gs::booleans()) {
        DType::Utf8(Nullability::from(nullable))
    } else {
        DType::Binary(Nullability::from(nullable))
    };

    let make_chunk = |tc: &TestCase, count: usize, ctx: &mut ExecutionCtx| -> ArrayRef {
        match kind {
            0 => {
                // Primitive i32 with low cardinality so duplicates occur frequently.
                if nullable {
                    let vals: Vec<Option<i32>> = (0..count)
                        .map(|_| {
                            if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                                None
                            } else {
                                Some(tc.draw(gs::integers::<i32>().min_value(-4).max_value(4)))
                            }
                        })
                        .collect();
                    PrimitiveArray::from_option_iter(vals).into_array()
                } else {
                    let vals: Vec<i32> = (0..count)
                        .map(|_| tc.draw(gs::integers::<i32>().min_value(-4).max_value(4)))
                        .collect();
                    PrimitiveArray::from_iter(vals).into_array()
                }
            }
            1 => {
                let dec_dtype = DecimalDType::new(decimal_precision, 2);
                let validity = if nullable {
                    Validity::from_iter(
                        (0..count).map(|_| tc.draw(gs::integers::<u8>().max_value(3)) != 0),
                    )
                } else {
                    Validity::NonNullable
                };
                match_each_decimal_value_type!(dec_dtype.canonical_values_type(), |T| {
                    let buf: Buffer<T> = (0..count)
                        .map(|_| {
                            let val = tc.draw(gs::integers::<i64>().min_value(-4).max_value(4));
                            <T as crate::dtype::BigCast>::from(val).vortex_expect("fits")
                        })
                        .collect();
                    DecimalArray::new(buf, dec_dtype, validity, ctx).into_array()
                })
            }
            _ => {
                // Bytes (Utf8 or Binary, inline <=12B and out-of-line >12B, VarBinView or VarBin).
                const POOL: [&str; 6] = [
                    "",
                    "a",
                    "hello",
                    "inline_12_ok",
                    "a_long_outlined_string_1",
                    "another_long_outlined_string_2",
                ];
                let items: Vec<Option<&[u8]>> = (0..count)
                    .map(|_| {
                        if nullable && tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                            None
                        } else {
                            let idx = tc.draw(gs::integers::<usize>().max_value(POOL.len() - 1));
                            Some(POOL[idx].as_bytes())
                        }
                    })
                    .collect();
                if tc.draw(gs::booleans()) {
                    VarBinViewArray::from_iter(items, bytes_dtype.clone()).into_array()
                } else {
                    VarBinArray::from_iter(items, bytes_dtype.clone()).into_array()
                }
            }
        }
    };

    let arr = make_chunk(&tc, len, &mut ctx);
    let distinct = distinct_scalars(&arr, &mut ctx);
    let expected_distinct = distinct.len();
    let dict_bytes = expected_dict_bytes(arr.dtype(), &distinct);

    // 1. dict_encode roundtrip + exact distinct count and first-seen dictionary order check.
    let dict = dict_encode(&arr, &mut ctx).vortex_expect("dict_encode");
    assert_eq!(dict.values().dtype(), arr.dtype());
    assert_eq!(dict.values().len(), expected_distinct);
    for (code, expected) in distinct.iter().enumerate() {
        assert_eq!(
            dict.values()
                .execute_scalar(code, &mut ctx)
                .vortex_expect("dict value"),
            *expected,
            "dict_encode values must preserve first-seen order at code {code}"
        );
    }
    assert_arrays_eq!(&dict, &arr, &mut ctx);

    // 2. dict_encode_by_frequency roundtrip + non-increasing code frequency check.
    let freq_dict = dict_encode_by_frequency(&arr, &mut ctx).vortex_expect("dict_encode_by_freq");
    assert!(freq_dict.has_all_values_referenced());
    assert_eq!(freq_dict.values().len(), expected_distinct);
    assert_arrays_eq!(&freq_dict, &arr, &mut ctx);
    let codes_prim = freq_dict
        .codes()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .vortex_expect("codes");
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for i in 0..codes_prim.len() {
        let code = usize::try_from(&codes_prim.execute_scalar(i, &mut ctx).vortex_expect("c"))
            .vortex_expect("usize");
        *counts.entry(code).or_default() += 1;
    }
    for code in 1..freq_dict.values().len() {
        assert!(
            counts.get(&(code - 1)).copied().unwrap_or(0)
                >= counts.get(&code).copied().unwrap_or(0),
            "frequency codes must be non-increasing"
        );
    }

    // 3. Constrained frequency encode: exact `Some`/`None` contract on `max_len` (the first entry
    // is always admitted so callers make progress even when `max_len == 0`), plus finite `max_bytes`.
    let max_len = tc.draw(gs::integers::<usize>().max_value(8));
    let len_constraints = DictConstraints {
        max_bytes: usize::MAX,
        max_len,
    };
    let constrained_by_len =
        dict_encode_by_frequency_with_constraints(&arr, &len_constraints, &mut ctx)
            .vortex_expect("constrained max_len");
    if expected_distinct <= max_len.max(1) {
        let constrained = constrained_by_len
            .vortex_expect("inputs within max_len.max(1) must succeed under unconstrained bytes");
        assert_eq!(constrained.values().len(), expected_distinct);
        assert_arrays_eq!(constrained, &arr, &mut ctx);
    } else {
        assert!(
            constrained_by_len.is_none(),
            "inputs exceeding max_len must return None"
        );
    }

    let max_bytes = tc.draw(gs::integers::<usize>().max_value(160));
    let byte_constraints = DictConstraints {
        max_bytes,
        max_len: usize::MAX,
    };
    let constrained_by_bytes =
        dict_encode_by_frequency_with_constraints(&arr, &byte_constraints, &mut ctx)
            .vortex_expect("constrained max_bytes");
    if expected_distinct <= 1 || dict_bytes <= max_bytes {
        let constrained = constrained_by_bytes
            .vortex_expect("inputs within max_bytes (or <=1 distinct entry) must succeed");
        assert_eq!(constrained.values().len(), expected_distinct);
        assert_arrays_eq!(constrained, &arr, &mut ctx);
    } else {
        assert!(
            constrained_by_bytes.is_none(),
            "inputs exceeding max_bytes must return None"
        );
    }

    // 4. Streaming multi-chunk encoder + reset() clearing stale dictionary entries.
    if len > 0 {
        let chunk2 = make_chunk(
            &tc,
            tc.draw(gs::integers::<usize>().min_value(1).max_value(8)),
            &mut ctx,
        );
        let chunk2_distinct = distinct_scalars(&chunk2, &mut ctx).len();

        let mut encoder = dict_encoder_in(&arr, &UNCONSTRAINED, ctx.allocator().clone());
        let c1 = encoder.encode(&arr, &mut ctx).vortex_expect("enc1");
        let c2 = encoder.encode(&chunk2, &mut ctx).vortex_expect("enc2");
        let values = encoder.reset();
        let combined_codes =
            ChunkedArray::from_iter([c1.into_array(), c2.into_array()]).into_array();
        let decoded = DictArray::try_new(combined_codes, values)
            .vortex_expect("dict")
            .into_array();
        let expected =
            ChunkedArray::try_new(vec![arr.clone(), chunk2.clone()], arr.dtype().clone())
                .vortex_expect("expected")
                .into_array();
        assert_arrays_eq!(decoded, expected, &mut ctx);

        // Second batch after reset() must start fresh: values contain only chunk2's distinct
        // entries and the first emitted code is 0.
        let c3 = encoder.encode(&chunk2, &mut ctx).vortex_expect("enc3");
        let values2 = encoder.reset();
        assert_eq!(
            values2.len(),
            chunk2_distinct,
            "reset() must clear stale dictionary entries from the previous batch"
        );
        assert_eq!(
            usize::try_from(&c3.execute_scalar(0, &mut ctx).vortex_expect("first code"))
                .vortex_expect("usize"),
            0,
            "codes after reset() must start at 0"
        );
        let decoded2 = DictArray::try_new(c3.into_array(), values2)
            .vortex_expect("dict2")
            .into_array();
        assert_arrays_eq!(decoded2, chunk2, &mut ctx);
    }
}

// --- API signature, allocator reuse, panic/error guards, and physical-layout tests ---

#[derive(Debug)]
struct CountingAllocator {
    allocations: Arc<AtomicUsize>,
}

// SAFETY: this forwards memory operations to Global and only counts allocations.
unsafe impl Allocator for CountingAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: ptr and layout came from Global.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

#[test]
#[allow(deprecated, unused_qualifications)]
fn deprecated_builder_constructors_keep_their_signatures() {
    let _: fn(Nullability) -> super::BoolBuilder = super::BoolBuilder::new;
    let _: fn(Nullability, usize) -> super::BoolBuilder = super::BoolBuilder::with_capacity;
    let _: fn(Nullability) -> super::PrimitiveBuilder<i32> = super::PrimitiveBuilder::<i32>::new;
    let _: fn(Nullability, usize) -> super::PrimitiveBuilder<i32> =
        super::PrimitiveBuilder::<i32>::with_capacity;
    let _: fn(DecimalDType, Nullability) -> super::DecimalBuilder = super::DecimalBuilder::new;
    let _: fn(usize, DecimalDType, Nullability) -> super::DecimalBuilder =
        super::DecimalBuilder::with_capacity;
    let _: fn(ExtDTypeRef) -> super::ExtensionBuilder = super::ExtensionBuilder::new;
    let _: fn(ExtDTypeRef, usize) -> super::ExtensionBuilder =
        super::ExtensionBuilder::with_capacity;
    let _: fn(Arc<DType>, u32, Nullability) -> super::FixedSizeListBuilder =
        super::FixedSizeListBuilder::new;
    let _: fn(Arc<DType>, u32, Nullability, usize) -> super::FixedSizeListBuilder =
        super::FixedSizeListBuilder::with_capacity;
    let _: fn(Arc<DType>, Nullability) -> super::ListBuilder<u32> = super::ListBuilder::<u32>::new;
    let _: fn(Arc<DType>, Nullability, usize, usize) -> super::ListBuilder<u32> =
        super::ListBuilder::<u32>::with_capacity;
    let _: fn(Arc<DType>, Nullability) -> super::ListViewBuilder<u32, u32> =
        super::ListViewBuilder::<u32, u32>::new;
    let _: fn(Arc<DType>, Nullability, usize, usize) -> super::ListViewBuilder<u32, u32> =
        super::ListViewBuilder::<u32, u32>::with_capacity;
    let _: fn(crate::dtype::MapDType, Nullability) -> super::MapBuilder<u32, u32> =
        super::MapBuilder::<u32, u32>::new;
    let _: fn(crate::dtype::MapDType, Nullability, usize) -> super::MapBuilder<u32, u32> =
        super::MapBuilder::<u32, u32>::with_capacity;
    let _: fn(StructFields, Nullability) -> super::StructBuilder = super::StructBuilder::new;
    let _: fn(StructFields, Nullability, usize) -> super::StructBuilder =
        super::StructBuilder::with_capacity;
    let _: fn(DType) -> super::VarBinBuilder<i32> = super::VarBinBuilder::<i32>::new;
    let _: fn(DType, usize) -> super::VarBinBuilder<i32> =
        super::VarBinBuilder::<i32>::with_capacity;
    let _: fn(DType, usize, usize) -> super::VarBinBuilder<i32> =
        super::VarBinBuilder::<i32>::with_capacity_bytes;
    let _: fn(DType, usize) -> super::VarBinViewBuilder = super::VarBinViewBuilder::with_capacity;
    let _: fn(DType, usize) -> super::VarBinViewBuilder =
        super::VarBinViewBuilder::with_buffer_deduplication;
    let _: fn(DType, usize, f64) -> super::VarBinViewBuilder =
        super::VarBinViewBuilder::with_compaction;
    let _: fn(
        DType,
        usize,
        super::CompletedBuffers,
        super::BufferGrowthStrategy,
        f64,
    ) -> super::VarBinViewBuilder = super::VarBinViewBuilder::new;
    let _: fn(&DType, usize) -> Box<dyn ArrayBuilder> = super::builder_with_capacity;
    let _: fn(&DType, usize, &BufferAllocatorRef) -> Box<dyn ArrayBuilder> =
        super::builder_with_capacity_in;
}

#[test]
fn builder_reuses_its_allocator_after_finish() {
    let allocations = Arc::new(AtomicUsize::new(0));
    let allocator = BufferAllocatorRef::new(CountingAllocator {
        allocations: Arc::clone(&allocations),
    });
    let dtype = DType::Struct(
        StructFields::from_iter([
            (
                "number",
                DType::Primitive(PType::I32, Nullability::Nullable),
            ),
            ("text", DType::Utf8(Nullability::Nullable)),
        ]),
        Nullability::Nullable,
    );
    let mut builder = builder_with_capacity_in(&dtype, 1, &allocator);

    builder.append_null();
    drop(builder.finish());
    let first = allocations.load(Ordering::Relaxed);

    builder.append_null();
    drop(builder.finish());
    let second = allocations.load(Ordering::Relaxed);

    assert!(
        second - first >= 5,
        "all nested buffers must reuse the allocator"
    );
}

#[rstest]
#[case::bool(DType::Bool(Nullability::NonNullable))]
#[case::i32(DType::Primitive(PType::I32, Nullability::NonNullable))]
#[case::f64(DType::Primitive(PType::F64, Nullability::NonNullable))]
#[case::utf8(DType::Utf8(Nullability::NonNullable))]
#[case::binary(DType::Binary(Nullability::NonNullable))]
#[case::decimal(DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable))]
#[case::list(DType::List(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    Nullability::NonNullable
))]
#[case::fixed_size_list(DType::FixedSizeList(
    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
    3,
    Nullability::NonNullable
))]
#[case::struct_type(DType::Struct(
    StructFields::from_iter([("a", DType::Primitive(PType::I32, Nullability::NonNullable))]),
    Nullability::NonNullable
))]
#[case::extension(DType::Extension(
    Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()
))]
#[should_panic(expected = "non-nullable")]
fn test_append_nulls_panics_on_non_nullable(#[case] dtype: DType, #[values(1, 3)] count: usize) {
    let mut builder = builder_with_capacity_in(&dtype, count, BufferAllocatorRef::static_ref());
    builder.append_nulls(count);
}

#[rstest]
#[case::null(DType::Null, Scalar::from(42i32))]
#[case::bool(DType::Bool(Nullability::NonNullable), Scalar::from(42i32))]
#[case::i32(
    DType::Primitive(PType::I32, Nullability::NonNullable),
    Scalar::from(true)
)]
#[case::decimal(
    DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable),
    Scalar::from(true)
)]
#[case::utf8(
    DType::Utf8(Nullability::NonNullable),
    Scalar::binary(vec![0u8, 1, 2], Nullability::NonNullable)
)]
#[case::struct_(
    DType::Struct(
        StructFields::from_iter([("a", DType::Primitive(PType::I32, Nullability::NonNullable))]),
        Nullability::NonNullable
    ),
    Scalar::from(42i32)
)]
#[case::list(
    DType::List(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        Nullability::NonNullable
    ),
    Scalar::from(42i32)
)]
#[case::fixed_size_list(
    DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        2,
        Nullability::NonNullable
    ),
    Scalar::from(42i32)
)]
#[case::extension(
    DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
    Scalar::from(true)
)]
fn test_append_scalar_wrong_dtype_rejection(#[case] dtype: DType, #[case] wrong: Scalar) {
    let mut builder = builder_with_capacity_in(&dtype, 1, BufferAllocatorRef::static_ref());
    assert!(
        builder.append_scalar(&wrong).is_err(),
        "builder for {dtype} should reject scalar of dtype {}",
        wrong.dtype()
    );
}

/// Builders only promise a canonical *top level*, so a child array that is long enough to be worth
/// a chunk must come back out of the builder in the encoding it went in with.
///
/// Each case appends the same array twice, which additionally checks that the two chunks are
/// stitched back together into a [`ChunkedArray`] rather than being decoded and concatenated.
#[rstest]
#[case::struct_field(
    StructArray::try_from_iter([("a", constant_i32())])
        .vortex_expect("struct array")
        .into_array(),
    |array: &ArrayRef| array.as_::<Struct>().unmasked_field(0).clone()
)]
#[case::list_elements(
    ListViewArray::new(
        constant_i32(),
        (0..CHUNK_LEN as u64).collect::<Buffer<_>>().into_array(),
        Buffer::full(1u64, CHUNK_LEN).into_array(),
        Validity::NonNullable,
        &mut SESSION.create_execution_ctx(),
    )
    .into_array(),
    |array: &ArrayRef| array.as_::<ListView>().elements().clone()
)]
#[case::fixed_size_list_elements(
    FixedSizeListArray::new(
        constant_i32(),
        2,
        Validity::NonNullable,
        CHUNK_LEN / 2,
    )
    .into_array(),
    |array: &ArrayRef| array.as_::<FixedSizeList>().elements().clone()
)]
#[case::extension_storage(
    ExtensionArray::new(
        Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        ConstantArray::new(
            Scalar::primitive(0i64, Nullability::NonNullable),
            CHUNK_LEN,
        )
        .into_array(),
    )
    .into_array(),
    |array: &ArrayRef| array.as_::<Extension>().storage().clone()
)]
fn test_children_are_not_canonicalized(
    #[case] array: ArrayRef,
    #[case] child_of: fn(&ArrayRef) -> ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let mut builder = builder_with_capacity_in(array.dtype(), 0, BufferAllocatorRef::static_ref());
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    let built = builder.finish();

    let child = child_of(&built);
    let chunked = child.as_::<Chunked>();
    assert_eq!(
        chunked.nchunks(),
        2,
        "expected one chunk per appended array"
    );
    assert!(
        chunked.iter_chunks().all(|chunk| chunk.is::<Constant>()),
        "the constant-encoded child was decoded by the builder",
    );

    let expected = ChunkedArray::try_new(vec![array.clone(), array], built.dtype().clone())?;
    assert_arrays_eq!(&built, &expected, &mut ctx);
    Ok(())
}

/// A child is chunked on the boundaries it is appended on, however small the appends. Appending
/// scalars instead is what asks the builder to copy the values into one canonical child.
#[test]
fn test_children_are_chunked_on_the_boundaries_they_are_appended_on() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let elements = ConstantArray::new(1i32, 2).into_array();
    let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 1).into_array();

    let mut builder = builder_with_capacity_in(array.dtype(), 0, BufferAllocatorRef::static_ref());
    for _ in 0..CHUNK_LEN {
        array.append_to_builder(builder.as_mut(), &mut ctx)?;
    }
    let built = builder.finish();

    assert_eq!(built.len(), CHUNK_LEN);
    assert_eq!(
        built
            .as_::<FixedSizeList>()
            .elements()
            .as_::<Chunked>()
            .nchunks(),
        CHUNK_LEN,
        "one chunk per appended array",
    );

    // The same values appended as scalars land in a single canonical child.
    let mut builder = builder_with_capacity_in(array.dtype(), 0, BufferAllocatorRef::static_ref());
    let scalar = array.execute_scalar(0, &mut ctx)?;
    for _ in 0..CHUNK_LEN {
        builder.append_scalar(&scalar)?;
    }
    let built_from_scalars = builder.finish();

    assert!(
        built_from_scalars
            .as_::<FixedSizeList>()
            .elements()
            .is::<Primitive>()
    );

    let expected = FixedSizeListArray::new(
        buffer![1i32; 2 * CHUNK_LEN].into_array(),
        2,
        Validity::NonNullable,
        CHUNK_LEN,
    );
    assert_arrays_eq!(built, &expected, &mut ctx);
    assert_arrays_eq!(built_from_scalars, expected, &mut ctx);
    Ok(())
}

/// A builder that mixes appended arrays with scalar appends must keep the two in order.
#[test]
fn test_struct_builder_interleaves_arrays_and_scalars() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let array = StructArray::try_from_iter([("a", constant_i32())])?.into_array();
    let scalar = Scalar::struct_(
        array.dtype().clone(),
        vec![Scalar::primitive(1i32, Nullability::NonNullable)],
    );

    let mut builder = builder_with_capacity_in(array.dtype(), 0, BufferAllocatorRef::static_ref());
    builder.append_scalar(&scalar)?;
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    builder.append_scalar(&scalar)?;
    let built = builder.finish();

    let values: PrimitiveArray = [1i32]
        .into_iter()
        .chain(std::iter::repeat_n(0i32, CHUNK_LEN))
        .chain([1i32])
        .collect();
    let independent = StructArray::try_from_iter([("a", values)])?;
    assert_arrays_eq!(built, independent, &mut ctx);
    Ok(())
}

/// An arbitrary array length. Nested builders treat no length specially, so the tests only need a
/// length long enough to tell chunks apart.
const CHUNK_LEN: usize = 64;

/// A nested builder's own validity is accumulated the same way its children are: an appended
/// array's validity is kept as it arrived rather than executed into a mask and copied bit by bit.
#[test]
fn test_appended_validity_is_not_materialized() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let array = StructArray::try_from_iter_with_validity(
        [("a", iota(CHUNK_LEN))],
        Validity::from_iter((0..CHUNK_LEN).map(|i| i % 3 != 0)),
    )?
    .into_array();

    let mut builder = builder_with_capacity_in(array.dtype(), 0, BufferAllocatorRef::static_ref());
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    let built = builder.finish();

    let Validity::Array(validity) = built.validity()? else {
        panic!("expected array-backed validity");
    };
    assert!(
        validity.is::<Chunked>(),
        "the two appended validities should have been concatenated, not copied into one buffer",
    );

    let expected = ChunkedArray::try_new(vec![array.clone(), array], built.dtype().clone())?;
    assert_arrays_eq!(&built, &expected, &mut ctx);
    Ok(())
}

/// A non-canonical array of [`CHUNK_LEN`] `i32` values.
fn constant_i32() -> ArrayRef {
    ConstantArray::new(0i32, CHUNK_LEN).into_array()
}

/// Two lists of [`CHUNK_LEN`] elements each, so that appending them chunks the elements.
fn two_lists_of_chunk_len() -> ListViewArray {
    let mut ctx = SESSION.create_execution_ctx();
    ListViewArray::new(
        iota(2 * CHUNK_LEN),
        u64s([0, CHUNK_LEN]),
        u64s([CHUNK_LEN, CHUNK_LEN]),
        Validity::NonNullable,
        &mut ctx,
    )
}

/// `0..n` as an `i32` array, so that the values of one chunk are distinguishable from the next.
fn iota(n: usize) -> ArrayRef {
    (0..n)
        .map(|i| i32::try_from(i).vortex_expect("iota value fits in an i32"))
        .collect::<Buffer<_>>()
        .into_array()
}

/// A `u64` array of list offsets or sizes.
fn u64s(values: impl IntoIterator<Item = usize>) -> ArrayRef {
    values
        .into_iter()
        .map(|v| u64::try_from(v).vortex_expect("list offset fits in a u64"))
        .collect::<Buffer<_>>()
        .into_array()
}

/// Once a list builder keeps its elements as chunks, `elements_builder.len()` is a running total
/// across those chunks — every offset appended afterwards has to be rebased onto it.
///
/// The two cases cover the two bulk paths into the elements builder: appending a `ListViewArray`
/// rebases the view's own offsets, while appending a `ListArray` slices the elements first.
#[rstest]
#[case::from_listview(two_lists_of_chunk_len().into_array())]
#[case::from_list(
    ListArray::new(
        iota(2 * CHUNK_LEN),
        u64s([0, CHUNK_LEN, 2 * CHUNK_LEN]),
        Validity::NonNullable,
        &mut SESSION.create_execution_ctx(),
    )
    .into_array()
)]
fn test_list_offsets_are_rebased_across_element_chunks(
    #[case] lists: ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let mut builder = builder_with_capacity_in(
        &DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable,
        ),
        0,
        BufferAllocatorRef::static_ref(),
    );
    lists.append_to_builder(builder.as_mut(), &mut ctx)?;
    lists.append_to_builder(builder.as_mut(), &mut ctx)?;
    let built = builder.finish();

    assert!(
        built.as_::<ListView>().elements().is::<Chunked>(),
        "the elements should have been kept as chunks",
    );

    let independent = ListArray::new(
        (0..128i32)
            .chain(0..128i32)
            .collect::<Buffer<_>>()
            .into_array(),
        buffer![0u32, 64, 128, 192, 256].into_array(),
        Validity::NonNullable,
        &mut ctx,
    );
    assert_arrays_eq!(built, independent, &mut ctx);
    Ok(())
}

/// `ListBuilder` computes each offset from the running element count as well, and reaches the
/// elements builder through `append_array_as_list` rather than a bulk append.
#[test]
fn test_list_builder_offsets_are_rebased_across_element_chunks() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

    let mut builder = ListBuilder::<u64>::with_capacity_in(
        element_dtype,
        Nullability::NonNullable,
        0,
        BufferAllocatorRef::static_ref(),
    );
    for value in 0..3i32 {
        builder
            .append_array_as_list(&ConstantArray::new(value, CHUNK_LEN).into_array(), &mut ctx)?;
    }
    let built = builder.finish();

    assert!(built.as_::<List>().elements().is::<Chunked>());

    let expected = ListArray::new(
        (0..3i32)
            .flat_map(|value| std::iter::repeat_n(value, CHUNK_LEN))
            .collect::<Buffer<_>>()
            .into_array(),
        u64s((0..=3).map(|i| i * CHUNK_LEN)),
        Validity::NonNullable,
        &mut ctx,
    );
    assert_arrays_eq!(&built, &expected, &mut ctx);
    Ok(())
}

/// A nested builder's own validity buffer is independent of its chunked child, so nulls appended
/// alongside chunks must survive.
#[rstest]
#[case::fixed_size_list(
    FixedSizeListArray::new(
        iota(CHUNK_LEN),
        4,
        Validity::from_iter((0..CHUNK_LEN / 4).map(|i| i % 3 != 0)),
        CHUNK_LEN / 4,
    )
    .into_array(),
    |array: &ArrayRef| array.as_::<FixedSizeList>().elements().clone()
)]
#[case::struct_(
    StructArray::try_from_iter_with_validity(
        [("a", iota(CHUNK_LEN))],
        Validity::from_iter((0..CHUNK_LEN).map(|i| i % 3 != 0)),
    )
    .vortex_expect("struct array")
    .into_array(),
    |array: &ArrayRef| array.as_::<Struct>().unmasked_field(0).clone()
)]
fn test_validity_survives_chunked_children(
    #[case] array: ArrayRef,
    #[case] child_of: fn(&ArrayRef) -> ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let mut builder = builder_with_capacity_in(array.dtype(), 0, BufferAllocatorRef::static_ref());
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    builder.append_nulls(1);
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    let built = builder.finish();

    assert!(child_of(&built).is::<Chunked>());
    assert_eq!(built.len(), array.len() * 2 + 1);

    let null = ConstantArray::new(Scalar::null(built.dtype().clone()), 1).into_array();
    let expected = ChunkedArray::try_new(vec![array.clone(), null, array], built.dtype().clone())?;
    assert_arrays_eq!(&built, &expected, &mut ctx);
    Ok(())
}

/// Consumers that genuinely need a fully-decoded tree ask for it, and must still get one.
#[test]
fn test_chunked_children_canonicalize_recursively() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let array = StructArray::try_from_iter([("a", constant_i32())])?.into_array();
    let mut builder = builder_with_capacity_in(array.dtype(), 0, BufferAllocatorRef::static_ref());
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    array.append_to_builder(builder.as_mut(), &mut ctx)?;
    let built = builder.finish();

    let recursive = built.execute::<RecursiveCanonical>(&mut ctx)?.0;
    assert!(
        recursive
            .clone()
            .into_array()
            .as_::<Struct>()
            .unmasked_field(0)
            .is::<Primitive>()
    );
    let independent =
        StructArray::try_from_iter([("a", buffer![0i32; 2 * CHUNK_LEN].into_array())])?;
    assert_arrays_eq!(&recursive.into_array(), &independent, &mut ctx);
    Ok(())
}

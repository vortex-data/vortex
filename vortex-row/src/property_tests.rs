// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Property-based differential tests for row-oriented encoding.
//!
//! Generates multi-column Vortex tables across all supported logical types, sort options, and
//! physical encodings, and checks that row encoding agrees with an independent logical comparator,
//! matches [`RowEncoder::row_sizes`], canonicalizes null parent payloads, and produces
//! byte-identical output for canonical vs. non-canonical (sliced, chunked, dict, constant,
//! widened) inputs.

use std::cmp::Ordering;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::WidenArray;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::RowEncoder;
use crate::RowSortField;
use crate::tests::collect_row_sizes;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Independent logical value model used as the reference oracle for row comparisons.
#[derive(Clone, Debug)]
enum LogicalValue {
    Null,
    Bool(bool),
    Int(i128),
    F16(f16),
    F32(f32),
    F64(f64),
    Decimal(i256),
    Bytes(Vec<u8>),
    Struct(Vec<LogicalValue>),
    FixedSizeList(Vec<LogicalValue>),
    List(Vec<LogicalValue>),
}

fn cmp_logical(lhs: &LogicalValue, rhs: &LogicalValue, field: RowSortField) -> Ordering {
    match (lhs, rhs) {
        (LogicalValue::Null, LogicalValue::Null) => Ordering::Equal,
        (LogicalValue::Null, _) => {
            if field.nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (_, LogicalValue::Null) => {
            if field.nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (LogicalValue::Struct(xs), LogicalValue::Struct(ys))
        | (LogicalValue::FixedSizeList(xs), LogicalValue::FixedSizeList(ys)) => {
            assert_eq!(xs.len(), ys.len());
            xs.iter()
                .zip(ys)
                .find_map(|(x, y)| match cmp_logical(x, y, field) {
                    Ordering::Equal => None,
                    ord => Some(ord),
                })
                .unwrap_or(Ordering::Equal)
        }
        (LogicalValue::List(xs), LogicalValue::List(ys)) => {
            for (x, y) in xs.iter().zip(ys) {
                let ord = cmp_logical(x, y, field);
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            let len_ord = xs.len().cmp(&ys.len());
            if field.descending {
                len_ord.reverse()
            } else {
                len_ord
            }
        }
        _ => {
            let ord = match (lhs, rhs) {
                (LogicalValue::Bool(x), LogicalValue::Bool(y)) => x.cmp(y),
                (LogicalValue::Int(x), LogicalValue::Int(y)) => x.cmp(y),
                (LogicalValue::F16(x), LogicalValue::F16(y)) => x.total_cmp(y),
                (LogicalValue::F32(x), LogicalValue::F32(y)) => x.total_cmp(y),
                (LogicalValue::F64(x), LogicalValue::F64(y)) => x.total_cmp(y),
                (LogicalValue::Decimal(x), LogicalValue::Decimal(y)) => x.cmp(y),
                (LogicalValue::Bytes(x), LogicalValue::Bytes(y)) => x.cmp(y),
                _ => unreachable!("mismatched logical value types: {lhs:?} vs {rhs:?}"),
            };
            if field.descending { ord.reverse() } else { ord }
        }
    }
}

fn collect_rows_and_total_bytes(
    encoded: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<Vec<u8>>, usize)> {
    let mut rows = Vec::with_capacity(encoded.len());
    for i in 0..encoded.len() {
        let slice = encoded.list_elements_at(i, ctx)?;
        rows.push(
            slice
                .execute::<PrimitiveArray>(ctx)?
                .as_slice::<u8>()
                .to_vec(),
        );
    }
    Ok((rows, encoded.elements().len()))
}

struct GeneratedColumn {
    canonical: ArrayRef,
    alternate: ArrayRef,
    logical: Vec<LogicalValue>,
}

fn finish_column(
    tc: &TestCase,
    canonical: ArrayRef,
    logical: Vec<LogicalValue>,
) -> VortexResult<GeneratedColumn> {
    let alternate = make_alternate_encoding(tc, &canonical)?;
    Ok(GeneratedColumn {
        canonical,
        alternate,
        logical,
    })
}

fn draw_validity(tc: &TestCase, len: usize) -> (Validity, Vec<bool>) {
    match tc.draw(gs::integers::<u8>().max_value(3)) {
        0 => (Validity::NonNullable, vec![true; len]),
        1 => (Validity::AllValid, vec![true; len]),
        2 => (Validity::AllInvalid, vec![false; len]),
        _ => {
            let bits = tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len));
            (
                Validity::from(BitBuffer::from_iter(bits.iter().copied())),
                bits,
            )
        }
    }
}

fn make_alternate_encoding(tc: &TestCase, canonical: &ArrayRef) -> VortexResult<ArrayRef> {
    let len = canonical.len();
    if len == 0 {
        return Ok(canonical.clone());
    }
    match tc.draw(gs::integers::<u8>().max_value(3)) {
        0 => Ok(canonical.clone()),
        // Sliced array from a larger concatenated array (non-zero slice offset).
        1 => {
            let prefix_len = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
            let mut ctx = SESSION.create_execution_ctx();
            let prefix =
                ConstantArray::new(canonical.execute_scalar(0, &mut ctx)?, prefix_len).into_array();
            let chunked =
                ChunkedArray::try_new(vec![prefix, canonical.clone()], canonical.dtype().clone())?
                    .into_array();
            chunked
                .execute::<Canonical>(&mut ctx)?
                .into_array()
                .slice(prefix_len..prefix_len + len)
        }
        // Chunked array split into 2 chunks.
        2 => {
            let split = tc.draw(gs::integers::<usize>().max_value(len));
            let left = canonical.slice(0..split)?;
            let right = canonical.slice(split..len)?;
            Ok(ChunkedArray::try_new(vec![left, right], canonical.dtype().clone())?.into_array())
        }
        // Dict-encoded array with identity codes.
        _ => {
            let codes =
                PrimitiveArray::from_iter((0..len).map(|i| u32::try_from(i).unwrap())).into_array();
            Ok(DictArray::try_new(codes, canonical.clone())?.into_array())
        }
    }
}

fn draw_varlen_bytes(tc: &TestCase, is_utf8: bool) -> Vec<u8> {
    const INTERESTING_LENS: &[usize] = &[0, 1, 11, 12, 13, 31, 32, 33, 63, 64, 65, 95, 96, 97, 128];
    let len = if tc.draw(gs::booleans()) {
        tc.draw(gs::sampled_from(INTERESTING_LENS))
    } else {
        tc.draw(gs::integers::<usize>().max_value(128))
    };
    if len == 0 {
        return Vec::new();
    }
    if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
        // Repeated byte pattern (often shared across rows to hit exact block boundary comparisons).
        let fill = if is_utf8 {
            tc.draw(gs::sampled_from(b"\0abz"))
        } else {
            tc.draw(gs::sampled_from(&[0x00u8, 0x01, 0x02, 0x7F, 0xFE, 0xFF]))
        };
        let mut out = vec![fill; len];
        if tc.draw(gs::booleans()) {
            let tail = if is_utf8 {
                tc.draw(gs::sampled_from(b"\0ab"))
            } else {
                tc.draw(gs::sampled_from(&[0x00u8, 0x01, 0xFF]))
            };
            *out.last_mut().unwrap() = tail;
        }
        out
    } else if is_utf8 {
        (0..len)
            .map(|_| tc.draw(gs::sampled_from(b"\0abcxyz")))
            .collect()
    } else {
        tc.draw(gs::vecs(gs::integers::<u8>()).min_size(len).max_size(len))
    }
}

// Float bit patterns covering -NaN, -Inf, -1.0, -0.0, +0.0, +1.0, +Inf, +qNaN, +sNaN.
fn draw_f16(tc: &TestCase) -> f16 {
    const SPECIALS: &[u16] = &[
        0xFC01, 0xFC00, 0xBC00, 0x8000, 0x0000, 0x3C00, 0x7C00, 0x7E00, 0x7C01,
    ];
    let bits = if tc.draw(gs::booleans()) {
        tc.draw(gs::sampled_from(SPECIALS))
    } else {
        tc.draw(gs::integers::<u16>())
    };
    f16::from_bits(bits)
}

fn draw_f32(tc: &TestCase) -> f32 {
    const SPECIALS: &[u32] = &[
        0xFFC0_0001,
        0xFF80_0000,
        0xBF80_0000,
        0x8000_0000,
        0x0000_0000,
        0x3F80_0000,
        0x7F80_0000,
        0x7FC0_0000,
        0x7F80_0001,
    ];
    let bits = if tc.draw(gs::booleans()) {
        tc.draw(gs::sampled_from(SPECIALS))
    } else {
        tc.draw(gs::integers::<u32>())
    };
    f32::from_bits(bits)
}

fn draw_f64(tc: &TestCase) -> f64 {
    const SPECIALS: &[u64] = &[
        0xFFF8_0000_0000_0001,
        0xFFF0_0000_0000_0000,
        0xBFF0_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0x7FF0_0000_0000_0000,
        0x7FF8_0000_0000_0000,
        0x7FF0_0000_0000_0001,
    ];
    let bits = if tc.draw(gs::booleans()) {
        tc.draw(gs::sampled_from(SPECIALS))
    } else {
        tc.draw(gs::integers::<u64>())
    };
    f64::from_bits(bits)
}

fn draw_primitive<T: NativePType>(
    tc: &TestCase,
    len: usize,
    mut draw_val: impl FnMut(&TestCase) -> T,
    to_logical: impl Fn(T) -> LogicalValue,
) -> VortexResult<GeneratedColumn> {
    let (validity, valid_bits) = draw_validity(tc, len);
    let vals: Vec<T> = (0..len).map(|_| draw_val(tc)).collect();
    let logical = vals
        .iter()
        .zip(&valid_bits)
        .map(|(&v, &ok)| {
            if ok {
                to_logical(v)
            } else {
                LogicalValue::Null
            }
        })
        .collect();
    let arr = PrimitiveArray::new(Buffer::from_iter(vals), validity).into_array();
    finish_column(tc, arr, logical)
}

fn draw_int<T: NativePType + Into<i128>>(
    tc: &TestCase,
    len: usize,
    draw_val: impl FnMut(&TestCase) -> T,
) -> VortexResult<GeneratedColumn> {
    draw_primitive(tc, len, draw_val, |v| LogicalValue::Int(v.into()))
}

fn draw_decimal_typed<T: NativeDecimalType>(
    tc: &TestCase,
    len: usize,
    dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
    mut draw_val: impl FnMut(&TestCase) -> T,
    to_i256: impl Fn(T) -> i256,
) -> VortexResult<GeneratedColumn> {
    let (validity, valid_bits) = draw_validity(tc, len);
    let raw: Vec<T> = (0..len).map(|_| draw_val(tc)).collect();
    let logical = raw
        .iter()
        .zip(&valid_bits)
        .map(|(&v, &ok)| {
            if ok {
                LogicalValue::Decimal(to_i256(v))
            } else {
                LogicalValue::Null
            }
        })
        .collect();
    let arr = DecimalArray::new(Buffer::from_iter(raw), dtype, validity, ctx).into_array();
    finish_column(tc, arr, logical)
}

fn draw_decimal_int<T: NativeDecimalType + Into<i128>>(
    tc: &TestCase,
    len: usize,
    dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
    draw_val: impl FnMut(&TestCase) -> T,
) -> VortexResult<GeneratedColumn> {
    draw_decimal_typed(tc, len, dtype, ctx, draw_val, |v| i256::from_i128(v.into()))
}

fn draw_decimal_column(
    tc: &TestCase,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<GeneratedColumn> {
    match tc.draw(gs::integers::<u8>().max_value(5)) {
        0 => draw_decimal_int(tc, len, DecimalDType::new(2, 1), ctx, |tc| {
            tc.draw(gs::integers::<i8>().min_value(-99).max_value(99))
        }),
        1 => draw_decimal_int(tc, len, DecimalDType::new(4, 2), ctx, |tc| {
            tc.draw(gs::integers::<i16>().min_value(-9999).max_value(9999))
        }),
        2 if tc.draw(gs::booleans()) && len > 0 => {
            let dtype = DecimalDType::new(7, 2);
            let (validity, valid_bits) = draw_validity(tc, len);
            let raw: Vec<i8> = (0..len)
                .map(|_| tc.draw(gs::integers::<i8>().min_value(-99).max_value(99)))
                .collect();
            let logical = raw
                .iter()
                .zip(&valid_bits)
                .map(|(&v, &ok)| {
                    if ok {
                        LogicalValue::Decimal(i256::from_i128(i128::from(v)))
                    } else {
                        LogicalValue::Null
                    }
                })
                .collect();
            let narrow = DecimalArray::new(
                Buffer::from_iter(raw.iter().copied()),
                DecimalDType::new(2, 2),
                validity.clone(),
                ctx,
            )
            .into_array();
            let widened =
                WidenArray::try_new(narrow, DType::Decimal(dtype, validity.nullability()))?
                    .into_array();
            let canonical = DecimalArray::new(
                Buffer::from_iter(raw.into_iter().map(i32::from)),
                dtype,
                validity,
                ctx,
            )
            .into_array();
            Ok(GeneratedColumn {
                canonical,
                alternate: widened,
                logical,
            })
        }
        2 => draw_decimal_int(tc, len, DecimalDType::new(7, 2), ctx, |tc| {
            tc.draw(
                gs::integers::<i32>()
                    .min_value(-9_999_999)
                    .max_value(9_999_999),
            )
        }),
        3 => draw_decimal_int(tc, len, DecimalDType::new(15, 3), ctx, |tc| {
            tc.draw(
                gs::integers::<i64>()
                    .min_value(-999_999_999_999_999)
                    .max_value(999_999_999_999_999),
            )
        }),
        4 => {
            let max_v: i128 = 10i128.pow(30) - 1;
            draw_decimal_int(tc, len, DecimalDType::new(30, 4), ctx, |tc| {
                tc.draw(gs::integers::<i128>().min_value(-max_v).max_value(max_v))
            })
        }
        _ => {
            let dtype = DecimalDType::new(50, 6);
            assert_eq!(dtype.canonical_values_type(), DecimalType::I256);
            draw_decimal_typed(
                tc,
                len,
                dtype,
                ctx,
                |tc| {
                    let lo = tc.draw(gs::integers::<u128>());
                    // Keep high limb small so the combined i256 fits within 50 decimal digits.
                    let hi = tc.draw(
                        gs::integers::<i128>()
                            .min_value(-1_000_000)
                            .max_value(1_000_000),
                    );
                    i256::from_parts(lo, hi)
                },
                |v| v,
            )
        }
    }
}

fn draw_varlen_column(tc: &TestCase, len: usize, is_utf8: bool) -> VortexResult<GeneratedColumn> {
    let (validity, valid_bits) = draw_validity(tc, len);
    let vals: Vec<Vec<u8>> = (0..len).map(|_| draw_varlen_bytes(tc, is_utf8)).collect();
    let logical = vals
        .iter()
        .zip(&valid_bits)
        .map(|(v, &ok)| {
            if ok {
                LogicalValue::Bytes(v.clone())
            } else {
                LogicalValue::Null
            }
        })
        .collect();
    let arr = if is_utf8 {
        let strs: Vec<&str> = vals
            .iter()
            .map(|b| std::str::from_utf8(b).unwrap())
            .collect();
        match validity {
            Validity::NonNullable => VarBinViewArray::from_iter_str(strs).into_array(),
            _ => VarBinViewArray::from_iter_nullable_str(
                strs.into_iter()
                    .zip(&valid_bits)
                    .map(|(s, &ok)| ok.then_some(s)),
            )
            .into_array(),
        }
    } else {
        match validity {
            Validity::NonNullable => {
                VarBinViewArray::from_iter_bin(vals.iter().map(Vec::as_slice)).into_array()
            }
            _ => VarBinViewArray::from_iter_nullable_bin(
                vals.iter()
                    .zip(&valid_bits)
                    .map(|(b, &ok)| ok.then_some(b.as_slice())),
            )
            .into_array(),
        }
    };
    finish_column(tc, arr, logical)
}

struct ListRanges {
    ranges: Vec<(usize, usize)>,
    offsets: ArrayRef,
    sizes: ArrayRef,
}

fn draw_list_ranges(
    tc: &TestCase,
    len: usize,
    pool_len: usize,
    max_item_len: usize,
) -> VortexResult<ListRanges> {
    let mut ranges = Vec::with_capacity(len);
    let mut offsets = Vec::with_capacity(len);
    let mut sizes = Vec::with_capacity(len);
    for _ in 0..len {
        let sz = if pool_len == 0 {
            0
        } else {
            tc.draw(gs::integers::<usize>().max_value(pool_len.min(max_item_len)))
        };
        let off = if sz == 0 {
            0
        } else {
            tc.draw(gs::integers::<usize>().max_value(pool_len - sz))
        };
        ranges.push((off, sz));
        offsets.push(u32::try_from(off)?);
        sizes.push(u32::try_from(sz)?);
    }
    Ok(ListRanges {
        ranges,
        offsets: PrimitiveArray::from_iter(offsets).into_array(),
        sizes: PrimitiveArray::from_iter(sizes).into_array(),
    })
}

fn draw_column(
    tc: &TestCase,
    len: usize,
    depth: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<GeneratedColumn> {
    let max_kind = if depth < 2 { 19 } else { 15 };
    match tc.draw(gs::integers::<u8>().max_value(max_kind)) {
        0 => finish_column(
            tc,
            NullArray::new(len).into_array(),
            vec![LogicalValue::Null; len],
        ),
        1 => {
            let (validity, valid_bits) = draw_validity(tc, len);
            let vals = tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len));
            let logical = vals
                .iter()
                .zip(&valid_bits)
                .map(|(&v, &ok)| {
                    if ok {
                        LogicalValue::Bool(v)
                    } else {
                        LogicalValue::Null
                    }
                })
                .collect();
            let arr = BoolArray::new(BitBuffer::from_iter(vals), validity).into_array();
            finish_column(tc, arr, logical)
        }
        2 => draw_int(tc, len, |tc| tc.draw(gs::integers::<u8>())),
        3 => draw_int(tc, len, |tc| tc.draw(gs::integers::<u16>())),
        4 => draw_int(tc, len, |tc| tc.draw(gs::integers::<u32>())),
        5 => draw_int(tc, len, |tc| tc.draw(gs::integers::<u64>())),
        6 => draw_int(tc, len, |tc| tc.draw(gs::integers::<i8>())),
        7 => draw_int(tc, len, |tc| tc.draw(gs::integers::<i16>())),
        8 => draw_int(tc, len, |tc| tc.draw(gs::integers::<i32>())),
        9 => draw_int(tc, len, |tc| tc.draw(gs::integers::<i64>())),
        10 => draw_primitive(tc, len, draw_f16, LogicalValue::F16),
        11 => draw_primitive(tc, len, draw_f32, LogicalValue::F32),
        12 => draw_primitive(tc, len, draw_f64, LogicalValue::F64),
        13 => draw_decimal_column(tc, len, ctx),
        14 => draw_varlen_column(tc, len, true),
        15 => draw_varlen_column(tc, len, false),
        16 => {
            // FixedSizeList (size 0..=4, nullable parent with differing garbage children)
            let list_size = tc.draw(gs::integers::<u32>().max_value(4));
            let (validity, valid_bits) = draw_validity(tc, len);
            let elem_col = draw_column(tc, len * list_size as usize, depth + 1, ctx)?;
            let canonical =
                FixedSizeListArray::try_new(elem_col.canonical, list_size, validity.clone(), len)?
                    .into_array();
            let alt_fsl =
                FixedSizeListArray::try_new(elem_col.alternate, list_size, validity, len)?
                    .into_array();
            let alternate = make_alternate_encoding(tc, &alt_fsl)?;
            let ls = list_size as usize;
            let logical = (0..len)
                .map(|i| {
                    if valid_bits[i] {
                        LogicalValue::FixedSizeList(elem_col.logical[i * ls..(i + 1) * ls].to_vec())
                    } else {
                        LogicalValue::Null
                    }
                })
                .collect();
            Ok(GeneratedColumn {
                canonical,
                alternate,
                logical,
            })
        }
        17 => {
            // Struct (0..=3 fields, nullable parent with differing underlying child payloads)
            let nfields = tc.draw(gs::integers::<usize>().max_value(3));
            let (validity, valid_bits) = draw_validity(tc, len);
            let mut names = Vec::with_capacity(nfields);
            let mut canon_fields = Vec::with_capacity(nfields);
            let mut alt_fields = Vec::with_capacity(nfields);
            let mut logical_fields = Vec::with_capacity(nfields);
            for f in 0..nfields {
                let child = draw_column(tc, len, depth + 1, ctx)?;
                names.push(FieldName::from(format!("f{f}")));
                canon_fields.push(child.canonical);
                alt_fields.push(child.alternate);
                logical_fields.push(child.logical);
            }
            let field_names = FieldNames::from(names);
            let canonical =
                StructArray::try_new(field_names.clone(), canon_fields, len, validity.clone())?
                    .into_array();
            let alt_struct =
                StructArray::try_new(field_names, alt_fields, len, validity)?.into_array();
            let alternate = make_alternate_encoding(tc, &alt_struct)?;
            let logical = (0..len)
                .map(|i| {
                    if valid_bits[i] {
                        LogicalValue::Struct(
                            logical_fields.iter().map(|lf| lf[i].clone()).collect(),
                        )
                    } else {
                        LogicalValue::Null
                    }
                })
                .collect();
            Ok(GeneratedColumn {
                canonical,
                alternate,
                logical,
            })
        }
        18 => {
            // ListView (variable-length lists, possibly out-of-order / overlapping offsets)
            let (validity, valid_bits) = draw_validity(tc, len);
            let pool_len = tc.draw(gs::integers::<usize>().max_value(12));
            let elem_col = draw_column(tc, pool_len, depth + 1, ctx)?;
            let ListRanges {
                ranges,
                offsets: offsets_arr,
                sizes: sizes_arr,
            } = draw_list_ranges(tc, len, pool_len, 4)?;
            let logical = ranges
                .into_iter()
                .zip(&valid_bits)
                .map(|((off, sz), &ok)| {
                    if ok {
                        LogicalValue::List(elem_col.logical[off..off + sz].to_vec())
                    } else {
                        LogicalValue::Null
                    }
                })
                .collect();
            let canonical = ListViewArray::new(
                elem_col.canonical,
                offsets_arr.clone(),
                sizes_arr.clone(),
                validity.clone(),
                ctx,
            )
            .into_array();
            let alt_list =
                ListViewArray::new(elem_col.alternate, offsets_arr, sizes_arr, validity, ctx)
                    .into_array();
            let alternate = make_alternate_encoding(tc, &alt_list)?;
            Ok(GeneratedColumn {
                canonical,
                alternate,
                logical,
            })
        }
        _ => {
            // Map (ListView of Struct { key (non-nullable), value })
            let (validity, valid_bits) = draw_validity(tc, len);
            let pool_len = tc.draw(gs::integers::<usize>().max_value(8));
            let keys_raw = tc.draw(
                gs::vecs(gs::integers::<i32>())
                    .min_size(pool_len)
                    .max_size(pool_len),
            );
            let keys_arr = PrimitiveArray::from_iter(keys_raw.clone()).into_array();
            let val_col = draw_column(tc, pool_len, depth + 1, ctx)?;
            let entry_names = FieldNames::from([FieldName::from("key"), FieldName::from("value")]);
            let entries = StructArray::try_new(
                entry_names.clone(),
                vec![keys_arr.clone(), val_col.canonical.clone()],
                pool_len,
                Validity::NonNullable,
            )?
            .into_array();
            let alt_entries = StructArray::try_new(
                entry_names,
                vec![keys_arr, val_col.alternate],
                pool_len,
                Validity::NonNullable,
            )?
            .into_array();
            let map_dtype = MapDType::try_new(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                val_col.canonical.dtype().clone(),
                false,
            )?;
            let ListRanges {
                ranges,
                offsets: offsets_arr,
                sizes: sizes_arr,
            } = draw_list_ranges(tc, len, pool_len, 3)?;
            let logical = ranges
                .into_iter()
                .zip(&valid_bits)
                .map(|((off, sz), &ok)| {
                    if ok {
                        let pairs = (off..off + sz)
                            .map(|idx| {
                                LogicalValue::Struct(vec![
                                    LogicalValue::Int(i128::from(keys_raw[idx])),
                                    val_col.logical[idx].clone(),
                                ])
                            })
                            .collect();
                        LogicalValue::List(pairs)
                    } else {
                        LogicalValue::Null
                    }
                })
                .collect();
            let entry_lists = ListViewArray::new(
                entries,
                offsets_arr.clone(),
                sizes_arr.clone(),
                validity.clone(),
                ctx,
            );
            let alt_entry_lists =
                ListViewArray::new(alt_entries, offsets_arr, sizes_arr, validity, ctx);
            let canonical = MapArray::new(map_dtype.clone(), entry_lists).into_array();
            let alt_map = MapArray::new(map_dtype, alt_entry_lists).into_array();
            let alternate = make_alternate_encoding(tc, &alt_map)?;
            Ok(GeneratedColumn {
                canonical,
                alternate,
                logical,
            })
        }
    }
}

fn check_table_invariants(
    cols_canon: &[ArrayRef],
    cols_alt: &[ArrayRef],
    logical_cols: &[Vec<LogicalValue>],
    fields: &[RowSortField],
    nrows: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let encoder = RowEncoder::new(fields.iter().copied());
    let encoded_canon = encoder.encode(cols_canon, ctx)?;
    let encoded_alt = encoder.encode(cols_alt, ctx)?;
    let (canon_rows, canon_total_bytes) = collect_rows_and_total_bytes(&encoded_canon, ctx)?;
    let (alt_rows, _) = collect_rows_and_total_bytes(&encoded_alt, ctx)?;

    // (d) Encoded/sliced/chunked inputs produce byte-identical output to canonical inputs.
    assert_eq!(
        alt_rows, canon_rows,
        "non-canonical inputs must produce byte-identical row keys"
    );

    // (b) Every row's encoded byte length equals RowEncoder::row_sizes, and total buffer length matches.
    let sizes_canon = collect_row_sizes(encoder.row_sizes(cols_canon, ctx)?, ctx)?;
    let sizes_alt = collect_row_sizes(encoder.row_sizes(cols_alt, ctx)?, ctx)?;
    assert_eq!(sizes_alt, sizes_canon);
    assert_eq!(canon_rows.len(), nrows);
    let mut sum_bytes = 0usize;
    for (i, (row, &expected_len)) in canon_rows.iter().zip(&sizes_canon).enumerate() {
        assert_eq!(
            row.len(),
            expected_len as usize,
            "row {i} byte length mismatch with RowEncoder::row_sizes"
        );
        sum_bytes += row.len();
    }
    assert_eq!(canon_total_bytes, sum_bytes);

    // (a) Pairwise encoded byte-slice ordering matches independent logical tuple comparator.
    // When logical tuples are Equal (including two null parent Struct/FSL rows with differing
    // underlying child payloads), Vec<u8>::cmp == Ordering::Equal also verifies (c) byte identity.
    for i in 0..nrows {
        for j in 0..nrows {
            let expected = fields
                .iter()
                .enumerate()
                .find_map(|(col_idx, &field)| {
                    match cmp_logical(&logical_cols[col_idx][i], &logical_cols[col_idx][j], field) {
                        Ordering::Equal => None,
                        ord => Some(ord),
                    }
                })
                .unwrap_or(Ordering::Equal);
            assert_eq!(
                canon_rows[i].cmp(&canon_rows[j]),
                expected,
                "row ordering mismatch for pair ({i}, {j}) under fields {fields:?}"
            );
        }
    }
    Ok(())
}

#[hegel::test(test_cases = 100, print_blob = true)]
fn multi_column_row_encode_oracle(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let nrows = tc.draw(gs::integers::<usize>().max_value(12));
    let ncols = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));

    let mut cols_canon = Vec::with_capacity(ncols);
    let mut cols_alt = Vec::with_capacity(ncols);
    let mut logical_cols = Vec::with_capacity(ncols);
    let mut fields = Vec::with_capacity(ncols);

    for _ in 0..ncols {
        let col =
            draw_column(&tc, nrows, 0, &mut ctx).vortex_expect("column generation must succeed");
        cols_canon.push(col.canonical);
        cols_alt.push(col.alternate);
        logical_cols.push(col.logical);
        fields.push(RowSortField::new(
            tc.draw(gs::booleans()),
            tc.draw(gs::booleans()),
        ));
    }

    check_table_invariants(
        &cols_canon,
        &cols_alt,
        &logical_cols,
        &fields,
        nrows,
        &mut ctx,
    )
    .vortex_expect("row encoding invariants must hold");
}

fn check_varlen_lengths_oracle(lengths: &[usize]) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Build adversarial byte payloads spanning exact requested lengths, including shared prefixes,
    // trailing NUL/0xFF bytes, and a reversed tiebreaker column so continuation/length marker
    // bytes are strictly tested at every 32-byte boundary.
    let mut utf8_vals: Vec<Option<String>> = vec![None];
    let mut bin_vals: Vec<Option<Vec<u8>>> = vec![None];
    for (idx, &len) in lengths.iter().enumerate() {
        let mut s = vec![b'a'; len];
        let mut b = vec![0xFFu8; len];
        if len > 0 && idx % 3 == 1 {
            s[len - 1] = b'\0';
            b[len - 1] = 0x00;
        } else if len > 0 && idx % 3 == 2 {
            s[len - 1] = b'b';
            b[len - 1] = 0x7F;
        }
        utf8_vals.push(Some(
            String::from_utf8(s).map_err(|err| vortex_err!("{err}"))?,
        ));
        bin_vals.push(Some(b));
    }

    let nrows = utf8_vals.len();
    let tiebreaker: Vec<i32> = (0..nrows)
        .map(|i| i32::try_from(nrows - i).unwrap())
        .collect();
    let tie_arr = PrimitiveArray::from_iter(tiebreaker.clone()).into_array();
    let tie_logical: Vec<LogicalValue> = tiebreaker
        .into_iter()
        .map(|v| LogicalValue::Int(i128::from(v)))
        .collect();

    let utf8_arr = VarBinViewArray::from_iter_nullable_str(utf8_vals.iter().map(Option::as_deref))
        .into_array();
    let utf8_logical = utf8_vals
        .iter()
        .map(|v| {
            v.as_ref().map_or(LogicalValue::Null, |s| {
                LogicalValue::Bytes(s.as_bytes().to_vec())
            })
        })
        .collect::<Vec<_>>();
    let bin_arr =
        VarBinViewArray::from_iter_nullable_bin(bin_vals.iter().map(Option::as_deref)).into_array();
    let bin_logical = bin_vals
        .iter()
        .map(|v| {
            v.as_ref()
                .map_or(LogicalValue::Null, |b| LogicalValue::Bytes(b.clone()))
        })
        .collect::<Vec<_>>();

    for descending in [false, true] {
        for nulls_first in [false, true] {
            let fields = [
                RowSortField::new(descending, nulls_first),
                RowSortField::ascending(),
            ];
            for (arr, logical) in [(&utf8_arr, &utf8_logical), (&bin_arr, &bin_logical)] {
                let cols = [arr.clone(), tie_arr.clone()];
                check_table_invariants(
                    &cols,
                    &cols,
                    &[logical.clone(), tie_logical.clone()],
                    &fields,
                    nrows,
                    &mut ctx,
                )?;
            }
        }
    }
    Ok(())
}

#[hegel::test(test_cases = 100, print_blob = true)]
#[hegel::explicit_test_case(lengths = vec![0usize, 1, 31, 32, 33, 63, 64, 65])]
#[hegel::explicit_test_case(lengths = vec![31usize, 32, 32, 33, 63, 64, 64, 65, 96, 128])]
fn varlen_boundary_oracle(tc: TestCase) {
    let lengths = tc.draw(
        gs::vecs(gs::integers::<usize>().max_value(128))
            .min_size(1)
            .max_size(12),
    );
    check_varlen_lengths_oracle(&lengths)
        .vortex_expect("varlen block encoding must preserve ordering and sizes at all boundaries");
}

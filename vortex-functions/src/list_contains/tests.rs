// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use num_traits::PrimInt;
use prost::Message;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Bool;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::compute::conformance::hegel::draw_validity_for;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::i256;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::proto::expr as pb;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::builtin::operators::CompareMode;
use vortex_array::scalar_fn::builtin::operators::FloatOrder;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::ArrayExprExt;
use vortex_expr::Expression;
use vortex_expr::bound;
use vortex_expr::col;
use vortex_expr::get_item;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_session::VortexSession;

use super::ListContains;
use super::ListContainsOptions;
use super::PreparedSet;
use super::PreparedSetData;
use super::process_matches;
use crate::expr::in_list;
use crate::expr::list_contains;
use crate::expr::list_contains_opts;
use crate::tests::draw_len;
use crate::tests::draw_nullability;
use crate::tests::slice_with_padding;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    crate::initialize(&session);
    session
});

const SQL: ListContainsOptions = ListContainsOptions {
    sql_null_semantics: true,
    compare_mode: CompareMode::TOTAL_ORDER,
};
const ZEROS_EQUAL: CompareMode = CompareMode::new(FloatOrder::ZerosEqual);
const CANONICAL: CompareMode = CompareMode::new(FloatOrder::Canonical);
const IEEE754: CompareMode = CompareMode::new(FloatOrder::Ieee754);

const ALL_COMPARE_MODES: [CompareMode; 4] =
    [CompareMode::TOTAL_ORDER, ZEROS_EQUAL, CANONICAL, IEEE754];

// Signaling NaN with payload bits distinct from `f64::NAN`.
const PAYLOAD_NAN_BITS: u64 = 0x7ff0_0000_0000_0001;

const STR_POOL: [&str; 8] = [
    "",
    "a",
    "ab",
    "abc",
    "twelve_bytes",
    "a long string with suffix A",
    "a long string with suffix B",
    "another long string > 12 bytes",
];

// ==================== Pure-Rust Reference Oracle ====================

#[derive(Clone, Debug)]
enum OracleVal {
    Bool(bool),
    I64(i64),
    F64(f64),
    Decimal(i256),
    Bytes(Vec<u8>),
    Nested(Vec<Option<OracleVal>>),
}

type OracleList = Option<Vec<Option<OracleVal>>>;

fn floats_equal(a: f64, b: f64, mode: CompareMode) -> bool {
    match mode.float_order() {
        FloatOrder::TotalOrder => a.to_bits() == b.to_bits(),
        FloatOrder::ZerosEqual => (a == 0.0 && b == 0.0) || a.to_bits() == b.to_bits(),
        FloatOrder::Canonical => {
            (a.is_nan() && b.is_nan()) || (a == 0.0 && b == 0.0) || a.to_bits() == b.to_bits()
        }
        FloatOrder::Ieee754 => a == b,
        other => unreachable!("unsupported float order {other}"),
    }
}

fn oracle_val_eq(lhs: &OracleVal, rhs: &OracleVal, mode: CompareMode) -> bool {
    match (lhs, rhs) {
        (OracleVal::Bool(x), OracleVal::Bool(y)) => x == y,
        (OracleVal::I64(x), OracleVal::I64(y)) => x == y,
        (OracleVal::F64(x), OracleVal::F64(y)) => floats_equal(*x, *y, mode),
        (OracleVal::Decimal(x), OracleVal::Decimal(y)) => x == y,
        (OracleVal::Bytes(x), OracleVal::Bytes(y)) => x == y,
        (OracleVal::Nested(xs), OracleVal::Nested(ys)) => {
            xs.len() == ys.len()
                && xs.iter().zip(ys.iter()).all(|(x, y)| match (x, y) {
                    (None, None) => true,
                    (Some(x), Some(y)) => oracle_val_eq(x, y, mode),
                    _ => false,
                })
        }
        _ => unreachable!("mismatched oracle value variants"),
    }
}

/// Independent nullability specification: a null list or needle can yield null, and under SQL null
/// semantics a nullable list element can also leave a non-match unknown.
fn expected_result_nullability(
    list_nullability: Nullability,
    elem_nullability: Nullability,
    needle_nullability: Nullability,
    sql_null_semantics: bool,
) -> Nullability {
    if list_nullability.is_nullable()
        || needle_nullability.is_nullable()
        || (sql_null_semantics && elem_nullability.is_nullable())
    {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    }
}

/// Exact row-by-row specification of `list_contains(list, needle, options)`.
fn eval_row_oracle(
    list: Option<&[Option<OracleVal>]>,
    needle: Option<&OracleVal>,
    options: ListContainsOptions,
) -> Option<bool> {
    let elements = list?;
    let Some(needle) = needle else {
        return (elements.is_empty() && !options.sql_null_semantics).then_some(false);
    };
    let any_match = elements
        .iter()
        .flatten()
        .any(|elem| oracle_val_eq(elem, needle, options.compare_mode));
    if any_match {
        Some(true)
    } else if options.sql_null_semantics && elements.iter().any(Option::is_none) {
        None
    } else {
        Some(false)
    }
}

fn oracle_bool_array(expected: &[Option<bool>], result_nullability: Nullability) -> BoolArray {
    let bits: BitBuffer = expected.iter().map(|v| v.unwrap_or(false)).collect();
    let validity = match result_nullability {
        Nullability::NonNullable => {
            assert!(expected.iter().all(Option::is_some));
            Validity::NonNullable
        }
        Nullability::Nullable => Validity::from_iter(expected.iter().map(Option::is_some)),
    };
    BoolArray::new(bits, validity)
}

fn assert_result_and_probes(
    tc: &TestCase,
    actual: &ArrayRef,
    expected: &BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(actual.len(), expected.len());
    assert_arrays_eq!(actual.clone(), expected.clone(), ctx);

    let len = actual.len();
    if len > 0 {
        for idx in [0, len - 1] {
            assert_eq!(
                actual.execute_scalar(idx, ctx)?,
                expected.execute_scalar(idx, ctx)?,
                "scalar mismatch at boundary row {idx}"
            );
        }
        if len > 2 {
            let probe = tc.draw(gs::integers::<usize>().max_value(len - 1));
            assert_eq!(
                actual.execute_scalar(probe, ctx)?,
                expected.execute_scalar(probe, ctx)?,
                "scalar mismatch at sampled row {probe}"
            );
        }
    }
    Ok(())
}

// ==================== Generators & Encoding Wrappers ====================

fn draw_f64(tc: &TestCase) -> f64 {
    let pool = [
        -0.0,
        0.0,
        1.0,
        -1.0,
        2.5,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(PAYLOAD_NAN_BITS),
    ];
    tc.draw(gs::sampled_from(&pool))
}

#[derive(Clone, Copy, Debug)]
enum ElemSpec {
    Bool,
    I64,
    F64,
    Decimal18,
    Utf8,
    Binary,
    FixedSizeListF64,
    StructMixed,
    ListI64,
    MapI64Utf8,
}

impl ElemSpec {
    fn contains_float(self) -> bool {
        matches!(self, Self::F64 | Self::FixedSizeListF64 | Self::StructMixed)
    }

    fn dtype(self, nullability: Nullability) -> DType {
        match self {
            Self::Bool => DType::Bool(nullability),
            Self::I64 => DType::Primitive(PType::I64, nullability),
            Self::F64 => DType::Primitive(PType::F64, nullability),
            Self::Decimal18 => DType::Decimal(DecimalDType::new(18, 1), nullability),
            Self::Utf8 => DType::Utf8(nullability),
            Self::Binary => DType::Binary(nullability),
            Self::FixedSizeListF64 => DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::F64, Nullability::Nullable)),
                2,
                nullability,
            ),
            Self::StructMixed => DType::Struct(
                StructFields::new(
                    ["a", "b"].into(),
                    vec![
                        DType::Primitive(PType::I64, Nullability::Nullable),
                        DType::Primitive(PType::F64, Nullability::Nullable),
                    ],
                ),
                nullability,
            ),
            Self::ListI64 => DType::List(
                Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
                nullability,
            ),
            Self::MapI64Utf8 => DType::map(
                DType::Primitive(PType::I64, Nullability::NonNullable),
                DType::Utf8(Nullability::Nullable),
                false,
                nullability,
            )
            .vortex_expect("valid map dtype"),
        }
    }
}

fn draw_oracle_val(tc: &TestCase, spec: ElemSpec) -> OracleVal {
    match spec {
        ElemSpec::Bool => OracleVal::Bool(tc.draw(gs::booleans())),
        ElemSpec::I64 => {
            let v = match tc.draw(gs::integers::<u8>().max_value(5)) {
                0 => i64::MIN,
                1 => i64::MAX,
                2 => -(1i64 << 40),
                3 => 1i64 << 40,
                _ => tc.draw(gs::integers::<i64>().min_value(-8).max_value(12)),
            };
            OracleVal::I64(v)
        }
        ElemSpec::F64 => OracleVal::F64(draw_f64(tc)),
        ElemSpec::Decimal18 => OracleVal::Decimal(i256::from_i128(i128::from(
            tc.draw(gs::integers::<i8>().min_value(-6).max_value(6)),
        ))),
        ElemSpec::Utf8 | ElemSpec::Binary => {
            OracleVal::Bytes(tc.draw(gs::sampled_from(&STR_POOL)).as_bytes().to_vec())
        }
        ElemSpec::FixedSizeListF64 => OracleVal::Nested(
            (0..2)
                .map(|_| {
                    tc.draw(gs::booleans())
                        .then(|| OracleVal::F64(draw_f64(tc)))
                })
                .collect(),
        ),
        ElemSpec::StructMixed => OracleVal::Nested(vec![
            tc.draw(gs::booleans())
                .then(|| OracleVal::I64(tc.draw(gs::integers::<i64>().min_value(-3).max_value(3)))),
            tc.draw(gs::booleans())
                .then(|| OracleVal::F64(draw_f64(tc))),
        ]),
        ElemSpec::ListI64 => {
            let n = tc.draw(gs::integers::<usize>().max_value(3));
            OracleVal::Nested(
                (0..n)
                    .map(|_| {
                        tc.draw(gs::booleans()).then(|| {
                            OracleVal::I64(
                                tc.draw(gs::integers::<i64>().min_value(-3).max_value(3)),
                            )
                        })
                    })
                    .collect(),
            )
        }
        ElemSpec::MapI64Utf8 => {
            let key = tc.draw(gs::integers::<i64>().min_value(0).max_value(3));
            let val = tc.draw(gs::booleans()).then(|| {
                OracleVal::Bytes(tc.draw(gs::sampled_from(&STR_POOL)).as_bytes().to_vec())
            });
            OracleVal::Nested(vec![Some(OracleVal::I64(key)), val])
        }
    }
}

fn oracle_val_to_scalar(
    val: Option<&OracleVal>,
    spec: ElemSpec,
    nullability: Nullability,
) -> Scalar {
    let dtype = spec.dtype(nullability);
    let Some(val) = val else {
        return Scalar::null(dtype);
    };
    match (spec, val) {
        (ElemSpec::Bool, OracleVal::Bool(b)) => Scalar::bool(*b, nullability),
        (ElemSpec::I64, OracleVal::I64(v)) => Scalar::primitive(*v, nullability),
        (ElemSpec::F64, OracleVal::F64(v)) => Scalar::primitive(*v, nullability),
        (ElemSpec::Decimal18, OracleVal::Decimal(v)) => {
            Scalar::decimal((*v).into(), DecimalDType::new(18, 1), nullability)
        }
        (ElemSpec::Utf8, OracleVal::Bytes(b)) => {
            Scalar::utf8(str::from_utf8(b).expect("utf8"), nullability)
        }
        (ElemSpec::Binary, OracleVal::Bytes(b)) => Scalar::binary(b.clone(), nullability),
        (ElemSpec::FixedSizeListF64, OracleVal::Nested(items)) => {
            let elem_dtype = Arc::new(DType::Primitive(PType::F64, Nullability::Nullable));
            let scalars = items
                .iter()
                .map(|x| oracle_val_to_scalar(x.as_ref(), ElemSpec::F64, Nullability::Nullable))
                .collect();
            Scalar::fixed_size_list(elem_dtype, scalars, nullability)
        }
        (ElemSpec::StructMixed, OracleVal::Nested(fields)) => Scalar::struct_(
            dtype,
            vec![
                oracle_val_to_scalar(fields[0].as_ref(), ElemSpec::I64, Nullability::Nullable),
                oracle_val_to_scalar(fields[1].as_ref(), ElemSpec::F64, Nullability::Nullable),
            ],
        ),
        (ElemSpec::ListI64, OracleVal::Nested(items)) => {
            let elem_dtype = Arc::new(DType::Primitive(PType::I64, Nullability::Nullable));
            let scalars = items
                .iter()
                .map(|x| oracle_val_to_scalar(x.as_ref(), ElemSpec::I64, Nullability::Nullable))
                .collect();
            Scalar::list(elem_dtype, scalars, nullability)
        }
        (ElemSpec::MapI64Utf8, OracleVal::Nested(kv)) => {
            let key = oracle_val_to_scalar(kv[0].as_ref(), ElemSpec::I64, Nullability::NonNullable);
            let value = oracle_val_to_scalar(kv[1].as_ref(), ElemSpec::Utf8, Nullability::Nullable);
            Scalar::map(dtype, [(key, value)])
        }
        _ => unreachable!("mismatched spec and oracle value"),
    }
}

fn build_values_array(
    tc: &TestCase,
    vals: &[Option<OracleVal>],
    spec: ElemSpec,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = spec.dtype(nullability);
    // Inject dirty physical values under nulls to verify null-value bit masking.
    if matches!(spec, ElemSpec::I64) && nullability.is_nullable() && tc.draw(gs::booleans()) {
        let raw: Vec<i64> = vals
            .iter()
            .map(|v| match v {
                Some(OracleVal::I64(x)) => *x,
                None => tc.draw(gs::integers::<i64>().min_value(-8).max_value(12)),
                _ => unreachable!(),
            })
            .collect();
        let validity = Validity::from_iter(vals.iter().map(Option::is_some));
        return Ok(PrimitiveArray::new(Buffer::from(raw), validity).into_array());
    }
    if matches!(spec, ElemSpec::Utf8) && tc.draw(gs::booleans()) {
        let strs = vals.iter().map(|v| {
            v.as_ref().map(|x| match x {
                OracleVal::Bytes(b) => str::from_utf8(b).expect("utf8"),
                _ => unreachable!(),
            })
        });
        return Ok(VarBinArray::from_iter(strs, dtype).into_array());
    }

    let mut builder = builder_with_capacity_in(&dtype, vals.len(), ctx.allocator());
    for v in vals {
        builder.append_scalar(&oracle_val_to_scalar(v.as_ref(), spec, nullability))?;
    }
    Ok(builder.finish())
}

/// Wraps an array in a randomly chosen needle/element encoding (`Canonical`, `Constant`, `Dict`,
/// `RunEnd`, `Chunked`, or sliced) while returning its exact row-by-row oracle values.
fn wrap_encoded_array(
    tc: &TestCase,
    canonical: ArrayRef,
    vals: Vec<Option<OracleVal>>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, Vec<Option<OracleVal>>)> {
    let len = vals.len();
    match tc.draw(gs::integers::<u8>().max_value(6)) {
        0 | 1 => Ok((canonical, vals)),
        2 if len > 0 => {
            let scalar = canonical.execute_scalar(0, ctx)?;
            Ok((
                ConstantArray::new(scalar, len).into_array(),
                vec![vals[0].clone(); len],
            ))
        }
        3 if len > 0 => {
            let dict_len = tc.draw(gs::integers::<usize>().min_value(1).max_value(len.min(6)));
            let dict_values = canonical.slice(0..dict_len)?;
            let max_code = u16::try_from(dict_len - 1).vortex_expect("fits u16");
            let slice_dict = tc.draw(gs::booleans()) && len >= 2;
            let codes_len = if slice_dict { len + 2 } else { len };
            let codes: Vec<u16> = tc.draw(
                gs::vecs(gs::integers::<u16>().max_value(max_code))
                    .min_size(codes_len)
                    .max_size(codes_len),
            );
            let codes_validity = if canonical.dtype().is_nullable() {
                draw_validity_for(tc, Nullability::Nullable, codes_len)
            } else {
                Validity::NonNullable
            };
            let codes_mask = codes_validity.execute_mask(codes_len, ctx)?;
            let codes_arr =
                PrimitiveArray::new(Buffer::from(codes.clone()), codes_validity).into_array();
            let mut dict = DictArray::try_new(codes_arr, dict_values)?.into_array();
            let mut out_vals: Vec<Option<OracleVal>> = codes
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    codes_mask
                        .value(i)
                        .then(|| vals[usize::from(c)].clone())
                        .flatten()
                })
                .collect();
            if slice_dict {
                let slice_len = tc.draw(gs::integers::<usize>().min_value(1).max_value(len));
                dict = dict.slice(1..1 + slice_len)?;
                out_vals = out_vals[1..1 + slice_len].to_vec();
            }
            Ok((dict, out_vals))
        }
        4 if len > 0 => {
            let mut ends = Vec::new();
            let mut run_indices = Vec::new();
            let mut run_oracle = Vec::new();
            let mut pos = 0usize;
            while pos < len {
                let run_len = tc.draw(gs::integers::<usize>().min_value(1).max_value(len - pos));
                let pick = tc.draw(gs::integers::<usize>().max_value(len - 1));
                pos += run_len;
                ends.push(u32::try_from(pos).vortex_expect("fits u32"));
                run_indices.push(u32::try_from(pick).vortex_expect("fits u32"));
                run_oracle.extend(std::iter::repeat_n(vals[pick].clone(), run_len));
            }
            let run_values = canonical.take(Buffer::from(run_indices).into_array())?;
            let mut ree =
                RunEnd::try_new(Buffer::from(ends).into_array(), run_values, ctx)?.into_array();
            if len > 2 && tc.draw(gs::booleans()) {
                let start = tc.draw(gs::integers::<usize>().max_value(len - 1));
                let end = tc.draw(gs::integers::<usize>().min_value(start).max_value(len));
                ree = ree.slice(start..end)?;
                run_oracle = run_oracle[start..end].to_vec();
            }
            Ok((ree, run_oracle))
        }
        5 => {
            let dtype = canonical.dtype().clone();
            let mid = len / 2;
            let c0 = canonical.slice(0..mid)?;
            let c_empty = canonical.slice(0..0)?;
            let c1 = canonical.slice(mid..len)?;
            let chunks = if tc.draw(gs::booleans()) && !c1.is_empty() {
                let dict_c1 = DictArray::try_new(
                    Buffer::from_iter(0..u32::try_from(c1.len()).vortex_expect("fits u32"))
                        .into_array(),
                    c1,
                )?
                .into_array();
                vec![c0, c_empty, dict_c1]
            } else {
                vec![c0, c_empty, c1]
            };
            Ok((ChunkedArray::try_new(chunks, dtype)?.into_array(), vals))
        }
        _ => Ok((slice_with_padding(tc, &canonical, ctx)?, vals)),
    }
}

fn draw_values(
    tc: &TestCase,
    spec: ElemSpec,
    nullability: Nullability,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, Vec<Option<OracleVal>>)> {
    let mask = draw_validity_for(tc, nullability, len).execute_mask(len, ctx)?;
    let vals: Vec<Option<OracleVal>> = (0..len)
        .map(|row| mask.value(row).then(|| draw_oracle_val(tc, spec)))
        .collect();
    let arr = build_values_array(tc, &vals, spec, nullability, ctx)?;
    Ok((arr, vals))
}

// ==================== Hegel Oracle 1: Variable Haystack ====================

const VAR_HAYSTACK_SPECS: [ElemSpec; 6] = [
    ElemSpec::Bool,
    ElemSpec::I64,
    ElemSpec::F64,
    ElemSpec::Decimal18,
    ElemSpec::Utf8,
    ElemSpec::FixedSizeListF64,
];

fn encode_index_buf(tc: &TestCase, vals: Vec<u64>) -> ArrayRef {
    match tc.draw(gs::integers::<u8>().max_value(2)) {
        0 => Buffer::from_iter(
            vals.iter()
                .map(|&x| u8::try_from(x).vortex_expect("fits u8")),
        )
        .into_array(),
        1 => Buffer::from_iter(
            vals.iter()
                .map(|&x| u32::try_from(x).vortex_expect("fits u32")),
        )
        .into_array(),
        _ => Buffer::from(vals).into_array(),
    }
}

fn build_variable_haystack(
    tc: &TestCase,
    spec: ElemSpec,
    elem_nullability: Nullability,
    list_nullability: Nullability,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, Vec<OracleList>)> {
    let list_validity = draw_validity_for(tc, list_nullability, len);
    let list_mask = list_validity.execute_mask(len, ctx)?;

    match tc.draw(gs::integers::<u8>().max_value(4)) {
        // ConstantArray of a list scalar (valid or null).
        0 => {
            let elem_dtype = spec.dtype(elem_nullability);
            if list_nullability.is_nullable() && tc.draw(gs::booleans()) {
                let scalar = Scalar::null(DType::List(Arc::new(elem_dtype), Nullability::Nullable));
                return Ok((
                    ConstantArray::new(scalar, len).into_array(),
                    vec![None; len],
                ));
            }
            let n = tc.draw(gs::integers::<usize>().max_value(5));
            let (_, elem_vals) = draw_values(tc, spec, elem_nullability, n, ctx)?;
            let scalars = elem_vals
                .iter()
                .map(|v| oracle_val_to_scalar(v.as_ref(), spec, elem_nullability))
                .collect();
            let list_scalar = Scalar::list(Arc::new(elem_dtype), scalars, list_nullability);
            Ok((
                ConstantArray::new(list_scalar, len).into_array(),
                vec![Some(elem_vals); len],
            ))
        }
        // Out-of-order / overlapping ListViewArray with u8/u32/u64 offsets & sizes.
        1 | 2 => {
            let pool_len = tc.draw(gs::integers::<usize>().max_value(16));
            let (raw_elems, raw_pool_vals) =
                draw_values(tc, spec, elem_nullability, pool_len, ctx)?;
            let (elems, pool_vals) = if pool_len > 0 && tc.draw(gs::booleans()) {
                wrap_encoded_array(tc, raw_elems, raw_pool_vals, ctx)?
            } else {
                (raw_elems, raw_pool_vals)
            };
            let actual_pool_len = pool_vals.len();

            let mut offsets_u64 = Vec::with_capacity(len);
            let mut sizes_u64 = Vec::with_capacity(len);
            let mut oracle_lists = Vec::with_capacity(len);
            for row in 0..len {
                let off = if actual_pool_len == 0 {
                    0
                } else {
                    tc.draw(gs::integers::<usize>().max_value(actual_pool_len))
                };
                if !list_mask.value(row) {
                    let sz = tc.draw(gs::integers::<usize>().max_value(actual_pool_len - off));
                    offsets_u64.push(off as u64);
                    sizes_u64.push(sz as u64);
                    oracle_lists.push(None);
                } else {
                    let sz =
                        tc.draw(gs::integers::<usize>().max_value((actual_pool_len - off).min(5)));
                    offsets_u64.push(off as u64);
                    sizes_u64.push(sz as u64);
                    oracle_lists.push(Some(pool_vals[off..off + sz].to_vec()));
                }
            }

            let lv = ListViewArray::try_new(
                elems,
                encode_index_buf(tc, offsets_u64),
                encode_index_buf(tc, sizes_u64),
                list_validity,
                ctx,
            )?
            .into_array();
            Ok((lv, oracle_lists))
        }
        // Standard ListArray, optionally wrapped in Chunked or Dict.
        _ => {
            let mut offsets = Vec::with_capacity(len + 1);
            offsets.push(0u32);
            let mut flat_vals = Vec::new();
            let mut oracle_lists = Vec::with_capacity(len);
            for row in 0..len {
                let sz = tc.draw(gs::integers::<usize>().max_value(4));
                let mut row_vals = Vec::with_capacity(sz);
                for _ in 0..sz {
                    let is_valid = !elem_nullability.is_nullable() || tc.draw(gs::booleans());
                    row_vals.push(is_valid.then(|| draw_oracle_val(tc, spec)));
                }
                flat_vals.extend(row_vals.iter().cloned());
                offsets.push(u32::try_from(flat_vals.len()).vortex_expect("fits u32"));
                oracle_lists.push(list_mask.value(row).then_some(row_vals));
            }
            let elems = build_values_array(tc, &flat_vals, spec, elem_nullability, ctx)?;
            let list_arr = ListArray::try_new(
                elems,
                Buffer::from(offsets).into_array(),
                list_validity,
                ctx,
            )?
            .into_array();
            let wrapped = match tc.draw(gs::integers::<u8>().max_value(2)) {
                0 => list_arr,
                1 => {
                    let mid = len / 2;
                    let dtype = list_arr.dtype().clone();
                    ChunkedArray::try_new(
                        vec![list_arr.slice(0..mid)?, list_arr.slice(mid..len)?],
                        dtype,
                    )?
                    .into_array()
                }
                _ if len > 0 => DictArray::try_new(
                    Buffer::from_iter(0..u32::try_from(len).vortex_expect("fits u32")).into_array(),
                    list_arr,
                )?
                .into_array(),
                _ => list_arr,
            };
            Ok((wrapped, oracle_lists))
        }
    }
}

fn check_variable_haystack_oracle(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = draw_len(tc, 28);
    let spec = tc.draw(gs::sampled_from(&VAR_HAYSTACK_SPECS));
    let elem_nullability = draw_nullability(tc);
    let list_nullability = draw_nullability(tc);
    let needle_nullability = draw_nullability(tc);
    let options = ListContainsOptions {
        sql_null_semantics: tc.draw(gs::booleans()),
        compare_mode: tc.draw(gs::sampled_from(&ALL_COMPARE_MODES)),
    };

    let (list_arr, oracle_lists) =
        build_variable_haystack(tc, spec, elem_nullability, list_nullability, len, &mut ctx)?;

    let (needle_arr, oracle_needles) = if tc.draw(gs::booleans()) {
        let val = (!needle_nullability.is_nullable() || tc.draw(gs::booleans()))
            .then(|| draw_oracle_val(tc, spec));
        let scalar = oracle_val_to_scalar(val.as_ref(), spec, needle_nullability);
        (ConstantArray::new(scalar, len).into_array(), vec![val; len])
    } else {
        let (raw, vals) = draw_values(tc, spec, needle_nullability, len, &mut ctx)?;
        wrap_encoded_array(tc, raw, vals, &mut ctx)?
    };

    let n = oracle_needles.len();
    let list_arr = if n != len {
        list_arr.slice(0..n)?
    } else {
        list_arr
    };
    let expected_nullability = expected_result_nullability(
        list_nullability,
        elem_nullability,
        needle_nullability,
        options.sql_null_semantics,
    );
    let expected_rows: Vec<Option<bool>> = oracle_lists[..n]
        .iter()
        .zip(oracle_needles.iter())
        .map(|(l, n)| eval_row_oracle(l.as_deref(), n.as_ref(), options))
        .collect();
    let expected = oracle_bool_array(&expected_rows, expected_nullability);

    let unopt =
        ListContains::try_new_opts(list_arr.clone(), needle_arr.clone(), options)?.into_array();
    let opt = unopt.clone().optimize_ctx(&SESSION)?;

    assert_result_and_probes(tc, &unopt, &expected, &mut ctx)?;
    assert_result_and_probes(tc, &opt, &expected, &mut ctx)?;

    // Constant haystack + constant needle folds when float normalization is identity.
    let identity_float_norm =
        !spec.contains_float() || options.compare_mode.float_order() == FloatOrder::TotalOrder;
    if list_arr.as_constant().is_some() && needle_arr.as_constant().is_some() && identity_float_norm
    {
        assert!(
            opt.as_constant().is_some(),
            "const list + const needle with identity normalization must fold"
        );
    }
    Ok(())
}

#[hegel::test(test_cases = 100, print_blob = true)]
fn hegel_list_contains_variable_haystack_oracle(tc: TestCase) {
    check_variable_haystack_oracle(&tc).vortex_expect("variable haystack oracle failed");
}

// ==================== Hegel Oracle 2: Constant Haystack / PreparedSet ====================

const PREPARED_SET_SPECS: [ElemSpec; 8] = [
    ElemSpec::Bool,
    ElemSpec::I64,
    ElemSpec::F64,
    ElemSpec::Decimal18,
    ElemSpec::Utf8,
    ElemSpec::Binary,
    ElemSpec::FixedSizeListF64,
    ElemSpec::StructMixed,
];

const NESTED_ROW_SPECS: [ElemSpec; 2] = [ElemSpec::ListI64, ElemSpec::MapI64Utf8];

fn check_prepared_set_oracle(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let spec = if tc.draw(gs::integers::<u8>().max_value(6)) == 0 {
        tc.draw(gs::sampled_from(&NESTED_ROW_SPECS))
    } else {
        tc.draw(gs::sampled_from(&PREPARED_SET_SPECS))
    };
    let elem_nullability = draw_nullability(tc);
    let list_nullability = draw_nullability(tc);
    let needle_nullability = draw_nullability(tc);
    let options = ListContainsOptions {
        sql_null_semantics: tc.draw(gs::booleans()),
        compare_mode: tc.draw(gs::sampled_from(&ALL_COMPARE_MODES)),
    };

    let set_len = match tc.draw(gs::integers::<u8>().max_value(4)) {
        0 => 0,
        1 => tc.draw(gs::integers::<usize>().min_value(1).max_value(4)),
        2 => tc.draw(gs::integers::<usize>().min_value(5).max_value(20)),
        _ => tc.draw(gs::integers::<usize>().max_value(12)),
    };
    let elem_dtype = spec.dtype(elem_nullability);

    let set_vals: Vec<Option<OracleVal>> = if elem_nullability.is_nullable()
        && set_len > 0
        && tc.draw(gs::integers::<u8>().max_value(5)) == 0
    {
        vec![None; set_len]
    } else {
        (0..set_len)
            .map(|_| {
                let valid = !elem_nullability.is_nullable() || tc.draw(gs::booleans());
                valid.then(|| draw_oracle_val(tc, spec))
            })
            .collect()
    };
    let list_scalar = Scalar::list(
        Arc::new(elem_dtype),
        set_vals
            .iter()
            .map(|v| oracle_val_to_scalar(v.as_ref(), spec, elem_nullability))
            .collect(),
        list_nullability,
    );

    let len = draw_len(tc, 36);
    let (raw_needles, raw_vals) = draw_values(tc, spec, needle_nullability, len, &mut ctx)?;
    let (needles, needle_vals) = wrap_encoded_array(tc, raw_needles, raw_vals, &mut ctx)?;
    let actual_len = needle_vals.len();

    let result_nullability = expected_result_nullability(
        list_nullability,
        elem_nullability,
        needle_nullability,
        options.sql_null_semantics,
    );
    let expected_rows: Vec<Option<bool>> = needle_vals
        .iter()
        .map(|n| eval_row_oracle(Some(&set_vals), n.as_ref(), options))
        .collect();
    let expected = oracle_bool_array(&expected_rows, result_nullability);

    // 1. Expression path (unoptimized + optimized with PreparedSet fixed-point & batch reuse).
    let expr = list_contains_opts(lit(list_scalar.clone()), root(), options);
    assert_result_and_probes(tc, &needles.clone().apply(&expr)?, &expected, &mut ctx)?;

    let bound = expr.bind(needles.dtype())?.optimize()?;
    assert_eq!(
        bound
            .child(0)
            .as_opt::<PreparedSet>()
            .map(PreparedSetData::list),
        Some(&list_scalar)
    );
    assert_eq!(bound.optimize()?, bound);
    assert_result_and_probes(
        tc,
        &needles.clone().apply_bound(&bound)?,
        &expected,
        &mut ctx,
    )?;

    if actual_len > 1 {
        let half = actual_len / 2;
        assert_arrays_eq!(
            needles.slice(half..actual_len)?.apply_bound(&bound)?,
            oracle_bool_array(&expected_rows[half..], result_nullability),
            &mut ctx
        );
    }

    // 2. Direct PreparedSetData::contains & contains_scalar.
    let prepared = PreparedSetData::try_new(list_scalar.clone())?;
    let direct = prepared.contains(&needles, &options, &mut ctx)?;
    if needles.as_constant().is_none() {
        assert!(
            direct.is::<Bool>(),
            "non-constant PreparedSet::contains must return BoolArray"
        );
    }
    assert_arrays_eq!(direct, expected.clone(), &mut ctx);
    if actual_len > 0 {
        let probe = tc.draw(gs::integers::<usize>().max_value(actual_len - 1));
        assert_eq!(
            prepared.contains_scalar(
                &needles.execute_scalar(probe, &mut ctx)?,
                &options,
                &mut ctx
            )?,
            expected.execute_scalar(probe, &mut ctx)?
        );
    }

    // 3. Array-level optimize_ctx path (Chunked / Dict / RunEnd reduce rules).
    let array_opt = ListContains::try_new_opts(
        ConstantArray::new(list_scalar, actual_len).into_array(),
        needles,
        options,
    )?
    .into_array()
    .optimize_ctx(&SESSION)?;
    assert_result_and_probes(tc, &array_opt, &expected, &mut ctx)?;
    Ok(())
}

#[hegel::test(test_cases = 100, print_blob = true)]
fn hegel_list_contains_prepared_set_oracle(tc: TestCase) {
    check_prepared_set_oracle(&tc).vortex_expect("prepared set oracle failed");
}

// ==================== Deterministic Haystack Regime Tests ====================

fn assert_integer_membership<T: NativePType + PrimInt + Into<PValue>>(
    set: Vec<T>,
) -> VortexResult<()> {
    let mut needles = vec![T::min_value(), T::max_value()];
    for &element in &set {
        needles.push(element);
        needles.extend(element.checked_sub(&T::one()));
        needles.extend(element.checked_add(&T::one()));
    }
    let expected = BoolArray::from_iter(needles.iter().map(|n| set.contains(n)));
    let list = Scalar::list(
        Arc::new(DType::Primitive(T::PTYPE, Nullability::NonNullable)),
        set.into_iter()
            .map(|v| Scalar::primitive(v, Nullability::NonNullable))
            .collect(),
        Nullability::NonNullable,
    );
    let mut ctx = SESSION.create_execution_ctx();
    let result = PrimitiveArray::from_iter(needles)
        .into_array()
        .apply(&list_contains(lit(list), root()))?;
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

/// Deterministically covers every `IntegerSet` storage regime: L1 bitmap (straddling zero & whole
/// `i8` / `u64::MAX` bounds), L2 bitmap (`>= 16` distinct elements spanning `4096..1<<20`),
/// small/large sparse `Sorted` binary search, and `i64::MIN..i64::MAX` overflow bounds.
#[test]
fn test_integer_set_haystack_regimes() -> VortexResult<()> {
    assert_integer_membership(vec![-3i32, -1, 0, 2, 5])?;
    assert_integer_membership(vec![i8::MIN, -1, 0, 1, i8::MAX])?;
    assert_integer_membership(vec![
        u64::MAX,
        u64::MAX - 2,
        u64::MAX - 5,
        u64::MAX - 7,
        u64::MAX - 9,
    ])?;
    assert_integer_membership(vec![
        i64::MIN,
        i64::MIN + 1,
        i64::MIN + 2,
        i64::MIN + 4,
        i64::MIN + 5,
    ])?;
    // L2 bitmap regime: >= 16 distinct elements with 4096 <= span < 1<<20.
    assert_integer_membership((0..32i64).map(|i| i * 2000).collect())?;
    // Sparse Sorted binary search (small, large, and full i64::MIN..i64::MAX span).
    assert_integer_membership(vec![-(1i64 << 40), 1, 1 << 40])?;
    assert_integer_membership((0..100u64).map(|v| v << 30).collect())?;
    assert_integer_membership(vec![i64::MIN, i64::MAX])
}

#[rstest]
fn test_decimal_haystack_all_storage_widths(
    #[values(2, 4, 9, 18, 38, 76)] precision: u8,
    #[values(false, true)] sql_null_semantics: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalDType::new(precision, 1);
    let dtype = DType::Decimal(decimal, Nullability::Nullable);
    let mut elements: Vec<_> = [1i8, -2, 1]
        .map(|v| Scalar::decimal(v.into(), decimal, Nullability::Nullable))
        .into();
    elements.push(Scalar::null(dtype.clone()));
    let set = PreparedSetData::try_new(Scalar::list(dtype, elements, Nullability::NonNullable))?;
    let needles =
        DecimalArray::from_option_iter([Some(-2i8), Some(0), Some(1), None], decimal).into_array();
    let options = ListContainsOptions {
        sql_null_semantics,
        ..Default::default()
    };
    let non_match = (!sql_null_semantics).then_some(false);
    assert_arrays_eq!(
        set.contains(&needles, &options, &mut ctx)?,
        BoolArray::from_iter([Some(true), non_match, Some(true), None]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::dense_i128(38, i256::from_i128(1i128 << 100), true)]
#[case::sparse_i128(38, i256::from_i128(1i128 << 100), false)]
#[case::dense_i256(76, i256::from_parts(0, 1i128 << 72), true)]
#[case::sparse_i256(76, i256::from_parts(0, 1i128 << 72), false)]
fn test_decimal_wide_values_low_64_bit_collision(
    #[case] precision: u8,
    #[case] base: i256,
    #[case] dense: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalDType::new(precision, 2);
    let second = if dense { base + i256::ONE } else { -base };
    let list = Scalar::list(
        DType::Decimal(decimal, Nullability::NonNullable),
        [second, base, second]
            .map(|v| Scalar::decimal(v.into(), decimal, Nullability::NonNullable))
            .into(),
        Nullability::NonNullable,
    );
    // `base + (1 << 64)` shares the low 64 bits of `base` and must not match.
    let needles = DecimalArray::from_option_iter::<i256, _>(
        [
            Some(base),
            Some(second),
            Some(base - i256::ONE),
            Some(i256::ZERO),
            Some(base + i256::from_i128(1i128 << 64)),
            None,
        ],
        decimal,
    )
    .into_array();
    let set = PreparedSetData::try_new(list)?;
    assert_arrays_eq!(
        set.contains(&needles, &ListContainsOptions::default(), &mut ctx)?,
        BoolArray::from_iter([
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            None,
        ]),
        &mut ctx
    );
    Ok(())
}

// ==================== Focused Metadata, Guard & Edge-Case Tests ====================

fn i32_set(values: Vec<Option<i32>>) -> Scalar {
    let element = DType::Primitive(PType::I32, Nullability::Nullable);
    Scalar::list(
        Arc::new(element.clone()),
        values
            .into_iter()
            .map(|v| match v {
                Some(v) => Scalar::primitive(v, Nullability::Nullable),
                None => Scalar::null(element.clone()),
            })
            .collect(),
        Nullability::NonNullable,
    )
}

#[test]
fn test_display_and_strictness() {
    assert_eq!(
        list_contains(get_item("tags", root()), lit("urgent")).to_string(),
        "vortex.list.contains($.tags, \"urgent\")"
    );
    assert_eq!(
        list_contains(root(), lit(42)).to_string(),
        "vortex.list.contains($, 42i32)"
    );

    let empty = i32_set(vec![]);
    let default = list_contains(lit(empty.clone()), root());
    let sql = in_list(root(), lit(empty));
    assert!(
        !default
            .as_scalar()
            .is_some_and(|f| f.signature().is_strict())
    );
    assert!(sql.as_scalar().is_some_and(|f| f.signature().is_strict()));
}

#[test]
fn test_options_serde_roundtrip_and_legacy_formats() -> VortexResult<()> {
    for options in [
        ListContainsOptions::default(),
        SQL,
        ListContainsOptions {
            sql_null_semantics: true,
            compare_mode: IEEE754,
        },
    ] {
        let bytes = ListContains
            .serialize(&options)?
            .ok_or_else(|| vortex_err!("list contains options serialize"))?;
        assert_eq!(ListContains.deserialize(&bytes, &SESSION)?, options);
    }

    assert_eq!(
        ListContains.deserialize(&[], &SESSION)?,
        ListContainsOptions::default()
    );

    let no_mode = pb::ListContainsOpts {
        sql_null_semantics: true,
        compare_mode: None,
    }
    .encode_to_vec();
    assert_eq!(ListContains.deserialize(&no_mode, &SESSION)?, SQL);
    Ok(())
}

#[rstest]
#[case::no_float_order(vec![], "compare mode has no float order")]
#[case::unknown_type(
    vec![pb::compare_mode::TypeOrder { order: None }],
    "compare mode has the order of an unknown type"
)]
fn test_deserialize_rejects_invalid_mode(
    #[case] orders: Vec<pb::compare_mode::TypeOrder>,
    #[case] message: &str,
) {
    let bytes = pb::ListContainsOpts {
        sql_null_semantics: false,
        compare_mode: Some(pb::CompareMode { orders }),
    }
    .encode_to_vec();
    let error = ListContains.deserialize(&bytes, &SESSION).unwrap_err();
    assert!(error.to_string().contains(message), "{error}");
}

#[rstest]
#[case::non_list_arg0(
    DType::Primitive(PType::I32, Nullability::NonNullable),
    DType::Primitive(PType::I32, Nullability::NonNullable),
    "First argument to ListContains must be a List"
)]
#[case::fixed_size_list_arg0(
    DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
        2,
        Nullability::NonNullable
    ),
    DType::Primitive(PType::I32, Nullability::NonNullable),
    "First argument to ListContains must be a List"
)]
#[case::mismatched_needle(
    DType::List(
        Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
        Nullability::NonNullable
    ),
    DType::Utf8(Nullability::NonNullable),
    "does not match search value"
)]
fn test_return_dtype_error_guards(
    #[case] list_dtype: DType,
    #[case] needle_dtype: DType,
    #[case] expected_msg: &str,
    #[values(ListContainsOptions::default(), SQL)] options: ListContainsOptions,
) {
    let err = ListContains
        .return_dtype(&options, &[list_dtype, needle_dtype])
        .unwrap_err();
    assert!(err.to_string().contains(expected_msg), "{err}");
}

#[test]
fn test_return_dtype_sql_nullability_widening() -> VortexResult<()> {
    let scope = DType::Struct(
        StructFields::new(
            ["a"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    );
    let set = lit(i32_set(vec![Some(1)]));
    assert_eq!(
        list_contains(set.clone(), col("a")).return_dtype(&scope)?,
        DType::Bool(Nullability::NonNullable)
    );
    assert_eq!(
        in_list(col("a"), set).return_dtype(&scope)?,
        DType::Bool(Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn test_validity_reduction_and_null_list_optimization() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::Nullable);
    let bound = |expr: Expression| expr.bind(&dtype);
    let needle = bound(root())?;
    let with_null = lit(i32_set(vec![Some(2), None]));
    let without_null = lit(i32_set(vec![Some(2)]));

    let expr = bound(list_contains(with_null.clone(), root()))?;
    assert_eq!(expr.validity()?, bound::is_not_null(needle.clone()));
    let expr = bound(in_list(root(), without_null))?;
    assert_eq!(expr.validity()?, bound::is_not_null(needle));
    let expr = bound(in_list(root(), with_null))?;
    assert_eq!(expr.validity()?, bound::is_not_null(expr));

    let expr = bound(list_contains(lit(i32_set(vec![])), root()))?;
    assert_eq!(expr.validity()?, bound::lit(true));

    let null_list = Scalar::null(DType::List(Arc::new(dtype.clone()), Nullability::Nullable));
    let null_expr = bound(list_contains(lit(null_list.clone()), root()))?;
    assert_eq!(null_expr.validity()?, bound::lit(false));
    assert_eq!(
        null_expr.optimize()?.child(0).as_opt::<Literal>(),
        Some(&null_list)
    );
    Ok(())
}

#[rstest]
#[case::default(ListContainsOptions::default(), Some(false))]
#[case::sql(SQL, None)]
fn test_constant_needle_in_constant_set_folds(
    #[case] options: ListContainsOptions,
    #[case] expected: Option<bool>,
) -> VortexResult<()> {
    let set = i32_set(vec![Some(2), None]);
    let needle = Scalar::primitive(3i32, Nullability::Nullable);
    let expected = match expected {
        Some(value) => Scalar::bool(value, Nullability::Nullable),
        None => Scalar::null(DType::Bool(Nullability::Nullable)),
    };

    let expr = list_contains_opts(lit(set.clone()), lit(needle.clone()), options)
        .bind(&DType::Primitive(PType::I32, Nullability::Nullable))?
        .optimize()?;
    assert_eq!(expr.as_opt::<Literal>(), Some(&expected));

    let array = ListContains::try_new_opts(
        ConstantArray::new(set, 3).into_array(),
        ConstantArray::new(needle, 3).into_array(),
        options,
    )?
    .into_array()
    .optimize_ctx(&SESSION)?;
    assert_eq!(array.as_constant(), Some(&expected));
    Ok(())
}

#[test]
fn test_fold_sliced_match_bitmap() {
    let mut ctx = SESSION.create_execution_ctx();
    let bits = BitBuffer::from_iter([true, false, true, false]).slice(1..);
    let result = process_matches::<u32, u32>(
        BoolArray::new(bits, Validity::NonNullable),
        3,
        PrimitiveArray::from_iter([0u32, 1, 2]),
        PrimitiveArray::from_iter([1u32, 1, 1]),
        &mut ctx,
    );
    assert_eq!(result, BitBuffer::from_iter([false, true, false]));
}

#[test]
fn test_u8_offset_size_boundary_at_256() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let list_array = ListViewArray::new(
        Buffer::from_iter(0i32..256).into_array(),
        buffer![0u8, 100, 200, 254].into_array(),
        buffer![50u8, 50, 54, 2].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )
    .into_array();

    assert_arrays_eq!(
        list_array
            .clone()
            .apply(&list_contains(root(), lit(255i32)))?,
        BoolArray::from_iter([false, false, false, true]),
        &mut ctx
    );
    assert_arrays_eq!(
        list_array.apply(&list_contains(root(), lit(0i32)))?,
        BoolArray::from_iter([true, false, false, false]),
        &mut ctx
    );
    Ok(())
}

/// A constant float needle against a non-constant overlapping ListView takes the scalar fast path
/// instead of gathering every overlapping view element.
#[rstest]
#[case::total_order(CompareMode::TOTAL_ORDER, false)]
#[case::zeros_equal(ZEROS_EQUAL, true)]
#[case::canonical(CANONICAL, true)]
#[case::ieee754(IEEE754, true)]
fn test_constant_float_needle_overlapping_list_view(
    #[case] mode: CompareMode,
    #[case] expected: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let n = 1024u32;
    let mut elements = vec![1.0f64; n as usize];
    elements[0] = 0.0;
    let lists = ListViewArray::try_new(
        PrimitiveArray::from_iter(elements).into_array(),
        PrimitiveArray::from_iter(vec![0u32; n as usize]).into_array(),
        PrimitiveArray::from_iter(vec![n; n as usize]).into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let needle = ConstantArray::new(-0.0f64, n as usize).into_array();
    let options = ListContainsOptions {
        sql_null_semantics: true,
        compare_mode: mode,
    };

    assert_arrays_eq!(
        ListContains::try_new_opts(lists, needle, options)?.into_array(),
        BoolArray::from_iter(vec![expected; n as usize]),
        &mut ctx
    );
    Ok(())
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared Hegel property-test harness for differential checking of array encodings.
//!
//! Encoding crates enable `vortex-array`'s `_test-harness-hegel` feature in `[dev-dependencies]`
//! and call [`check_encoding_roundtrip_and_ops`] to compare an encoded array against a canonical
//! reference across bulk execution, scalar probing, optional flatbuffer serde, optional scalar
//! comparison, differential `cast`, scalar aggregates, `search_sorted`, and chained
//! `slice` / `take` / `filter` / `mask` operations.

use std::ops::Range;

use hegel::TestCase;
use hegel::generators as gs;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::ArrayContext;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::all_non_distinct::variant_scalars_equal;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::nan_count::nan_count;
use crate::aggregate_fn::fns::sum::sum;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::IntegerPType;
use crate::dtype::MAX_PRECISION;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::operators::CompareOperator;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedArray;
use crate::search_sorted::SearchSortedSide;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

const ALL_COMPARE_OPERATORS: [CompareOperator; 6] = [
    CompareOperator::Eq,
    CompareOperator::NotEq,
    CompareOperator::Lt,
    CompareOperator::Lte,
    CompareOperator::Gt,
    CompareOperator::Gte,
];

const ALL_PTYPES: [PType; 11] = [
    PType::U8,
    PType::U16,
    PType::U32,
    PType::U64,
    PType::I8,
    PType::I16,
    PType::I32,
    PType::I64,
    PType::F16,
    PType::F32,
    PType::F64,
];

/// Configuration for [`check_encoding_roundtrip_and_ops`].
#[derive(Clone, Debug)]
pub struct EncodingCheckOptions {
    /// Whether to check flatbuffer serialize/deserialize round trip through `ctx.session()`.
    pub serde_roundtrip: bool,
    /// Whether to check scalar `compare` against a constant when the array's dtype is comparable.
    pub compare: bool,
    /// Whether to check differential `cast` against canonical on root and post-op arrays.
    pub cast: bool,
    /// Whether to check differential aggregates (`is_constant`, `min_max`, `nan_count`, `sum`).
    pub aggregates: bool,
    /// Whether to check differential `search_sorted` on sorted arrays or sorted prefixes.
    pub search_sorted: bool,
    /// Maximum number of chained operations (`slice`, `take`, `filter`, `mask`) to run per case.
    pub max_ops: usize,
    /// Maximum number of random scalar indices to probe in addition to boundary indices.
    pub max_scalar_probes: usize,
}

impl Default for EncodingCheckOptions {
    fn default() -> Self {
        Self {
            serde_roundtrip: true,
            compare: true,
            cast: true,
            aggregates: true,
            search_sorted: true,
            max_ops: 3,
            max_scalar_probes: 6,
        }
    }
}

/// Draws a boolean pattern of exact length `len` mixing constant masks, single-bit pulses,
/// short-pattern cycling (for periodic/word-boundary coverage), and dense random draws.
pub fn draw_bool_pattern(tc: &TestCase, len: usize) -> Vec<bool> {
    if len == 0 {
        return Vec::new();
    }

    match tc.draw(gs::integers::<u8>().max_value(4)) {
        0 => vec![true; len],
        1 => vec![false; len],
        2 => {
            let background = tc.draw(gs::booleans());
            let mut bits = vec![background; len];
            let pos = tc.draw(gs::integers::<usize>().max_value(len - 1));
            bits[pos] = !background;
            bits
        }
        3 => {
            let period = tc.draw(gs::integers::<usize>().min_value(1).max_value(len.min(8)));
            let pattern: Vec<bool> =
                tc.draw(gs::vecs(gs::booleans()).min_size(period).max_size(period));
            let phase = tc.draw(gs::integers::<usize>().max_value(period - 1));
            (0..len).map(|i| pattern[(i + phase) % period]).collect()
        }
        _ => tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len)),
    }
}

/// Draws a [`Mask`] of length `len` using [`draw_bool_pattern`].
pub fn draw_mask(tc: &TestCase, len: usize) -> Mask {
    Mask::from_iter(draw_bool_pattern(tc, len))
}

/// Draws a nullable [`Validity`] of length `len` covering `AllValid`, `AllInvalid`, and patterned
/// bitmaps.
pub fn draw_validity(tc: &TestCase, len: usize) -> Validity {
    match tc.draw(gs::integers::<u8>().max_value(4)) {
        0 => Validity::AllValid,
        1 => Validity::AllInvalid,
        _ => Validity::from_iter(draw_bool_pattern(tc, len)),
    }
}

/// Draws a [`Validity`] matching `nullability` and `len`.
pub fn draw_validity_for(tc: &TestCase, nullability: Nullability, len: usize) -> Validity {
    match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => draw_validity(tc, len),
    }
}

/// Serializes `array` with `session` and decodes it back via [`SerializedArray`].
pub fn serialize_and_decode(array: &ArrayRef, session: &VortexSession) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let serialized = array.serialize(&array_ctx, session, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }
    SerializedArray::try_from(concat.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        session,
    )
}

/// Checks an `encoded` array differentially against its `canonical` reference.
///
/// Exercises:
/// 1. Length, dtype, and bulk execution equality (`assert_arrays_eq!`).
/// 2. Boundary and random `execute_scalar` probes.
/// 3. Optional flatbuffer serde round trip (`options.serde_roundtrip`).
/// 4. Optional scalar `compare` against a constant (`options.compare`).
/// 5. Optional differential `cast` (`options.cast`).
/// 6. Optional differential aggregates (`options.aggregates`).
/// 7. Optional differential `search_sorted` (`options.search_sorted`).
/// 8. Chained `slice`, `take`, `filter`, and `mask` operations (both lazy and post-optimization),
///    re-running `compare`, `cast`, `aggregates`, and `search_sorted` on the resulting array.
pub fn check_encoding_roundtrip_and_ops(
    tc: &TestCase,
    encoded: &ArrayRef,
    canonical: &ArrayRef,
    options: &EncodingCheckOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    assert_eq!(encoded.len(), canonical.len(), "encoded length mismatch");
    assert_eq!(encoded.dtype(), canonical.dtype(), "encoded dtype mismatch");
    assert_arrays_eq!(encoded.clone(), canonical.clone(), ctx);

    check_scalar_probes(tc, encoded, canonical, options.max_scalar_probes, ctx)?;

    if options.serde_roundtrip {
        let decoded = serialize_and_decode(encoded, ctx.session())?;
        assert_eq!(decoded.len(), canonical.len(), "serde length mismatch");
        assert_eq!(decoded.dtype(), canonical.dtype(), "serde dtype mismatch");
        assert_arrays_eq!(decoded.clone(), canonical.clone(), ctx);
        if !decoded.is_empty() {
            let idx = tc.draw(gs::integers::<usize>().max_value(decoded.len() - 1));
            assert_scalar_eq(
                &decoded.execute_scalar(idx, ctx)?,
                &canonical.execute_scalar(idx, ctx)?,
                &format!("serde scalar_at mismatch at index {idx}"),
            );
        }
    }

    if options.compare {
        check_compare(tc, encoded, canonical, ctx)?;
    }

    if options.cast {
        check_cast(tc, encoded, canonical, ctx)?;
    }

    if options.aggregates {
        check_aggregates(encoded, canonical, ctx)?;
    }

    if options.search_sorted {
        check_search_sorted(tc, encoded, canonical, ctx)?;
    }

    if options.max_ops == 0 {
        return Ok(());
    }

    let num_ops = tc.draw(
        gs::integers::<usize>()
            .min_value(1)
            .max_value(options.max_ops),
    );
    let mut cur_encoded = encoded.clone();
    let mut cur_canonical = canonical.clone();

    for step in 0..num_ops {
        let (next_encoded, next_canonical) = apply_random_op(tc, &cur_encoded, &cur_canonical)?;
        // Keep the reference materialized so post-op differential checks compare the encoded
        // chain against a true canonical array rather than a second lazy op tree.
        let next_canonical = next_canonical
            .execute::<RecursiveCanonical>(ctx)?
            .0
            .into_array();

        assert_step_eq(tc, &next_encoded, &next_canonical, step, "op", ctx)?;

        // Mix unoptimized chains (which accumulate ops lazily before execution) with
        // intermediate `optimize_ctx` steps (which exercise eager parent-reduce rewrites).
        cur_encoded = if tc.draw(gs::booleans()) {
            let optimized = next_encoded.optimize_ctx(ctx.session())?;
            assert_step_eq(tc, &optimized, &next_canonical, step, "optimized op", ctx)?;
            optimized
        } else {
            next_encoded
        };
        cur_canonical = next_canonical;
    }

    if options.compare && !cur_encoded.is_empty() {
        check_compare(tc, &cur_encoded, &cur_canonical, ctx)?;
    }

    if options.cast {
        check_cast(tc, &cur_encoded, &cur_canonical, ctx)?;
    }

    if options.aggregates {
        check_aggregates(&cur_encoded, &cur_canonical, ctx)?;
    }

    if options.search_sorted {
        check_search_sorted(tc, &cur_encoded, &cur_canonical, ctx)?;
    }

    Ok(())
}

fn assert_scalar_eq(actual: &Scalar, expected: &Scalar, context: &str) {
    if matches!(expected.dtype(), DType::Variant(_)) {
        assert!(
            variant_scalars_equal(actual, expected),
            "{context}: actual={actual:?} expected={expected:?}"
        );
    } else {
        assert_eq!(actual, expected, "{context}");
    }
}

fn assert_step_eq(
    tc: &TestCase,
    encoded: &ArrayRef,
    canonical: &ArrayRef,
    step: usize,
    label: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    assert_eq!(
        encoded.len(),
        canonical.len(),
        "{label} step {step} length mismatch"
    );
    assert_eq!(
        encoded.dtype(),
        canonical.dtype(),
        "{label} step {step} dtype mismatch"
    );
    assert_arrays_eq!(encoded.clone(), canonical.clone(), ctx);

    if !encoded.is_empty() {
        let probe = tc.draw(gs::integers::<usize>().max_value(encoded.len() - 1));
        assert_scalar_eq(
            &encoded.execute_scalar(probe, ctx)?,
            &canonical.execute_scalar(probe, ctx)?,
            &format!("{label} step {step} scalar_at mismatch at index {probe}"),
        );
    }

    Ok(())
}

fn check_scalar_probes(
    tc: &TestCase,
    encoded: &ArrayRef,
    canonical: &ArrayRef,
    max_probes: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let len = encoded.len();
    if len == 0 {
        return Ok(());
    }

    // Always probe endpoints so off-by-one and boundary errors are caught immediately.
    for idx in [0, len - 1] {
        assert_scalar_eq(
            &encoded.execute_scalar(idx, ctx)?,
            &canonical.execute_scalar(idx, ctx)?,
            &format!("scalar_at mismatch at boundary index {idx}"),
        );
    }

    if max_probes == 0 || len <= 2 {
        return Ok(());
    }

    let extra = tc.draw(gs::integers::<usize>().max_value(max_probes.min(len)));
    for _ in 0..extra {
        let idx = tc.draw(gs::integers::<usize>().max_value(len - 1));
        assert_scalar_eq(
            &encoded.execute_scalar(idx, ctx)?,
            &canonical.execute_scalar(idx, ctx)?,
            &format!("scalar_at mismatch at index {idx}"),
        );
    }
    Ok(())
}

fn is_comparable_dtype(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_)
    )
}

fn supports_compare(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(ext) => is_comparable_dtype(ext.storage_dtype()),
        _ => is_comparable_dtype(dtype),
    }
}

fn check_compare(
    tc: &TestCase,
    encoded: &ArrayRef,
    canonical: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let len = encoded.len();
    if len == 0 || !supports_compare(encoded.dtype()) {
        return Ok(());
    }

    let idx = tc.draw(gs::integers::<usize>().max_value(len - 1));
    let mut pivot = canonical.execute_scalar(idx, ctx)?;
    if tc.draw(gs::booleans()) {
        pivot = pivot.into_nullable();
    }
    let rhs = ConstantArray::new(pivot, len).into_array();
    let op = tc.draw(gs::sampled_from(&ALL_COMPARE_OPERATORS));

    let actual = encoded.compare(rhs.clone(), op)?;
    let expected = canonical.compare(rhs, op)?;
    assert_arrays_eq!(actual, expected, ctx);
    Ok(())
}

fn cast_and_execute(
    array: &ArrayRef,
    target: DType,
    optimize: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let casted = array.cast(target)?;
    let casted = if optimize {
        casted.optimize_ctx(ctx.session())?
    } else {
        casted
    };
    Ok(casted.execute::<RecursiveCanonical>(ctx)?.0.into_array())
}

fn draw_cast_targets(tc: &TestCase, dtype: &DType) -> Vec<DType> {
    let flipped = Nullability::from(!dtype.is_nullable());
    let mut targets = Vec::with_capacity(2);

    match dtype {
        DType::Null => {
            let candidates = [
                DType::Bool(Nullability::Nullable),
                DType::Primitive(PType::I32, Nullability::Nullable),
                DType::Primitive(PType::F64, Nullability::Nullable),
                DType::Utf8(Nullability::Nullable),
                DType::Primitive(PType::I32, Nullability::NonNullable),
            ];
            targets.push(tc.draw(gs::sampled_from(&candidates)));
        }
        DType::Bool(_) => {
            targets.push(DType::Bool(flipped));
        }
        DType::Primitive(..) => {
            targets.push(dtype.with_nullability(flipped));
            let target_ptype = tc.draw(gs::sampled_from(&ALL_PTYPES));
            let target_nullability = if tc.draw(gs::booleans()) {
                dtype.nullability()
            } else {
                flipped
            };
            targets.push(DType::Primitive(target_ptype, target_nullability));
        }
        DType::Decimal(dec, _) => {
            targets.push(DType::Decimal(*dec, flipped));
            let target_nullability = if tc.draw(gs::booleans()) {
                dtype.nullability()
            } else {
                flipped
            };
            let min_p = u8::try_from(dec.scale().max(1)).unwrap_or(1);
            let new_precision = match tc.draw(gs::integers::<u8>().max_value(2)) {
                0 => dec.precision().saturating_add(2).min(MAX_PRECISION),
                1 => dec.precision().saturating_sub(2).max(min_p),
                _ => dec.precision(),
            };
            if let Ok(target_dec) = DecimalDType::try_new(new_precision, dec.scale()) {
                targets.push(DType::Decimal(target_dec, target_nullability));
            }
        }
        DType::Utf8(_) => {
            targets.push(DType::Utf8(flipped));
        }
        DType::Binary(_) => {
            targets.push(DType::Binary(flipped));
        }
        DType::Variant(_) => {}
        _ => {
            targets.push(dtype.with_nullability(flipped));
        }
    }

    targets
}

fn check_cast(
    tc: &TestCase,
    encoded: &ArrayRef,
    canonical: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    for target in draw_cast_targets(tc, encoded.dtype()) {
        match cast_and_execute(canonical, target.clone(), false, ctx) {
            Ok(expected) => {
                let actual = cast_and_execute(encoded, target.clone(), false, ctx)?;
                assert_arrays_eq!(actual, expected, ctx);

                let optimized = cast_and_execute(encoded, target, true, ctx)?;
                assert_arrays_eq!(optimized, expected, ctx);
            }
            Err(_) => {
                assert!(
                    cast_and_execute(encoded, target.clone(), false, ctx).is_err(),
                    "encoded cast to {target} succeeded where canonical failed"
                );
                assert!(
                    cast_and_execute(encoded, target.clone(), true, ctx).is_err(),
                    "optimized encoded cast to {target} succeeded where canonical failed"
                );
            }
        }
    }
    Ok(())
}

fn supports_minmax(dtype: &DType) -> bool {
    match dtype {
        DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        DType::Extension(ext) => supports_minmax(ext.storage_dtype()),
        _ => false,
    }
}

fn supports_exact_sum(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Bool(_)
            | DType::Primitive(
                PType::U8
                    | PType::U16
                    | PType::U32
                    | PType::U64
                    | PType::I8
                    | PType::I16
                    | PType::I32,
                _
            )
            | DType::Decimal(..)
    )
}

fn check_aggregates(
    encoded: &ArrayRef,
    canonical: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let dtype = encoded.dtype();

    if !matches!(dtype, DType::Null | DType::Variant(..)) {
        assert_eq!(
            is_constant(encoded, ctx)?,
            is_constant(canonical, ctx)?,
            "is_constant mismatch for {dtype}"
        );
    }

    if supports_minmax(dtype) {
        let opts = NumericalAggregateOpts::default();
        assert_eq!(
            min_max(encoded, ctx, opts)?,
            min_max(canonical, ctx, opts)?,
            "min_max mismatch for {dtype}"
        );
    }

    if dtype.is_float() {
        assert_eq!(
            nan_count(encoded, ctx)?,
            nan_count(canonical, ctx)?,
            "nan_count mismatch for {dtype}"
        );
    }

    if is_comparable_dtype(dtype) {
        assert_eq!(
            is_sorted(encoded, ctx)?,
            is_sorted(canonical, ctx)?,
            "is_sorted mismatch for {dtype}"
        );
    }

    if supports_exact_sum(dtype) {
        assert_eq!(
            sum(encoded, ctx)?,
            sum(canonical, ctx)?,
            "sum mismatch for {dtype}"
        );
    }

    Ok(())
}

fn neighbor_scalars(scalar: &Scalar) -> Vec<Scalar> {
    let mut out = Vec::new();
    let nullability = scalar.dtype().nullability();
    match scalar.dtype() {
        DType::Bool(_) => {
            out.push(Scalar::bool(false, nullability));
            out.push(Scalar::bool(true, nullability));
        }
        DType::Primitive(ptype, _) => {
            macro_rules! push_int_neighbors {
                ($v:expr) => {{
                    out.push(Scalar::primitive($v.saturating_sub(1), nullability));
                    out.push(Scalar::primitive($v.saturating_add(1), nullability));
                }};
            }
            if let Some(pvalue) = scalar.as_primitive().pvalue() {
                match pvalue {
                    PValue::U8(v) => push_int_neighbors!(v),
                    PValue::U16(v) => push_int_neighbors!(v),
                    PValue::U32(v) => push_int_neighbors!(v),
                    PValue::U64(v) => push_int_neighbors!(v),
                    PValue::I8(v) => push_int_neighbors!(v),
                    PValue::I16(v) => push_int_neighbors!(v),
                    PValue::I32(v) => push_int_neighbors!(v),
                    PValue::I64(v) => push_int_neighbors!(v),
                    PValue::F16(v) if v.is_finite() => {
                        out.push(Scalar::primitive(
                            half::f16::from_f32(v.to_f32() - 1.0),
                            nullability,
                        ));
                        out.push(Scalar::primitive(
                            half::f16::from_f32(v.to_f32() + 1.0),
                            nullability,
                        ));
                    }
                    PValue::F32(v) if v.is_finite() => {
                        out.push(Scalar::primitive(v - 1.0, nullability));
                        out.push(Scalar::primitive(v + 1.0, nullability));
                    }
                    PValue::F64(v) if v.is_finite() => {
                        out.push(Scalar::primitive(v - 1.0, nullability));
                        out.push(Scalar::primitive(v + 1.0, nullability));
                    }
                    PValue::F16(_) | PValue::F32(_) | PValue::F64(_) => {}
                }
            }
            out.push(Scalar::default_value(&DType::Primitive(
                *ptype,
                nullability,
            )));
        }
        DType::Decimal(dec, _) => {
            if let Some(dval) = scalar.as_decimal().decimal_value() {
                for delta in [-1i128, 1i128] {
                    if let Some(cand) = dval.checked_add(&DecimalValue::I128(delta))
                        && cand.fits_in_precision(*dec)
                    {
                        out.push(Scalar::decimal(cand, *dec, nullability));
                    }
                }
            }
        }
        DType::Utf8(_) => {
            out.push(Scalar::utf8("", nullability));
            if let Some(s) = scalar.as_utf8().value() {
                let mut extended = s.as_str().to_owned();
                extended.push('z');
                out.push(Scalar::utf8(extended, nullability));
            }
        }
        DType::Binary(_) => {
            out.push(Scalar::binary(ByteBuffer::empty(), nullability));
            if let Some(b) = scalar.as_binary().value() {
                let mut extended = b.as_slice().to_vec();
                extended.push(0xFF);
                out.push(Scalar::binary(ByteBuffer::from(extended), nullability));
            }
        }
        _ => {}
    }
    out
}

fn probe_search_sorted(
    tc: &TestCase,
    encoded: &ArrayRef,
    canonical: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let len = encoded.len();
    let mut needles = Vec::with_capacity(8);

    if len == 0 {
        needles.push(Scalar::default_value(encoded.dtype()));
    } else {
        let first = canonical.execute_scalar(0, ctx)?;
        let last = canonical.execute_scalar(len - 1, ctx)?;
        let mid =
            canonical.execute_scalar(tc.draw(gs::integers::<usize>().max_value(len - 1)), ctx)?;

        needles.extend(neighbor_scalars(&first));
        needles.extend(neighbor_scalars(&last));
        needles.push(first);
        needles.push(mid);
        needles.push(last);
    }

    for needle in &needles {
        for side in [SearchSortedSide::Left, SearchSortedSide::Right] {
            let actual = SearchSortedArray::new(encoded, ctx).search_sorted(needle, side)?;
            let expected = SearchSortedArray::new(canonical, ctx).search_sorted(needle, side)?;
            assert_eq!(
                actual, expected,
                "search_sorted mismatch for needle={needle:?} side={side}"
            );
        }
    }

    Ok(())
}

fn check_search_sorted(
    tc: &TestCase,
    encoded: &ArrayRef,
    canonical: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if !is_comparable_dtype(encoded.dtype()) {
        return Ok(());
    }

    if is_sorted(canonical, ctx)? {
        return probe_search_sorted(tc, encoded, canonical, ctx);
    }

    // Even when the full draw is unsorted, probe a sorted prefix when at least two elements
    // are in non-decreasing order so `search_sorted` exercises sliced encoded probes too.
    if encoded.len() >= 2 {
        let mut prefix_len = 1usize;
        let mut prev = canonical.execute_scalar(0, ctx)?;
        for i in 1..encoded.len().min(16) {
            let cur = canonical.execute_scalar(i, ctx)?;
            if prev.partial_cmp(&cur).is_some_and(|o| o.is_le()) {
                prefix_len = i + 1;
                prev = cur;
            } else {
                break;
            }
        }
        if prefix_len >= 2 {
            let enc_prefix = encoded.slice(0..prefix_len)?;
            let can_prefix = canonical.slice(0..prefix_len)?;
            if is_sorted(&can_prefix, ctx)? {
                probe_search_sorted(tc, &enc_prefix, &can_prefix, ctx)?;
            }
        }
    }

    Ok(())
}

fn apply_random_op(
    tc: &TestCase,
    encoded: &ArrayRef,
    canonical: &ArrayRef,
) -> VortexResult<(ArrayRef, ArrayRef)> {
    let len = encoded.len();
    // `Mask::execute` on an all-true constant mask calls `input.cast(input.dtype().as_nullable())`,
    // which `cast_canonical` rejects for `DType::Variant` ("Variant arrays don't support casting").
    let max_op = if matches!(encoded.dtype(), DType::Variant(Nullability::NonNullable)) {
        2
    } else {
        3
    };
    match tc.draw(gs::integers::<u8>().max_value(max_op)) {
        // Slice (including empty slices at non-zero offsets and singletons)
        0 => {
            let range = draw_slice_range(tc, len);
            Ok((encoded.slice(range.clone())?, canonical.slice(range)?))
        }
        // Take (across fitting signed/unsigned index widths and nullable/non-nullable indices)
        1 => {
            let indices = draw_take_indices(tc, len);
            Ok((encoded.take(indices.clone())?, canonical.take(indices)?))
        }
        // Filter (short-pattern cycling, constant, sparse, and dense masks)
        2 => {
            let mask = draw_mask(tc, len);
            Ok((encoded.filter(mask.clone())?, canonical.filter(mask)?))
        }
        // Mask (sets rows to null where the boolean array is false)
        _ => {
            let keep = draw_mask(tc, len).into_array();
            Ok((
                encoded.clone().mask(keep.clone())?,
                canonical.clone().mask(keep)?,
            ))
        }
    }
}

fn draw_slice_range(tc: &TestCase, len: usize) -> Range<usize> {
    if len == 0 {
        return 0..0;
    }

    match tc.draw(gs::integers::<u8>().max_value(3)) {
        0 => {
            let pos = tc.draw(gs::integers::<usize>().max_value(len));
            pos..pos
        }
        1 => {
            let pos = tc.draw(gs::integers::<usize>().max_value(len - 1));
            pos..pos + 1
        }
        _ => {
            let a = tc.draw(gs::integers::<usize>().max_value(len));
            let b = tc.draw(gs::integers::<usize>().max_value(len));
            a.min(b)..a.max(b)
        }
    }
}

fn draw_take_indices(tc: &TestCase, len: usize) -> ArrayRef {
    let index_ptype = draw_fitting_index_ptype(tc, len);
    let take_len = if len == 0 {
        0
    } else {
        match tc.draw(gs::integers::<u8>().max_value(3)) {
            0 => 0,
            1 => 1,
            _ => tc.draw(gs::integers::<usize>().max_value(len.min(32))),
        }
    };

    let raw_indices: Vec<usize> = if take_len == 0 {
        Vec::new()
    } else {
        tc.draw(
            gs::vecs(gs::integers::<usize>().max_value(len - 1))
                .min_size(take_len)
                .max_size(take_len),
        )
    };

    let validity = if tc.draw(gs::booleans()) {
        Validity::NonNullable
    } else {
        draw_validity(tc, take_len)
    };

    match_each_integer_ptype!(index_ptype, |I| {
        build_index_array::<I>(raw_indices, validity)
    })
}

fn build_index_array<I: IntegerPType>(raw_indices: Vec<usize>, validity: Validity) -> ArrayRef {
    let buffer = Buffer::<I>::from_iter(
        raw_indices
            .into_iter()
            .map(|idx| num_traits::cast::<usize, I>(idx).vortex_expect("index fits ptype")),
    );
    PrimitiveArray::new(buffer, validity).into_array()
}

fn draw_fitting_index_ptype(tc: &TestCase, len: usize) -> PType {
    let max_idx = len.saturating_sub(1);
    let mut candidates = Vec::with_capacity(8);
    candidates.push(PType::U64);
    candidates.push(PType::I64);
    if max_idx <= u32::MAX as usize {
        candidates.push(PType::U32);
    }
    if max_idx <= i32::MAX as usize {
        candidates.push(PType::I32);
    }
    if max_idx <= u16::MAX as usize {
        candidates.push(PType::U16);
    }
    if max_idx <= i16::MAX as usize {
        candidates.push(PType::I16);
    }
    if max_idx <= u8::MAX as usize {
        candidates.push(PType::U8);
    }
    if max_idx <= i8::MAX as usize {
        candidates.push(PType::I8);
    }
    tc.draw(gs::sampled_from(&candidates))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use hegel::TestCase;
    use hegel::generators as gs;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[hegel::test(test_cases = 50, print_blob = true)]
    fn test_dict_array_passes_hegel_encoding_harness(tc: TestCase) {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<i32> = tc.draw(gs::vecs(gs::integers::<i32>()).min_size(1).max_size(12));
        let values_validity = if tc.draw(gs::booleans()) {
            Validity::NonNullable
        } else {
            draw_validity(&tc, values.len())
        };
        let values_len = values.len();
        let values_mask = values_validity
            .execute_mask(values_len, &mut ctx)
            .vortex_expect("values mask");
        let values_array =
            PrimitiveArray::new(Buffer::from(values.clone()), values_validity.clone()).into_array();

        let codes_len = tc.draw(gs::integers::<usize>().max_value(32));
        let codes: Vec<u16> = if codes_len == 0 {
            Vec::new()
        } else {
            tc.draw(
                gs::vecs(
                    gs::integers::<u16>()
                        .max_value(u16::try_from(values_len - 1).vortex_expect("fits u16")),
                )
                .min_size(codes_len)
                .max_size(codes_len),
            )
        };
        let codes_validity = if tc.draw(gs::booleans()) {
            Validity::NonNullable
        } else {
            draw_validity(&tc, codes_len)
        };
        let codes_mask = codes_validity
            .execute_mask(codes_len, &mut ctx)
            .vortex_expect("codes mask");
        let codes_array =
            PrimitiveArray::new(Buffer::from(codes.clone()), codes_validity.clone()).into_array();

        let expected_values: Vec<i32> = codes.iter().map(|&c| values[usize::from(c)]).collect();
        let expected_validity = if matches!(codes_validity, Validity::NonNullable)
            && matches!(values_validity, Validity::NonNullable)
        {
            Validity::NonNullable
        } else {
            Validity::from_iter(
                codes
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| codes_mask.value(i) && values_mask.value(usize::from(c))),
            )
        };
        let canonical =
            PrimitiveArray::new(Buffer::from(expected_values), expected_validity).into_array();

        let dict = DictArray::try_new(codes_array, values_array)
            .vortex_expect("valid dict")
            .into_array();

        check_encoding_roundtrip_and_ops(
            &tc,
            &dict,
            &canonical,
            &EncodingCheckOptions::default(),
            &mut ctx,
        )
        .vortex_expect("dict harness check");
    }
}

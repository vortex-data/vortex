// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A probe that decides from one hash pass over a sample if a column gets a dictionary layout.
//!
//! The probe does not run a compressor. A cost model compares the estimated size of a dictionary
//! with the estimated size of the best encoding without a dictionary. The constants of the model
//! are estimates of the ratios of the compressor schemes.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builders::dict::DictConstraints;
use vortex_array::builders::dict::dict_encode_with_constraints;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_sequence::sequence_encode;

/// The estimated ratio of a string scheme, such as FSST, on string data.
const STRING_RATIO: f64 = 3.0;

/// The estimated bytes per row of the lengths of a string encoding.
const STRING_LENGTH_BYTES: f64 = 2.0;

/// The estimated ratio of a float scheme, such as ALP, on floats with few distinct values.
const FLOAT_RATIO: f64 = 2.0;

/// The bits that run-length encoding adds to each run for its run end.
const RUN_END_BITS: f64 = 32.0;

/// The result of a probe of a sample.
#[derive(Clone, Copy, Debug)]
pub(super) struct DictProbe {
    /// The bytes of the uncompressed sample.
    pub(super) uncompressed_bytes: u64,
    /// The estimated compressed bytes of the sample.
    pub(super) estimated_bytes: u64,
    /// The uncompressed and the estimated bytes of the codes, if the column gets a dictionary.
    pub(super) codes: Option<(u64, u64)>,
}

/// Probes the canonical `sample`, which must not be empty, with the dictionary `constraints`.
pub(super) fn probe_dictionary(
    sample: &ArrayRef,
    constraints: &DictConstraints,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictProbe> {
    let rows = sample.len() as u64;
    let uncompressed_bytes = uncompressed_size_in_bytes(sample, ctx)? as u64;
    let plain = |estimate: f64| DictProbe {
        uncompressed_bytes,
        estimated_bytes: whole_bytes(estimate),
        codes: None,
    };

    if sample.all_invalid(ctx)? {
        return Ok(plain(0.0));
    }

    if let Some(primitive) = sample.as_opt::<Primitive>()
        && !primitive.ptype().is_float()
        && let Some(sequence) = sequence_encode(primitive, ctx)?
    {
        return Ok(plain(sequence.nbytes() as f64));
    }

    // A dictionary with more entries than half the rows does not pay. Stop the build there, so
    // that a column of distinct values costs only half a pass.
    let constraints = DictConstraints {
        max_bytes: constraints.max_bytes,
        max_len: constraints.max_len.min(sample.len() / 2 + 1),
    };
    let dict = dict_encode_with_constraints(sample, &constraints, ctx)?;

    if dict.len() != sample.len() {
        let estimate = match sample.dtype() {
            DType::Utf8(_) | DType::Binary(_) => string_bytes(rows, string_payload(sample, ctx)?),
            DType::Primitive(ptype, _) if ptype.is_float() => uncompressed_bytes as f64,
            _ => match range_bits(sample, ctx)? {
                Some(bits) => rows as f64 * bits / 8.0,
                None => uncompressed_bytes as f64,
            },
        };
        return Ok(plain(estimate));
    }

    let values = dict.values().clone();
    let entries = values.len() as u64;
    let values_bytes = uncompressed_size_in_bytes(&values, ctx)? as u64;
    if entries <= 1 {
        return Ok(plain(values_bytes as f64));
    }

    let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
    let runs = match_each_unsigned_integer_ptype!(codes.ptype(), |T| {
        count_runs(codes.as_slice::<T>())
    });
    let run_length = |bits: f64| runs as f64 * (bits + RUN_END_BITS);

    let (values_estimate, plain_estimate) = match sample.dtype() {
        DType::Utf8(_) | DType::Binary(_) => (
            string_bytes(entries, string_payload(&values, ctx)?),
            string_bytes(rows, string_payload(sample, ctx)?),
        ),
        DType::Primitive(ptype, _) if ptype.is_float() => {
            let bits = (ptype.byte_width() * 8) as f64;
            let packed = rows as f64 * bits / FLOAT_RATIO;
            (
                values_bytes as f64 / FLOAT_RATIO,
                packed.min(run_length(bits)) / 8.0,
            )
        }
        _ => {
            let bits =
                range_bits(&values, ctx)?.unwrap_or(values_bytes as f64 * 8.0 / entries as f64);
            let packed = rows as f64 * bits;
            (
                entries as f64 * bits / 8.0,
                packed.min(run_length(bits)) / 8.0,
            )
        }
    };

    let code_bits = (entries as f64).log2().ceil();
    let codes_estimate = (rows as f64 * code_bits).min(run_length(code_bits)) / 8.0;
    let dict_estimate = values_estimate + codes_estimate;

    let valid_rows = rows - sample.invalid_count(ctx)? as u64;
    if entries > valid_rows / 2 || dict_estimate >= plain_estimate {
        return Ok(plain(plain_estimate));
    }

    let codes_bytes = rows * codes.ptype().byte_width() as u64;
    Ok(DictProbe {
        uncompressed_bytes,
        estimated_bytes: whole_bytes(dict_estimate),
        codes: Some((codes_bytes, whole_bytes(codes_estimate))),
    })
}

/// Estimates the compressed bytes of `count` strings with `payload` bytes of string data.
fn string_bytes(count: u64, payload: u64) -> f64 {
    payload as f64 / STRING_RATIO + count as f64 * STRING_LENGTH_BYTES
}

/// Returns the bytes of the valid strings in `strings`.
///
/// The uncompressed size of a view array does not give this value: it counts only the 16-byte
/// view of a string that is inlined in its view.
fn string_payload(strings: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let strings = strings.clone().execute::<VarBinViewArray>(ctx)?;
    let views = strings.views();

    let payload = match strings.validity()?.execute_mask(strings.len(), ctx)? {
        Mask::AllTrue(_) => views.iter().map(|view| u64::from(view.len())).sum(),
        Mask::AllFalse(_) => 0,
        Mask::Values(valid) => views
            .iter()
            .zip(valid.bit_buffer().iter())
            .filter(|(_, is_valid)| *is_valid)
            .map(|(view, _)| u64::from(view.len()))
            .sum(),
    };

    Ok(payload)
}

fn count_runs<T: NativePType>(codes: &[T]) -> u64 {
    let changes = codes
        .windows(2)
        .filter(|pair| !pair[0].is_eq(pair[1]))
        .count() as u64;

    changes + u64::from(!codes.is_empty())
}

fn range_bits(values: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<f64>> {
    let Some(result) = min_max(values, ctx, NumericalAggregateOpts::default())? else {
        return Ok(None);
    };

    let (Some(min), Some(max)) = (as_i128(&result.min), as_i128(&result.max)) else {
        return Ok(None);
    };

    let bits = u128::BITS - max.abs_diff(min).leading_zeros();
    Ok(Some(f64::from(bits.max(1))))
}

fn as_i128(scalar: &Scalar) -> Option<i128> {
    match scalar.dtype() {
        DType::Primitive(..) => scalar.as_primitive().pvalue()?.as_i128(),
        DType::Decimal(..) => scalar.as_decimal().decimal_value()?.cast::<i128>(),
        _ => None,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "estimates are not negative and far below u64::MAX"
)]
fn whole_bytes(bytes: f64) -> u64 {
    bytes.ceil() as u64
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use hegel::TestCase;
    use hegel::generators as gs;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::builders::dict::DictConstraints;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::DictProbe;
    use super::probe_dictionary;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    const CONSTRAINTS: DictConstraints = DictConstraints {
        max_bytes: 1 << 20,
        max_len: u16::MAX as usize,
    };

    fn probe(sample: ArrayRef) -> VortexResult<DictProbe> {
        probe_dictionary(&sample, &CONSTRAINTS, &mut SESSION.create_execution_ctx())
    }

    #[hegel::test]
    fn estimates_do_not_exceed_the_input(tc: TestCase) {
        let max: i32 = tc.draw(gs::integers().min_value(0).max_value(1 << 20));
        let values: Vec<Option<i32>> = tc.draw(
            gs::vecs(gs::optional(gs::integers().min_value(0).max_value(max)))
                .min_size(1)
                .max_size(2048),
        );
        let distinct = values.iter().flatten().collect::<HashSet<_>>().len();
        let valid = values.iter().flatten().count();

        let result =
            probe(PrimitiveArray::from_option_iter(values).into_array()).vortex_expect("probe");

        assert!(result.estimated_bytes <= result.uncompressed_bytes);
        if let Some((codes_bytes, codes_estimate)) = result.codes {
            assert!(2 * distinct <= valid);
            assert!(codes_estimate <= codes_bytes);
        }
    }

    #[rstest]
    #[case::sequence(PrimitiveArray::from_iter(0i64..4096).into_array(), Some(0), false)]
    #[case::distinct_floats(
        PrimitiveArray::from_iter((0..4096).map(|i| f64::from(i).sqrt())).into_array(),
        Some(32768),
        false
    )]
    #[case::all_null(
        PrimitiveArray::from_option_iter([None::<i32>; 64]).into_array(),
        Some(0),
        false
    )]
    #[case::constant(VarBinViewArray::from_iter_str(["abc"; 64]).into_array(), None, false)]
    #[case::categories(
        VarBinViewArray::from_iter_str((0..4096).map(|i| ["books", "garden", "toys"][i % 3]))
            .into_array(),
        None,
        true
    )]
    #[case::short_strings(
        VarBinViewArray::from_iter_str((0..8192).map(|i| format!("{:012}", i % 3000)))
            .into_array(),
        None,
        true
    )]
    fn probes_corner_cases(
        #[case] sample: ArrayRef,
        #[case] estimate: Option<u64>,
        #[case] dict: bool,
    ) -> VortexResult<()> {
        let result = probe(sample)?;

        if let Some(estimate) = estimate {
            assert_eq!(result.estimated_bytes, estimate);
        }
        assert_eq!(result.codes.is_some(), dict);
        Ok(())
    }
}

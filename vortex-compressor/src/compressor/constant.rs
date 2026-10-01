// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in constant detection and encoding.
//!
//! Constant arrays are not compressed through a pluggable [`Scheme`]: the compressor always
//! detects constant leaf arrays itself, before evaluating any registered scheme. Detection is
//! skipped while compressing samples, since a constant sample does not imply that the full array
//! is constant.
//!
//! [`Scheme`]: crate::scheme::Scheme

use vortex_array::ArrayInput;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::aggregates;
use crate::aggregates::FloatDistinct;
use crate::aggregates::IntegerFrequencies;
use crate::aggregates::VarBinViewPrefixDistinct;
use crate::scheme::CompressorContext;
use crate::scheme::SchemeId;

/// Synthetic scheme ID reported in traces when the compressor's built-in constant encoding wins.
pub(crate) const CONSTANT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.compressor.constant",
};

/// Returns `true` if all valid values of the canonical array are equal, meaning the array can be
/// encoded by [`compress_constant`].
///
/// The caller must have already handled empty and all-null arrays.
///
/// Uses the cheapest available evidence per type: distinct counts when another scheme already
/// requested them, integer extrema or boolean counts where possible, and otherwise a vectorized
/// equality scan via [`IsConstant`].
///
/// Note that for types where the check falls through to [`IsConstant`] (floats without distinct
/// counts, strings, binary, decimals, and extension types), arrays that contain any nulls are
/// reported as not constant, while numeric summaries detect constant valid values under nulls.
/// This mirrors the behavior of the per-type constant schemes this module replaced.
pub(crate) fn is_constant_for_compression(
    data: &ArrayInput,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let dtype = data.array().dtype();

    if matches!(dtype, DType::Bool(_)) {
        let value_count = aggregates::valid_count(data, exec_ctx);
        let true_count = data
            .compute_result(&Sum.bind(NumericalAggregateOpts::skip_nans()), exec_ctx)?
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("nonempty valid boolean input has a sum");
        return Ok(value_count > 0 && (true_count == 0 || true_count == u64::from(value_count)));
    }

    if dtype.is_int() {
        if compress_ctx.requests_aggregate(&IntegerFrequencies.bind(EmptyOptions)) {
            return Ok(aggregates::integer_frequencies(data, exec_ctx).distinct_count() == 1);
        }
        return Ok(aggregates::integer_range(data, exec_ctx).max_minus_min() == 0);
    }

    if dtype.is_float() && compress_ctx.requests_aggregate(&FloatDistinct.bind(EmptyOptions)) {
        return Ok(aggregates::float_distinct(data, exec_ctx).distinct_count() == 1);
    }

    if (dtype.is_utf8() || dtype.is_binary())
        && compress_ctx.requests_aggregate(&VarBinViewPrefixDistinct.bind(EmptyOptions))
        && aggregates::view_prefix_distinct(data, exec_ctx) > 1
    {
        return Ok(false);
    }

    // The generic constant contract includes nulls. Preserve the compressor's existing fallback
    // policy for floats without distinct requests, strings, decimals, and extension leaves.
    if aggregates::null_count(data, exec_ctx) > 0 {
        return Ok(false);
    }
    Ok(data
        .compute_result(&IsConstant.bind(EmptyOptions), exec_ctx)?
        .as_bool()
        .value()
        .unwrap_or(false))
}

/// Encodes an array whose valid values are all equal.
///
/// Returns a [`ConstantArray`], wrapped in a [`MaskedArray`] when the array has some nulls, or a
/// null [`ConstantArray`] when the array is all-null.
///
/// # Errors
///
/// Returns an error if computing validity or extracting the constant scalar fails.
pub(crate) fn compress_constant(
    source: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = source.validity()?;
    let mask = validity.execute_mask(source.len(), ctx)?;

    let Some(first_valid) = mask.first() else {
        return Ok(
            ConstantArray::new(Scalar::null(source.dtype().clone()), source.len()).into_array(),
        );
    };

    let scalar = source.execute_scalar(first_valid, ctx)?;
    let const_arr = ConstantArray::new(scalar, source.len()).into_array();

    if mask.all_true() {
        Ok(const_arr)
    } else {
        Ok(MaskedArray::try_new(const_arr, validity)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::Masked;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::CascadingCompressor;
    use crate::builtins::FloatDictScheme;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// Constant detection is built into the compressor, so it must work with no schemes at all.
    fn empty_compressor() -> CascadingCompressor {
        CascadingCompressor::new(Vec::new())
    }

    #[test]
    fn constant_int_compresses_without_schemes() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![7i64; 100], Validity::NonNullable).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn constant_int_with_nulls_compresses_to_masked_constant() -> VortexResult<()> {
        let validity =
            Validity::Array(BoolArray::from_iter((0..100).map(|i| i % 10 != 0)).into_array());
        let array = PrimitiveArray::new(buffer![7i64; 100], validity).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Masked>());
        Ok(())
    }

    #[rstest::rstest]
    #[case::no_schemes(false)]
    #[case::dictionary(true)]
    fn nullable_float_constant_uses_requested_distinctness(
        #[case] dictionary: bool,
    ) -> VortexResult<()> {
        let validity =
            Validity::Array(BoolArray::from_iter((0..100).map(|i| i % 10 != 0)).into_array());
        let array = PrimitiveArray::new(buffer![7.0f32; 100], validity).into_array();
        let compressor = CascadingCompressor::new(if dictionary {
            vec![&FloatDictScheme]
        } else {
            Vec::new()
        });
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = compressor.compress(&array, &mut ctx)?;
        assert_eq!(compressed.is::<Masked>(), dictionary);
        Ok(())
    }

    #[test]
    fn constant_string_compresses_without_schemes() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str(std::iter::repeat_n("hello", 100)).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn constant_bool_compresses_without_schemes() -> VortexResult<()> {
        let array = BoolArray::from_iter(std::iter::repeat_n(true, 100)).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn constant_decimal_compresses_without_schemes() -> VortexResult<()> {
        let array = DecimalArray::new(
            buffer![123_456i128; 100],
            DecimalDType::new(20, 2),
            Validity::NonNullable,
        )
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn constant_timestamp_compresses_without_schemes() -> VortexResult<()> {
        let ts = PrimitiveArray::from_iter(std::iter::repeat_n(1_704_067_200_000i64, 100));
        let array = TemporalArray::new_timestamp(ts.into_array(), TimeUnit::Milliseconds, None)
            .into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn non_constant_int_is_left_canonical_without_schemes() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(0..100i64).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(!compressed.is::<Constant>());
        assert_eq!(compressed.dtype(), array.dtype());
        Ok(())
    }
}

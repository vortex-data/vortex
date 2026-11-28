// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Property tests for ALP `compare` and `between` kernels at encoded integer boundaries.

use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use num_traits::Bounded;
use num_traits::One;
use num_traits::SaturatingAdd;
use num_traits::SaturatingSub;
use num_traits::Zero;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::patches::Patches;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::compare::CompareKernel;
use vortex_array::scalar_fn::builtin::operators::CompareMode;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_functions::between::BetweenKernel;
use vortex_functions::between::BetweenOptions;
use vortex_functions::between::BetweenReduce;
use vortex_functions::between::StrictComparison;
use vortex_functions::builtins::ArrayFunctions;
use vortex_session::VortexSession;

use crate::ALP;
use crate::ALPArray;
use crate::ALPArrayExt;
use crate::ALPFloat;
use crate::Exponents;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

const ALL_OPERATORS: [CompareOperator; 6] = [
    CompareOperator::Eq,
    CompareOperator::NotEq,
    CompareOperator::Lt,
    CompareOperator::Lte,
    CompareOperator::Gt,
    CompareOperator::Gte,
];

const ALL_BETWEEN_OPTIONS: [BetweenOptions; 4] = [
    BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::NonStrict,
    },
    BetweenOptions {
        lower_strict: StrictComparison::Strict,
        upper_strict: StrictComparison::NonStrict,
    },
    BetweenOptions {
        lower_strict: StrictComparison::NonStrict,
        upper_strict: StrictComparison::Strict,
    },
    BetweenOptions {
        lower_strict: StrictComparison::Strict,
        upper_strict: StrictComparison::Strict,
    },
];

trait BoundaryFloat:
    ALPFloat<
        ALPInt: NativePType + Bounded + One + Zero + SaturatingAdd + SaturatingSub + gs::Integer,
    > + NativePType
    + Into<PValue>
    + gs::Float
    + 'static
{
    /// Smallest integer above which adjacent integers begin rounding to the same float.
    const PRECISION_CLIFF: Self::ALPInt;

    const SPECIALS: &'static [Self];

    fn next_up(self) -> Self;
    fn next_down(self) -> Self;

    fn step_ulps(self, steps: i8) -> Self {
        let mut value = self;
        for _ in 0..steps.unsigned_abs() {
            value = if steps > 0 {
                value.next_up()
            } else {
                value.next_down()
            };
        }
        value
    }
}

impl BoundaryFloat for f32 {
    const PRECISION_CLIFF: i32 = 1 << 24;

    const SPECIALS: &'static [f32] = &[
        -0.0,
        0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::MIN,
        f32::MAX,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_0000), // negative quiet NaN
    ];

    fn next_up(self) -> Self {
        f32::next_up(self)
    }

    fn next_down(self) -> Self {
        f32::next_down(self)
    }
}

impl BoundaryFloat for f64 {
    const PRECISION_CLIFF: i64 = 1 << 53;

    const SPECIALS: &'static [f64] = &[
        -0.0,
        0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MIN,
        f64::MAX,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NAN,
        f64::from_bits(0xfff8_0000_0000_0000), // negative quiet NaN
    ];

    fn next_up(self) -> Self {
        f64::next_up(self)
    }

    fn next_down(self) -> Self {
        f64::next_down(self)
    }
}

/// Draws an integer biased toward ALP boundary regions: zero, integer limits, the float mantissa
/// precision cliff where several integers decode to one float, and arbitrary integers.
fn draw_anchor_int<F: BoundaryFloat>(tc: &TestCase) -> F::ALPInt {
    let min = F::ALPInt::min_value();
    let max = F::ALPInt::max_value();
    let zero = F::ALPInt::zero();
    let cliff = F::PRECISION_CLIFF;
    let neg_cliff = zero.saturating_sub(&cliff);

    match tc.draw(gs::integers::<u8>().max_value(7)) {
        0 => zero,
        1 => min,
        2 => max,
        3 => cliff,
        4 => neg_cliff,
        _ => tc.draw(gs::integers::<F::ALPInt>()),
    }
}

/// Draws a comparison cutoff relative to `anchor`, including 1-ULP neighbors, special floats
/// (`-0.0`, subnormals, `±inf`, `±NaN`), and unconstrained floats.
fn draw_boundary_float<F: BoundaryFloat>(tc: &TestCase, anchor: F) -> F {
    match tc.draw(gs::integers::<u8>().max_value(5)) {
        0..=3 => {
            let ulps = tc.draw(gs::integers::<i8>().min_value(-2).max_value(2));
            anchor.step_ulps(ulps)
        }
        4 => tc.draw(gs::sampled_from(F::SPECIALS)),
        _ => tc.draw(gs::floats::<F>()),
    }
}

/// Builds a small ALP array directly from encoded integers around `anchor` plus the integer
/// extremes, with optional nulls and patches, so tests hit boundaries that `alp_encode` alone
/// rarely emits.
fn draw_boundary_alp<F: BoundaryFloat>(
    tc: &TestCase,
    exponents: Exponents,
    anchor: F::ALPInt,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ALPArray> {
    let one = F::ALPInt::one();
    let two = one.saturating_add(&one);
    let ints = [
        anchor.saturating_sub(&two),
        anchor.saturating_sub(&one),
        anchor,
        anchor.saturating_add(&one),
        anchor.saturating_add(&two),
        F::ALPInt::min_value(),
        F::ALPInt::zero(),
        F::ALPInt::max_value(),
    ];
    let len = ints.len();

    let (validity, valid_mask) = if tc.draw(gs::booleans()) {
        let mask: Vec<bool> = tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len));
        (Validity::from_iter(mask.iter().copied()), Some(mask))
    } else {
        (Validity::NonNullable, None)
    };
    let encoded = PrimitiveArray::new(Buffer::copy_from(ints), validity).into_array();

    let patches = if tc.draw(gs::booleans()) {
        let decoded_anchor = F::decode_single(anchor, exponents);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for idx in 0..len {
            let is_valid = valid_mask.as_ref().is_none_or(|m| m[idx]);
            if is_valid && tc.draw(gs::booleans()) {
                indices.push(u32::try_from(idx)?);
                values.push(draw_boundary_float::<F>(tc, decoded_anchor));
            }
        }
        if indices.is_empty() {
            None
        } else {
            let patch_validity = if valid_mask.is_some() {
                Validity::AllValid
            } else {
                Validity::NonNullable
            };
            let patch_values =
                PrimitiveArray::new(Buffer::from(values), patch_validity).into_array();
            Some(Patches::new(
                len,
                0,
                Buffer::from(indices).into_array(),
                patch_values,
                None,
                ctx,
            )?)
        }
    } else {
        None
    };

    ALP::try_new(encoded, exponents, patches)
}

fn check_compare<F: BoundaryFloat>(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let exponents = Exponents {
        e: tc.draw(gs::integers::<u8>().max_value(F::MAX_EXPONENT)),
        f: tc.draw(gs::integers::<u8>().max_value(F::MAX_EXPONENT)),
    };
    let anchor = draw_anchor_int::<F>(tc);
    let alp = draw_boundary_alp::<F>(tc, exponents, anchor, &mut ctx)?;
    let canonical = alp
        .clone()
        .into_array()
        .execute::<PrimitiveArray>(&mut ctx)?;

    let cutoff = draw_boundary_float::<F>(tc, F::decode_single(anchor, exponents));
    let rhs_nullability = if tc.draw(gs::booleans()) {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let rhs =
        ConstantArray::new(Scalar::primitive(cutoff, rhs_nullability), alp.len()).into_array();

    for operator in ALL_OPERATORS {
        let actual = <ALP as CompareKernel>::compare(alp.as_view(), &rhs, operator, &mut ctx)?
            .vortex_expect("ALP compare kernel must engage");
        let expected = canonical.clone().into_array().compare(
            rhs.clone(),
            operator,
            CompareMode::TOTAL_ORDER,
        )?;
        assert_arrays_eq!(actual, expected, &mut ctx);
    }
    Ok(())
}

fn check_between<F: BoundaryFloat>(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let exponents = Exponents {
        e: tc.draw(gs::integers::<u8>().max_value(F::MAX_EXPONENT)),
        f: tc.draw(gs::integers::<u8>().max_value(F::MAX_EXPONENT)),
    };
    let anchor = draw_anchor_int::<F>(tc);
    let alp = draw_boundary_alp::<F>(tc, exponents, anchor, &mut ctx)?;
    let canonical = alp
        .clone()
        .into_array()
        .execute::<PrimitiveArray>(&mut ctx)?;

    let decoded_anchor = F::decode_single(anchor, exponents);
    let lower_val = draw_boundary_float::<F>(tc, decoded_anchor);
    let upper_anchor = if tc.draw(gs::booleans()) {
        decoded_anchor
    } else {
        F::decode_single(draw_anchor_int::<F>(tc), exponents)
    };
    let upper_val = draw_boundary_float::<F>(tc, upper_anchor);

    let lower_nullability = if tc.draw(gs::booleans()) {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let upper_nullability = if tc.draw(gs::booleans()) {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let lower =
        ConstantArray::new(Scalar::primitive(lower_val, lower_nullability), alp.len()).into_array();
    let upper =
        ConstantArray::new(Scalar::primitive(upper_val, upper_nullability), alp.len()).into_array();

    for options in ALL_BETWEEN_OPTIONS {
        let actual = if alp.patches().is_some() {
            <ALP as BetweenKernel>::between(alp.as_view(), &lower, &upper, &options, &mut ctx)?
                .vortex_expect("ALP between kernel must engage for patched arrays")
        } else {
            <ALP as BetweenReduce>::between(alp.as_view(), &lower, &upper, &options)?
                .vortex_expect("ALP between reduce must engage for unpatched arrays")
        };
        let expected =
            canonical
                .clone()
                .into_array()
                .between(lower.clone(), upper.clone(), options)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
    }
    Ok(())
}

/// ALP scalar comparison matches comparing the decoded primitive array across exponents, integer
/// limits, precision plateaus, ULP neighbors, non-finites, array nulls, and patches.
#[hegel::test(test_cases = 100, print_blob = true)]
fn compare_matches_canonical_at_boundaries(tc: TestCase) {
    if tc.draw(gs::booleans()) {
        check_compare::<f32>(&tc)
    } else {
        check_compare::<f64>(&tc)
    }
    .vortex_expect("ALP compare boundary property");
}

/// ALP between matches between over the decoded primitive array across exponents, integer limits,
/// precision plateaus, ULP neighbors, non-finites, array nulls, and patches.
#[hegel::test(test_cases = 100, print_blob = true)]
fn between_matches_canonical_at_boundaries(tc: TestCase) {
    if tc.draw(gs::booleans()) {
        check_between::<f32>(&tc)
    } else {
        check_between::<f64>(&tc)
    }
    .vortex_expect("ALP between boundary property");
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::expr;
pub use crate::expr::BetweenOptions;
pub use crate::expr::StrictComparison;

/// Compute between (a <= x <= b).
///
/// This is an optimized implementation that is equivalent to `(a <= x) AND (x <= b)`.
///
/// The `BetweenOptions` defines if the lower or upper bounds are strict (exclusive) or non-strict
/// (inclusive).
pub fn between(
    arr: &dyn Array,
    lower: &dyn Array,
    upper: &dyn Array,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef> {
    expr::between_canonical(
        arr,
        lower,
        upper,
        options,
        &mut ExecutionCtx::new(VortexSession::empty()),
    )
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_scalar::Scalar;

    use super::*;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::ConstantArray;
    use crate::compute::conformance::search_sorted::rstest;
    use crate::test_harness::to_int_indices;

    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict, vec![0, 1, 2, 3])]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict, vec![0, 1])]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict, vec![0, 2])]
    #[case(StrictComparison::Strict, StrictComparison::Strict, vec![0])]
    fn test_bounds(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
        #[case] expected: Vec<u64>,
    ) {
        let lower = buffer![0, 0, 0, 0, 2].into_array();
        let array = buffer![1, 0, 1, 0, 1].into_array();
        let upper = buffer![2, 1, 1, 0, 0].into_array();

        let matches = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict,
                upper_strict,
            },
        )
        .unwrap()
        .to_bool();

        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, expected);
    }

    #[test]
    fn test_constants() {
        let lower = buffer![0, 0, 2, 0, 2].into_array();
        let array = buffer![1, 0, 1, 0, 1].into_array();

        // upper is null
        let upper = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            5,
        );

        let matches = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
        .unwrap()
        .to_bool();

        let indices = to_int_indices(matches).unwrap();
        assert!(indices.is_empty());

        // upper is a fixed constant
        let upper = ConstantArray::new(Scalar::from(2), 5);
        let matches = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
        .unwrap()
        .to_bool();
        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, vec![0, 1, 3]);

        // lower is also a constant
        let lower = ConstantArray::new(Scalar::from(0), 5);

        let matches = between(
            array.as_ref(),
            lower.as_ref(),
            upper.as_ref(),
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        )
        .unwrap()
        .to_bool();
        let indices = to_int_indices(matches).unwrap();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }
}

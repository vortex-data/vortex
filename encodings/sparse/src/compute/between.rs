// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_error::VortexResult;

use crate::Sparse;
use crate::SparseExt as _;

/// Sparse-specific between kernel.
///
/// `lower <= x <= upper` (with per-bound strictness) over a Sparse column with constant
/// bounds is itself sparse: every unpatched position resolves to `between(F, lo, hi)` and
/// every patched position to `between(patch, lo, hi)`. We push the range check into the
/// patches and rebuild a `Sparse<Bool>` with the new fill, preserving downstream sparsity.
///
/// Declines (falls back to canonical) unless both bounds are constants.
impl BetweenKernel for Sparse {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lo), Some(hi)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        let patches = array.patches();

        let fill_bool = ConstantArray::new(array.fill_scalar().clone(), 1)
            .into_array()
            .between(
                ConstantArray::new(lo.clone(), 1).into_array(),
                ConstantArray::new(hi.clone(), 1).into_array(),
                options.clone(),
            )?
            .execute_scalar(0, ctx)?;

        let new_patches = patches.map_values(|values| {
            let len = values.len();
            values.between(
                ConstantArray::new(lo.clone(), len).into_array(),
                ConstantArray::new(hi.clone(), len).into_array(),
                options.clone(),
            )
        })?;

        Ok(Some(
            Sparse::try_new_from_patches(new_patches, fill_bool)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::Sparse;
    use crate::initialize;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let mut builder = vortex_array::default_session_builder();
        initialize(&mut builder);
        builder.build()
    });

    #[rstest]
    #[case(0i32, 100i32, StrictComparison::NonStrict, StrictComparison::NonStrict)]
    #[case(5i32, 25i32, StrictComparison::Strict, StrictComparison::Strict)]
    #[case(1i32, 20i32, StrictComparison::NonStrict, StrictComparison::Strict)]
    fn between_matches_canonical(
        #[case] lo: i32,
        #[case] hi: i32,
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
    ) {
        let array = Sparse::try_new(
            buffer![1u64, 3, 5].into_array(),
            buffer![10i32, 20, 30].into_array(),
            8,
            Scalar::from(1i32),
        )
        .unwrap()
        .into_array();
        let len = array.len();
        let options = BetweenOptions {
            lower_strict,
            upper_strict,
        };

        let lower = ConstantArray::new(Scalar::from(lo), len).into_array();
        let upper = ConstantArray::new(Scalar::from(hi), len).into_array();

        let mut ctx = SESSION.create_execution_ctx();

        // Kernel path: between pushes through the Sparse encoding.
        let kernel = array
            .clone()
            .between(lower.clone(), upper.clone(), options.clone())
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap();

        // Baseline: canonicalize the input first so between runs on a PrimitiveArray.
        let canonical_input = array.execute::<Canonical>(&mut ctx).unwrap().into_array();
        let baseline = canonical_input
            .between(lower, upper, options)
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap();

        assert_arrays_eq!(kernel, baseline, &mut ctx);
    }
}

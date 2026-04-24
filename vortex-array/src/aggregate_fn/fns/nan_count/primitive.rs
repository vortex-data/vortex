// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::match_each_float_ptype;

pub(super) fn accumulate_primitive(
    count: &mut u64,
    p: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match_each_float_ptype!(p.ptype(), |F| {
        *count += compute_nan_count_with_validity(
            p.as_slice::<F>(),
            p.as_ref().validity()?.execute_mask(p.as_ref().len(), ctx)?,
        ) as u64;
    });
    Ok(())
}

fn compute_nan_count_with_validity<T: NativePType>(values: &[T], validity: Mask) -> usize {
    match validity {
        Mask::AllTrue(_) => values.iter().filter(|v| v.is_nan()).count(),
        Mask::AllFalse(_) => 0,
        Mask::Values(v) => values
            .iter()
            .zip(v.bit_buffer().iter())
            .filter_map(|(v, m)| m.then_some(v))
            .filter(|v| v.is_nan())
            .count(),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::nan_count::nan_count;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[test]
    fn primitive_nan_count() -> VortexResult<()> {
        let p = PrimitiveArray::new(
            buffer![
                -f32::NAN,
                f32::NAN,
                0.1,
                1.1,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY
            ],
            Validity::NonNullable,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(nan_count(&p.into_array(), &mut ctx)?, 2);
        Ok(())
    }

    #[test]
    fn primitive_nan_count_with_nulls() -> VortexResult<()> {
        let p = PrimitiveArray::from_option_iter([Some(f64::NAN), None, Some(f64::NAN), Some(1.0)]);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(nan_count(&p.into_array(), &mut ctx)?, 2);
        Ok(())
    }

    #[test]
    fn primitive_nan_count_all_valid_no_nans() -> VortexResult<()> {
        let p = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(nan_count(&p.into_array(), &mut ctx)?, 0);
        Ok(())
    }
}

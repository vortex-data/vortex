// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::vtable::VTable;

/// An optimizer rule that tries to reduce/replace a parent array where the implementer is a
/// child array in the `CHILD_IDX` position of the parent array.
pub trait ReduceParent<Parent: VTable, const CHILD_IDX: usize>: VTable {
    /// Try to reduce/replace the given parent array based on this child array.
    ///
    /// If no reduction is possible, return None.
    fn reduce_parent(array: &Self::Array, parent: &Parent::Array)
    -> VortexResult<Option<ArrayRef>>;
}

/// A generic optimizer rule that can be applied to an array to try to optimize it.
pub trait OptimizerRule {
    /// Try to optimize the given array, returning a replacement if successful.
    ///
    /// If no optimization is possible, return None.
    fn optimize(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{bitbuffer, buffer};
    use vortex_dtype::PTypeDowncast;
    use vortex_vector::VectorOps;

    use crate::arrays::{BoolArray, MaskedArray, PrimitiveArray};
    use crate::expr::session::ExprSession;
    use crate::expr::transform::ExprOptimizer;
    use crate::validity::Validity;
    use crate::{ArraySession, IntoArray};

    #[test]
    fn test_masked_pushdown() {
        let array = PrimitiveArray::from_iter([0u32, 1, 2, 3]);
        assert!(!array.dtype().is_nullable());

        let masked = MaskedArray::try_new(
            array.into_array(),
            Validity::Array(BoolArray::from(bitbuffer![0 1 0 1]).into_array()),
        )
        .unwrap();

        let masked_dtype = masked.dtype().clone();

        // Use the new ArrayOptimizer via ArraySession
        let array_session = ArraySession::default();
        let expr_session = ExprSession::default();
        let expr_optimizer = ExprOptimizer::new(&expr_session);
        let optimizer = array_session.optimizer(expr_optimizer);

        let result = optimizer.optimize_array(masked.into_array()).unwrap();
        assert_eq!(&masked_dtype, result.dtype());
        assert!(result.dtype().is_nullable());

        let vector = result.execute().unwrap().into_primitive().into_u32();
        assert_eq!(vector.elements(), &buffer![0, 1, 2, 3]);
        assert_eq!(vector.validity().to_bit_buffer(), bitbuffer![0 1 0 1]);
    }
}

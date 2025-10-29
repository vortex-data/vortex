// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::vtable::VTable;
use crate::{Array, ArrayRef, ArrayVisitor};

pub trait ArrayOptimizeExt {
    /// Optimize the entire tree in a single bottom-up pass
    fn optimize(&self) -> VortexResult<ArrayRef>;
}

impl ArrayOptimizeExt for ArrayRef {
    fn optimize(&self) -> VortexResult<ArrayRef> {
        let children = self.children();

        let mut new_children = Vec::with_capacity(children.len());
        let mut children_modified = false;
        for (idx, child) in children.iter().enumerate() {
            let child = child.optimize()?;

            // Check if the child can reduce us (its parent), and if so bail early.
            if let Some(reduced) = child.reduce_parent(self, idx)? {
                return Ok(reduced);
            }

            if !Arc::ptr_eq(&child, &children[idx]) {
                children_modified = true;
            }
            new_children.push(child);
        }

        if children_modified {
            return self.with_children(&new_children);
        }

        Ok(self.to_array())
    }
}

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
    use crate::optimizer::ArrayOptimizeExt;
    use crate::validity::Validity;
    use crate::{ArrayOperator, IntoArray};

    #[test]
    fn test_masked_pushdown() {
        let array = PrimitiveArray::from_iter([0u32, 1, 2, 3]);
        assert!(!array.dtype().is_nullable());

        let masked = MaskedArray::try_new(
            array.into_array(),
            Validity::Array(BoolArray::from(bitbuffer![0 1 0 1]).into_array()),
        )
        .unwrap()
        .into_array();

        let result = masked.optimize().unwrap();
        assert_eq!(masked.dtype(), result.dtype());
        assert!(result.dtype().is_nullable());

        let vector = result.execute().unwrap().into_primitive().into_u32();
        assert_eq!(vector.elements(), &buffer![0, 1, 2, 3]);
        assert_eq!(vector.validity().to_bit_buffer(), bitbuffer![0 1 0 1]);
    }
}

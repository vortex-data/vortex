// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorOps;

use crate::Array;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::executor::CanonicalOutput;
use crate::executor::VectorExecutor;
use crate::expr::ExecutionArgs;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ScalarFnVTable> for ScalarFnVTable {
    fn is_valid(array: &ScalarFnArray, index: usize) -> bool {
        // inlined to remove a cycle `is_valid()` and `scalar_at()`
        assert!(index < array.len(), "index {index} out of bounds");
        let input_datums: Vec<_> = array
            .children()
            .iter()
            .map(|c| c.scalar_at(index))
            .map(|scalar| Datum::from(scalar.to_vector_scalar()))
            .collect();

        let ctx = ExecutionArgs {
            datums: input_datums,
            dtypes: array.children().iter().map(|c| c.dtype().clone()).collect(),
            row_count: 1,
            return_dtype: array.dtype.clone(),
        };

        let result = array
            .scalar_fn
            .execute(ctx)
            .vortex_expect("Scalar function execution should be fallible")
            .into_scalar()
            .vortex_expect("Scalar function execution should return scalar");

        result.is_valid()
    }

    fn all_valid(array: &ScalarFnArray) -> bool {
        match array.scalar_fn.signature().is_null_sensitive() {
            true => {
                // If the function is null sensitive, we cannot guarantee all valid without evaluating
                // the function
                false
            }
            false => {
                // If the function is not null sensitive, we can guarantee all valid if all children
                // are all valid
                array.children().iter().all(|child| child.all_valid())
            }
        }
    }

    fn all_invalid(array: &ScalarFnArray) -> bool {
        match array.scalar_fn.signature().is_null_sensitive() {
            true => {
                // If the function is null sensitive, we cannot guarantee all invalid without evaluating
                // the function
                false
            }
            false => {
                // If the function is not null sensitive, we can guarantee all invalid if any child
                // is all invalid
                array.children().iter().any(|child| child.all_invalid())
            }
        }
    }

    fn validity(array: &ScalarFnArray) -> VortexResult<Validity> {
        // TODO(ngates): we should add an execute_validity function to ScalarFn.
        //  Or have more descriptive null sensitivity metadata.
        Ok(Validity::from_mask(
            Self::validity_mask(array),
            Nullability::Nullable,
        ))
    }

    fn validity_mask(array: &ScalarFnArray) -> Mask {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let output = array
            .to_array()
            .execute_output(&mut ctx)
            .vortex_expect("Validity mask computation should be fallible");
        match output {
            CanonicalOutput::Constant(c) => Mask::new(array.len, c.scalar().is_valid()),
            CanonicalOutput::Array(a) => a
                .to_vector(&mut ctx)
                .vortex_expect("Failed to convert canonical to vector")
                .validity()
                .clone(),
        }
    }
}

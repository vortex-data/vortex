// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::executor::CanonicalOutput;
use crate::expr::ExecutionArgs;
use crate::expr::Expression;
use crate::expr::lit;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ScalarFnVTable> for ScalarFnVTable {
    fn is_valid(array: &ScalarFnArray, index: usize) -> bool {
        // inlined to remove a cycle `is_valid()` and `scalar_at()`
        assert!(index < array.len(), "index {index} out of bounds");
        let inputs: Arc<[_]> = array
            .children
            .iter()
            .map(|child| lit(child.scalar_at(index)))
            .collect::<_>();

        let result = array
            .scalar_fn
            .evaluate(
                &Expression::try_new(array.scalar_fn.clone(), inputs)
                    .vortex_expect("create expr must not fail"),
                &array.to_array(),
            )
            .vortex_expect("execute cannot fail");

        let result = result.as_constant().unwrap_or_else(|| {
            tracing::info!(
                "Scalar function {} returned non-constant array from execution over all scalar inputs",
                array.scalar_fn,
            );
            result.scalar_at(0)
        });

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
            .execute::<CanonicalOutput>(&mut ctx)
            .vortex_expect("Validity mask computation should be fallible");
        match output {
            CanonicalOutput::Constant(c) => Mask::new(array.len, c.scalar().is_valid()),
            CanonicalOutput::Array(a) => a
                .into_array()
                .validity()
                .vortex_expect("cannot fail")
                .to_mask(array.len()),
        }
    }
}

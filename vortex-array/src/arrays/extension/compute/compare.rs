// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

impl CompareKernel for ExtensionVTable {
    fn compare(
        lhs: &ExtensionArray,
        rhs: &ArrayRef,
        operator: CompareOperator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the RHS is a constant, we can extract the storage scalar.
        if let Some(const_ext) = rhs.as_constant() {
            let storage_scalar = const_ext.as_extension().to_storage_scalar();
            return lhs
                .storage()
                .to_array()
                .binary(
                    ConstantArray::new(storage_scalar, lhs.len()).to_array(),
                    Operator::from(operator),
                )
                .map(Some);
        }

        // If the RHS is an extension array matching ours, we can extract the storage.
        if let Some(rhs_ext) = rhs.as_opt::<ExtensionVTable>() {
            return lhs
                .storage()
                .to_array()
                .binary(rhs_ext.storage().to_array(), Operator::from(operator))
                .map(Some);
        }

        // Otherwise, we need the RHS to handle this comparison.
        Ok(None)
    }
}

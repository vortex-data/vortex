// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::struct_::compute::rules::struct_get_item;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::pack::Pack;
use crate::validity::Validity;

impl ArrayRef {
    /// Apply a bound expression to this array, producing a new array in constant time.
    pub fn apply_bound(self, expr: &BoundExpression) -> VortexResult<ArrayRef> {
        let BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } = expr
        else {
            return Ok(self);
        };

        if let Some(scalar) = scalar_fn.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        let children: Vec<_> = children
            .iter()
            .map(|child| self.clone().apply_bound(child))
            .try_collect()?;

        if let Some(field_name) = scalar_fn.as_opt::<GetItem>()
            && let [child] = children.as_slice()
            && let Some(array) = child.as_opt::<Struct>()
        {
            return struct_get_item(array, field_name);
        }

        if let Some(pack) = scalar_fn.as_opt::<Pack>() {
            let validity = match pack.nullability {
                crate::dtype::Nullability::NonNullable => Validity::NonNullable,
                crate::dtype::Nullability::Nullable => Validity::AllValid,
            };
            return Ok(
                StructArray::try_new(pack.names.clone(), children, self.len(), validity)?
                    .into_array(),
            );
        }

        let array =
            ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, self.len())?.into_array();

        array.optimize()
    }

    /// Apply the expression to this array, producing a new array in constant time.
    pub fn apply(self, expr: &Expression) -> VortexResult<ArrayRef> {
        // If the expression is a root, return self.
        if expr.is_root() {
            return Ok(self);
        }

        // Manually convert literals to ConstantArray.
        if let Some(scalar) = expr.as_opt::<Literal>() {
            return Ok(ConstantArray::new(scalar.clone(), self.len()).into_array());
        }

        // Otherwise, collect the child arrays.
        let children: Vec<_> = expr
            .children()
            .iter()
            .map(|e| self.clone().apply(e))
            .try_collect()?;

        // And wrap the scalar function up in an array.
        let scalar_fn = expr
            .as_scalar()
            .vortex_expect("root and literal were handled above, so this is a scalar node");
        let array =
            ScalarFnArray::try_new_with_len(scalar_fn.clone(), children, self.len())?.into_array();

        // Optimize the resulting array's root.
        array.optimize()
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::AnyScalarFn;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ConstantVTable;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::expr::Binary;
use vortex_array::expr::Operator;
use vortex_array::matchers::Exact;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_scalar::PrimitiveScalar;
use vortex_scalar::Scalar;

use crate::FoRArray;
use crate::FoRVTable;

pub(super) const PARENT_RULES: ParentRuleSet<FoRVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FoRFilterPushDownRule),
    ParentRuleSet::lift(&FoRBinaryComparisonPushDownRule),
]);

#[derive(Debug)]
struct FoRFilterPushDownRule;

impl ArrayParentReduceRule<FoRVTable> for FoRFilterPushDownRule {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn reduce_parent(
        &self,
        child: &FoRArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_array = unsafe {
            FoRArray::new_unchecked(
                child.encoded.filter(parent.filter_mask().clone())?,
                child.reference.clone(),
            )
        };
        Ok(Some(new_array.into_array()))
    }
}

/// Push down binary comparison operations through FoR encoding.
///
/// For FoR encoding where `value = encoded + reference`, a comparison like
/// `value <= constant` can be transformed to `encoded <= (constant - reference)`.
#[derive(Debug)]
struct FoRBinaryComparisonPushDownRule;

impl ArrayParentReduceRule<FoRVTable> for FoRBinaryComparisonPushDownRule {
    type Parent = AnyScalarFn;

    fn parent(&self) -> Self::Parent {
        AnyScalarFn
    }

    #[allow(clippy::cognitive_complexity)]
    fn reduce_parent(
        &self,
        child: &FoRArray,
        parent: &ScalarFnArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle Binary scalar functions with comparison operators
        let Some(op) = parent.scalar_fn().as_opt::<Binary>() else {
            return Ok(None);
        };
        if op.maybe_cmp_operator().is_none() {
            return Ok(None);
        }

        // Only handle case where sibling is a constant
        let sibling = &parent.children()[1 - child_idx];
        let Some(constant_array) = sibling.as_opt::<ConstantVTable>() else {
            return Ok(None);
        };

        // Try to transform the constant
        let transformed = transform_constant_for_comparison(
            constant_array.scalar(),
            child.reference_scalar(),
            *op,
            child.len(),
        )?;

        let Some(transformed_constant) = transformed else {
            return Ok(None);
        };

        // Build the new ScalarFnArray with encoded values and transformed constant
        let new_children =
            build_new_children(child, &transformed_constant, parent.len(), child_idx);

        Ok(Some(
            ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children, parent.len())?
                .into_array(),
        ))
    }
}

/// Transform a constant for FoR comparison by subtracting the reference.
/// Returns the transformed constant as a ConstantArray, or None if transformation isn't possible.
fn transform_constant_for_comparison(
    constant_scalar: &Scalar,
    reference_scalar: &Scalar,
    _op: Operator,
    len: usize,
) -> VortexResult<Option<ArrayRef>> {
    let Ok(constant_prim) = PrimitiveScalar::try_from(constant_scalar) else {
        return Ok(None);
    };
    let Ok(reference_prim) = PrimitiveScalar::try_from(reference_scalar) else {
        return Ok(None);
    };

    if constant_prim.ptype() != reference_prim.ptype() {
        return Ok(None);
    }

    let nullability = constant_scalar.dtype().nullability();

    match_each_integer_ptype!(constant_prim.ptype(), |T| {
        let constant_val: T = constant_prim
            .typed_value::<T>()
            .ok_or_else(|| vortex_error::vortex_err!("Null constant not supported"))?;
        let reference_val: T = reference_prim
            .typed_value::<T>()
            .ok_or_else(|| vortex_error::vortex_err!("Null reference not supported"))?;

        // When constant < reference, the wrapping subtraction would give incorrect ordering
        // semantics, so we fall back to the default comparison path
        if constant_val < reference_val {
            return Ok(None);
        }

        let transformed: T = constant_val.wrapping_sub(reference_val);
        let scalar = Scalar::primitive(transformed, nullability);
        Ok(Some(ConstantArray::new(scalar, len).into_array()))
    })
}

/// Build the new children array for the transformed comparison.
fn build_new_children(
    child: &FoRArray,
    transformed_constant: &ArrayRef,
    _len: usize,
    child_idx: usize,
) -> Vec<ArrayRef> {
    if child_idx == 0 {
        vec![child.encoded().clone(), transformed_constant.clone()]
    } else {
        vec![transformed_constant.clone(), child.encoded().clone()]
    }
}

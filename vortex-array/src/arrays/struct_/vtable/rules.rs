// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExactScalarFn;
use crate::arrays::ScalarFnArrayExt;
use crate::arrays::ScalarFnArrayView;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::builtins::ArrayBuiltins;
use crate::expr::Cast;
use crate::expr::EmptyOptions;
use crate::expr::GetItem;
use crate::expr::Mask;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

pub(super) const PARENT_RULES: ParentRuleSet<StructVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&StructCastPushDownRule),
    ParentRuleSet::lift(&StructGetItemRule),
]);

/// Rule to push down cast into struct fields
#[derive(Debug)]
struct StructCastPushDownRule;
impl ArrayParentReduceRule<StructVTable> for StructCastPushDownRule {
    type Parent = ExactScalarFn<Cast>;

    fn parent(&self) -> Self::Parent {
        ExactScalarFn::from(&Cast)
    }

    fn reduce_parent(
        &self,
        array: &StructArray,
        parent: ScalarFnArrayView<Cast>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let target_fields = parent.options.as_struct_fields();

        let mut new_fields = Vec::with_capacity(target_fields.nfields());
        for (field_array, field_dtype) in array.fields.iter().zip(target_fields.fields()) {
            new_fields.push(field_array.cast(field_dtype)?)
        }

        let new_struct = unsafe {
            StructArray::new_unchecked(
                new_fields,
                target_fields.clone(),
                array.len(),
                array.validity().clone(),
            )
        };

        Ok(Some(new_struct.into_array()))
    }
}

/// Rule to flatten get_item from struct by field name
#[derive(Debug)]
pub(crate) struct StructGetItemRule;
impl ArrayParentReduceRule<StructVTable> for StructGetItemRule {
    type Parent = ExactScalarFn<GetItem>;

    fn parent(&self) -> ExactScalarFn<GetItem> {
        ExactScalarFn::from(&GetItem)
    }

    fn reduce_parent(
        &self,
        child: &StructArray,
        parent: ScalarFnArrayView<'_, GetItem>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let field_name = parent.options;
        let Some(field) = child.field_by_name_opt(field_name) else {
            return Ok(None);
        };

        match child.validity() {
            Validity::NonNullable | Validity::AllValid => {
                // If the struct is non-nullable or all valid, the field's validity is unchanged
                Ok(Some(field.clone()))
            }
            Validity::AllInvalid => {
                // If everything is invalid, the field is also all invalid
                Ok(Some(
                    ConstantArray::new(
                        vortex_scalar::Scalar::null(field.dtype().clone()),
                        field.len(),
                    )
                    .into_array(),
                ))
            }
            Validity::Array(mask) => {
                // If the validity is an array, we need to combine it with the field's validity
                Mask.try_new_array(field.len(), EmptyOptions, [field.clone(), mask.clone()])
                    .map(Some)
            }
        }
    }
}

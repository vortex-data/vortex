// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

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

/// Rule to push down cast into struct fields.
///
/// TODO(joe/rob): should be have this in casts.
///
/// This rule supports schema evolution by allowing new nullable fields to be added
/// at the end of the struct, filled with null values.
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
        let source_field_count = array.fields.len();
        let target_field_count = target_fields.nfields();

        // Target must have at least as many fields as source
        vortex_ensure!(
            target_field_count >= source_field_count,
            "Cannot cast struct: target has fewer fields ({}) than source ({})",
            target_field_count,
            source_field_count
        );

        let mut new_fields = Vec::with_capacity(target_field_count);

        // Cast existing source fields to target types
        for (field_array, field_dtype) in array
            .fields
            .iter()
            .zip(target_fields.fields().take(source_field_count))
        {
            new_fields.push(field_array.cast(field_dtype)?);
        }

        // Add null arrays for any extra target fields (schema evolution)
        for field_dtype in target_fields.fields().skip(source_field_count) {
            vortex_ensure!(
                field_dtype.is_nullable(),
                "Cannot add non-nullable field during struct cast (schema evolution only supports nullable fields)"
            );
            new_fields.push(
                ConstantArray::new(vortex_scalar::Scalar::null(field_dtype), array.len())
                    .into_array(),
            );
        }

        let validity = if parent.options.is_nullable() {
            array.validity().clone().into_nullable()
        } else {
            array
                .validity()
                .clone()
                .into_non_nullable(array.len)
                .ok_or_else(|| vortex_err!("Failed to cast nullable struct to non-nullable"))?
        };

        let new_struct = unsafe {
            StructArray::new_unchecked(new_fields, target_fields.clone(), array.len(), validity)
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

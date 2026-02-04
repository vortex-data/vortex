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

    fn reduce_parent(
        &self,
        array: &StructArray,
        parent: ScalarFnArrayView<Cast>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let target_fields = parent.options.as_struct_fields();
        let mut new_fields = Vec::with_capacity(target_fields.nfields());

        for (target_name, target_dtype) in target_fields.names().iter().zip(target_fields.fields())
        {
            match array.unmasked_field_by_name(target_name).ok() {
                Some(field) => {
                    new_fields.push(field.cast(target_dtype)?);
                }
                None => {
                    // Not found - create NULL array (schema evolution)
                    vortex_ensure!(
                        target_dtype.is_nullable(),
                        "Cannot add non-nullable field '{}' during struct cast",
                        target_name
                    );
                    new_fields.push(
                        ConstantArray::new(vortex_scalar::Scalar::null(target_dtype), array.len())
                            .into_array(),
                    );
                }
            }
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

    fn reduce_parent(
        &self,
        child: &StructArray,
        parent: ScalarFnArrayView<'_, GetItem>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let field_name = parent.options;
        let Some(field) = child.unmasked_field_by_name_opt(field_name) else {
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

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::FieldNames;
    use vortex_dtype::Nullability;
    use vortex_dtype::StructFields;
    use vortex_scalar::Scalar;

    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::canonical::ToCanonical;
    use crate::validity::Validity;

    #[test]
    fn test_struct_cast_field_reorder() {
        // Source: {a, b}, Target: {c, b, a} - reordered + new null field
        let source = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![
                VarBinViewArray::from_iter_str(["A"]).into_array(),
                VarBinViewArray::from_iter_str(["B"]).into_array(),
            ],
            1,
            Validity::NonNullable,
        )
        .unwrap();

        let utf8_null = DType::Utf8(Nullability::Nullable);
        let target = DType::Struct(
            StructFields::new(
                FieldNames::from(["c", "b", "a"]),
                vec![utf8_null.clone(); 3],
            ),
            Nullability::NonNullable,
        );

        // Use ArrayBuiltins::cast which goes through the optimizer and applies StructCastPushDownRule
        let result = source.into_array().cast(target).unwrap().to_struct();
        assert_arrays_eq!(
            result.unmasked_field_by_name("a").unwrap(),
            VarBinViewArray::from_iter_nullable_str([Some("A")])
        );
        assert_arrays_eq!(
            result.unmasked_field_by_name("b").unwrap(),
            VarBinViewArray::from_iter_nullable_str([Some("B")])
        );
        assert_arrays_eq!(
            result.unmasked_field_by_name("c").unwrap(),
            ConstantArray::new(Scalar::null(utf8_null), 1)
        );
    }
}

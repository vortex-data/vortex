// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExactScalarFn;
use crate::arrays::ExpressionArray;
use crate::arrays::ExpressionVTable;
use crate::arrays::ScalarFnArrayExt;
use crate::arrays::ScalarFnArrayView;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::builtins::ArrayBuiltins;
use crate::builtins::ExprBuiltins;
use crate::expr::Cast;
use crate::expr::EmptyOptions;
use crate::expr::GetItem;
use crate::expr::Mask;
use crate::expr::Pack;
use crate::expr::Root;
use crate::expr::annotate_scope_access;
use crate::expr::root;
use crate::expr::transform::partition;
use crate::expr::transform::replace;
use crate::matchers::Exact;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

pub(super) const PARENT_RULES: ParentRuleSet<StructVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&StructCastPushDownRule),
    ParentRuleSet::lift(&StructGetItemRule),
    ParentRuleSet::lift(&StructPartitionRule),
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

/// Rule to partition a parent expression over the fields of a StructArray.
#[derive(Debug)]
pub struct StructPartitionRule;
impl ArrayParentReduceRule<StructVTable> for StructPartitionRule {
    type Parent = Exact<ExpressionVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn reduce_parent(
        &self,
        array: &StructArray,
        parent: &ExpressionArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the parent expression is a get_item over root, we can directly return the field.
        if let Some(field_name) = parent.expression().as_opt::<GetItem>() {
            if parent.expression().child(0).is::<Root>() {
                let field_idx = array.struct_fields().find(field_name).ok_or_else(|| {
                    vortex_err!(
                        "Field '{}' not found in struct for direct get_item",
                        field_name
                    )
                })?;
                return Ok(Some(array.masked_field(field_idx)?));
            }
        }

        // Partition the expression into field accesses.
        let partitioned = partition(
            parent.expression().clone(),
            array.dtype(),
            annotate_scope_access(array.struct_fields()),
        )?;

        // If there's only one partition, we can return that field directly.
        if partitioned.partitions.len() == 1 {
            let field_name = &partitioned.partition_annotations[0];
            let field_idx = array.struct_fields().find(field_name).ok_or_else(|| {
                vortex_err!(
                    "Field '{}' not found in struct for partitioning",
                    field_name
                )
            })?;
            let field = array.masked_field(field_idx)?;

            // Replace the field access in the expression with root.
            let field_expr = replace(
                parent.expression().clone(),
                &root().get_item(field_name.clone())?,
                root(),
            );

            // Wrap up the field array in an ExpressionArray.
            return Ok(Some(
                ExpressionArray::try_new(field_expr, field)?.into_array(),
            ));
        }

        // FIXME(ngates): we need a better way to avoid terminating here..
        //  For now, the partitioner returns Pack expressions for direct field accesses.
        if partitioned.partitions.iter().all(|p| p.is::<Pack>()) {
            return Ok(None);
        }

        // Otherwise, we need to handle multiple partitions.
        let mut partitions = Vec::with_capacity(partitioned.partitions.len());
        for (idx, partition) in partitioned.partitions.iter().enumerate() {
            let part_field_name = &partitioned.partition_annotations[idx];
            let part_field_idx = array.struct_fields().find(part_field_name).ok_or_else(|| {
                vortex_err!(
                    "Field '{}' not found in struct for partitioning",
                    part_field_name
                )
            })?;
            let part_field = array.masked_field(part_field_idx)?;

            let part_expr = replace(
                partition.clone(),
                &root().get_item(partitioned.partition_names[idx].clone())?,
                root(),
            );

            partitions.push(ExpressionArray::try_new(part_expr, part_field)?.into_array());
        }

        // Combine the partitioned expressions back into a single expression.
        let combined = ExpressionArray::try_new(
            partitioned.root,
            StructArray::try_new(
                partitioned.partition_names.clone(),
                partitions,
                array.len,
                Validity::NonNullable,
            )?
            .into_array(),
        )?
        .into_array();

        Ok(Some(combined))
    }
}

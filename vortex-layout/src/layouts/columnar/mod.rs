// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared column storage for layouts with row-aligned named fields and optional validity.

mod reader;

use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

pub(super) use reader::ColumnarReader;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutRef;
use crate::lowering::slots_from_layout;
use crate::plan::physical::PackPlan;
use crate::plan::physical::PlanRef;

/// Both layouts use the pack slot convention: validity at zero, then one slot per field.
/// Only present slots have a child on disk.
pub(super) struct Columns<'a> {
    fields: &'a StructFields,
    nullability: Nullability,
}

/// Only the children declared by the layout participate in validation.
pub(super) fn validate_child_rows(
    args: &LayoutDeserializeArgs<'_>,
    count: usize,
) -> VortexResult<()> {
    for idx in 0..count {
        vortex_ensure_eq!(
            args.children.child_row_count(idx),
            args.row_count,
            "Column {idx} row count does not match parent"
        );
    }
    Ok(())
}

impl<'a> Columns<'a> {
    pub(super) fn of(storage: &'a DType) -> Self {
        Self {
            fields: storage.as_struct_fields(),
            nullability: storage.nullability(),
        }
    }

    pub(super) fn nslots(&self) -> usize {
        1 + self.fields.nfields()
    }

    pub(super) fn slot_to_child(&self, slot: usize) -> Option<usize> {
        match slot {
            0 => self.nullability.is_nullable().then_some(0),
            slot if slot < self.nslots() => {
                Some(slot - 1 + usize::from(self.nullability.is_nullable()))
            }
            _ => None,
        }
    }

    pub(super) fn child_dtype(&self, slot: usize) -> VortexResult<DType> {
        if slot == 0 {
            Ok(DType::Bool(Nullability::NonNullable))
        } else {
            self.fields
                .field_by_index(slot - 1)
                .ok_or_else(|| vortex_err!("Missing field {}", slot - 1))
        }
    }

    pub(super) fn child_type(&self, slot: usize) -> LayoutChildType {
        if slot == 0 {
            LayoutChildType::Auxiliary("validity".into())
        } else {
            LayoutChildType::Field(
                self.fields
                    .field_name(slot - 1)
                    .vortex_expect("field slot out of bounds")
                    .clone(),
            )
        }
    }

    pub(super) fn pack_plan(&self, layout: &LayoutRef) -> PlanRef {
        // SAFETY: both layout vtables use these columns for slot presence and dtypes and
        // validate the row counts of their required children on deserialization.
        unsafe {
            PackPlan::from_slots_unchecked(
                self.fields.clone(),
                self.nullability,
                layout.row_count(),
                slots_from_layout(layout),
            )
        }
        .into_plan()
    }

    pub(super) fn matching_fields(
        &self,
        field_mask: &[FieldMask],
        mut per_child: impl FnMut(FieldMask, usize) -> VortexResult<()>,
    ) -> VortexResult<()> {
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for idx in 0..self.fields.nfields() {
                per_child(FieldMask::All, idx)?;
            }
            return Ok(());
        }

        for path in field_mask {
            let Some(field) = path.starting_field()? else {
                continue;
            };
            let Field::Name(field_name) = field else {
                vortex_bail!("Expected field name, got {field:?}");
            };
            let idx = self
                .fields
                .find(field_name)
                .ok_or_else(|| vortex_err!("Field not found: {field_name}"))?;
            per_child(path.clone().step_into()?, idx)?;
        }
        Ok(())
    }
}

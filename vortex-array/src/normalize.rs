// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::Id;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayRef;
use crate::ExecutionCtx;

/// Options for normalizing an array.
pub struct NormalizeOptions<'a> {
    /// The set of allowed array encodings (in addition to the canonical ones) that are permitted
    /// in the normalized array.
    pub allowed: &'a HashSet<Id>,
    /// The operation to perform when a non-allowed encoding is encountered.
    pub operation: Operation<'a>,
}

/// The operation to perform when a non-allowed encoding is encountered.
pub enum Operation<'a> {
    Error,
    Execute(&'a mut ExecutionCtx),
}

impl ArrayRef {
    /// Normalize the array according to given options.
    ///
    /// This operation performs a recursive traversal of the array. Any non-allowed encoding is
    /// normalized per the configured operation.
    pub fn normalize(self, options: &mut NormalizeOptions) -> VortexResult<ArrayRef> {
        match &mut options.operation {
            Operation::Error => {
                self.normalize_with_error(options.allowed)?;
                // Note this takes ownership so we can at a later date remove non-allowed encodings.
                Ok(self)
            }
            Operation::Execute(ctx) => self.normalize_with_execution(options.allowed, ctx),
        }
    }

    fn normalize_with_error(&self, allowed: &HashSet<Id>) -> VortexResult<()> {
        if !self.is_allowed_encoding(allowed) {
            vortex_bail!(AssertionFailed: "normalize forbids encoding ({})", self.encoding_id())
        }

        for child in self.children() {
            child.normalize_with_error(allowed)?
        }
        Ok(())
    }

    fn normalize_with_execution(
        self,
        allowed: &HashSet<Id>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let mut normalized = self;

        // Top-first execute the array tree while we hit non-allowed encodings.
        while !normalized.is_allowed_encoding(allowed) {
            normalized = normalized.execute(ctx)?;
        }

        // Now we've normalized the root, we need to ensure the children are normalized also.
        let slots = normalized.slots();
        let mut normalized_slots = Vec::with_capacity(slots.len());
        let mut any_slot_changed = false;

        for slot in slots {
            match slot {
                Some(child) => {
                    let normalized_child = child.clone().normalize(&mut NormalizeOptions {
                        allowed,
                        operation: Operation::Execute(ctx),
                    })?;
                    any_slot_changed |= !ArrayRef::ptr_eq(child, &normalized_child);
                    normalized_slots.push(Some(normalized_child));
                }
                None => normalized_slots.push(None),
            }
        }

        if any_slot_changed {
            normalized = normalized.with_slots(normalized_slots)?;
        }

        Ok(normalized)
    }

    fn is_allowed_encoding(&self, allowed: &HashSet<Id>) -> bool {
        allowed.contains(&self.encoding_id()) || self.is_canonical()
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::NormalizeOptions;
    use super::Operation;
    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::array::VTable;
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::Slice;
    use crate::arrays::SliceArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::validity::Validity;

    #[test]
    fn normalize_with_execution_keeps_parent_when_children_are_unchanged() -> VortexResult<()> {
        let field = PrimitiveArray::from_iter(0i32..4).into_array();
        let array = StructArray::try_new(
            ["field"].into(),
            vec![field.clone()],
            field.len(),
            Validity::NonNullable,
        )?
        .into_array();
        let allowed = HashSet::from_iter([array.encoding_id(), field.encoding_id()]);
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let normalized = array.clone().normalize(&mut NormalizeOptions {
            allowed: &allowed,
            operation: Operation::Execute(&mut ctx),
        })?;

        assert!(ArrayRef::ptr_eq(&array, &normalized));
        Ok(())
    }

    #[test]
    fn normalize_with_error_allows_canonical_arrays() -> VortexResult<()> {
        let field = PrimitiveArray::from_iter(0i32..4).into_array();
        let array = StructArray::try_new(
            ["field"].into(),
            vec![field.clone()],
            field.len(),
            Validity::NonNullable,
        )?
        .into_array();
        let allowed = HashSet::default();

        let normalized = array.clone().normalize(&mut NormalizeOptions {
            allowed: &allowed,
            operation: Operation::Error,
        })?;

        assert!(ArrayRef::ptr_eq(&array, &normalized));
        Ok(())
    }

    #[test]
    fn normalize_with_execution_rebuilds_parent_when_a_child_changes() -> VortexResult<()> {
        let unchanged = PrimitiveArray::from_iter(0i32..4).into_array();
        let sliced =
            SliceArray::new(PrimitiveArray::from_iter(10i32..20).into_array(), 2..6).into_array();
        let array = StructArray::try_new(
            ["lhs", "rhs"].into(),
            vec![unchanged.clone(), sliced],
            unchanged.len(),
            Validity::NonNullable,
        )?
        .into_array();
        let allowed = HashSet::from_iter([array.encoding_id(), unchanged.encoding_id()]);
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let normalized = array.clone().normalize(&mut NormalizeOptions {
            allowed: &allowed,
            operation: Operation::Execute(&mut ctx),
        })?;

        assert!(!ArrayRef::ptr_eq(&array, &normalized));

        let original_children = array.children();
        let normalized_children = normalized.children();
        assert!(ArrayRef::ptr_eq(
            &original_children[0],
            &normalized_children[0]
        ));
        assert!(!ArrayRef::ptr_eq(
            &original_children[1],
            &normalized_children[1]
        ));
        assert_arrays_eq!(normalized_children[1], PrimitiveArray::from_iter(12i32..16));

        Ok(())
    }

    #[test]
    fn normalize_slice_of_dict_returns_dict() -> VortexResult<()> {
        let codes = PrimitiveArray::from_iter(vec![0u32, 1, 0, 1, 2]).into_array();
        let values = PrimitiveArray::from_iter(vec![10i32, 20, 30]).into_array();
        let dict = DictArray::try_new(codes, values)?.into_array();

        // Slice the dict array to get a SliceArray wrapping a DictArray.
        let sliced = SliceArray::new(dict, 1..4).into_array();
        assert_eq!(sliced.encoding_id(), Slice.id());

        let allowed = HashSet::from_iter([Dict.id(), Primitive.id()]);
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let normalized = sliced.normalize(&mut NormalizeOptions {
            allowed: &allowed,
            operation: Operation::Execute(&mut ctx),
        })?;

        // The normalized result should be a DictArray, not a SliceArray.
        assert_eq!(normalized.encoding_id(), Dict.id());
        assert_eq!(normalized.len(), 3);

        // Verify the data: codes [1,0,1] -> values [20, 10, 20]
        assert_arrays_eq!(
            normalized.to_canonical()?,
            PrimitiveArray::from_iter(vec![20i32, 10, 20])
        );

        Ok(())
    }
}

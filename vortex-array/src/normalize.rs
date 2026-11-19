// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prepares array trees for consumers that support a subset of encodings.
//!
//! In-memory consumers check the encoding of each array. Writers check the wire ID chosen by
//! each serializer, since a serializer may reuse an older wire representation.

use smallvec::SmallVec;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayContext;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::VortexSessionExecute;
use crate::array::ArraySerialization;
use crate::arrays::write_parent_reductions;
use crate::executor::max_iterations;
use crate::serde::WireRejection;
use crate::session::ArraySessionExt;

/// Options for normalizing an array.
pub struct NormalizeOptions<'a> {
    /// The in-memory encoding IDs the caller accepts in addition to canonical encodings.
    pub allowed: &'a HashSet<Id>,
    /// The operation to perform when a non-allowed encoding is encountered.
    pub operation: Operation<'a>,
}

/// The operation to perform when a non-allowed encoding is encountered.
pub enum Operation<'a> {
    /// Return an error at the first disallowed encoding.
    Error,
    /// Execute disallowed encodings until the remaining array tree is allowed.
    Execute(&'a mut ExecutionCtx),
}

#[derive(Clone, Copy)]
enum NormalizeTarget<'a> {
    InMemory {
        allowed_encoding_ids: &'a HashSet<Id>,
    },
    Wire {
        write_context: &'a ArrayContext,
    },
}

/// A source array whose root can be serialized to the target.
struct WireRoot {
    array: ArrayRef,
    /// The emitted children when a serializer projects a different child list. Normalization
    /// matches those children against source descendants. `None` visits every source slot.
    projected_children: Option<Vec<ArrayRef>>,
}

impl WireRoot {
    fn new(array: ArrayRef, children: Vec<ArrayRef>) -> Self {
        // Most serializers reuse every source child in order, so a linear comparison suffices.
        // A serializer that projects children uses a scan of its existing list on the retry.
        let uses_source_children = array.nchildren() == children.len()
            && array
                .children_iter()
                .zip(&children)
                .all(|(source, written)| ArrayRef::ptr_eq(source, written));

        Self {
            array,
            projected_children: (!uses_source_children).then_some(children),
        }
    }

    fn normalize_children(
        self,
        write_context: &ArrayContext,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.array.normalize_slots(
            self.projected_children.as_deref(),
            NormalizeTarget::Wire { write_context },
            ctx,
        )
    }
}

impl ArrayRef {
    /// Visit the root and its children, applying the configured [operation](Operation) to
    /// disallowed in-memory encodings.
    pub fn normalize(self, options: &mut NormalizeOptions) -> VortexResult<ArrayRef> {
        match &mut options.operation {
            Operation::Error => {
                self.normalize_with_error(options.allowed)?;
                Ok(self)
            }
            Operation::Execute(ctx) => self.normalize_with_execution(
                NormalizeTarget::InMemory {
                    allowed_encoding_ids: options.allowed,
                },
                ctx,
            ),
        }
    }

    fn normalize_with_error(&self, allowed: &HashSet<Id>) -> VortexResult<()> {
        if !self.is_allowed_encoding(allowed) {
            vortex_bail!("normalize forbids encoding ({})", self.encoding_id())
        }

        for child in self.children() {
            child.normalize_with_error(allowed)?
        }
        Ok(())
    }

    /// Normalize the source array before retrying a write rejected by the wire allowlist.
    pub(crate) fn normalize_for_write(
        self,
        write_context: &ArrayContext,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        self.normalize_with_execution(
            NormalizeTarget::Wire { write_context },
            &mut session.create_execution_ctx(),
        )
    }

    fn normalize_with_execution(
        self,
        target: NormalizeTarget<'_>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match target {
            NormalizeTarget::InMemory {
                allowed_encoding_ids,
            } => {
                let mut array = self;
                while !array.is_allowed_encoding(allowed_encoding_ids) {
                    array = array.execute(ctx)?;
                }

                array.normalize_slots(None, target, ctx)
            }
            NormalizeTarget::Wire { write_context } => normalize_wire(self, write_context, ctx),
        }
    }

    fn normalize_slots(
        self,
        projected_children: Option<&[ArrayRef]>,
        target: NormalizeTarget<'_>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let slots = self.slots();
        let mut normalized_slots = SmallVec::with_capacity(slots.len());
        let mut changed = false;

        for slot in slots {
            let Some(child) = slot else {
                normalized_slots.push(None);
                continue;
            };

            let normalized_child = match projected_children {
                Some(children) if !children.iter().any(|emitted| Self::ptr_eq(child, emitted)) => {
                    // A plugin can expose a descendant while omitting its in-memory parent.
                    // Walk through the parent without applying wire rules to that parent.
                    child.clone().normalize_slots(Some(children), target, ctx)?
                }
                _ => child.clone().normalize_with_execution(target, ctx)?,
            };
            changed |= !Self::ptr_eq(child, &normalized_child);
            normalized_slots.push(Some(normalized_child));
        }

        if !changed {
            return Ok(self);
        }

        // SAFETY: each replacement is the same source slot after a physical rewrite or
        // execution. Both preserve its logical values. `with_slots` checks shape and dtype.
        unsafe { self.with_slots(normalized_slots) }
    }

    fn is_allowed_encoding(&self, allowed: &HashSet<Id>) -> bool {
        allowed.contains(&self.encoding_id()) || self.is_canonical()
    }
}

fn normalize_wire(
    mut array: ArrayRef,
    write_context: &ArrayContext,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = array.dtype().clone();
    let len = array.len();

    for _ in 0..max_iterations() {
        let session = ctx.session();
        match serialize_if_registered(&array, session)? {
            Some(serialized) if write_context.permits(&serialized.serialized_id) => {
                return WireRoot::new(array, serialized.children)
                    .normalize_children(write_context, ctx);
            }
            Some(serialized) if array.is_canonical() => {
                return Err(VortexError::external(WireRejection::DisallowedId(
                    serialized.serialized_id,
                )));
            }
            None if array.is_canonical() => {
                return Err(VortexError::external(WireRejection::UnsupportedArray(
                    array.encoding_id(),
                )));
            }
            Some(_) => {
                if let Some(rewritten) = try_rewrite_wire_root(&array, write_context, ctx)? {
                    return Ok(rewritten);
                }
            }
            None => {}
        }

        let next = array.clone().execute::<ArrayRef>(ctx)?;
        vortex_ensure_eq!(next.dtype(), &dtype);
        vortex_ensure_eq!(next.len(), len);
        next.statistics().inherit_from(array.statistics());
        array = next;
    }

    vortex_bail!("Exceeded execution iteration limit while normalizing for writing")
}

/// Try one parent reduction per occupied slot. A rejected candidate leaves the source
/// available for ordinary execution.
fn try_rewrite_wire_root(
    array: &ArrayRef,
    write_context: &ArrayContext,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let kernels = write_parent_reductions();

    for (slot_idx, child) in array.slots().iter().enumerate() {
        let Some(child) = child else { continue };
        let Some(rewritten) = kernels.reduce_parent(child, array, slot_idx)? else {
            continue;
        };

        rewritten.statistics().inherit_from(array.statistics());
        let Some(serialized) = serialize_if_registered(&rewritten, ctx.session())? else {
            continue;
        };
        if !write_context.permits(&serialized.serialized_id) {
            continue;
        }

        let root = WireRoot::new(rewritten, serialized.children);
        let normalized = match root.normalize_children(write_context, ctx) {
            Ok(normalized) => normalized,
            Err(error) if WireRejection::from_error(&error).is_some() => continue,
            Err(error) => return Err(error),
        };

        // Rebuilding the source can change the wire representation or leave a forbidden
        // child synthesized by the serializer. Verify the candidate before committing to it.
        if wire_tree_is_permitted(&normalized, write_context, ctx.session())? {
            return Ok(Some(normalized));
        }
    }

    Ok(None)
}

fn wire_tree_is_permitted(
    array: &ArrayRef,
    write_context: &ArrayContext,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(serialized) = serialize_if_registered(array, session)? else {
        return Ok(false);
    };
    if !write_context.permits(&serialized.serialized_id) {
        return Ok(false);
    }

    for child in serialized.children {
        if !wire_tree_is_permitted(&child, write_context, session)? {
            return Ok(false);
        }
    }

    Ok(true)
}

fn serialize_if_registered(
    array: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<Option<ArraySerialization>> {
    // An execution step may yield a lazy intermediate with no serializer.
    if !session.arrays().has_serializer(&array.encoding_id()) {
        return Ok(None);
    }

    session.array_serialize(array)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::NormalizeOptions;
    use super::Operation;
    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
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

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
        let mut ctx = SESSION.create_execution_ctx();

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
        let mut ctx = SESSION.create_execution_ctx();

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
        assert!(normalized_children[1].is::<Primitive>());
        assert_arrays_eq!(
            normalized,
            StructArray::try_from_iter([
                ("lhs", PrimitiveArray::from_iter(0i32..4).into_array()),
                ("rhs", PrimitiveArray::from_iter(12i32..16).into_array()),
            ])?,
            &mut ctx
        );

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
        let mut ctx = SESSION.create_execution_ctx();

        let normalized = sliced.normalize(&mut NormalizeOptions {
            allowed: &allowed,
            operation: Operation::Execute(&mut ctx),
        })?;

        // The normalized result should be a DictArray, not a SliceArray.
        assert_eq!(normalized.encoding_id(), Dict.id());
        assert_eq!(normalized.len(), 3);

        // Verify the data: codes [1,0,1] -> values [20, 10, 20]
        let normalized_canonical = normalized.execute::<Canonical>(&mut ctx)?;
        assert_arrays_eq!(
            normalized_canonical,
            PrimitiveArray::from_iter(vec![20i32, 10, 20]),
            &mut ctx
        );

        Ok(())
    }
}

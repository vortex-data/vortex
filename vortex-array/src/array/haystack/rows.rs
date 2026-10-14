// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A lookup array of any values, found by binary search over their sorted rows.

use smallvec::smallvec;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::registry::CachedId;

use super::HaystackOps;
use super::HaystackVTable;
use super::haystack_vtable_common;
use super::slot;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar_fn::fns::binary::build_row_comparator;
use crate::validity::Validity;

/// Builds the row haystack of the non-null `members`.
pub(super) fn try_new(
    members: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Array<RowsHaystack>> {
    let elements = members.execute::<RecursiveCanonical>(ctx)?.0.into_array();
    let mut order: Vec<u64> = (0..elements.len() as u64).collect();
    if !order.is_empty() {
        let compare = build_row_comparator(&elements, &elements, ctx)?;
        let compare = |lhs: &u64, rhs: &u64| compare(row(*lhs), row(*rhs));
        if !order.is_sorted_by(|lhs, rhs| compare(lhs, rhs).is_le()) {
            order.sort_unstable_by(compare);
        }
        order.dedup_by(|lhs, rhs| compare(lhs, rhs).is_eq());
    }
    let order = PrimitiveArray::new(Buffer::from(order), Validity::NonNullable).into_array();

    Array::try_from_parts(ArrayParts::new(
        RowsHaystack,
        elements.dtype().clone(),
        order.len(),
        EmptyArrayData,
        smallvec![Some(elements), Some(order)],
    ))
}

/// A lookup array of any values, found by binary search with the same comparator as equality.
///
/// Its children are the elements, recursively canonical, and the indices of the distinct
/// elements in sorted order. Its rows are the distinct elements in that order. The comparator
/// materializes child buffers and validity once per batch of needles, including decimal widening
/// and nested offsets. Each search then reads those buffers directly.
#[derive(Clone, Debug)]
pub(crate) struct RowsHaystack;

impl RowsHaystack {
    const ELEMENTS: usize = 0;
    const ORDER: usize = 1;
    const NAMES: [&'static str; 2] = ["elements", "order"];
}

impl VTable for RowsHaystack {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = HaystackOps;
    type ValidityVTable = HaystackOps;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.haystack.rows");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), 2, "A row haystack has two children");
        let (Some(elements), Some(order)) = (&slots[Self::ELEMENTS], &slots[Self::ORDER]) else {
            vortex_bail!("A row haystack needs its elements and their order");
        };
        vortex_ensure_eq!(
            elements.dtype(),
            dtype,
            "A row haystack has its elements' dtype"
        );
        vortex_ensure_eq!(
            order.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
            "A row haystack orders its elements by u64 indices"
        );
        vortex_ensure_eq!(
            order.len(),
            len,
            "A row haystack has a row per distinct element"
        );
        Ok(())
    }

    haystack_vtable_common!("RowsHaystack");

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        Self::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let elements = slot(array.slots(), Self::ELEMENTS);
        let order = slot(array.slots(), Self::ORDER).clone();
        Ok(ExecutionResult::done(elements.take(order)?))
    }
}

impl HaystackVTable for RowsHaystack {
    fn contains(
        array: ArrayView<'_, Self>,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        if array.is_empty() {
            return Ok((
                BitBuffer::full_in(false, needles.len(), ctx.allocator().clone()),
                needles.validity()?,
            ));
        }

        let elements = slot(array.slots(), Self::ELEMENTS);
        let order = slot(array.slots(), Self::ORDER)
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let order = order.as_slice::<u64>();
        let needles = needles
            .clone()
            .execute::<RecursiveCanonical>(ctx)?
            .0
            .into_array();

        let compare = build_row_comparator(elements, &needles, ctx)?;
        let bits = BitBuffer::collect_bool_in(
            needles.len(),
            |needle| {
                order
                    .binary_search_by(|&element| compare(row(element), needle))
                    .is_ok()
            },
            ctx.allocator().clone(),
        );
        Ok((bits, needles.validity()?))
    }
}

/// A row index of the elements, which fit in memory and so in `usize`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the elements of a haystack fit in memory"
)]
fn row(index: u64) -> usize {
    index as usize
}

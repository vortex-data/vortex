// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::EmptyArrayData;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::dtype::DType;
use vortex_array::dtype::UnionVariants;
use vortex_error::VortexResult;

/// A Vortex array encoded as a [`DenseUnion`].
pub type DenseUnionArray = Array<DenseUnion>;

/// Slot layout of a dense union array.
#[array_slots(DenseUnion)]
#[allow(dead_code)]
pub struct DenseUnionSlots {
    /// The row-aligned type IDs selecting union variants.
    #[slot(0)]
    pub type_ids: ArrayRef,
    /// The row-aligned offsets into the selected compact child.
    #[slot(1)]
    pub offsets: ArrayRef,
    /// The compact children in variant order.
    #[slot(2..)]
    pub children: Vec<ArrayRef>,
}

pub(crate) fn make_parts(
    type_ids: ArrayRef,
    offsets: ArrayRef,
    variants: UnionVariants,
    children: impl IntoIterator<Item = ArrayRef>,
) -> ArrayParts<DenseUnion> {
    let len = type_ids.len();
    let nullability = type_ids.dtype().nullability();
    let children = children.into_iter();
    let (lower, _) = children.size_hint();
    let mut slots = ArraySlots::with_capacity(DenseUnionSlots::CHILDREN_OFFSET + lower);
    slots.push(Some(type_ids));
    slots.push(Some(offsets));
    slots.extend(children.map(Some));

    ArrayParts::new(
        DenseUnion,
        DType::Union(variants, nullability),
        len,
        EmptyArrayData,
    )
    .with_slots(slots)
}

/// Accessors for a dense union array.
pub trait DenseUnionArrayExt: DenseUnionArraySlotsExt {
    /// Return the union's variant schema.
    fn variants(&self) -> &UnionVariants {
        match self.as_ref().dtype() {
            DType::Union(variants, _) => variants,
            _ => unreachable!("DenseUnionArrayExt requires a union dtype"),
        }
    }

    /// Iterate over compact children in variant order.
    fn iter_children(&self) -> impl ExactSizeIterator<Item = &ArrayRef> + '_ {
        self.children().iter()
    }

    /// Return a compact child by variant index.
    fn child(&self, index: usize) -> Option<&ArrayRef> {
        self.children().get(index)
    }
}

impl<T: TypedArrayRef<DenseUnion>> DenseUnionArrayExt for T {}

/// The dense physical encoding for the logical [`DType::Union`] type.
#[derive(Clone, Debug)]
pub struct DenseUnion;

impl DenseUnion {
    /// Try to construct a dense union array.
    ///
    /// The logical union's nullability is inherited from `type_ids`; nullable type IDs represent
    /// outer union nulls. `type_ids` must be a nullable or non-nullable `u8` array, `offsets` must
    /// be a non-nullable `i32` array of the same length, and the compact children must match the
    /// variant count and dtypes. Type IDs and offsets are structurally validated, but their
    /// individual values are checked only when the array is accessed or converted to its canonical
    /// sparse representation.
    ///
    /// # Errors
    ///
    /// Returns an error when the selector arrays or compact children do not satisfy these
    /// structural invariants.
    pub fn try_new(
        type_ids: ArrayRef,
        offsets: ArrayRef,
        variants: UnionVariants,
        children: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<DenseUnionArray> {
        Array::try_from_parts(make_parts(type_ids, offsets, variants, children))
    }
}

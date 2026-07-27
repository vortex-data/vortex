// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::PrimitiveArray;
use crate::arrays::Union;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::UnionVariants;

/// Slot layout of a canonical sparse union array.
#[array_slots(Union)]
pub struct UnionSlots {
    /// The row-aligned array of type IDs selecting a union child.
    #[slot(0)]
    pub type_ids: ArrayRef,
    /// The row-aligned sparse children in variant order.
    #[slot(1..)]
    pub children: Vec<ArrayRef>,
}

pub(super) fn make_union_parts(
    type_ids: ArrayRef,
    variants: UnionVariants,
    children: impl IntoIterator<Item = ArrayRef>,
) -> ArrayParts<Union> {
    let len = type_ids.len();
    let nullability = type_ids.dtype().nullability();
    let children = children.into_iter();
    let (lower, _) = children.size_hint();
    let mut slots = ArraySlots::with_capacity(UnionSlots::CHILDREN_OFFSET + lower);
    slots.push(Some(type_ids));
    slots.extend(children.map(Some));

    ArrayParts::new(
        Union,
        DType::Union(variants, nullability),
        len,
        EmptyArrayData,
    )
    .with_slots(slots)
}

/// Concrete parts of a [`UnionArray`](super::UnionArray).
pub struct UnionDataParts {
    /// The union variant schema.
    pub variants: UnionVariants,
    /// The row-aligned type IDs.
    pub type_ids: ArrayRef,
    /// The row-aligned sparse children in variant order.
    pub children: Vec<ArrayRef>,
}

/// Accessors for a canonical sparse union array.
///
/// Slot accessors (`type_ids`, `children`, `slots_view`) live on the generated
/// [`UnionArraySlotsExt`] supertrait; this trait layers union-specific lookups on top.
pub trait UnionArrayExt: UnionArraySlotsExt {
    /// The union's variant schema.
    fn variants(&self) -> &UnionVariants {
        match self.as_ref().dtype() {
            DType::Union(variants, _) => variants,
            _ => unreachable!("UnionArrayExt requires a union dtype"),
        }
    }

    /// Iterate over sparse children in variant order.
    fn iter_children(&self) -> impl ExactSizeIterator<Item = &ArrayRef> + '_ {
        self.children().iter()
    }

    /// Return a sparse child by its variant index.
    fn child(&self, index: usize) -> Option<&ArrayRef> {
        self.children().get(index)
    }

    /// Return a sparse child selected by a data-level type ID.
    fn child_by_type_id(&self, type_id: u8) -> Option<&ArrayRef> {
        self.child(self.variants().tag_to_child_index(type_id)?)
    }

    /// Return a sparse child selected by its variant name, if present.
    fn child_by_name_opt(&self, name: impl AsRef<str>) -> Option<&ArrayRef> {
        self.child(self.variants().find(name)?)
    }

    /// Return a sparse child selected by its variant name.
    fn child_by_name(&self, name: impl AsRef<str>) -> VortexResult<&ArrayRef> {
        let name = name.as_ref();
        self.child_by_name_opt(name).ok_or_else(|| {
            vortex_err!(
                "Variant {name} not found in union array with names {:?}",
                self.variants().names()
            )
        })
    }
}
impl<T: TypedArrayRef<Union>> UnionArrayExt for T {}

impl Array<Union> {
    /// Construct a canonical sparse union array.
    ///
    /// # Panics
    ///
    /// Panics if the components do not satisfy the invariants documented on
    /// [`Self::new_unchecked`].
    pub fn new(
        type_ids: ArrayRef,
        variants: UnionVariants,
        children: impl IntoIterator<Item = ArrayRef>,
    ) -> Self {
        Self::try_new(type_ids, variants, children).vortex_expect("UnionArray construction failed")
    }

    /// Try to construct a canonical sparse union array.
    ///
    /// Type ID values are not validated during construction. Accessing a non-null row whose type
    /// ID is not declared by `variants` will panic.
    ///
    /// # Errors
    ///
    /// Returns an error if `type_ids` is not a `u8` array, or if the sparse children do not match
    /// the variant schema and outer array length.
    pub fn try_new(
        type_ids: ArrayRef,
        variants: UnionVariants,
        children: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            matches!(type_ids.dtype(), DType::Primitive(PType::U8, _)),
            "UnionArray type_ids must be u8, got {}",
            type_ids.dtype()
        );

        Array::try_from_parts(make_union_parts(type_ids, variants, children))
    }

    /// Construct a canonical sparse union array without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure `type_ids` is a `u8` array, every child has the corresponding variant
    /// dtype, and all arrays have the same length. Null type IDs represent outer union nulls.
    pub unsafe fn new_unchecked(
        type_ids: ArrayRef,
        variants: UnionVariants,
        children: impl IntoIterator<Item = ArrayRef>,
    ) -> Self {
        unsafe { Array::from_parts_unchecked(make_union_parts(type_ids, variants, children)) }
    }

    /// Deconstruct this array into its type IDs, variant schema, and sparse children.
    pub fn into_data_parts(self) -> UnionDataParts {
        let variants = self.variants().clone();
        let type_ids = self.type_ids().clone();
        let children = self.iter_children().cloned().collect();
        UnionDataParts {
            variants,
            type_ids,
            children,
        }
    }

    /// Create an empty array for a union dtype.
    pub(crate) fn empty(variants: UnionVariants, nullability: Nullability) -> Self {
        let type_ids = PrimitiveArray::empty::<u8>(nullability).into_array();
        let children: Vec<_> = variants
            .variants()
            .map(|dtype| crate::Canonical::empty(&dtype).into_array())
            .collect();

        Self::new(type_ids, variants, children)
    }
}

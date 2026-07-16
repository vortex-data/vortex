// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::arrays::PrimitiveArray;
use crate::arrays::Union;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::UnionVariants;
use crate::legacy_session;

/// The row-aligned array of type IDs selecting a union child.
pub(super) const TYPE_IDS_SLOT: usize = 0;
/// The offset at which the sparse child arrays begin in the slots vector.
pub(super) const CHILDREN_OFFSET: usize = 1;

pub(super) fn make_union_parts(
    type_ids: ArrayRef,
    variants: UnionVariants,
    children: &[ArrayRef],
) -> ArrayParts<Union> {
    let len = type_ids.len();
    let nullability = type_ids.dtype().nullability();
    let slots: ArraySlots = once(Some(type_ids))
        .chain(children.iter().cloned().map(Some))
        .collect();

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
    pub children: Arc<[ArrayRef]>,
}

/// Accessors for a canonical sparse union array.
pub trait UnionArrayExt: TypedArrayRef<Union> {
    /// The union's variant schema.
    fn variants(&self) -> &UnionVariants {
        match self.as_ref().dtype() {
            DType::Union(variants, _) => variants,
            _ => unreachable!("UnionArrayExt requires a union dtype"),
        }
    }

    /// The row-aligned `u8` type IDs whose nulls represent outer union nulls.
    fn type_ids(&self) -> &ArrayRef {
        self.as_ref().slots()[TYPE_IDS_SLOT]
            .as_ref()
            .vortex_expect("UnionArray type_ids slot")
    }

    /// Iterate over sparse children in variant order.
    fn iter_children(&self) -> impl ExactSizeIterator<Item = &ArrayRef> + '_ {
        self.as_ref().slots()[CHILDREN_OFFSET..]
            .iter()
            .map(|slot| slot.as_ref().vortex_expect("UnionArray child slot"))
    }

    /// Return the sparse children in variant order.
    fn children(&self) -> Arc<[ArrayRef]> {
        self.iter_children().cloned().collect()
    }

    /// Return a sparse child by its variant index.
    fn child(&self, index: usize) -> Option<&ArrayRef> {
        self.as_ref().slots().get(CHILDREN_OFFSET + index)?.as_ref()
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

/// Validate host-resident type ID values eagerly.
///
/// Non-host arrays remain valid structurally and are checked when their values are accessed.
#[allow(clippy::disallowed_methods)]
fn validate_type_id_values(array: &Array<Union>) -> VortexResult<()> {
    if !array.type_ids().is_host() {
        return Ok(());
    }

    let mut ctx = legacy_session().create_execution_ctx();
    let type_ids = array
        .type_ids()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    let values = type_ids.as_slice::<u8>();
    let validity = type_ids.validity()?.execute_mask(array.len(), &mut ctx)?;
    let valid_indices: Box<dyn Iterator<Item = usize> + '_> = match &validity {
        vortex_mask::Mask::AllTrue(_) => Box::new(0..values.len()),
        vortex_mask::Mask::AllFalse(_) => Box::new(std::iter::empty()),
        vortex_mask::Mask::Values(mask) => Box::new(mask.indices().iter().copied()),
    };
    for index in valid_indices {
        let type_id = values[index];
        vortex_ensure!(
            array.variants().tag_to_child_index(type_id).is_some(),
            "UnionArray type ID {type_id} is not present in {:?}",
            array.variants().type_ids()
        );
    }

    Ok(())
}

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
        children: impl Into<Arc<[ArrayRef]>>,
    ) -> Self {
        Self::try_new(type_ids, variants, children).vortex_expect("UnionArray construction failed")
    }

    /// Try to construct a canonical sparse union array.
    ///
    /// # Errors
    ///
    /// Returns an error if the type IDs and sparse children do not match the variant schema or do
    /// not all have the same length.
    pub fn try_new(
        type_ids: ArrayRef,
        variants: UnionVariants,
        children: impl Into<Arc<[ArrayRef]>>,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            matches!(
                type_ids.dtype(),
                DType::Primitive(crate::dtype::PType::U8, _)
            ),
            "UnionArray type_ids must be u8, got {}",
            type_ids.dtype()
        );

        let children = children.into();
        let array = Array::try_from_parts(make_union_parts(type_ids, variants, &children))?;
        validate_type_id_values(&array)?;

        Ok(array)
    }

    /// Construct a canonical sparse union array without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure every non-null type ID is a `u8` value declared by `variants`, every
    /// child has the corresponding variant dtype, and all arrays have the same length. Null type
    /// IDs represent outer union nulls.
    pub unsafe fn new_unchecked(
        type_ids: ArrayRef,
        variants: UnionVariants,
        children: impl Into<Arc<[ArrayRef]>>,
    ) -> Self {
        let children = children.into();
        unsafe { Array::from_parts_unchecked(make_union_parts(type_ids, variants, &children)) }
    }

    /// Deconstruct this array into its type IDs, variant schema, and sparse children.
    pub fn into_data_parts(self) -> UnionDataParts {
        let variants = self.variants().clone();
        let type_ids = self.type_ids().clone();
        let children = self.children();
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

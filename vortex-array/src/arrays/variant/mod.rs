// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vtable;

use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

pub use self::vtable::Variant;
pub use self::vtable::VariantArray;
use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::dtype::DType;

pub(super) const CORE_STORAGE_SLOT: usize = 0;
pub(super) const SHREDDED_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["core_storage", "shredded"];

pub trait VariantArrayExt: TypedArrayRef<Variant> {
    /// Returns the raw storage that preserves the full variant value for every row.
    fn core_storage(&self) -> &ArrayRef {
        self.as_ref().slots()[CORE_STORAGE_SLOT]
            .as_ref()
            .vortex_expect("validated variant core_storage slot")
    }

    /// Returns the optional row-aligned typed shredded tree for selected variant paths.
    fn shredded(&self) -> Option<&ArrayRef> {
        self.as_ref().slots()[SHREDDED_SLOT].as_ref()
    }

    /// Returns the raw storage child.
    ///
    /// This is a compatibility shim for the previous one-child canonical shape.
    fn child(&self) -> &ArrayRef {
        self.core_storage()
    }
}
impl<T: TypedArrayRef<Variant>> VariantArrayExt for T {}

impl Array<Variant> {
    /// Creates a new `VariantArray` with raw core storage and optional shredded storage.
    pub fn try_new(core_storage: ArrayRef, shredded: Option<ArrayRef>) -> VortexResult<Self> {
        let dtype = core_storage.dtype().clone();
        vortex_ensure!(
            matches!(dtype, DType::Variant(_)),
            "VariantArray core_storage dtype must be Variant, found {dtype}"
        );
        let len = core_storage.len();
        let stats = core_storage.statistics().to_owned();
        Ok(Array::try_from_parts(
            ArrayParts::new(Variant, dtype, len, EmptyArrayData)
                .with_slots(vec![Some(core_storage), shredded]),
        )?
        .with_stats_set(stats))
    }

    /// Creates a new `VariantArray`.
    pub fn new(core_storage: ArrayRef) -> Self {
        Self::try_new(core_storage, None).vortex_expect("invalid VariantArray core_storage")
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VariantArray;
    use crate::arrays::variant::VariantArrayExt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    fn core_storage(len: usize) -> crate::ArrayRef {
        ConstantArray::new(
            Scalar::variant(Scalar::primitive(1i32, Nullability::NonNullable)),
            len,
        )
        .into_array()
    }

    #[test]
    fn try_new_exposes_core_storage_without_shredded() -> VortexResult<()> {
        let core_storage = core_storage(2);

        let variant = VariantArray::try_new(core_storage.clone(), None)?;

        assert_eq!(variant.dtype(), core_storage.dtype());
        assert_eq!(variant.len(), 2);
        assert_eq!(variant.core_storage().dtype(), core_storage.dtype());
        assert!(variant.shredded().is_none());

        Ok(())
    }

    #[test]
    fn try_new_exposes_core_storage_and_shredded() -> VortexResult<()> {
        let core_storage = core_storage(3);
        let shredded = buffer![10i32, 20, 30].into_array();

        let variant = VariantArray::try_new(core_storage.clone(), Some(shredded.clone()))?;

        assert_eq!(variant.dtype(), &DType::Variant(Nullability::NonNullable));
        assert_eq!(variant.len(), 3);
        assert_eq!(variant.core_storage().dtype(), core_storage.dtype());
        assert_eq!(variant.core_storage().len(), core_storage.len());
        assert_eq!(
            variant.shredded().map(|child| child.dtype()),
            Some(shredded.dtype())
        );
        assert_eq!(
            variant.shredded().map(|child| child.len()),
            Some(shredded.len())
        );
        assert_eq!(variant.as_ref().slot_name(0), "core_storage");
        assert_eq!(variant.as_ref().slot_name(1), "shredded");

        Ok(())
    }

    #[test]
    fn try_new_rejects_non_variant_core_storage() {
        let core_storage = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();

        assert!(VariantArray::try_new(core_storage, None).is_err());
    }

    #[test]
    fn try_new_rejects_shredded_length_mismatch() {
        let core_storage = core_storage(3);
        let shredded = buffer![10i32, 20].into_array();

        assert!(VariantArray::try_new(core_storage, Some(shredded)).is_err());
    }
}

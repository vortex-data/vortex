// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vtable;

pub(crate) mod compute;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

pub use self::vtable::Variant;
pub use self::vtable::VariantArray;

pub(crate) fn initialize(session: &vortex_session::VortexSession) {
    vtable::initialize(session);
}

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

/// Accessors for canonical variant storage.
///
/// A canonical variant array keeps the full variant value for every row in `core_storage` and may
/// carry a row-aligned, storage-agnostic `shredded` typed tree for selected paths.
///
/// `core_storage` is a logical `DType::Variant` array, not a specific physical encoding: it may be
/// chunked, constant, or otherwise encoded. Callers must use normal array operations instead of
/// assuming a particular slot layout. The shredded child may have any dtype; its dtype is recorded
/// during serialization and validated by normal child deserialization.
pub trait VariantArrayExt: TypedArrayRef<Variant> {
    /// Returns the logical variant storage that preserves the full value for every row.
    fn core_storage(&self) -> &ArrayRef {
        self.as_ref().slots()[CORE_STORAGE_SLOT]
            .as_ref()
            .vortex_expect("validated variant core_storage slot")
    }

    /// Returns the optional row-aligned typed shredded tree for selected variant paths.
    /// This functions returns `Some` only if the array was canonicalized and the shredded data
    /// was pulled out of the underlying variant storage.
    fn shredded(&self) -> Option<&ArrayRef> {
        self.as_ref().slots()[SHREDDED_SLOT].as_ref()
    }
}
impl<T: TypedArrayRef<Variant>> VariantArrayExt for T {}

impl Array<Variant> {
    /// Creates a new `VariantArray` with logical variant core storage and optional shredded storage.
    ///
    /// `core_storage` must have `DType::Variant`, but it may use any Variant-typed physical
    /// encoding. See [`VariantArrayExt`] for the higher-level storage contract.
    ///
    /// `shredded`, when present, must be row-aligned with `core_storage` and stores typed values for
    /// selected variant paths.
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
                .with_slots(vec![Some(core_storage), shredded].into()),
        )?
        .with_stats_set(stats))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VariantArray;
    use crate::arrays::variant::VariantArrayExt;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::root;
    use crate::expr::variant_get;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::variant_get::VariantPath;

    fn core_storage(len: usize) -> ArrayRef {
        ConstantArray::new(
            Scalar::variant(Scalar::primitive(1i32, Nullability::NonNullable)),
            len,
        )
        .into_array()
    }

    fn row_storage(values: impl IntoIterator<Item = i32>) -> VortexResult<ArrayRef> {
        let chunks = values
            .into_iter()
            .map(|value| {
                ConstantArray::new(
                    Scalar::variant(Scalar::primitive(value, Nullability::NonNullable)),
                    1,
                )
                .into_array()
            })
            .collect();

        Ok(ChunkedArray::try_new(chunks, DType::Variant(Nullability::NonNullable))?.into_array())
    }

    fn variant_with_shredded(
        core_values: impl IntoIterator<Item = i32>,
        shredded_values: impl IntoIterator<Item = i32>,
    ) -> VortexResult<VariantArray> {
        VariantArray::try_new(
            row_storage(core_values)?,
            Some(PrimitiveArray::from_iter(shredded_values).into_array()),
        )
    }

    fn execute_variant(array: ArrayRef) -> VortexResult<VariantArray> {
        let mut ctx = array_session().create_execution_ctx();
        let Canonical::Variant(variant) = array.execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant array"));
        };
        Ok(variant)
    }

    fn assert_variant_rows(
        array: &VariantArray,
        expected_core: &[Option<i32>],
        expected_shredded: &[Option<i32>],
    ) -> VortexResult<()> {
        assert_variant_core_rows(array, expected_core)?;
        assert_eq!(array.len(), expected_shredded.len());

        let shredded = array
            .shredded()
            .ok_or_else(|| vortex_err!("expected shredded child"))?;
        let mut ctx = array_session().create_execution_ctx();
        let shredded = shredded.clone().execute::<PrimitiveArray>(&mut ctx)?;
        let expected_shredded_array = if let Some(values) = expected_shredded
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
        {
            PrimitiveArray::from_iter(values)
        } else {
            PrimitiveArray::from_option_iter(expected_shredded.iter().copied())
        };
        assert_arrays_eq!(shredded, expected_shredded_array, &mut ctx);

        Ok(())
    }

    fn assert_variant_core_rows(
        array: &VariantArray,
        expected_core: &[Option<i32>],
    ) -> VortexResult<()> {
        assert_eq!(array.len(), expected_core.len());

        let mut ctx = array_session().create_execution_ctx();
        for (idx, expected) in expected_core.iter().enumerate() {
            let scalar = array.core_storage().execute_scalar(idx, &mut ctx)?;
            let variant = scalar.as_variant();
            match expected {
                Some(expected) => {
                    let value = variant
                        .value()
                        .ok_or_else(|| vortex_err!("expected non-null variant row"))?;
                    assert_eq!(value.as_primitive().typed_value::<i32>(), Some(*expected));
                }
                None => assert!(variant.is_null()),
            }
        }

        Ok(())
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

    #[test]
    fn scalar_at_merges_shredded_with_core_storage() -> VortexResult<()> {
        let dtype = DType::Variant(Nullability::Nullable);
        let core_chunks = [Some(1i32), None, Some(3)]
            .into_iter()
            .map(|value| {
                let scalar = match value {
                    Some(value) => {
                        Scalar::variant(Scalar::primitive(value, Nullability::NonNullable))
                            .cast(&dtype)?
                    }
                    None => Scalar::null(dtype.clone()),
                };
                Ok(ConstantArray::new(scalar, 1).into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let core_storage = ChunkedArray::try_new(core_chunks, dtype)?.into_array();
        let shredded = PrimitiveArray::from_option_iter([Some(10i32), Some(20), None]).into_array();
        let variant = VariantArray::try_new(core_storage, Some(shredded))?;

        let mut ctx = array_session().create_execution_ctx();
        for (idx, expected) in [Some(10i32), None, Some(3)].into_iter().enumerate() {
            let scalar = variant.execute_scalar(idx, &mut ctx)?;
            let variant = scalar.as_variant();
            match expected {
                Some(expected) => {
                    let value = variant
                        .value()
                        .ok_or_else(|| vortex_err!("expected non-null variant row"))?;
                    assert_eq!(value.as_primitive().typed_value::<i32>(), Some(expected));
                }
                None => assert!(variant.is_null()),
            }
        }

        Ok(())
    }

    #[test]
    fn slice_preserves_core_storage_and_shredded_rows() -> VortexResult<()> {
        let variant = variant_with_shredded(0..5, 10..15)?;

        let sliced = execute_variant(variant.into_array().slice(1..4)?)?;

        assert_variant_rows(
            &sliced,
            &[Some(1), Some(2), Some(3)],
            &[Some(11), Some(12), Some(13)],
        )
    }

    #[test]
    fn filter_preserves_core_storage_and_shredded_rows() -> VortexResult<()> {
        let variant = variant_with_shredded(0..5, 10..15)?;

        let filtered = execute_variant(
            variant
                .into_array()
                .filter(Mask::from_iter([true, false, true, false, true]))?,
        )?;

        assert_variant_rows(
            &filtered,
            &[Some(0), Some(2), Some(4)],
            &[Some(10), Some(12), Some(14)],
        )
    }

    #[test]
    fn take_preserves_core_storage_and_shredded_rows() -> VortexResult<()> {
        let variant = variant_with_shredded(0..5, 10..15)?;

        let taken = execute_variant(
            variant
                .into_array()
                .take(buffer![4u64, 1, 3].into_array())?,
        )?;

        assert_variant_rows(
            &taken,
            &[Some(4), Some(1), Some(3)],
            &[Some(14), Some(11), Some(13)],
        )
    }

    #[test]
    fn mask_preserves_core_storage_and_shredded_rows() -> VortexResult<()> {
        let variant = variant_with_shredded(0..5, 10..15)?;
        let mask = BoolArray::from_iter([true, false, true, false, true]).into_array();

        let masked = execute_variant(variant.into_array().mask(mask)?)?;

        assert_variant_rows(
            &masked,
            &[Some(0), None, Some(2), None, Some(4)],
            &[Some(10), None, Some(12), None, Some(14)],
        )
    }

    #[test]
    fn mask_preserves_chunked_core_storage_validity() -> VortexResult<()> {
        let dtype = DType::Variant(Nullability::Nullable);
        let core_chunks = [Some(1i32), None, Some(3), Some(4)]
            .into_iter()
            .map(|value| {
                let scalar = match value {
                    Some(value) => {
                        Scalar::variant(Scalar::primitive(value, Nullability::NonNullable))
                            .cast(&dtype)?
                    }
                    None => Scalar::null(dtype.clone()),
                };
                Ok(ConstantArray::new(scalar, 1).into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let core_storage = ChunkedArray::try_new(core_chunks, dtype)?.into_array();
        let variant = VariantArray::try_new(core_storage, None)?;
        let mask = BoolArray::from_iter([true, true, false, true]).into_array();

        let masked = execute_variant(variant.into_array().mask(mask)?)?;

        assert_variant_core_rows(&masked, &[Some(1), None, None, Some(4)])
    }

    #[test]
    fn variant_get_keeps_valid_shredded_rows_for_matching_dtype() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let core_storage = row_storage([1, 2, 3])?;
        let shredded = StructArray::try_from_iter([(
            "a",
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
        )])?;
        let variant = VariantArray::try_new(core_storage, Some(shredded.into_array()))?;
        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );

        let result = variant
            .into_array()
            .apply(&expr)?
            .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_treats_value_and_typed_value_as_logical_field_names() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let core_storage = row_storage([1, 2, 3])?;
        let shredded = StructArray::try_from_iter([
            (
                "value",
                PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            ),
            (
                "typed_value",
                PrimitiveArray::from_iter([40i32, 50, 60]).into_array(),
            ),
        ])?;
        let variant = VariantArray::try_new(core_storage, Some(shredded.into_array()))?;

        let value_expr = variant_get(
            root(),
            VariantPath::field("value"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let value_result = variant
            .clone()
            .into_array()
            .apply(&value_expr)?
            .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())?;
        assert_arrays_eq!(
            value_result,
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30)]),
            &mut ctx
        );

        let typed_value_expr = variant_get(
            root(),
            VariantPath::field("typed_value"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let typed_value_result = variant
            .into_array()
            .apply(&typed_value_expr)?
            .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())?;
        assert_arrays_eq!(
            typed_value_result,
            PrimitiveArray::from_option_iter([Some(40i32), Some(50), Some(60)]),
            &mut ctx
        );
        Ok(())
    }
}

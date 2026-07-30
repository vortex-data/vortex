// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`MapScalar`] typed view implementation.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::MapDType;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing an ordered sequence of map key/value entries.
///
/// Map keys are non-null and each entry has exactly one key and one value. Duplicate keys are
/// preserved because map logical types do not enforce key uniqueness.
#[derive(Debug, Clone, Copy)]
pub struct MapScalar<'a> {
    dtype: &'a DType,
    entries: Option<&'a [Option<ScalarValue>]>,
}

impl Display for MapScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.is_null() {
            return write!(f, "null");
        }

        write!(
            f,
            "{{{}}}",
            self.entries()
                .map(|(key, value)| format!("{key}: {value}"))
                .format(", ")
        )
    }
}

impl PartialEq for MapScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.entries == other.entries
    }
}

impl Eq for MapScalar<'_> {}

impl Hash for MapScalar<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dtype.hash_ignore_nullability(state);
        self.entries.hash(state);
    }
}

impl<'a> MapScalar<'a> {
    /// Creates a map scalar view from a dtype and optional scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error when `dtype` is not [`DType::Map`].
    pub fn try_new(dtype: &'a DType, value: Option<&'a ScalarValue>) -> VortexResult<Self> {
        if !dtype.is_map() {
            vortex_bail!("Expected map scalar, found {dtype}")
        }

        Ok(Self {
            dtype,
            entries: value.map(ScalarValue::as_list),
        })
    }

    /// Returns the map dtype.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns the map type details.
    #[inline]
    pub fn map_dtype(&self) -> &'a MapDType {
        self.dtype
            .as_map_opt()
            .vortex_expect("MapScalar always has a map dtype")
    }

    /// Returns the number of entries, or zero for a null map.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.map_or(0, <[Option<ScalarValue>]>::len)
    }

    /// Returns whether the map has no entries or is null.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns whether the entire map scalar is null.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.entries.is_none()
    }

    /// Returns the entry at `index`, or `None` when the map is null or the index is out of bounds.
    pub fn entry(&self, index: usize) -> Option<(Scalar, Scalar)> {
        let values = self.entries?.get(index)?.as_ref()?.as_list();
        Some(self.entry_scalars(values))
    }

    /// Iterates over `(key, value)` entries. A null map yields no entries.
    pub fn entries(&self) -> impl Iterator<Item = (Scalar, Scalar)> + '_ {
        self.entries.into_iter().flatten().map(|entry| {
            self.entry_scalars(
                entry
                    .as_ref()
                    .vortex_expect("map entry is non-null")
                    .as_list(),
            )
        })
    }

    /// Iterates over map keys. A null map yields no keys.
    pub fn keys(&self) -> impl Iterator<Item = Scalar> + '_ {
        self.entries().map(|(key, _)| key)
    }

    /// Iterates over map values. A null map yields no values.
    pub fn values(&self) -> impl Iterator<Item = Scalar> + '_ {
        self.entries().map(|(_, value)| value)
    }

    /// Casts this map scalar to another map dtype.
    ///
    /// # Errors
    ///
    /// Returns an error when `dtype` is not a map, its key/value dtypes cannot be cast, or the
    /// target claims sorted keys when this scalar's dtype does not make that assertion. Also
    /// returns an error for direct null-map casts; callers should use [`Scalar::cast`] so null
    /// handling and sortedness checks stay centralized.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let target = dtype
            .as_map_opt()
            .ok_or_else(|| vortex_err!("Cannot cast map to {dtype}: target must be a map"))?;

        if target.keys_sorted() && !self.map_dtype().keys_sorted() {
            vortex_bail!(
                "Cannot cast {} to {dtype}: source does not assert sorted map keys",
                self.dtype
            );
        }

        let Some(entries) = self.entries else {
            vortex_bail!(
                "Cannot cast null map {} to {dtype}: Scalar::cast should handle nulls first",
                self.dtype
            );
        };

        let target_key = target.key_dtype();
        let target_value = target.value_dtype();
        let entries = entries
            .iter()
            .map(|entry| {
                let (key, value) = self.entry_scalars(
                    entry
                        .as_ref()
                        .vortex_expect("map entry is non-null")
                        .as_list(),
                );
                Ok(Some(ScalarValue::Tuple(vec![
                    key.cast(&target_key)?.into_value(),
                    value.cast(&target_value)?.into_value(),
                ])))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Scalar::try_new(dtype.clone(), Some(ScalarValue::Tuple(entries)))
    }

    fn entry_scalars(&self, values: &[Option<ScalarValue>]) -> (Scalar, Scalar) {
        let key = values.first().vortex_expect("map entry has a key").clone();
        let value = values.get(1).vortex_expect("map entry has a value").clone();

        // SAFETY: MapScalar only views a Scalar that has passed Scalar::validate, which enforces
        // the entry shape and the key/value dtypes.
        (
            unsafe { Scalar::new_unchecked(self.map_dtype().key_dtype(), key) },
            unsafe { Scalar::new_unchecked(self.map_dtype().value_dtype(), value) },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;

    use vortex_error::VortexResult;
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    fn dtype() -> VortexResult<DType> {
        DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            false,
            Nullability::Nullable,
        )
    }

    #[test]
    fn entries_and_display() -> VortexResult<()> {
        let scalar = Scalar::try_map(
            dtype()?,
            [
                (
                    Scalar::primitive(1i32, Nullability::NonNullable),
                    Scalar::utf8("one", Nullability::Nullable),
                ),
                (
                    Scalar::primitive(2i32, Nullability::NonNullable),
                    Scalar::null(DType::Utf8(Nullability::Nullable)),
                ),
            ],
        )?;

        let map = scalar.as_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map.keys().count(), 2);
        assert_eq!(map.values().count(), 2);
        assert_eq!(
            map.entry(0).unwrap().0,
            Scalar::primitive(1i32, Nullability::NonNullable)
        );
        assert_eq!(format!("{map}"), "{1i32: \"one\", 2i32: null}");

        Ok(())
    }

    #[test]
    fn null_map_is_distinct_from_empty_map() -> VortexResult<()> {
        let dtype = dtype()?;
        let empty = Scalar::try_map(dtype.clone(), [])?;
        let null = Scalar::null(dtype);

        assert!(empty.as_map().is_empty());
        assert!(!empty.as_map().is_null());
        assert!(null.as_map().is_null());
        assert_ne!(empty, null);

        Ok(())
    }

    #[test]
    fn rejects_malformed_entries() -> VortexResult<()> {
        let dtype = dtype()?;
        let malformed = Scalar::try_new(
            dtype,
            Some(ScalarValue::Tuple(vec![Some(ScalarValue::Tuple(vec![
                Some(ScalarValue::Primitive(1i32.into())),
            ]))])),
        );

        assert!(malformed.is_err());
        Ok(())
    }

    #[test]
    fn rejects_null_keys() -> VortexResult<()> {
        let malformed = Scalar::try_new(
            dtype()?,
            Some(ScalarValue::Tuple(vec![Some(ScalarValue::Tuple(vec![
                None,
                Some(ScalarValue::Utf8("value".into())),
            ]))])),
        );

        assert!(malformed.is_err());
        Ok(())
    }

    #[test]
    fn cast_can_drop_a_sortedness_assertion() -> VortexResult<()> {
        let source_dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            true,
            Nullability::NonNullable,
        )?;
        let target_dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            false,
            Nullability::Nullable,
        )?;
        let scalar = Scalar::try_map(
            source_dtype,
            [(
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::utf8("one", Nullability::Nullable),
            )],
        )?;

        let cast = scalar.cast(&target_dtype)?;
        assert_eq!(cast.dtype(), &target_dtype);
        assert_eq!(cast.as_map().entry(0), scalar.as_map().entry(0));

        Ok(())
    }

    #[test]
    fn cast_cannot_create_a_sortedness_assertion() -> VortexResult<()> {
        let target_dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            true,
            Nullability::Nullable,
        )?;
        let scalar = Scalar::try_map(
            dtype()?,
            [(
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::utf8("one", Nullability::Nullable),
            )],
        )?;

        assert!(scalar.cast(&target_dtype).is_err());

        Ok(())
    }

    #[test]
    fn equal_maps_with_different_nested_nullability_hash_equal() -> VortexResult<()> {
        let key_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let nullable_dtype = DType::map(
            key_dtype.clone(),
            DType::Utf8(Nullability::Nullable),
            false,
            Nullability::Nullable,
        )?;
        let nonnullable_dtype = DType::map(
            key_dtype,
            DType::Utf8(Nullability::NonNullable),
            false,
            Nullability::NonNullable,
        )?;
        let nullable_scalar = Scalar::try_map(
            nullable_dtype,
            [(
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::utf8("one", Nullability::Nullable),
            )],
        )?;
        let nonnullable_scalar = Scalar::try_map(
            nonnullable_dtype,
            [(
                Scalar::primitive(1i32, Nullability::NonNullable),
                Scalar::utf8("one", Nullability::NonNullable),
            )],
        )?;
        let nullable_map = nullable_scalar.as_map();
        let nonnullable_map = nonnullable_scalar.as_map();

        assert_eq!(nullable_map, nonnullable_map);

        let mut nullable_hash = DefaultHasher::new();
        nullable_map.hash(&mut nullable_hash);
        let mut nonnullable_hash = DefaultHasher::new();
        nonnullable_map.hash(&mut nonnullable_hash);
        assert_eq!(nullable_hash.finish(), nonnullable_hash.finish());

        let mut set = HashSet::new();
        set.insert(nullable_map);
        assert!(set.contains(&nonnullable_map));

        Ok(())
    }

    #[test]
    fn direct_null_map_cast_returns_error() -> VortexResult<()> {
        let target_dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            false,
            Nullability::NonNullable,
        )?;
        let scalar = Scalar::null(dtype()?);

        assert!(scalar.as_map().cast(&target_dtype).is_err());

        Ok(())
    }
}

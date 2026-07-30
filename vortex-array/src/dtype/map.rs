// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;
use crate::dtype::FieldDType;
use crate::dtype::Nullability;

/// Logical type information for a map's entries.
///
/// A map has ordered key/value entries. Keys must be non-nullable, while values may be nullable.
/// `keys_sorted` is a producer assertion matching Arrow's map type; it is not validated against
/// data at type construction time.
#[allow(
    clippy::derived_hash_with_manual_eq,
    reason = "manual PartialEq adds Arc::ptr_eq fast path only"
)]
#[derive(Clone, Eq, Hash)]
pub struct MapDType(Arc<MapDTypeInner>);

impl PartialEq for MapDType {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl fmt::Debug for MapDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapDType")
            .field("key", &self.0.key)
            .field("value", &self.0.value)
            .field("keys_sorted", &self.0.keys_sorted)
            .finish()
    }
}

impl fmt::Display for MapDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "map({}, {}, keys_sorted={})",
            self.key_dtype(),
            self.value_dtype(),
            self.keys_sorted()
        )
    }
}

struct MapDTypeInner {
    key: FieldDType,
    value: FieldDType,
    keys_sorted: bool,
}

impl PartialEq for MapDTypeInner {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.value == other.value && self.keys_sorted == other.keys_sorted
    }
}

impl Eq for MapDTypeInner {}

impl Hash for MapDTypeInner {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
        self.value.hash(state);
        self.keys_sorted.hash(state);
    }
}

impl MapDType {
    pub(crate) fn eq_ignore_nullability(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.key_dtype().eq_ignore_nullability(&other.key_dtype())
                && self
                    .value_dtype()
                    .eq_ignore_nullability(&other.value_dtype())
                && self.keys_sorted() == other.keys_sorted())
    }

    pub(crate) fn hash_ignore_nullability<H: Hasher>(&self, state: &mut H) {
        self.key_dtype().hash_ignore_nullability(state);
        self.value_dtype().hash_ignore_nullability(state);
        self.keys_sorted().hash(state);
    }

    /// Creates a map dtype from its key and value dtypes.
    ///
    /// # Errors
    ///
    /// Returns an error when `key` is nullable. Arrow map keys cannot be null.
    pub fn try_new(key: DType, value: DType, keys_sorted: bool) -> VortexResult<Self> {
        Self::try_from_fields(key.into(), value.into(), keys_sorted)
    }

    pub(crate) fn try_from_fields(
        key: FieldDType,
        value: FieldDType,
        keys_sorted: bool,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            !key.value()?.is_nullable(),
            "map key dtype must be non-nullable"
        );

        Ok(Self(Arc::new(MapDTypeInner {
            key,
            value,
            keys_sorted,
        })))
    }

    /// Returns the dtype of the map keys.
    pub fn key_dtype(&self) -> DType {
        self.0
            .key
            .value()
            .vortex_expect("map key dtype must be valid")
    }

    /// Returns the dtype of the map values.
    pub fn value_dtype(&self) -> DType {
        self.0
            .value
            .value()
            .vortex_expect("map value dtype must be valid")
    }

    /// Returns whether producers assert that keys are sorted within every map value.
    pub fn keys_sorted(&self) -> bool {
        self.0.keys_sorted
    }

    /// Returns the non-nullable `{key, value}` struct dtype used for map entries.
    pub fn entries_dtype(&self) -> DType {
        DType::struct_(
            [("key", self.key_dtype()), ("value", self.value_dtype())],
            Nullability::NonNullable,
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::MapDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    #[test]
    fn rejects_nullable_keys() {
        let result = MapDType::try_new(
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Utf8(Nullability::Nullable),
            false,
        );

        assert!(result.is_err());
    }

    #[test]
    fn accepts_nullable_values() -> VortexResult<()> {
        let dtype = MapDType::try_new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            true,
        )?;

        assert_eq!(
            dtype.entries_dtype(),
            DType::struct_(
                [
                    (
                        "key",
                        DType::Primitive(PType::I32, Nullability::NonNullable)
                    ),
                    ("value", DType::Utf8(Nullability::Nullable)),
                ],
                Nullability::NonNullable,
            )
        );
        assert!(dtype.keys_sorted());
        assert_eq!(dtype.to_string(), "map(i32, utf8?, keys_sorted=true)");

        Ok(())
    }

    #[test]
    fn outer_nullability_is_independent_of_map_details() -> VortexResult<()> {
        let dtype = DType::map(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            true,
            Nullability::NonNullable,
        )?;
        let nullable = dtype.as_nullable();

        assert!(dtype.is_map());
        assert!(!dtype.is_nullable());
        assert!(nullable.is_nullable());
        assert!(dtype.eq_ignore_nullability(&nullable));
        assert_eq!(dtype.as_map_opt(), nullable.as_map_opt());
        assert!(nullable.as_map_opt().unwrap().keys_sorted());

        Ok(())
    }
}

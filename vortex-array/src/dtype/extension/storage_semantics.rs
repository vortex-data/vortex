// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The extension types whose storage gives the equality and the order of their values.
//!
//! Extension types are opaque by default, but a few types in this repository take their equality,
//! and most also their order, from their storage. For example, two timestamps of one dtype are
//! equal exactly when their `i64` storage values are equal, and they order the same way. Extension
//! types cannot declare this themselves yet, so this module lists them until
//! <https://github.com/vortex-data/vortex/issues/11115> lets them. The lists name types by
//! extension ID, because `Uuid`, `Vector`, and `FixedShapeTensor` live in crates that depend on
//! this one. A type outside this repository cannot be listed.
//!
//! Other code answers the same question with a list of its own, such as `register_storage_ordered`,
//! the `Uuid` check of the DuckDB aggregate pushdown, and the temporal check of the DataFusion
//! statistics.

use vortex_session::registry::CachedId;

use crate::dtype::DType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ForeignExtDType;

/// The extension types whose storage gives the order of their values, and so also their equality.
static ORDER_IDS: [CachedId; 4] = [
    CachedId::new("vortex.date"),
    CachedId::new("vortex.time"),
    CachedId::new("vortex.timestamp"),
    CachedId::new("vortex.uuid"),
];

/// The extension types whose storage gives the equality of their values, but not their order.
static EQUALITY_ONLY_IDS: [CachedId; 2] = [
    CachedId::new("vortex.tensor.vector"),
    CachedId::new("vortex.tensor.fixed_shape_tensor"),
];

/// Returns whether values of `dtype` are equal exactly when their storage values are equal.
///
/// The answer ignores nullability and covers every depth:
///
/// - It is `true` for [`DType::Null`], [`DType::Bool`], [`DType::Primitive`], [`DType::Decimal`],
///   [`DType::Utf8`], and [`DType::Binary`].
/// - It is `true` for a struct, list, fixed-size list, or map when it is `true` for every child.
/// - It is `false` for [`DType::Union`] and [`DType::Variant`].
/// - It is `true` for an extension dtype when its type is [`Date`], [`Time`], [`Timestamp`],
///   `Uuid`, `Vector`, or `FixedShapeTensor`, whatever its storage dtype. It is `false` for a
///   placeholder of an extension type that the session does not know, even under a listed ID.
///
/// [`Date`]: crate::extension::datetime::Date
/// [`Time`]: crate::extension::datetime::Time
/// [`Timestamp`]: crate::extension::datetime::Timestamp
pub fn storage_gives_equality(dtype: &DType) -> bool {
    holds_at_every_depth(dtype, |id| {
        ORDER_IDS
            .iter()
            .chain(&EQUALITY_ONLY_IDS)
            .any(|listed| **listed == id)
    })
}

/// Returns whether values of `dtype` order like their storage values.
///
/// The answer follows the rules of [`storage_gives_equality`], except that `Vector` and
/// `FixedShapeTensor` are not listed, because their values have no natural order. So a dtype whose
/// storage gives its order also gives its equality.
pub fn storage_gives_order(dtype: &DType) -> bool {
    holds_at_every_depth(dtype, |id| ORDER_IDS.iter().any(|listed| **listed == id))
}

/// Applies the rules of [`storage_gives_equality`], where `is_listed` names the extension types
/// that hold the property.
fn holds_at_every_depth(dtype: &DType, is_listed: fn(ExtId) -> bool) -> bool {
    match dtype {
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        DType::List(element, _) | DType::FixedSizeList(element, ..) => {
            holds_at_every_depth(element, is_listed)
        }
        DType::Map(map, _) => {
            holds_at_every_depth(&map.key_dtype(), is_listed)
                && holds_at_every_depth(&map.value_dtype(), is_listed)
        }
        DType::Struct(fields, _) => fields
            .field_dtypes()
            .all(|field| holds_at_every_depth(field, is_listed)),
        DType::Union(..) | DType::Variant(_) => false,
        // A placeholder keeps the ID of the type that it stands in for, but not its semantics.
        DType::Extension(ext) => !ext.is::<ForeignExtDType>() && is_listed(ext.id()),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::storage_gives_equality;
    use super::storage_gives_order;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::UnionVariants;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtVTable;
    use crate::dtype::extension::ForeignExtDType;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::extension::tests::divisible_int::DivisibleInt;
    use crate::extension::tests::divisible_int::Divisor;

    fn timestamp() -> DType {
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullable).erased())
    }

    /// An extension dtype that is on neither list, although its `u64` storage gives both.
    fn unlisted() -> VortexResult<DType> {
        let dtype = ExtDType::<DivisibleInt>::try_new(Divisor(2), PType::U64.into())?;

        Ok(DType::Extension(dtype.erased()))
    }

    /// A date dtype as read by a session that does not know dates.
    fn date_placeholder() -> VortexResult<DType> {
        let metadata = Date.serialize_metadata(&TimeUnit::Days)?;
        let dtype = ForeignExtDType::from_parts(Date.id(), metadata, PType::I32.into())?;

        Ok(DType::Extension(dtype))
    }

    /// A struct of a timestamp and `field`.
    fn struct_with(field: DType) -> DType {
        DType::struct_([("at", timestamp()), ("field", field)], NonNullable)
    }

    /// A union whose variants both give equality and order.
    fn union() -> VortexResult<DType> {
        let dtypes = vec![timestamp(), PType::I64.into()];
        let variants = UnionVariants::new(["at", "count"].into(), dtypes)?;

        Ok(DType::Union(variants, Nullable))
    }

    #[rstest]
    #[case::date(Ok(DType::Extension(Date::new(TimeUnit::Days, NonNullable).erased())), true)]
    #[case::time(Ok(DType::Extension(Time::new(TimeUnit::Seconds, Nullable).erased())), true)]
    #[case::timestamp(Ok(timestamp()), true)]
    #[case::unlisted(unlisted(), false)]
    #[case::date_placeholder(date_placeholder(), false)]
    #[case::list(Ok(DType::list(timestamp(), Nullable)), true)]
    #[case::list_of_placeholder(
        date_placeholder().map(|dtype| DType::list(dtype, Nullable)),
        false
    )]
    #[case::map(
        DType::map(DType::Utf8(NonNullable), timestamp(), false, Nullable),
        true
    )]
    #[case::map_of_variant(
        DType::map(DType::Utf8(NonNullable), DType::Variant(Nullable), false, Nullable),
        false
    )]
    #[case::map_with_placeholder_key(
        date_placeholder().and_then(|key| DType::map(key, timestamp(), false, Nullable)),
        false
    )]
    #[case::struct_(Ok(struct_with(PType::I64.into())), true)]
    #[case::struct_with_variant(Ok(struct_with(DType::Variant(Nullable))), false)]
    #[case::union(union(), false)]
    fn test_storage_gives_equality_and_order(
        #[case] dtype: VortexResult<DType>,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let dtype = dtype?;

        assert_eq!(storage_gives_equality(&dtype), expected);
        assert_eq!(storage_gives_order(&dtype), expected);

        Ok(())
    }
}

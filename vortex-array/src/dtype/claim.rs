// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Claims that allow operations on a dtype to use its storage semantics.

use std::fmt;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::dtype::DType;

/// A promise about the relationship between values of one dtype and their storage values.
///
/// Storage values compare the way Vortex compute compares their dtype. Floats use a total order
/// and bitwise equality, so `-0.0` sorts before `0.0` and NaNs with the same bits are equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(enum_iterator::Sequence))]
#[non_exhaustive]
pub enum Claim {
    /// Two values are equal exactly when their storage values are equal, so hashing can use storage.
    Eq,
    /// Values have the same order as their storage values. This implies [`Self::Eq`].
    Ord,
}

impl Claim {
    /// Every claim, each before the claims that it implies, so that a failed check names the claim
    /// that a type declared.
    pub(crate) const ALL: [Self; 2] = [Self::Ord, Self::Eq];
}

impl fmt::Display for Claim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Eq => "Eq",
            Self::Ord => "Ord",
        })
    }
}

/// The claims that an extension type declares for one metadata value.
///
/// Use [`DType::holds`] to query a claim. `holds(Claim::Ord)` always means that the storage
/// order is the value order. File statistics and engine pushdown rely on this guarantee.
// Keep the derives minimal so the fields can carry more than a yes or no in the future.
#[derive(Clone, Debug)]
pub struct Claims {
    eq: bool,
    ord: bool,
}

impl Claims {
    /// Declares no claims, so the type is opaque to operations that require a claim.
    pub fn none() -> Self {
        Self {
            eq: false,
            ord: false,
        }
    }

    /// Adds `claim` and every claim that it implies. Adding [`Claim::Ord`] also adds [`Claim::Eq`].
    #[must_use]
    pub fn with(mut self, claim: Claim) -> Self {
        match claim {
            Claim::Eq => self.eq = true,
            Claim::Ord => {
                self.eq = true;
                self.ord = true;
            }
        }

        self
    }

    pub(crate) fn holds(&self, claim: Claim) -> bool {
        match claim {
            Claim::Eq => self.eq,
            Claim::Ord => self.ord,
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = Claim> + '_ {
        Claim::ALL.into_iter().filter(|&claim| self.holds(claim))
    }
}

impl DType {
    /// Returns whether this dtype holds `claim`, regardless of its top-level nullability.
    ///
    /// Structs, lists, fixed-size lists, and maps require the claim on every child dtype.
    /// An extension dtype holds the claims it declares and any claims that they imply.
    pub fn holds(&self, claim: Claim) -> bool {
        match self {
            Self::Null
            | Self::Bool(_)
            | Self::Primitive(..)
            | Self::Decimal(..)
            | Self::Utf8(_)
            | Self::Binary(_) => match claim {
                Claim::Eq | Claim::Ord => true,
            },
            Self::List(element, _) | Self::FixedSizeList(element, ..) => element.holds(claim),
            Self::Map(map, _) => map.key_dtype().holds(claim) && map.value_dtype().holds(claim),
            Self::Struct(fields, _) => fields.field_dtypes().all(|dtype| dtype.holds(claim)),
            Self::Union(..) | Self::Variant(_) => false,
            Self::Extension(ext) => ext.claims().holds(claim),
        }
    }

    /// Returns an error unless this dtype holds `claim`.
    ///
    /// # Errors
    ///
    /// Returns an error naming `function`, this dtype, and `claim` when the claim is missing.
    pub fn require_claim(&self, claim: Claim, function: impl fmt::Display) -> VortexResult<()> {
        vortex_ensure!(
            self.holds(claim),
            "{function} requires the {claim} claim, but {self} does not hold it"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::DeserializeMetadata;
    use crate::EmptyMetadata;
    use crate::dtype::Claim;
    use crate::dtype::Claims;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::dtype::UnionVariants;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtDTypePlugin;
    use crate::dtype::extension::ExtId;
    use crate::dtype::extension::ExtVTable;
    use crate::extension::tests::DivisibleInt;
    use crate::extension::tests::Divisor;
    use crate::extension::tests::Opaque;
    use crate::scalar::ScalarValue;

    fn i64_dtype() -> DType {
        DType::Primitive(PType::I64, NonNullable)
    }

    fn opaque_dtype() -> VortexResult<DType> {
        Ok(DType::Extension(
            ExtDType::<Opaque>::try_new(EmptyMetadata, NonNullable)?.erased(),
        ))
    }

    fn divisible_int_dtype() -> VortexResult<DType> {
        Ok(DType::Extension(
            ExtDType::<DivisibleInt>::try_new(Divisor(3), NonNullable)?.erased(),
        ))
    }

    /// Declares equality even when the supplied storage lacks it.
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct ClaimsEq;

    impl ExtVTable for ClaimsEq {
        type Metadata = DType;
        type StoredMetadata = EmptyMetadata;
        type NativeValue<'a> = &'a ScalarValue;

        #[expect(clippy::disallowed_methods, reason = "test-only id")]
        fn id(&self) -> ExtId {
            ExtId::new("test.claims_eq")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::StoredMetadata> {
            EmptyMetadata::deserialize(bytes)
        }

        fn metadata_from_storage(
            &self,
            _stored_metadata: Self::StoredMetadata,
            storage_dtype: &DType,
        ) -> VortexResult<Self::Metadata> {
            Ok(storage_dtype.clone())
        }

        fn storage_dtype(metadata: &Self::Metadata) -> DType {
            metadata.clone()
        }

        fn claims(_metadata: &Self::Metadata) -> Claims {
            Claims::none().with(Claim::Eq)
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok(storage_value)
        }
    }

    #[rstest]
    #[case::i64(i64_dtype(), true)]
    #[case::f64(DType::Primitive(PType::F64, NonNullable), true)]
    #[case::bool(DType::Bool(NonNullable), true)]
    #[case::utf8(DType::Utf8(NonNullable), true)]
    #[case::binary(DType::Binary(NonNullable), true)]
    #[case::decimal(DType::Decimal(DecimalDType::new(8, 2), NonNullable), true)]
    #[case::null(DType::Null, true)]
    #[case::variant(DType::Variant(NonNullable), false)]
    #[case::union(DType::Union(UnionVariants::new(["a"].into(), vec![i64_dtype()]).unwrap(), NonNullable), false)]
    #[case::struct_i64(DType::struct_([("a", i64_dtype())], NonNullable), true)]
    #[case::list_i64(DType::list(i64_dtype(), NonNullable), true)]
    #[case::fixed_size_list_i64(DType::FixedSizeList(i64_dtype().into(), 2, NonNullable), true)]
    #[case::struct_variant(DType::struct_([("a", DType::Variant(NonNullable))], NonNullable), false)]
    fn builtin_dtypes_hold_claims(
        #[case] dtype: DType,
        #[case] expected: bool,
        #[values(Claim::Eq, Claim::Ord)] claim: Claim,
    ) {
        assert_eq!(dtype.holds(claim), expected);
    }

    #[rstest]
    #[case::ordered_entries(i64_dtype(), i64_dtype(), true)]
    #[case::opaque_key(DType::Variant(NonNullable), i64_dtype(), false)]
    #[case::opaque_value(i64_dtype(), DType::Variant(NonNullable), false)]
    fn maps_hold_claims_of_both_children(
        #[case] key: DType,
        #[case] value: DType,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let dtype = DType::map(key, value, false, NonNullable)?;

        assert_eq!(dtype.holds(Claim::Eq), expected);
        assert_eq!(dtype.holds(Claim::Ord), expected);

        Ok(())
    }

    #[rstest]
    #[case::none(Claims::none(), false, false)]
    #[case::eq(Claims::none().with(Claim::Eq), true, false)]
    #[case::ord(Claims::none().with(Claim::Ord), true, true)]
    #[case::eq_then_ord(Claims::none().with(Claim::Eq).with(Claim::Ord), true, true)]
    #[case::ord_then_eq(Claims::none().with(Claim::Ord).with(Claim::Eq), true, true)]
    fn claims_builder(#[case] claims: Claims, #[case] eq: bool, #[case] ord: bool) {
        assert_eq!(claims.holds(Claim::Eq), eq);
        assert_eq!(claims.holds(Claim::Ord), ord);
    }

    #[test]
    fn extension_holds_declared_and_implied_claims() -> VortexResult<()> {
        let divisible_int = divisible_int_dtype()?;
        let opaque = opaque_dtype()?;
        let opaque_list = DType::list(opaque.clone(), NonNullable);
        let divisible_int_struct = DType::struct_([("a", divisible_int.clone())], NonNullable);

        assert!(divisible_int.holds(Claim::Ord));
        assert!(divisible_int.holds(Claim::Eq));
        assert!(!opaque.holds(Claim::Ord));
        assert!(!opaque.holds(Claim::Eq));
        assert!(!opaque_list.holds(Claim::Ord));
        assert!(!opaque_list.holds(Claim::Eq));
        assert!(divisible_int_struct.holds(Claim::Ord));

        Ok(())
    }

    #[rstest]
    #[case::divisible_int(divisible_int_dtype())]
    #[case::opaque(opaque_dtype())]
    #[case::opaque_list(opaque_dtype().map(|opaque| DType::list(opaque, NonNullable)))]
    #[case::divisible_int_struct(divisible_int_dtype().map(|dtype| DType::struct_([("a", dtype)], NonNullable)))]
    fn claims_ignore_top_level_nullability(#[case] dtype: VortexResult<DType>) -> VortexResult<()> {
        let dtype = dtype?;
        let nullable = dtype.with_nullability(Nullability::Nullable);

        assert_eq!(dtype.holds(Claim::Eq), nullable.holds(Claim::Eq));
        assert_eq!(dtype.holds(Claim::Ord), nullable.holds(Claim::Ord));

        Ok(())
    }

    #[test]
    fn try_new_rejects_a_claim_that_storage_lacks() -> VortexResult<()> {
        let storage = DType::list(opaque_dtype()?, NonNullable);
        let from_metadata = ExtDType::<ClaimsEq>::try_new(storage.clone(), NonNullable)
            .expect_err("List<Opaque> cannot support Eq");
        let from_storage = ExtDTypePlugin::deserialize(&ClaimsEq, &[], storage.clone())
            .expect_err("List<Opaque> cannot support Eq when read");

        for error in [from_metadata, from_storage] {
            let message = error.to_string();
            assert!(message.contains("test.claims_eq"), "{message}");
            assert!(message.contains("Eq"), "{message}");
            assert!(message.contains(&storage.to_string()), "{message}");
        }

        Ok(())
    }

    #[test]
    fn require_claim_error_names_function_dtype_and_claim() -> VortexResult<()> {
        let dtype = opaque_dtype()?;
        let error = dtype
            .require_claim(Claim::Ord, "min")
            .expect_err("Opaque does not hold Ord");
        let expected = format!("min requires the Ord claim, but {dtype} does not hold it");

        assert_eq!(error.to_string().lines().next(), Some(expected.as_str()));

        Ok(())
    }

    #[test]
    fn all_lists_every_claim() {
        for claim in enum_iterator::all::<Claim>() {
            assert!(Claim::ALL.contains(&claim), "Claim::ALL is missing {claim}");
        }
    }
}

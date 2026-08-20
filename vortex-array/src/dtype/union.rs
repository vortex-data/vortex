// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::dtype::DType;
use crate::dtype::FieldDType;
use crate::dtype::FieldNames;

/// Type information for a union array.
///
/// A `UnionVariants` describes the possible alternative types for each row of a union, along with a
/// per-variant `u8` type tag. A union must contain at least one variant. We use the term
/// **variants** (rather than "fields") because a union is a sum type: each row chooses exactly one
/// alternative.
///
/// By default, tag `i` selects the child at offset `i` (`type_ids = [0, 1, ..., N-1]`). Vortex uses
/// unsigned tags and supports up to 256 variants.
///
/// Schemas may also use non-consecutive tags (e.g. `[0, 5, 7]`), in which case the value of
/// `type_ids[i]` is the tag used in the data to select the child at offset `i`. Supporting
/// non-consecutive tags lets the schema remove children without renumbering the remaining tags.
///
/// Variant names must be distinct. Unlike [`StructFields`](crate::dtype::StructFields), which
/// permits duplicate field names for Arrow/Parquet round-trip fidelity, duplicates have no
/// meaningful semantics in a sum type and are rejected at construction.
///
/// Note that the Arrow spec does not require distinct names (children are selected by the type-id
/// buffer, never by name), but DuckDB requires distinct member names for its `UNION` type and
/// resolves members by name (e.g. in `union_extract`), so permitting duplicates would break DuckDB
/// round-trips while enabling no useful semantics.
///
/// ```
/// use vortex_array::dtype::{DType, Nullability, PType, UnionVariants};
///
/// let variants = UnionVariants::new(
///     ["a", "b"].into(),
///     vec![
///         DType::Primitive(PType::I32, Nullability::NonNullable),
///         DType::Utf8(Nullability::NonNullable),
///     ],
/// )
/// .unwrap();
///
/// assert_eq!(variants.len(), 2);
/// assert_eq!(variants.type_ids(), &[0, 1]);
/// ```
#[allow(
    clippy::derived_hash_with_manual_eq,
    reason = "manual PartialEq adds Arc::ptr_eq fast path only"
)]
#[derive(Clone, Eq, Hash)]
pub struct UnionVariants(Arc<UnionVariantsInner>);

impl PartialEq for UnionVariants {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl fmt::Debug for UnionVariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnionVariants")
            .field("names", &self.0.names)
            .field("dtypes", &self.0.dtypes)
            .field("type_ids", &self.0.type_ids)
            .finish()
    }
}

impl fmt::Display for UnionVariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Surface non-consecutive type tags so they aren't hidden by the format. Consecutive
        // `[0, 1, ..., N-1]` is the common case and is left implicit.
        let show_tags = !self.is_consecutive();
        write!(
            f,
            "{}",
            self.names()
                .iter()
                .zip(self.variants())
                .zip(self.type_ids().iter())
                .map(|((name, dt), tag)| {
                    if show_tags {
                        format!("{name}@{tag}={dt}")
                    } else {
                        format!("{name}={dt}")
                    }
                })
                .join(", ")
        )
    }
}

struct UnionVariantsInner {
    /// The names of the variants. This is called `FieldNames` because it is shared with the
    /// [`StructFields`](crate::dtype::StructFields) implementation.
    names: FieldNames,

    /// The types of each of the variants.
    dtypes: Arc<[FieldDType]>,

    /// One tag per variant, in variant order. The common case where children are referenced by
    /// consecutive offsets is `[0, 1, ..., N-1]`.
    ///
    /// For schemas with explicit `typeIds` indirection (e.g. `[0, 5, 7]`), this stores those tags.
    type_ids: Arc<[u8]>,
}

impl UnionVariantsInner {
    fn from_fields(names: FieldNames, dtypes: Arc<[FieldDType]>, type_ids: Arc<[u8]>) -> Self {
        Self {
            names,
            dtypes,
            type_ids,
        }
    }
}

impl PartialEq for UnionVariantsInner {
    fn eq(&self, other: &Self) -> bool {
        self.names == other.names && self.dtypes == other.dtypes && self.type_ids == other.type_ids
    }
}

impl Eq for UnionVariantsInner {}

impl Hash for UnionVariantsInner {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.names.hash(state);
        self.dtypes.hash(state);
        self.type_ids.hash(state);
    }
}

impl UnionVariants {
    /// Check if these union variants are equal, ignoring variant dtype nullability recursively.
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.names == other.0.names
                && self
                    .variants()
                    .zip_eq(other.variants())
                    .all(|(lhs, rhs)| lhs.eq_ignore_nullability(&rhs))
                && self.0.type_ids == other.0.type_ids)
    }

    /// Hash these union variants using the same equivalence relation as
    /// [`Self::eq_ignore_nullability`].
    pub(crate) fn hash_ignore_nullability<H: Hasher>(&self, state: &mut H) {
        self.0.names.hash(state);
        for variant in self.variants() {
            variant.hash_ignore_nullability(state);
        }
        self.0.type_ids.hash(state);
    }
}

impl UnionVariants {
    /// Validate that `names`, `dtypes`, and `type_ids` are mutually consistent.
    fn validate_shape(names: &FieldNames, n_dtypes: usize, type_ids: &[u8]) -> VortexResult<()> {
        vortex_ensure_eq!(
            names.len(),
            n_dtypes,
            "length mismatch between names ({}) and dtypes ({})",
            names.len(),
            n_dtypes
        );
        vortex_ensure_eq!(
            names.len(),
            type_ids.len(),
            "length mismatch between names ({}) and type_ids ({})",
            names.len(),
            type_ids.len()
        );
        vortex_ensure!(
            !names.is_empty(),
            "union must have at least one variant (for now)"
        );

        vortex_ensure!(
            type_ids.iter().all_unique(),
            "type_ids must be distinct, got {:?}",
            type_ids
        );
        vortex_ensure!(
            type_ids.len() <= u8::MAX as usize + 1,
            "union supports at most {} variants, got {}",
            u8::MAX as usize + 1,
            type_ids.len()
        );
        vortex_ensure!(
            names.iter().all_unique(),
            "union variant names must be distinct, got {:?}",
            names
        );

        Ok(())
    }

    /// Create a new [`UnionVariants`] with explicit `type_ids`.
    ///
    /// # Errors
    ///
    /// Returns an error if the union has no variants, if names, dtypes, or type IDs do not all have
    /// the same length, if there are any duplicate names or type IDs, or if there are more than 256
    /// variants.
    pub fn try_new(names: FieldNames, dtypes: Vec<DType>, type_ids: Vec<u8>) -> VortexResult<Self> {
        Self::validate_shape(&names, dtypes.len(), &type_ids)?;

        let dtypes: Arc<[FieldDType]> = dtypes.into_iter().map(FieldDType::from).collect();
        let type_ids: Arc<[u8]> = Arc::from(type_ids);

        Ok(Self(Arc::new(UnionVariantsInner::from_fields(
            names, dtypes, type_ids,
        ))))
    }

    /// Create a new [`UnionVariants`] with consecutive `type_ids = [0, 1, ..., N-1]`.
    ///
    /// # Errors
    ///
    /// The union must have at least one variant, `names` and `dtypes` must have the same length, and
    /// `names.len()` cannot be more than `u8::MAX as usize + 1` (256).
    pub fn new(names: FieldNames, dtypes: Vec<DType>) -> VortexResult<Self> {
        const MAX_VARIANTS: usize = u8::MAX as usize + 1;
        vortex_ensure!(
            names.len() <= MAX_VARIANTS,
            "union supports at most {} consecutive variants, got {}",
            MAX_VARIANTS,
            names.len()
        );

        #[expect(
            clippy::cast_possible_truncation,
            reason = "the MAX_VARIANTS bound above guarantees `i as u8` is in range"
        )]
        let type_ids: Vec<u8> = (0..names.len()).map(|i| i as u8).collect();

        Self::try_new(names, dtypes, type_ids)
    }

    /// Create a new [`UnionVariants`] from pre-constructed [`FieldDType`]s, which may be owned or
    /// backed by a flatbuffer view.
    ///
    /// Used by deserialization paths where the children may be lazily backed by a flatbuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the union has no variants, if names, dtypes, or type IDs do not all have
    /// the same length, if there are any duplicate names or type IDs, or if there are more than 256
    /// variants.
    pub(crate) fn try_from_fields(
        names: FieldNames,
        dtypes: Vec<FieldDType>,
        type_ids: Vec<u8>,
    ) -> VortexResult<Self> {
        Self::validate_shape(&names, dtypes.len(), &type_ids)?;

        Ok(Self(Arc::new(UnionVariantsInner::from_fields(
            names,
            dtypes.into(),
            Arc::from(type_ids),
        ))))
    }

    /// Get the names of the variants in the union.
    pub fn names(&self) -> &FieldNames {
        &self.0.names
    }

    /// Returns the number of variants in the union.
    pub fn len(&self) -> usize {
        self.0.names.len()
    }

    /// Returns true if the union has no variants.
    pub fn is_empty(&self) -> bool {
        self.0.names.is_empty()
    }

    /// Returns the per-variant type tag vector. Entry `i` is the tag that the data uses to
    /// select the variant at offset `i`.
    pub fn type_ids(&self) -> &[u8] {
        &self.0.type_ids
    }

    /// Returns `true` if the type tags are the consecutive sequence `[0, 1, ..., N-1]`.
    pub fn is_consecutive(&self) -> bool {
        self.0
            .type_ids
            .iter()
            .enumerate()
            .all(|(i, &tag)| u8::try_from(i).is_ok_and(|i| i == tag))
    }

    /// Find the offset of a variant by name. Returns `None` if no variant has the name.
    ///
    /// This is a linear scan over [`Self::names`]. A union has at most 256 variants (and usually
    /// far fewer), so a linear scan beats building and probing a `HashMap` in practice.
    pub fn find(&self, name: impl AsRef<str>) -> Option<usize> {
        let name = name.as_ref();
        self.0.names.iter().position(|n| n.as_ref() == name)
    }

    /// Get the [`DType`] of a variant by name. Returns `None` if no variant has the name.
    pub fn variant(&self, name: impl AsRef<str>) -> Option<DType> {
        let index = self.find(name)?;
        Some(
            self.0.dtypes[index]
                .value()
                .vortex_expect("variant DType must be valid"),
        )
    }

    /// Get the [`DType`] of a variant by offset, or [`None`] if `index` is out of bounds.
    pub fn variant_by_index(&self, index: usize) -> Option<DType> {
        Some(
            self.0
                .dtypes
                .get(index)?
                .value()
                .vortex_expect("variant DType must be valid"),
        )
    }

    /// Returns an ordered iterator over the variants.
    pub fn variants(&self) -> impl ExactSizeIterator<Item = DType> + '_ {
        self.0
            .dtypes
            .iter()
            .map(|dt| dt.value().vortex_expect("variant DType must be valid"))
    }

    /// Convert a data-level type tag to a child offset. Returns `None` if the tag is not in
    /// `type_ids`.
    ///
    /// This is a linear scan over [`Self::type_ids`]. The number of variants is bounded by
    /// 256, so a linear scan is faster than a `HashMap` lookup in practice for the
    /// small unions typically encountered.
    pub fn tag_to_child_index(&self, tag: u8) -> Option<usize> {
        self.0.type_ids.iter().position(|&t| t == tag)
    }

    /// Convert a child offset to its data-level type tag.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    pub fn child_index_to_tag(&self, index: usize) -> u8 {
        self.0.type_ids[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::UnionVariants;

    fn i32_variants() -> UnionVariants {
        UnionVariants::new(
            ["int", "str"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_consecutive_type_ids() {
        let variants = i32_variants();
        assert_eq!(variants.type_ids(), &[0, 1]);
        assert!(variants.is_consecutive());
        assert_eq!(variants.tag_to_child_index(0), Some(0));
        assert_eq!(variants.tag_to_child_index(1), Some(1));
        assert_eq!(variants.tag_to_child_index(2), None);
        assert_eq!(variants.child_index_to_tag(0), 0);
        assert_eq!(variants.child_index_to_tag(1), 1);
    }

    #[test]
    fn test_type_id_indirection() {
        let variants = UnionVariants::try_new(
            ["a", "b", "c"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
                DType::Bool(Nullability::NonNullable),
            ],
            vec![0, 5, 7],
        )
        .unwrap();

        assert_eq!(variants.type_ids(), &[0, 5, 7]);
        assert!(!variants.is_consecutive());
        assert_eq!(variants.tag_to_child_index(0), Some(0));
        assert_eq!(variants.tag_to_child_index(5), Some(1));
        assert_eq!(variants.tag_to_child_index(7), Some(2));
        assert_eq!(variants.tag_to_child_index(1), None);
    }

    #[test]
    fn test_find() {
        let variants = i32_variants();
        assert_eq!(variants.find("int"), Some(0));
        assert_eq!(variants.find("str"), Some(1));
        assert!(variants.find("missing").is_none());

        let value = variants.variant("int").unwrap();
        assert_eq!(
            value,
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_duplicate_names_rejected() {
        let result = UnionVariants::new(
            ["dup", "dup"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Primitive(PType::I64, Nullability::NonNullable),
            ],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("distinct"));
    }

    #[test]
    fn test_high_type_id() {
        let variants = UnionVariants::try_new(
            ["high"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![u8::MAX],
        )
        .unwrap();
        assert_eq!(variants.type_ids(), &[u8::MAX]);
    }

    #[test]
    fn test_length_mismatch_rejected() {
        let result = UnionVariants::try_new(
            ["a", "b"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![0, 1],
        );
        assert!(result.is_err());

        let result = UnionVariants::try_new(
            ["a"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![0, 1],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_type_ids_rejected() {
        let result = UnionVariants::try_new(
            ["a", "b"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
            vec![3, 3],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_union_nullability_is_independent() {
        let inner = DType::Union(
            UnionVariants::new(["value"].into(), vec![DType::Utf8(Nullability::Nullable)]).unwrap(),
            Nullability::NonNullable,
        );
        let outer = DType::Union(
            UnionVariants::new(
                ["nested", "number"].into(),
                vec![
                    inner,
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                ],
            )
            .unwrap(),
            Nullability::Nullable,
        );

        assert_eq!(outer.nullability(), Nullability::Nullable);
        let variants = outer.as_union_variants_opt().unwrap();
        assert_eq!(
            variants.variant_by_index(0).unwrap().nullability(),
            Nullability::NonNullable
        );
    }

    #[test]
    fn test_with_nullability_changes_only_outer_union() {
        let nonnullable = DType::Union(i32_variants(), Nullability::NonNullable);
        assert_eq!(nonnullable.nullability(), Nullability::NonNullable);

        let nullable = nonnullable.as_nullable();
        assert_eq!(nullable.nullability(), Nullability::Nullable);
        assert_eq!(
            nullable.as_union_variants_opt(),
            nonnullable.as_union_variants_opt()
        );
        assert_eq!(nullable.as_nonnullable(), nonnullable);
    }

    #[test]
    fn test_display() {
        let variants = i32_variants();
        let dtype = DType::Union(variants, Nullability::NonNullable);
        assert_eq!(dtype.to_string(), "union(int=i32, str=utf8)");

        let nullable = DType::Union(
            UnionVariants::new(
                ["int", "maybe_str"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::Nullable),
                ],
            )
            .unwrap(),
            Nullability::Nullable,
        );
        assert_eq!(nullable.to_string(), "union(int=i32, maybe_str=utf8?)?");
        assert_eq!(
            nullable.as_nonnullable().to_string(),
            "union(int=i32, maybe_str=utf8?)"
        );
    }

    #[test]
    fn test_display_with_type_id_indirection() {
        let variants = UnionVariants::try_new(
            ["a", "b", "c"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
                DType::Bool(Nullability::NonNullable),
            ],
            vec![0, 5, 7],
        )
        .unwrap();
        let dtype = DType::Union(variants, Nullability::NonNullable);
        assert_eq!(dtype.to_string(), "union(a@0=i32, b@5=utf8, c@7=bool)");
    }

    #[test]
    fn test_new_max_size() {
        let names: Vec<String> = (0..256).map(|i| format!("v{i}")).collect();
        let dtypes: Vec<DType> = (0..256)
            .map(|_| DType::Primitive(PType::I32, Nullability::NonNullable))
            .collect();
        let names: FieldNames = names.into_iter().collect();
        let variants = UnionVariants::new(names, dtypes).unwrap();
        assert_eq!(variants.len(), 256);
        assert_eq!(variants.type_ids()[255], 255);
        assert!(variants.is_consecutive());
    }

    #[test]
    fn test_new_too_large_rejected() {
        let names: Vec<String> = (0..257).map(|i| format!("v{i}")).collect();
        let dtypes: Vec<DType> = (0..257)
            .map(|_| DType::Primitive(PType::I32, Nullability::NonNullable))
            .collect();
        let names: FieldNames = names.into_iter().collect();
        let result = UnionVariants::new(names, dtypes);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at most 256 consecutive variants")
        );
    }

    #[test]
    fn test_empty_rejected() {
        let error = UnionVariants::new(FieldNames::default(), vec![]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("union must have at least one variant")
        );

        let error = UnionVariants::try_new(FieldNames::default(), vec![], vec![]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("union must have at least one variant")
        );
    }
}

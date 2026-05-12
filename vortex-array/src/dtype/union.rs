// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::OnceLock;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::dtype::FieldDType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;

/// Type information for a union array.
///
/// A `UnionVariants` describes the possible alternative types for each row of a union, along with a
/// per-variant `i8` type tag. We use the term **variants** (rather than "fields") because a union
/// is a sum type: each row chooses exactly one alternative.
///
/// Per Arrow's spec, the per-row type tag is an `int8`. By default, tag `i` selects the child at
/// offset `i` (`type_ids = [0, 1, ..., N-1]`). Schemas may also use non-consecutive tags (e.g.
/// `[0, 5, 7]`), in which case the value of `type_ids[i]` is the tag used in the data to select the
/// child at offset `i`. Supporting non-consecutive tags from v1 lets the schema remove children
/// without renumbering the remaining tags.
///
/// Variant names must be distinct. Unlike [`StructFields`](crate::dtype::StructFields), which
/// permits duplicate field names for Arrow/Parquet round-trip fidelity, duplicates have no
/// meaningful semantics in a sum type and are rejected at construction.
///
/// ```
/// use vortex_array::dtype::{DType, Nullability, PType, UnionVariants};
///
/// let variants = UnionVariants::new_consecutive(
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
    type_ids: Arc<[i8]>,

    /// Derived from `names`, maps from variant name to index.
    /// This is excluded from the `PartialEq`, `Eq`, `Hash`, and serde serialization.
    indices: OnceLock<HashMap<FieldName, usize>>,
}

impl UnionVariantsInner {
    fn from_fields(names: FieldNames, dtypes: Arc<[FieldDType]>, type_ids: Arc<[i8]>) -> Self {
        Self {
            names,
            dtypes,
            type_ids,
            indices: OnceLock::new(),
        }
    }

    fn indices(&self) -> &HashMap<FieldName, usize> {
        self.indices.get_or_init(|| {
            // Uniqueness is enforced by `validate_shape`, so a plain `insert` is safe.
            let mut map = HashMap::with_capacity(self.names.len());
            for (idx, name) in self.names.iter().enumerate() {
                map.insert(name.clone(), idx);
            }
            map
        })
    }
}

impl PartialEq for UnionVariantsInner {
    fn eq(&self, other: &Self) -> bool {
        self.names == other.names && self.dtypes == other.dtypes && self.type_ids == other.type_ids
    }
}

impl Eq for UnionVariantsInner {}

impl Hash for UnionVariantsInner {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.names.hash(state);
        self.dtypes.hash(state);
        self.type_ids.hash(state);
    }
}

impl Default for UnionVariants {
    fn default() -> Self {
        Self::empty()
    }
}

impl UnionVariants {
    /// The variants of the empty union.
    pub fn empty() -> Self {
        Self(Arc::new(UnionVariantsInner::from_fields(
            FieldNames::default(),
            Arc::from([]),
            Arc::from([]),
        )))
    }

    /// Validate that `names`, `dtypes`, and `type_ids` are mutually consistent.
    fn validate_shape(names: &FieldNames, n_dtypes: usize, type_ids: &[i8]) -> VortexResult<()> {
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
            type_ids.iter().all_unique(),
            "type_ids must be distinct, got {:?}",
            type_ids
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
    /// Returns an error if names, dtypes, or type IDs do not all have the same length, or if there
    /// are any duplicate names or type ids.
    pub fn try_new(names: FieldNames, dtypes: Vec<DType>, type_ids: Vec<i8>) -> VortexResult<Self> {
        Self::validate_shape(&names, dtypes.len(), &type_ids)?;

        let dtypes: Arc<[FieldDType]> = dtypes.into_iter().map(FieldDType::from).collect();
        let type_ids: Arc<[i8]> = Arc::from(type_ids);

        Ok(Self(Arc::new(UnionVariantsInner::from_fields(
            names, dtypes, type_ids,
        ))))
    }

    /// Create a new [`UnionVariants`] with consecutive `type_ids = [0, 1, ..., N-1]`.
    ///
    /// # Errors
    ///
    /// `names` and `dtypes` must have the same length, and `names.len()` cannot be more than
    /// `i8::MAX as usize + 1` (128).
    pub fn new_consecutive(names: FieldNames, dtypes: Vec<DType>) -> VortexResult<Self> {
        const MAX_CONSECUTIVE: usize = i8::MAX as usize + 1;
        vortex_ensure!(
            names.len() <= MAX_CONSECUTIVE,
            "union supports at most {} consecutive variants, got {}",
            MAX_CONSECUTIVE,
            names.len()
        );

        #[expect(
            clippy::cast_possible_truncation,
            reason = "the MAX_CONSECUTIVE bound above guarantees `i as i8` is in range"
        )]
        let type_ids: Vec<i8> = (0..names.len()).map(|i| i as i8).collect();

        Self::try_new(names, dtypes, type_ids)
    }

    /// Create a new [`UnionVariants`] from pre-constructed [`FieldDType`]s, which may be owned or
    /// backed by a flatbuffer view.
    ///
    /// Used by deserialization paths where the children may be lazily backed by a flatbuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if names, dtypes, or type IDs do not all have the same length, or if there
    /// are any duplicate names or type ids.
    pub(crate) fn try_from_fields(
        names: FieldNames,
        dtypes: Vec<FieldDType>,
        type_ids: Vec<i8>,
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
    pub fn type_ids(&self) -> &[i8] {
        &self.0.type_ids
    }

    /// Returns `true` if the type tags are the consecutive sequence `[0, 1, ..., N-1]`.
    pub fn is_consecutive(&self) -> bool {
        self.0
            .type_ids
            .iter()
            .enumerate()
            .all(|(i, &tag)| i8::try_from(i).is_ok_and(|i| i == tag))
    }

    /// Find the offset of a variant by name. Returns `None` if no variant has the name.
    pub fn find(&self, name: impl AsRef<str>) -> Option<usize> {
        self.0.indices().get(name.as_ref()).copied()
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

    /// Get the [`DType`] of a variant by offset.
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
    /// `i8::MAX + 1 = 128`, which fits in two cache lines, so a linear scan is faster than a
    /// `HashMap` lookup in practice.
    pub fn tag_to_child_index(&self, tag: i8) -> Option<usize> {
        self.0.type_ids.iter().position(|&t| t == tag)
    }

    /// Convert a child offset to its data-level type tag.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    pub fn child_index_to_tag(&self, index: usize) -> i8 {
        self.0.type_ids[index]
    }

    /// Checks whether this set of variants satisfies the constraint imposed by a union-level
    /// `Nullability`:
    ///
    /// - `Nullable`: at least one variant is `DType::Null` or has nullable nullability.
    /// - `NonNullable`: no `DType::Null` variants and no nullable variants.
    ///
    /// The check materializes each [`FieldDType`] (so it may be expensive if some are still
    /// flatbuffer-backed views).
    ///
    /// It is intentionally not part of `UnionVariants` construction since `Nullability` lives on
    /// the `DType::Union(_, Nullability)` enum variant, not on `UnionVariants` itself.
    pub fn nullability_constraints_satisfied(&self, union_nullability: Nullability) -> bool {
        let has_null_or_nullable = self
            .variants()
            .any(|dt| matches!(dt, DType::Null) || dt.is_nullable());

        match union_nullability {
            Nullability::Nullable => has_null_or_nullable,
            Nullability::NonNullable => !has_null_or_nullable,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::UnionVariants;

    fn i32_variants() -> UnionVariants {
        UnionVariants::new_consecutive(
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
        let result = UnionVariants::new_consecutive(
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

    #[rstest]
    #[case::nullable_with_null_child(
        vec![
            DType::Null,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        ],
        Nullability::Nullable,
        true,
    )]
    #[case::nullable_with_nullable_child(
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
        ],
        Nullability::Nullable,
        true,
    )]
    #[case::nullable_with_no_null_or_nullable(
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::NonNullable),
        ],
        Nullability::Nullable,
        false,
    )]
    #[case::nonnullable_with_null_child(
        vec![
            DType::Null,
            DType::Primitive(PType::I32, Nullability::NonNullable),
        ],
        Nullability::NonNullable,
        false,
    )]
    #[case::nonnullable_with_nullable_child(
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
        ],
        Nullability::NonNullable,
        false,
    )]
    #[case::nonnullable_clean(
        vec![
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::NonNullable),
        ],
        Nullability::NonNullable,
        true,
    )]
    fn test_nullability_constraints(
        #[case] dtypes: Vec<DType>,
        #[case] nullability: Nullability,
        #[case] expected: bool,
    ) {
        let names: Vec<&str> = (0..dtypes.len()).map(|i| ["a", "b", "c", "d"][i]).collect();
        let variants = UnionVariants::new_consecutive(names.as_slice().into(), dtypes).unwrap();
        assert_eq!(
            variants.nullability_constraints_satisfied(nullability),
            expected
        );
    }

    #[test]
    fn test_display() {
        let variants = i32_variants();
        let dtype = DType::Union(variants, Nullability::NonNullable);
        assert_eq!(dtype.to_string(), "union(int=i32, str=utf8)");

        let nullable = DType::Union(
            UnionVariants::new_consecutive(
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
    fn test_new_consecutive_max_size() {
        // 128 variants is the maximum for consecutive type_ids: tags 0..=127 all fit in i8.
        let names: Vec<String> = (0..128).map(|i| format!("v{i}")).collect();
        let dtypes: Vec<DType> = (0..128)
            .map(|_| DType::Primitive(PType::I32, Nullability::NonNullable))
            .collect();
        let names: FieldNames = names.into_iter().collect();
        let variants = UnionVariants::new_consecutive(names, dtypes).unwrap();
        assert_eq!(variants.len(), 128);
        assert_eq!(variants.type_ids()[127], 127);
        assert!(variants.is_consecutive());
    }

    #[test]
    fn test_new_consecutive_too_large_rejected() {
        // 129 variants exceeds i8::MAX + 1 = 128.
        let names: Vec<String> = (0..129).map(|i| format!("v{i}")).collect();
        let dtypes: Vec<DType> = (0..129)
            .map(|_| DType::Primitive(PType::I32, Nullability::NonNullable))
            .collect();
        let names: FieldNames = names.into_iter().collect();
        let result = UnionVariants::new_consecutive(names, dtypes);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("at most 128 consecutive variants")
        );
    }

    #[test]
    fn test_empty() {
        let v = UnionVariants::empty();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert_eq!(v.type_ids(), &[] as &[i8]);
        assert!(v.is_consecutive());
    }
}

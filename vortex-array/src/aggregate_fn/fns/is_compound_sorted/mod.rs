// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checks that rows are lexicographically sorted by a declared sequence of fields.
//!
//! # About sort keys
//!
//! A sort key is a compound key that describes how the rows in a file are sorted.
//! For example, these rows are lexicographically sorted by
//! both (user_id, city) and (timestamp, user_id):
//!
//! ```text
//! user_id | city       | timestamp
//! --------+------------+----------
//! 1       | Bangkok    | 10
//! 1       | Mumbai     | 20
//! 2       | Tokyo      | 20
//! 3       | Copenhagen | 30
//! ```
//!
//! The first sort key orders cities within each user. The second key orders user IDs
//! within each timestamp.
//!
//! # Difference from sortedness stats
//!
//! The difference between `IsSorted` / `IsStrictSorted` and sort keys
//! is that they describe the sortedness of one field, while a compound key
//! requires sortedness across the file only for its first field. The remaining
//! fields only need to be sorted where all preceding fields are equal.
//!
//! The ordering must also hold across chunks. For example, each chunk below
//! is sorted by (user_id, city):
//!
//! ```text
//!          user_id | city
//!          --------+-----------
//! Chunk 1        1 | Bangkok
//!                2 | Tokyo
//!          --------+-----------
//! Chunk 2        2 | Mumbai      <- breaks the ordering
//!                3 | Copenhagen
//! ```
//!
//! But together, they break the sort key: (2, Tokyo) comes before (2, Mumbai),
//! even though Mumbai should come first.
//!
//! ## Notes
//!
//! Only `ASC` is supported for now. Whole-field checks reuse [`IsSorted`],
//! which only supports `ASC`. `DESC` is reserved in the options but rejected for now.
//! `DESC` support needs direction-aware checks to avoid reversing and materializing
//! the input.

use std::cmp::Ordering;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::LazyLock;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::operators::Operator;

/// The comparison direction of a field in a declared sort key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SortDirection {
    /// Ascending order, with nulls first.
    #[default]
    Asc,
    /// Descending order, reserved for future support and currently unsupported.
    Desc,
}

/// Options for [`IsCompoundSorted`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IsCompoundSortedOptions {
    /// Top-level struct fields and their directions, in priority order.
    /// Currently only ascending directions are supported.
    pub fields: Vec<(FieldName, SortDirection)>,
}

impl IsCompoundSortedOptions {
    /// Declare a lexicographic ordering over the given fields.
    pub fn new(fields: impl IntoIterator<Item = (impl Into<FieldName>, SortDirection)>) -> Self {
        Self {
            fields: fields
                .into_iter()
                .map(|(name, direction)| (name.into(), direction))
                .collect(),
        }
    }

    /// Validates that fields are non-empty, have no duplicates,
    /// and use only `ASC`.
    fn validate_fields(&self) -> VortexResult<()> {
        vortex_ensure!(
            !self.fields.is_empty(),
            "A sort key must contain at least one field"
        );

        let mut seen = HashSet::<&FieldName>::default();
        for (name, direction) in &self.fields {
            vortex_ensure!(
                *direction == SortDirection::Asc,
                "IsCompoundSorted currently supports ASC only"
            );
            vortex_ensure!(seen.insert(name), "Duplicate sort field: {}", name);
        }

        Ok(())
    }

    /// Validates the declared ordering against a struct input dtype, and returns the dtype
    /// of a row's sort key.
    ///
    /// Dtypes must be supported by [`IsSorted`] (not strict).
    fn key_dtype(&self, dtype: &DType) -> VortexResult<DType> {
        // Ideally this should be checked once during new
        self.validate_fields()?;

        let fields = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("IsCompoundSorted requires a struct input"))?;

        let (names, dtypes) = self
            .fields
            .iter()
            .map(|(name, _)| {
                let dtype = fields.field(name).ok_or_else(|| {
                    vortex_err!("Sort field {} does not exist in the input dtype", name)
                })?;
                vortex_ensure!(
                    IsSorted
                        .return_dtype(&IsSortedOptions { strict: false }, &dtype)
                        .is_some(),
                    "Sort field {} has unsupported dtype {}",
                    name,
                    dtype
                );
                Ok((name.clone(), dtype.as_nullable()))
            })
            .collect::<VortexResult<(Vec<_>, Vec<_>)>>()?;

        Ok(DType::Struct(
            StructFields::new(names.into(), dtypes),
            Nullability::Nullable,
        ))
    }

    /// Compares lexicographically two scalars (rows) in `ASC` order with nulls first.
    fn compare_sort_keys(&self, previous: &Scalar, next: &Scalar) -> VortexResult<Ordering> {
        for index in 0..self.fields.len() {
            let previous = previous
                .as_struct()
                .field_by_idx(index)
                .vortex_expect("key field");

            let next = next
                .as_struct()
                .field_by_idx(index)
                .vortex_expect("key field");

            let ordering = match (previous.is_null(), next.is_null()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                (false, false) => previous.partial_cmp(&next).ok_or_else(|| {
                    vortex_err!("Sort field {} cannot be compared", self.fields[index].0)
                })?,
            };

            // Tie-break
            if ordering != Ordering::Equal {
                return Ok(ordering);
            }
        }

        Ok(Ordering::Equal)
    }
}

impl Display for IsCompoundSortedOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (index, (name, direction)) in self.fields.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }

            let direction = match direction {
                SortDirection::Asc => "ASC",
                SortDirection::Desc => "DESC",
            };

            write!(f, "{} {}", name, direction)?;
        }
        Ok(())
    }
}

/// Checks that the rows inside one batch are sorted by the key fields (ASC, nulls
/// first). Equal neighbours are fine.
///
/// `sort_fields` are the key columns in priority order, all the same length (at least
/// 2 rows), and there must be at least one. Only rows *within* the batch are compared.
/// The seam with the previous batch is checked separately by
/// [`IsCompoundSortedOptions::compare_sort_keys`].
///
/// # How it works
///
/// A batch is sorted if every pair of neighbouring rows is in order.
/// For a compound key, "in order" depends on the field:
///
/// - On the first field, `row[i + 1]` must not be smaller than `row[i]`.
/// - On later fields, that only matters if the two rows tied on every field before
///   it. If they differ on an earlier field, they belong to different groups and the
///   later field doesn't constrain them.
///
/// So we keep a mask `tied` with one entry per pair. `tied[i]` is true while `row[i]` and `row[i + 1]`
/// agree on every field looked at so far.
///
/// Example, sorted by `(country, city)`:
///
/// ```text
/// row | country | city
/// ----+---------+-------
///  0  | DE      | Berlin
///  1  | DE      | Munich
///  2  | FR      | Lyon
///  3  | FR      | Paris
///  4  | FR      | Nice     <- Paris > Nice, so this batch is not sorted
/// ```
///
/// There are 4 pairs, `(0,1) (1,2) (2,3) (3,4)`, and `tied` starts as `[T, T, T, T]`.
///
/// **Field 0 (`country`).** Nothing is tied-off yet, so the whole column must simply
/// be sorted. We hand it to [`is_sorted`] to get its per-encoding fast paths. Then
/// we narrow `tied` to the pairs with equal countries:
///
/// ```text
/// pair:     (0,1) (1,2) (2,3) (3,4)
/// equal:      T     F     T     T      (DE=DE, DE!=FR, FR=FR, FR=FR)
/// tied:       T     F     T     T
/// ```
///
/// **Field 1 (`city`).** Now we look for pairs where `city[i] > city[i + 1]`, but
/// only among tied pairs:
///
/// ```text
/// pair:        (0,1)    (1,2)    (2,3)    (3,4)
/// cities:      B < M    M > L    L < P    P > N
/// out of order:  F        T        F        T
/// tied:          T        F        T        T
/// both:          F        F        F        T    <- violation, return false
/// ```
///
/// Pair `(1,2)` is out of order too, but it crosses from DE to FR, so it starts a
/// new group and is ignored.
///
/// After each field, `tied` is narrowed to the pairs that are also equal on that
/// field. If no pairs are tied anymore, later fields can't matter and we stop early.
///
/// # Nulls
///
/// Nulls sort first, and two nulls count as equal. So "out of order" also includes a
/// valid value followed by a null, and "equal" also includes null followed by null.
///
/// These rules must match `compare_sort_keys`. Together the two define the result for the
/// whole input.
fn check_chunk_sortedness(sort_fields: &[ArrayRef], ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let len = sort_fields[0].len();

    let pair_count = len - 1;
    let mut tied = Mask::new_true(pair_count);

    for (index, field) in sort_fields.iter().enumerate() {
        let all_pairs_tied = tied.all_true();
        let is_last_field = index + 1 == sort_fields.len();

        // If every pair is tied, reuse the encoding-specific whole-column check.
        if all_pairs_tied {
            if !is_sorted(field, ctx)? {
                return Ok(false);
            }

            if is_last_field {
                return Ok(true);
            }
        }

        // Align each row with the next row. For city in the example above:
        // pair:      (0,1)    (1,2)    (2,3)    (3,4)
        // previous:  Berlin   Munich   Lyon     Paris
        // next:      Munich   Lyon     Paris    Nice
        let previous = field.slice(0..pair_count)?;
        let next = field.slice(1..len)?;

        // For each pair, mark whether the previous and next values are non-null.
        let previous_valid = previous.validity()?.execute_mask(pair_count, ctx)?;
        let next_valid = next.validity()?.execute_mask(pair_count, ctx)?;

        // Otherwise, this field only constrains pairs tied on all earlier fields.
        if !all_pairs_tied {
            // Given that `IsCompoundSorted` uses only `ASC` and `NULLS` first,
            // a valid value followed by null is also a violation.
            let null_violation = &previous_valid & &!&next_valid;
            let out_of_order = previous
                .binary(next.clone(), Operator::Gt)?
                .execute::<BoolArray>(ctx)?
                .to_mask_fill_null_false(ctx)
                | &null_violation;

            if (&tied & &out_of_order).true_count() != 0 {
                return Ok(false);
            }

            if is_last_field {
                return Ok(true);
            }
        }

        // Later fields matter only for pairs that are also equal on this field.
        // Both values are null when neither side is valid.
        let both_null = !(&previous_valid | &next_valid);
        let equal_pairs = previous
            .binary(next, Operator::Eq)?
            .execute::<BoolArray>(ctx)?
            .to_mask_fill_null_false(ctx)
            | &both_null;

        tied = tied & &equal_pairs;
        if tied.all_false() {
            break;
        }
    }

    Ok(true)
}

/// Checks a declared lexicographic ordering across all accumulated batches.
///
/// It is not `strict`, so equal row keys are allowed. Only `ASC` is supported for now,
/// DESC remains reserved in the options.
///
/// # Note
///
/// As a file statistic, it requires `VortexWriteOptions::include_nested_statistics` to be
/// enabled and a nullable root struct. This is because the result is stored in the root
/// entry, and only nested statistics over a nullable root have one. Legacy statistics have
/// one slot per top-level field and no slot for the root struct.
#[derive(Clone, Debug)]
pub struct IsCompoundSorted;

/// Running verdict and boundary keys for [`IsCompoundSorted`].
pub struct IsCompoundSortedPartial {
    sorted: bool,
    // None means no rows. A row whose key fields are null still has a non-null key tuple.
    first: Option<Scalar>,
    last: Option<Scalar>,
}

impl IsCompoundSortedPartial {
    /// The state of a group with no accumulated values.
    fn empty() -> Self {
        Self {
            sorted: true,
            first: None,
            last: None,
        }
    }
}

static NAMES: LazyLock<FieldNames> =
    LazyLock::new(|| FieldNames::from(["is_sorted", "first", "last"]));

impl AggregateFnVTable for IsCompoundSorted {
    type Options = IsCompoundSortedOptions;
    type Partial = IsCompoundSortedPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.is_compound_sorted");
        *ID
    }

    /// Serializes [`IsCompoundSortedOptions`] as protobuf.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        options.validate_fields()?;
        Ok(Some(
            pb::IsCompoundSortedOpts {
                fields: options
                    .fields
                    .iter()
                    .map(|(name, direction)| pb::is_compound_sorted_opts::SortField {
                        name: name.to_string(),
                        direction: match direction {
                            SortDirection::Asc => pb::is_compound_sorted_opts::SortDirection::Asc,
                            SortDirection::Desc => pb::is_compound_sorted_opts::SortDirection::Desc,
                        } as i32,
                    })
                    .collect(),
            }
            .encode_to_vec(),
        ))
    }

    /// Deserializes [`IsCompoundSortedOptions`] from protobuf.
    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let proto = pb::IsCompoundSortedOpts::decode(metadata)?;
        let fields = proto
            .fields
            .into_iter()
            .map(|field| {
                let direction =
                    match pb::is_compound_sorted_opts::SortDirection::try_from(field.direction)? {
                        pb::is_compound_sorted_opts::SortDirection::Asc => SortDirection::Asc,
                        pb::is_compound_sorted_opts::SortDirection::Desc => SortDirection::Desc,
                    };
                Ok((field.name, direction))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let options = IsCompoundSortedOptions::new(fields);
        options.validate_fields()?;
        Ok(options)
    }

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        options.key_dtype(input_dtype).ok()?;
        Some(DType::Bool(Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        let key = options.key_dtype(input_dtype).ok()?;

        Some(DType::Struct(
            StructFields::new(
                NAMES.clone(),
                vec![DType::Bool(Nullability::NonNullable), key.clone(), key],
            ),
            Nullability::Nullable,
        ))
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(IsCompoundSortedPartial::empty())
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        // A null struct means the producing accumulator was empty.
        if scalar.is_null() {
            return Ok(IsCompoundSortedPartial::empty());
        }

        let fields = scalar.as_struct();
        let first = fields.field_by_idx(1).vortex_expect("first key");
        let last = fields.field_by_idx(2).vortex_expect("last key");

        vortex_ensure!(
            !first.is_null() && !last.is_null(),
            "Non-empty sortedness partials require boundary keys"
        );

        Ok(IsCompoundSortedPartial {
            sorted: fields
                .field_by_idx(0)
                .vortex_expect("sorted verdict")
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("Sortedness partials require a non-null verdict"))?,
            first: Some(first),
            last: Some(last),
        })
    }

    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        mut acc: Self::Partial,
        partial: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        if !acc.sorted {
            return Ok(acc);
        }

        acc.sorted &= partial.sorted;

        // Given that we only handle `ASC` for now,
        // the compare only checks that the incoming partial first key
        // is greater than the last acc. key. For `DESC` it would be the inverse.
        if acc.sorted
            && let (Some(previous), Some(next)) = (&acc.last, &partial.first)
        {
            acc.sorted = args.options.compare_sort_keys(previous, next)? != Ordering::Greater;
        }

        if acc.first.is_none() {
            acc.first = partial.first;
        }

        if partial.last.is_some() {
            acc.last = partial.last;
        }

        Ok(acc)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let Some(first) = &partial.first else {
            return Ok(Scalar::null(args.partial_dtype.clone()));
        };

        Ok(Scalar::struct_(
            args.partial_dtype.clone(),
            vec![
                Scalar::bool(partial.sorted, Nullability::NonNullable),
                first.clone(),
                partial
                    .last
                    .clone()
                    .vortex_expect("last key accompanies first key"),
            ],
        ))
    }

    /// As with `IsSorted`, an unsorted partial makes the global result unsorted (saturated).
    #[inline]
    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        !partial.sorted
    }

    /// Handled here instead of in `accumulate` so we can read just the key fields from
    /// the batch's original encoding, rather than decoding the whole struct to
    /// [`Columnar`] first. `accumulate` would give the same result, only slower.
    fn try_accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if batch.is_empty() || !partial.sorted {
            return Ok(true);
        }

        let sort_fields = args
            .options
            .fields
            .iter()
            .map(|(name, _)| batch.get_item(name.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        let key_dtype = args
            .partial_dtype
            .as_struct_fields_opt()
            .and_then(|fields| fields.field_by_index(1))
            .vortex_expect("IsCompoundSorted partial contains the first key dtype");

        let first = Scalar::struct_(
            key_dtype.clone(),
            sort_fields
                .iter()
                .map(|field| field.execute_scalar(0, ctx).map(Scalar::into_nullable))
                .collect::<VortexResult<Vec<_>>>()?,
        );

        let last = Scalar::struct_(
            key_dtype,
            sort_fields
                .iter()
                .map(|field| {
                    field
                        .execute_scalar(batch.len() - 1, ctx)
                        .map(Scalar::into_nullable)
                })
                .collect::<VortexResult<Vec<_>>>()?,
        );

        if let Some(previous) = &partial.last {
            partial.sorted = args.options.compare_sort_keys(previous, &first)? != Ordering::Greater;
        }

        // Check fields sortedness
        if partial.sorted && batch.len() > 1 {
            partial.sorted = check_chunk_sortedness(&sort_fields, ctx)?;
        }

        if partial.first.is_none() {
            partial.first = Some(first);
        }

        partial.last = Some(last);

        Ok(true)
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &mut Self::Partial,
        _batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        unreachable!("IsCompoundSorted::try_accumulate handles all arrays")
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        partials
            .get_item(NAMES.get(0).vortex_expect("out of bounds").clone())?
            .fill_null(Scalar::bool(true, Nullability::NonNullable))
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::bool(partial.sorted, Nullability::NonNullable))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use hegel::TestCase;
    use hegel::generators as gs;
    use prost::Message;
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;

    use super::IsCompoundSorted;
    use super::IsCompoundSortedOptions;
    use super::SortDirection;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::DictArray;
    use crate::arrays::PiecewiseSequenceArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::RunEnd;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::proto::expr as pb;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    // The following is a Hegel test with different properties how sortedness
    // works in a couple of subtypes of `DType` which includes all primitives
    // and VarBin/UTF-8.
    const MAX_ROWS: usize = 32;
    const MAX_COLUMNS: usize = 5;
    const MAX_CHUNKS: usize = 4;
    const MAX_VALUE: u8 = 3;
    const PTYPES: [PType; 11] = [
        PType::U8,
        PType::U16,
        PType::U32,
        PType::U64,
        PType::I8,
        PType::I16,
        PType::I32,
        PType::I64,
        PType::F16,
        PType::F32,
        PType::F64,
    ];

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn result_matches_compound_sortedness(tc: TestCase) {
        check_random_table(&tc).vortex_expect("IsCompoundSorted");
    }

    fn draw_dtype(tc: &TestCase) -> DType {
        let nullability = Nullability::from(tc.draw(gs::booleans()));
        let dtypes: Vec<DType> = PTYPES
            .iter()
            .map(|&ptype| DType::Primitive(ptype, nullability))
            .chain([DType::Utf8(nullability)])
            .collect();
        tc.draw(gs::sampled_from(dtypes))
    }

    fn draw_row(tc: &TestCase, dtypes: &[DType]) -> Vec<Option<u8>> {
        dtypes
            .iter()
            .map(|dtype| {
                let value = gs::integers::<u8>().max_value(MAX_VALUE);
                if dtype.is_nullable() {
                    tc.draw(gs::optional(value))
                } else {
                    Some(tc.draw(value))
                }
            })
            .collect()
    }

    fn column_into_array(
        dtype: &DType,
        values: impl Iterator<Item = Option<u8>>,
    ) -> VortexResult<ArrayRef> {
        let array = match dtype {
            DType::Primitive(..) => PrimitiveArray::from_option_iter(values).into_array(),
            DType::Utf8(_) => {
                VarBinViewArray::from_iter_nullable_str(values.map(|v| v.map(|v| v.to_string())))
                    .into_array()
            }
            other => vortex_bail!("No column generator for {other}"),
        };
        array.cast(dtype.clone())
    }

    fn check_random_table(tc: &TestCase) -> VortexResult<()> {
        let column_count = tc.draw(gs::integers::<usize>().min_value(1).max_value(MAX_COLUMNS));
        let dtypes: Vec<DType> = (0..column_count).map(|_| draw_dtype(tc)).collect();
        let rows_count = tc.draw(gs::integers::<usize>().max_value(MAX_ROWS));
        let mut rows: Vec<Vec<Option<u8>>> =
            (0..rows_count).map(|_| draw_row(tc, &dtypes)).collect();

        let sort_columns: Vec<usize> = tc.draw(
            gs::vecs(gs::integers::<usize>().max_value(column_count - 1))
                .max_size(column_count)
                .unique(true),
        );

        let mut names: Vec<String> = sort_columns
            .iter()
            .map(|index| format!("f{index}"))
            .collect();

        // Randomly add a non-existing column in the source array ref, so we can assert the agg. fn.
        // behaviour when a non-existent column is requested for sorting.
        let add_missing = tc.draw(gs::booleans());
        if add_missing {
            let at = tc.draw(gs::integers::<usize>().max_value(names.len()));
            names.insert(at, format!("f{column_count}"));
        }

        let options = IsCompoundSortedOptions::new(
            names.iter().map(|name| (name.as_str(), SortDirection::Asc)),
        );

        // Many times rows will be unsorted because we are creating them randomly
        // with hegel, so force some of them to be sorted.
        let row_key = |row: &Vec<Option<u8>>| -> Vec<Option<u8>> {
            sort_columns.iter().map(|&column| row[column]).collect()
        };

        let force_sort = tc.draw(gs::booleans());
        if force_sort {
            rows.sort_by_key(row_key);
        }
        let expected = rows.is_sorted_by_key(row_key);

        let fields = dtypes
            .iter()
            .enumerate()
            .map(|(column, dtype)| {
                let values = rows.iter().map(|row| row[column]);
                Ok((format!("f{column}"), column_into_array(dtype, values)?))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let array = StructArray::from_fields(&fields)?.into_array();
        let dtype = array.dtype().clone();

        // Both cases should err.
        if sort_columns.is_empty() || add_missing {
            assert!(IsCompoundSorted.return_dtype(&options, &dtype).is_none());
            assert!(Accumulator::try_new(IsCompoundSorted, options, dtype).is_err());

            return Ok(());
        }

        // Split the rows into chunks so we can also test combining different partials
        // and testing if the boundaries are also sorted.
        let chunks_count = tc.draw(gs::integers::<usize>().min_value(1).max_value(MAX_CHUNKS));
        let mut chunks_boundaries = Vec::new();
        let mut last_index = 0;
        for _ in 1..chunks_count {
            let end = tc.draw(gs::integers::<usize>().min_value(last_index).max_value(rows_count));
            chunks_boundaries.push(last_index..end);
            last_index = end;
        }
        chunks_boundaries.push(last_index..rows_count);

        let combine_partials = tc.draw(gs::booleans());

        let mut ctx = SESSION.create_execution_ctx();
        let mut acc =
            Accumulator::try_new(IsCompoundSorted, options.clone(), dtype.clone())?;
        for boundaries in chunks_boundaries {
            let batch = array.slice(boundaries)?;
            if combine_partials {
                let mut chunk_acc =
                    Accumulator::try_new(IsCompoundSorted, options.clone(), dtype.clone())?;
                chunk_acc.accumulate(&batch, &mut ctx)?;

                let partial = chunk_acc.partial_scalar()?;
                acc.combine_partials(&partial)?;
            } else {
                acc.accumulate(&batch, &mut ctx)?;
            }
        }

        assert_eq!(
            acc.finish()?,
            Scalar::bool(expected, Nullability::NonNullable)
        );

        Ok(())
    }

    fn order_lines(values: &[(i32, i32)]) -> VortexResult<ArrayRef> {
        // Unsorted metadata before the key fields catches accidental field selection.
        let is_expedited =
            BoolArray::from_iter(values.iter().enumerate().map(|(index, _)| index % 2 == 0))
                .into_array();
        let order_ids = PrimitiveArray::from_iter(values.iter().map(|row| row.0)).into_array();
        let line_item_ids = PrimitiveArray::from_iter(values.iter().map(|row| row.1)).into_array();

        Ok(StructArray::from_fields(&[
            ("is_expedited", is_expedited),
            ("order_id", order_ids),
            ("line_item_id", line_item_ids),
        ])?
        .into_array())
    }

    fn order_line_options() -> IsCompoundSortedOptions {
        IsCompoundSortedOptions::new([
            ("order_id", SortDirection::Asc),
            ("line_item_id", SortDirection::Asc),
        ])
    }

    fn check_batches(
        array: ArrayRef,
        options: IsCompoundSortedOptions,
        one_row_batches: bool,
        expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut accumulator =
            Accumulator::try_new(IsCompoundSorted, options, array.dtype().clone())?;

        if one_row_batches {
            for index in 0..array.len() {
                let batch = array.slice(index..index + 1)?;
                accumulator.accumulate(&batch, &mut ctx)?;
            }
        } else {
            accumulator.accumulate(&array, &mut ctx)?;
        }

        assert_eq!(
            accumulator.finish()?,
            Scalar::bool(expected, Nullability::NonNullable)
        );
        Ok(())
    }

    #[rstest]
    #[case::null_first(
        vec![(100, -100), (2, 7), (3, 9)],
        vec![false, true, true],
        true,
    )]
    #[case::null_last(
        vec![(2, 7), (3, 9), (-100, 100)],
        vec![true, true, false],
        false,
    )]
    fn parent_nulls_mask_sort_fields(
        #[case] values: Vec<(i32, i32)>,
        #[case] valid: Vec<bool>,
        #[case] expected: bool,
        #[values(false, true)] one_row_batches: bool,
    ) -> VortexResult<()> {
        let array = order_lines(&values)?.mask(BoolArray::from_iter(valid).into_array())?;

        check_batches(array, order_line_options(), one_row_batches, expected)
    }

    #[rstest]
    #[case::nan_at_end(buffer![1.0, 2.0, f64::NAN], true)]
    #[case::nan_before_finite(buffer![1.0, f64::NAN, 2.0], false)]
    #[case::signed_zero(buffer![-0.0, 0.0], true)]
    fn float_order_matches_across_batches(
        #[case] prices: Buffer<f64>,
        #[case] expected: bool,
        #[values(false, true)] one_row_batches: bool,
    ) -> VortexResult<()> {
        let array = StructArray::from_fields(&[("price", prices.into_array())])?.into_array();

        check_batches(
            array,
            IsCompoundSortedOptions::new([("price", SortDirection::Asc)]),
            one_row_batches,
            expected,
        )
    }

    #[rstest]
    #[case::null_prefix(
        PrimitiveArray::from_option_iter([None::<f64>, None]).into_array(),
    )]
    #[case::nan_prefix(buffer![f64::NAN; 2].into_array())]
    fn equal_null_and_nan_prefixes_require_sorted_secondary(
        #[case] prefix: ArrayRef,
    ) -> VortexResult<()> {
        let array = StructArray::from_fields(&[
            ("price", prefix),
            ("line_item_id", buffer![2i32, 1].into_array()),
        ])?
        .into_array();

        check_batches(
            array,
            IsCompoundSortedOptions::new([
                ("price", SortDirection::Asc),
                ("line_item_id", SortDirection::Asc),
            ]),
            false,
            false,
        )
    }

    #[test]
    fn options_roundtrip() -> VortexResult<()> {
        let options = order_line_options();
        let bytes = IsCompoundSorted
            .serialize(&options)?
            .vortex_expect("serializable options");

        assert_eq!(IsCompoundSorted.deserialize(&bytes, &SESSION)?, options);
        assert_eq!(options.to_string(), "order_id ASC, line_item_id ASC");

        assert!(
            IsCompoundSorted
                .zone_stat_default(&DType::from(PType::I32))
                .is_none()
        );

        Ok(())
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::duplicate(vec![("order_id", 0), ("order_id", 0)])]
    #[case::unknown_direction(vec![("order_id", 0), ("line_item_id", 99)])]
    #[case::descending_primary(vec![("order_id", 1), ("line_item_id", 0)])]
    #[case::descending_secondary(vec![("order_id", 0), ("line_item_id", 1)])]
    fn invalid_serialized_options_are_rejected(#[case] fields: Vec<(&str, i32)>) {
        let bytes = pb::IsCompoundSortedOpts {
            fields: fields
                .into_iter()
                .map(|(name, direction)| pb::is_compound_sorted_opts::SortField {
                    name: name.into(),
                    direction,
                })
                .collect(),
        }
        .encode_to_vec();

        assert!(IsCompoundSorted.deserialize(&bytes, &SESSION).is_err());
    }

    /// The property test never draws a repeated field, so duplicates are checked here.
    #[test]
    fn duplicate_keys_are_rejected() -> VortexResult<()> {
        let dtype = order_lines(&[])?.dtype().clone();
        let options = IsCompoundSortedOptions::new([
            ("order_id", SortDirection::Asc),
            ("order_id", SortDirection::Asc),
        ]);

        assert!(options.key_dtype(&dtype).is_err());
        assert!(Accumulator::try_new(IsCompoundSorted, options, dtype).is_err());

        Ok(())
    }

    #[test]
    fn unsupported_input_and_field_are_rejected() -> VortexResult<()> {
        let options = IsCompoundSortedOptions::new([("order_id", SortDirection::Asc)]);
        let primitive_input = buffer![1i32].into_array();
        let nested_input =
            StructArray::from_fields(&[("order_id", order_lines(&[(1, 2)])?)])?.into_array();

        assert!(options.key_dtype(primitive_input.dtype()).is_err());
        assert!(options.key_dtype(nested_input.dtype()).is_err());

        Ok(())
    }

    #[test]
    fn descending_is_reserved_but_rejected() -> VortexResult<()> {
        let dtype = order_lines(&[])?.dtype().clone();
        let options = IsCompoundSortedOptions::new([
            ("order_id", SortDirection::Asc),
            ("line_item_id", SortDirection::Desc),
        ]);
        let error = options
            .key_dtype(&dtype)
            .expect_err("DESC is not implemented");

        assert_eq!(options.to_string(), "order_id ASC, line_item_id DESC");
        assert!(error.to_string().contains("ASC only"));
        assert!(Accumulator::try_new(IsCompoundSorted, options.clone(), dtype).is_err());
        assert!(IsCompoundSorted.serialize(&options).is_err());

        Ok(())
    }

    #[rstest]
    #[case::distinct_zero(buffer![-0.0, 0.0], true)]
    #[case::equal_zero(buffer![-0.0; 2], false)]
    fn float_prefix_equality_matches_total_order(
        #[case] prices: Buffer<f64>,
        #[case] expected: bool,
        #[values(false, true)] one_row_batches: bool,
    ) -> VortexResult<()> {
        let array = StructArray::from_fields(&[
            ("price", prices.into_array()),
            ("line_item_id", buffer![2i32, 1].into_array()),
        ])?
        .into_array();

        check_batches(
            array,
            IsCompoundSortedOptions::new([
                ("price", SortDirection::Asc),
                ("line_item_id", SortDirection::Asc),
            ]),
            one_row_batches,
            expected,
        )
    }

    #[rstest]
    #[case::binary(
        VarBinViewArray::from_iter_bin([b"B2", b"C3", b"A1", b"B2"]).into_array(),
        true,
    )]
    #[case::boolean(
        BoolArray::from_iter([false, true, false, true]).into_array(),
        true,
    )]
    #[case::bad_boolean(
        BoolArray::from_iter([true, false, false, true]).into_array(),
        false,
    )]
    #[case::decimal(
        DecimalArray::new(
            buffer![1i64, 2, 0, 1],
            DecimalDType::new(10, 0),
            Validity::NonNullable,
            &mut SESSION.create_execution_ctx(),
        )
        .into_array(),
        true,
    )]
    #[case::bad_decimal(
        DecimalArray::new(
            buffer![2i64, 1, 0, 1],
            DecimalDType::new(10, 0),
            Validity::NonNullable,
            &mut SESSION.create_execution_ctx(),
        )
        .into_array(),
        false,
    )]
    fn typed_secondary_fields(
        #[case] values: ArrayRef,
        #[case] expected: bool,
        #[values(false, true)] one_row_batches: bool,
    ) -> VortexResult<()> {
        let array = StructArray::from_fields(&[
            ("order_id", buffer![1i32, 1, 2, 2].into_array()),
            ("item_value", values),
        ])?
        .into_array();

        check_batches(
            array,
            IsCompoundSortedOptions::new([
                ("order_id", SortDirection::Asc),
                ("item_value", SortDirection::Asc),
            ]),
            one_row_batches,
            expected,
        )
    }

    #[rstest]
    fn encoded_and_sliced_fields(
        #[values("constant", "dict", "runend", "sequence")] encoding: &str,
        #[values(false, true)] sliced: bool,
        #[values(false, true)] one_row_batches: bool,
        #[values(false, true)] reverse_secondary: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let order_ids = match encoding {
            "constant" => ConstantArray::new(1u64, 6).into_array(),
            "dict" => DictArray::try_new(
                buffer![0u32, 1, 1, 2, 2, 3].into_array(),
                buffer![0u64, 1, 2, 3].into_array(),
            )?
            .into_array(),
            "runend" => RunEnd::try_new(
                buffer![1u64, 3, 5, 6].into_array(),
                buffer![0u64, 1, 2, 3].into_array(),
                &mut ctx,
            )?
            .into_array(),
            "sequence" => PiecewiseSequenceArray::try_new(
                ConstantArray::new(0u64, 1).into_array(),
                ConstantArray::new(6u64, 1).into_array(),
                ConstantArray::new(1u64, 1).into_array(),
                6,
            )?
            .into_array(),
            _ => vortex_bail!("Unknown encoding fixture: {}", encoding),
        };

        let mut line_item_ids = if encoding == "constant" {
            buffer_mut![0i32, 1, 2, 3, 4, 5]
        } else {
            buffer_mut![99i32, 1, 2, 0, 1, 99]
        };
        if reverse_secondary {
            line_item_ids.swap(1, 2);
        }

        let mut array = StructArray::from_fields(&[
            ("order_id", order_ids),
            ("line_item_id", line_item_ids.into_array()),
        ])?
        .into_array();

        if sliced {
            array = array.slice(1..5)?;
        }

        // The sequence has distinct order IDs, so line-item IDs are unconstrained.
        let expected = !reverse_secondary || encoding == "sequence";
        check_batches(array, order_line_options(), one_row_batches, expected)
    }
}

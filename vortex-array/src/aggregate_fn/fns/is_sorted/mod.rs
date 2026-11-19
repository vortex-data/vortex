// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod decimal;
mod fixed_size_list;
mod primitive;
mod varbin;

use std::cmp::Reverse;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use self::bool::check_bool_sorted;
use self::decimal::check_decimal_sorted;
use self::fixed_size_list::check_fixed_size_bytes_sorted;
use self::primitive::check_primitive_sorted;
use self::varbin::check_varbinview_sorted;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::fns::fixed_size_bytes::is_fixed_size_bytes;
use crate::arrays::Constant;
use crate::arrays::Null;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::Scalar;

/// The order of non-null values, independent of where nulls sort.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SortDirection {
    /// Increasing values.
    #[default]
    Asc,
    /// Decreasing values.
    Desc,
}

/// Where nulls sort relative to non-null values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NullOrder {
    /// Nulls come before all values, whatever the direction.
    #[default]
    First,
    /// Nulls come after all values, whatever the direction.
    Last,
}

/// Options for the [`IsSorted`] aggregate function.
///
/// # Serialization
///
/// Options are serialized as a single byte, with one bit per field:
///
/// ```text
///  bit   7  6  5  4  3      2            1           0
///      ┌────────────────┬────────────┬───────────┬────────┐
///      │ reserved (0)   │ null order │ direction │ strict │
///      └────────────────┴────────────┴───────────┴────────┘
///                         0 = first    0 = asc     0 = no
///                         1 = last     1 = desc    1 = yes
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct IsSortedOptions {
    /// If true, require strict order (no duplicates).
    pub strict: bool,
    /// The direction to check.
    pub direction: SortDirection,
    /// Where nulls must appear, independent of the direction.
    pub null_order: NullOrder,
}

impl IsSortedOptions {
    /// Returns these options with `strict` replaced.
    pub fn with_strict(self, strict: bool) -> Self {
        Self { strict, ..self }
    }

    /// Returns these options with `direction` replaced.
    pub fn with_direction(self, direction: SortDirection) -> Self {
        Self { direction, ..self }
    }

    /// Returns these options with `null_order` replaced.
    pub fn with_null_order(self, null_order: NullOrder) -> Self {
        Self { null_order, ..self }
    }
}

// Options bits position inside a single byte.
//
// As a reference check [IsSortedOptions] doc.
const STRICT_BIT: u8 = 1 << 0;
const DESC_BIT: u8 = 1 << 1;
const NULLS_LAST_BIT: u8 = 1 << 2;

impl Display for SortDirection {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Asc => "asc",
            Self::Desc => "desc",
        })
    }
}

impl Display for NullOrder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::First => "first",
            Self::Last => "last",
        })
    }
}

impl Display for IsSortedOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "strict={}, direction={}, nulls={}",
            self.strict, self.direction, self.null_order
        )
    }
}

impl IsSortedOptions {
    fn is_sorted<I: Iterator>(&self, iter: I) -> bool
    where
        I::Item: PartialOrd,
    {
        match (self.direction, self.strict) {
            (SortDirection::Asc, false) => iter.is_sorted(),
            (SortDirection::Asc, true) => iter.is_strict_sorted(),
            (SortDirection::Desc, false) => iter.is_sorted_by(|a, b| a >= b),
            (SortDirection::Desc, true) => iter.is_sorted_by(|a, b| a > b),
        }
    }

    /// Checks if a nullable iterator is sorted, with nulls placed according to `null_order`.
    ///
    /// Every case is checked as ascending. [`Reverse`] around the values flips their order, and
    /// around the whole `Option` it also moves the nulls after every value. `Reverse` is free at
    /// runtime, so all four cases cost one `Option` comparison per pair.
    ///
    /// TODO (joacoc): I'm looking for alternatives to avoid using reverse here. Alternatives
    /// are for example, checking nulls in another place, instead of comparing `Option`s here. In a
    /// sorted array the nulls form one block at the end chosen by `null_order`, so a popcount over
    /// that block checks their placement, and the values in between can be checked as a plain
    /// slice, without `Reverse` or per-element validity.
    fn is_nullable_sorted<T: PartialOrd>(&self, iter: impl Iterator<Item = Option<T>>) -> bool {
        let ascending = self.with_direction(SortDirection::Asc);
        match (self.direction, self.null_order) {
            // `None` already compares below every `Some`.
            (SortDirection::Asc, NullOrder::First) => ascending.is_sorted(iter),
            // Reversing the values inside `Option` flips their order and keeps the nulls first.
            (SortDirection::Desc, NullOrder::First) => {
                ascending.is_sorted(iter.map(|value| value.map(Reverse)))
            }
            // Reversing the whole `Option` flips the values and moves the nulls last.
            (SortDirection::Desc, NullOrder::Last) => ascending.is_sorted(iter.map(Reverse)),
            // Reversing twice keeps the values ascending, and the outer `Reverse` moves the nulls
            // last.
            (SortDirection::Asc, NullOrder::Last) => {
                ascending.is_sorted(iter.map(|value| Reverse(value.map(Reverse))))
            }
        }
    }

    fn boundary_is_sorted(&self, previous: &Scalar, next: &Scalar) -> bool {
        match (previous.is_null(), next.is_null()) {
            (true, true) => !self.strict,
            (true, false) => self.null_order == NullOrder::First,
            (false, true) => self.null_order == NullOrder::Last,
            (false, false) => match (self.direction, self.strict) {
                (SortDirection::Asc, false) => previous <= next,
                (SortDirection::Asc, true) => previous < next,
                (SortDirection::Desc, false) => previous >= next,
                (SortDirection::Desc, true) => previous > next,
            },
        }
    }
}

/// Compute whether an array is sorted in non-decreasing order.
///
/// Returns `true` for empty arrays and arrays of length 1.
/// Returns `false` for struct, list, and fixed-size list arrays, except fixed-size lists of
/// non-nullable `u8`, which are ordered by byte.
pub fn is_sorted(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    is_sorted_with_options(array, IsSortedOptions::default(), ctx)
}

/// Compute whether an array is strictly sorted in increasing order (no duplicates).
///
/// Returns `true` for empty arrays and arrays of length 1.
/// Returns `false` for struct, list, and fixed-size list arrays, except fixed-size lists of
/// non-nullable `u8`, which are ordered by byte.
pub fn is_strict_sorted(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    is_sorted_with_options(array, IsSortedOptions::default().with_strict(true), ctx)
}

/// Checks sortedness.
///
/// Properties:
/// - Empty arrays and arrays of length one are always sorted, including in strict mode.
/// - Only ascending results use cached sortedness stats, and with nulls last only when the
///   array has no nulls.
pub fn is_sorted_with_options(
    array: &ArrayRef,
    options: IsSortedOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    // Arrays with 0 or 1 elements are (strict) sorted. Check this before the cached
    // statistics so that a stale hint cannot contradict it.
    if array.len() <= 1 {
        return Ok(true);
    }

    // Short-circuit using cached array statistics.
    if let Some(value) = cached_sortedness(array, &options) {
        return Ok(value);
    }

    // Constant and null arrays are always sorted, but not strict sorted.
    if array.is::<Constant>() || array.is::<Null>() {
        let result = !options.strict;
        set_sortedness_cache(array, &options, result);
        return Ok(result);
    }

    // We don't support sorting struct arrays.
    if array.dtype().is_struct() {
        return Ok(false);
    }

    // Short-circuit for unsupported dtypes.
    if IsSorted.return_dtype(&options, array.dtype()).is_none() {
        return Ok(false);
    }

    // Enforce strictness before we even try to check if the array is sorted.
    if options.strict {
        let invalid_count = array.invalid_count(ctx)?;
        match invalid_count {
            // We can keep going
            0 => {}
            // A single null position depends on the null order.
            1 => {
                let null_index = match options.null_order {
                    NullOrder::First => 0,
                    NullOrder::Last => array.len() - 1,
                };
                if !array.is_invalid(null_index, ctx)? {
                    set_sortedness_cache(array, &options, false);
                    return Ok(false);
                }
            }
            _ => {
                set_sortedness_cache(array, &options, false);
                return Ok(false);
            }
        }
    }

    // Compute using Accumulator<IsSorted>.
    let mut acc = Accumulator::try_new(IsSorted, options, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result_scalar = acc.finish()?;

    let result = result_scalar.as_bool().value().unwrap_or(false);

    // Cache the computed result as statistics.
    set_sortedness_cache(array, &options, result);

    Ok(result)
}

/// Returns true if the options can use the cache.
///
/// The current cache cannot discern between different options,
/// e.g. between asc and desc options. So, it is not possible to
/// use the cache for them. The only valid case is `ASC`, strict or
/// non-strict, with nulls first. Nulls last can also use it when the
/// array has no nulls, because then the null order doesn't make a
/// difference.
fn can_use_sortedness_cache(array: &ArrayRef, options: &IsSortedOptions) -> bool {
    if options.direction != SortDirection::Asc {
        return false;
    }

    match options.null_order {
        NullOrder::First => true,
        NullOrder::Last => {
            !array.dtype().is_nullable()
                || array.statistics().get_as::<usize>(Stat::NullCount) == Precision::Exact(0)
        }
    }
}

/// Returns cached sortedness if possible.
///
/// If aggregate options are not default or there is no
/// cached value, returns `None`.
fn cached_sortedness(array: &ArrayRef, options: &IsSortedOptions) -> Option<bool> {
    if !can_use_sortedness_cache(array, options) {
        return None;
    }

    let stat = if options.strict {
        Stat::IsStrictSorted
    } else {
        Stat::IsSorted
    };

    array.statistics().get_as::<bool>(stat).as_exact()
}

/// Cache sortedness into the array stats.
fn set_sortedness_cache(array: &ArrayRef, options: &IsSortedOptions, result: bool) {
    if !can_use_sortedness_cache(array, options) {
        return;
    }

    let array_stats = array.statistics();
    if options.strict {
        if result {
            array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
            array_stats.set(Stat::IsStrictSorted, Precision::Exact(true.into()));
        } else {
            array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
        }
    } else if result {
        array_stats.set(Stat::IsSorted, Precision::Exact(true.into()));
    } else {
        array_stats.set(Stat::IsSorted, Precision::Exact(false.into()));
        array_stats.set(Stat::IsStrictSorted, Precision::Exact(false.into()));
    }
}

/// Aggregate function vtable for `is_sorted`.
///
/// Returns `Bool(NonNullable)` scalar.
/// The partial state is a nullable struct `{is_sorted: Bool(NN), first_value: T?, last_value: T?}`.
/// A null struct means the accumulator has seen no data yet (empty).
#[derive(Clone, Debug)]
pub struct IsSorted;

impl IsSorted {
    /// Build a partial scalar from a kernel's `is_sorted` result.
    ///
    /// Kernels that compute `is_sorted` by delegating to child arrays can call this
    /// to package the boolean result into the partial struct format expected by the
    /// accumulator, avoiding duplicated boilerplate.
    pub fn make_partial(
        batch: &ArrayRef,
        is_sorted: bool,
        strict: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let partial_dtype = make_is_sorted_partial_dtype(batch.dtype());
        if batch.is_empty() {
            return Ok(Scalar::null(partial_dtype));
        }
        let first_value = batch.execute_scalar(0, ctx)?.into_nullable();
        let last_value = batch.execute_scalar(batch.len() - 1, ctx)?.into_nullable();
        // SAFETY: We constructed partial_dtype and the children match its field dtypes exactly.
        Ok(unsafe {
            Scalar::struct_unchecked(
                partial_dtype,
                [
                    Scalar::bool(is_sorted, Nullability::NonNullable),
                    Scalar::bool(strict, Nullability::NonNullable),
                    first_value,
                    last_value,
                ],
            )
        })
    }
}

/// Partial accumulator state for is_sorted.
pub struct IsSortedPartial {
    is_sorted: bool,
    /// None = empty (no values seen).
    first_value: Option<Scalar>,
    last_value: Option<Scalar>,
}

impl IsSortedPartial {
    /// The state of a group with no accumulated values.
    fn empty() -> Self {
        Self {
            is_sorted: true,
            first_value: None,
            last_value: None,
        }
    }
}

static NAMES: std::sync::LazyLock<FieldNames> = std::sync::LazyLock::new(|| {
    FieldNames::from(["is_sorted", "strict", "first_value", "last_value"])
});

pub fn make_is_sorted_partial_dtype(element_dtype: &DType) -> DType {
    DType::Struct(
        StructFields::new(
            NAMES.clone(),
            vec![
                DType::Bool(Nullability::NonNullable),
                DType::Bool(Nullability::NonNullable),
                element_dtype.as_nullable(),
                element_dtype.as_nullable(),
            ],
        ),
        Nullability::Nullable,
    )
}

impl AggregateFnVTable for IsSorted {
    type Options = IsSortedOptions;
    type Partial = IsSortedPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.is_sorted");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let mut byte = 0;
        if options.strict {
            byte |= STRICT_BIT;
        }
        if options.direction == SortDirection::Desc {
            byte |= DESC_BIT;
        }
        if options.null_order == NullOrder::Last {
            byte |= NULLS_LAST_BIT;
        }

        Ok(Some(vec![byte]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let &[byte] = metadata else {
            vortex_bail!("Invalid IsSorted options: expected a single byte");
        };

        if byte & !(STRICT_BIT | DESC_BIT | NULLS_LAST_BIT) != 0 {
            vortex_bail!("Invalid IsSorted options: reserved bits set in {byte:#010b}");
        }

        Ok(IsSortedOptions {
            strict: byte & STRICT_BIT != 0,
            direction: if byte & DESC_BIT != 0 {
                SortDirection::Desc
            } else {
                SortDirection::Asc
            },
            null_order: if byte & NULLS_LAST_BIT != 0 {
                NullOrder::Last
            } else {
                NullOrder::First
            },
        })
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::FixedSizeList(..) if is_fixed_size_bytes(input_dtype) => {
                Some(DType::Bool(Nullability::NonNullable))
            }
            DType::Null
            | DType::List(..)
            | DType::FixedSizeList(..)
            | DType::Map(..)
            | DType::Struct(..)
            | DType::Union(..)
            | DType::Variant(..)
            | DType::Extension(_) => None,
            DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_) => Some(DType::Bool(Nullability::NonNullable)),
        }
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::FixedSizeList(..) if is_fixed_size_bytes(input_dtype) => {
                Some(make_is_sorted_partial_dtype(input_dtype))
            }
            DType::Null
            | DType::List(..)
            | DType::FixedSizeList(..)
            | DType::Map(..)
            | DType::Struct(..)
            | DType::Union(..)
            | DType::Variant(..)
            | DType::Extension(_) => None,
            DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_) => Some(make_is_sorted_partial_dtype(input_dtype)),
        }
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(IsSortedPartial::empty())
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        // A null struct means the producing accumulator was empty.
        if scalar.is_null() {
            return Ok(IsSortedPartial::empty());
        }

        let is_sorted = scalar
            .as_struct()
            .field_by_idx(0)
            .map(|s| s.as_bool().value().unwrap_or(false))
            .unwrap_or(false);

        // The scalar's own strict flag is ignored: strictness comes from the options.
        Ok(IsSortedPartial {
            is_sorted,
            first_value: scalar.as_struct().field_by_idx(2),
            last_value: scalar.as_struct().field_by_idx(3),
        })
    }

    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        mut acc: Self::Partial,
        partial: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        if !acc.is_sorted {
            return Ok(acc);
        }

        if !partial.is_sorted {
            // An unsorted partial settles the verdict whether or not it observed boundaries.
            acc.is_sorted = false;
            if let Some(last) = partial.last_value.or_else(|| partial.first_value.clone()) {
                acc.last_value = Some(last);
            }
            if acc.first_value.is_none() {
                acc.first_value = partial.first_value;
            }
            return Ok(acc);
        }

        // A sorted partial without a first value is empty and contributes nothing.
        let Some(first) = partial.first_value else {
            return Ok(acc);
        };
        // A partial that saw a single value carries it as both boundaries.
        let last = partial.last_value.unwrap_or_else(|| first.clone());

        if let Some(acc_last) = &acc.last_value
            && !args.options.boundary_is_sorted(acc_last, &first)
        {
            acc.is_sorted = false;
        }

        if acc.first_value.is_none() {
            acc.first_value = Some(first);
        }
        acc.last_value = Some(last);

        Ok(acc)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let dtype = args.partial_dtype.clone();
        // Only a sorted partial that saw no values is the empty (null) state: an unsorted verdict
        // stands regardless of which boundaries were observed.
        if partial.is_sorted && partial.first_value.is_none() {
            return Ok(Scalar::null(dtype));
        }
        let first_value = partial
            .first_value
            .clone()
            .unwrap_or_else(|| Scalar::null(args.dtype.as_nullable()));
        // A partial that saw a single value carries it as both boundaries.
        let last_value = partial
            .last_value
            .clone()
            .unwrap_or_else(|| first_value.clone());
        Ok(Scalar::struct_(
            dtype,
            vec![
                Scalar::bool(partial.is_sorted, Nullability::NonNullable),
                Scalar::bool(args.options.strict, Nullability::NonNullable),
                first_value,
                last_value,
            ],
        ))
    }

    #[inline]
    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        !partial.is_sorted
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if !partial.is_sorted {
            return Ok(());
        }

        match batch {
            Columnar::Constant(c) => {
                if c.is_empty() {
                    return Ok(());
                }

                // Constant arrays are sorted but not strict sorted (if len > 1).
                let value = c.scalar().clone().into_nullable();
                if args.options.strict && c.len() > 1 {
                    partial.is_sorted = false;
                }

                // Check boundary with previous chunk.
                if let Some(self_last) = &partial.last_value
                    && !args.options.boundary_is_sorted(self_last, &value)
                {
                    partial.is_sorted = false;
                }

                if partial.first_value.is_none() {
                    partial.first_value = Some(value.clone());
                }
                partial.last_value = Some(value);
                Ok(())
            }
            Columnar::Canonical(c) => {
                if c.is_empty() {
                    return Ok(());
                }

                let array_ref = c.clone().into_array();

                // Check boundary with previous chunk.
                let first_value = array_ref.execute_scalar(0, ctx)?.into_nullable();
                if let Some(self_last) = &partial.last_value
                    && !args.options.boundary_is_sorted(self_last, &first_value)
                {
                    partial.is_sorted = false;
                    partial.last_value = Some(
                        array_ref
                            .execute_scalar(array_ref.len() - 1, ctx)?
                            .into_nullable(),
                    );
                    if partial.first_value.is_none() {
                        partial.first_value = Some(first_value);
                    }
                    return Ok(());
                }

                // Check within-batch sortedness.
                let batch_is_sorted = match c {
                    _ if c.len() == 1 => true,
                    Canonical::Primitive(p) => check_primitive_sorted(p, args.options, ctx)?,
                    Canonical::Bool(b) => check_bool_sorted(b, args.options, ctx)?,
                    Canonical::VarBinView(v) => check_varbinview_sorted(v, args.options, ctx)?,
                    Canonical::Decimal(d) => check_decimal_sorted(d, args.options, ctx)?,
                    Canonical::FixedSizeList(f) => {
                        check_fixed_size_bytes_sorted(f, args.options, ctx)?
                    }
                    Canonical::Null(_) => !args.options.strict,
                    // Struct, List, other FixedSizeLists and Extension should have been filtered
                    // out by return_dtype
                    _ => unreachable!(),
                };

                if !batch_is_sorted {
                    partial.is_sorted = false;
                }

                if partial.first_value.is_none() {
                    partial.first_value = Some(first_value);
                }
                partial.last_value = Some(
                    array_ref
                        .execute_scalar(array_ref.len() - 1, ctx)?
                        .into_nullable(),
                );
                Ok(())
            }
        }
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        // A null partial is the empty state of an empty or null group, which is vacuously sorted.
        partials
            .get_item(NAMES.get(0).vortex_expect("out of bounds").clone())?
            .fill_null(true)
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        // The empty state is vacuously sorted, so the verdict stands on its own.
        Ok(Scalar::bool(partial.is_sorted, Nullability::NonNullable))
    }
}

#[expect(
    clippy::wrong_self_convention,
    reason = "is_* naming follows Iterator::is_sorted convention"
)]
/// Helper methods to check sortedness with strictness.
pub trait IsSortedIteratorExt: Iterator
where
    <Self as Iterator>::Item: PartialOrd,
{
    fn is_strict_sorted(self) -> bool
    where
        Self: Sized,
        Self::Item: PartialOrd,
    {
        self.is_sorted_by(|a, b| a < b)
    }
}

impl<T> IsSortedIteratorExt for T
where
    T: Iterator + ?Sized,
    T::Item: PartialOrd,
{
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateDTypes;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::fns::is_sorted::IsSorted;
    use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use crate::aggregate_fn::fns::is_sorted::IsSortedPartial;
    use crate::aggregate_fn::fns::is_sorted::NullOrder;
    use crate::aggregate_fn::fns::is_sorted::SortDirection;
    use crate::aggregate_fn::fns::is_sorted::is_sorted;
    use crate::aggregate_fn::fns::is_sorted::is_sorted_with_options;
    use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::DictArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::runend::RunEnd;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProviderExt;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn singleton_ignores_stale_sortedness_stats() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = buffer![7i32].into_array();
        array
            .statistics()
            .set(Stat::IsSorted, Precision::Exact(false.into()));
        array
            .statistics()
            .set(Stat::IsStrictSorted, Precision::Exact(false.into()));

        assert!(is_sorted(&array, &mut ctx)?);
        assert!(is_strict_sorted(&array, &mut ctx)?);
        Ok(())
    }

    // Tests migrated from compute/is_sorted.rs
    #[test]
    fn test_is_sorted() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let arr = PrimitiveArray::new(buffer!(0, 1, 2, 3), Validity::AllValid).into_array();
        assert!(is_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 2, 3),
            Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
        )
        .into_array();
        assert!(is_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 2, 3),
            Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
        )
        .into_array();
        assert!(!is_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(buffer!(0, 1, 3, 2), Validity::AllValid).into_array();
        assert!(!is_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 3, 2),
            Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
        )
        .into_array();
        assert!(!is_sorted(&arr, &mut ctx)?);

        Ok(())
    }

    #[test]
    fn test_is_strict_sorted() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let arr = PrimitiveArray::new(buffer!(0, 1, 2, 3), Validity::AllValid).into_array();
        assert!(is_strict_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 2, 3),
            Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
        )
        .into_array();
        assert!(is_strict_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 2, 3),
            Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
        )
        .into_array();
        assert!(!is_strict_sorted(&arr, &mut ctx)?);

        let arr = PrimitiveArray::new(
            buffer!(0, 1, 3, 2),
            Validity::Array(BoolArray::from_iter([false, true, true, true]).into_array()),
        )
        .into_array();
        assert!(!is_strict_sorted(&arr, &mut ctx)?);

        Ok(())
    }

    // Tests migrated from arrays/primitive/compute/is_sorted.rs
    #[rstest]
    #[case(PrimitiveArray::from_iter([1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_iter([1, 1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2)]), true)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(1)]), true)]
    #[case(PrimitiveArray::from_option_iter([None, Some(5_u8), None]), false)]
    fn test_primitive_is_sorted(#[case] array: PrimitiveArray, #[case] expected: bool) {
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            is_sorted(&array.into_array(), &mut ctx)
                .vortex_expect("operation should succeed in test"),
            expected
        );
    }

    #[rstest]
    #[case(PrimitiveArray::from_iter([1, 2, 3, 4, 5]), true)]
    #[case(PrimitiveArray::from_iter([1, 1, 2, 3, 4, 5]), false)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(2), None]), false)]
    #[case(PrimitiveArray::from_option_iter([None, None, Some(1i32), Some(1), None]), false)]
    #[case(PrimitiveArray::from_option_iter([None, Some(5_u8), None]), false)]
    fn test_primitive_is_strict_sorted(#[case] array: PrimitiveArray, #[case] expected: bool) {
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            is_strict_sorted(&array.into_array(), &mut ctx)
                .vortex_expect("operation should succeed in test"),
            expected
        );
    }

    // Tests migrated from arrays/decimal/compute/is_sorted.rs
    #[test]
    fn test_decimal_is_sorted() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DecimalDType::new(19, 2);
        // "100.00" and "200.00" at scale 2.
        let i100 = 10_000i128;
        let i200 = 20_000i128;

        let sorted = buffer![i100, i200, i200];
        let unsorted = buffer![i200, i100, i200];

        let sorted_array = DecimalArray::new(sorted, dtype, Validity::NonNullable, &mut ctx);
        let unsorted_array = DecimalArray::new(unsorted, dtype, Validity::NonNullable, &mut ctx);

        assert!(is_sorted(&sorted_array.into_array(), &mut ctx)?);
        assert!(!is_sorted(&unsorted_array.into_array(), &mut ctx)?);

        Ok(())
    }

    #[test]
    fn test_decimal_is_strict_sorted() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // "100.00", "200.00" and "300.00" at scale 2.
        let i100 = 10_000i128;
        let i200 = 20_000i128;
        let i300 = 30_000i128;

        let strict_sorted = buffer![i100, i200, i300];
        let sorted = buffer![i100, i200, i200];

        let dtype = DecimalDType::new(19, 2);

        let strict_sorted_array =
            DecimalArray::new(strict_sorted, dtype, Validity::NonNullable, &mut ctx);
        let sorted_array = DecimalArray::new(sorted, dtype, Validity::NonNullable, &mut ctx);

        assert!(is_strict_sorted(
            &strict_sorted_array.into_array(),
            &mut ctx
        )?);
        assert!(!is_strict_sorted(&sorted_array.into_array(), &mut ctx)?);

        Ok(())
    }

    #[test]
    fn empty_constant_leaves_partial_empty() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = ConstantArray::new(Scalar::from(99i32), 0).into_array();
        let options = IsSortedOptions::default();
        let mut acc = Accumulator::try_new(IsSorted, options, array.dtype().clone())?;

        acc.accumulate(&array, &mut ctx)?;

        assert!(acc.partial_scalar()?.is_null());
        Ok(())
    }

    /// Merging an unsorted partial into a materialized empty one must keep the false verdict in
    /// both the finalized result and the partial scalar.
    #[test]
    fn unsorted_merged_into_empty_keeps_verdict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let options = IsSortedOptions::default();
        let mut empty = Accumulator::try_new(IsSorted, options, dtype.clone())?;
        let mut unsorted = Accumulator::try_new(IsSorted, options, dtype)?;

        // An empty batch materializes the empty partial in place.
        empty.accumulate(
            &PrimitiveArray::new(Buffer::<i32>::empty(), Validity::NonNullable).into_array(),
            &mut ctx,
        )?;
        unsorted.accumulate(&buffer![3i32, 1].into_array(), &mut ctx)?;
        empty.merge_from(&mut unsorted)?;

        assert!(!empty.partial_scalar()?.is_null());
        assert_eq!(
            empty.finish()?,
            Scalar::bool(false, Nullability::NonNullable)
        );
        Ok(())
    }

    /// An unsorted verdict without observed boundaries is not the empty state.
    #[test]
    fn unsorted_partial_without_boundaries_is_not_empty() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let options = IsSortedOptions::default();
        let owned = AggregateDTypes::try_new(&IsSorted, &options, dtype)?;
        let args = owned.args(&options);
        let partial = IsSortedPartial {
            is_sorted: false,
            first_value: None,
            last_value: None,
        };

        let scalar = IsSorted.to_scalar(args, &partial)?;
        assert!(!scalar.is_null());
        let parsed = IsSorted.partial_from_scalar(args, &scalar)?;
        assert_eq!(
            IsSorted.finalize_scalar(args, &parsed)?,
            Scalar::bool(false, Nullability::NonNullable)
        );
        Ok(())
    }

    #[rstest]
    #[case::primitive(PrimitiveArray::from_option_iter([None::<i32>]).into_array())]
    #[case::bool(BoolArray::from_iter([None::<bool>]).into_array())]
    #[case::decimal(
        DecimalArray::new(buffer![0i128], DecimalDType::new(19, 2), Validity::AllInvalid, &mut SESSION.create_execution_ctx()).into_array()
    )]
    #[case::string(VarBinViewArray::from_iter_nullable_str([None::<&str>]).into_array())]
    #[case::constant(ConstantArray::new(Scalar::null_native::<i32>(), 1).into_array())]
    fn singleton_null_is_strict_sorted(#[case] array: ArrayRef) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(buffer![0u32].into_array(), array.clone())?.into_array();
        let runend =
            RunEnd::try_new(buffer![1u32].into_array(), array.clone(), &mut ctx)?.into_array();

        for array in [array, dict, runend] {
            let mut acc = Accumulator::try_new(
                IsSorted,
                IsSortedOptions::default().with_strict(true),
                array.dtype().clone(),
            )?;
            acc.accumulate(&array, &mut ctx)?;

            assert_eq!(acc.finish()?, Scalar::bool(true, Nullability::NonNullable));
            assert!(is_strict_sorted(&array, &mut ctx)?);
        }

        Ok(())
    }

    #[rstest]
    #[case::empty(vec![], true)]
    #[case::null_then_valid(vec![None, Some(1)], true)]
    #[case::valid_then_null(vec![Some(1), None], false)]
    #[case::duplicate_nulls(vec![None, None], false)]
    #[case::duplicate_values(vec![Some(1), Some(1)], false)]
    #[case::increasing_values(vec![Some(1), Some(2)], true)]
    fn strict_sorted_batch_boundaries(
        #[case] values: Vec<Option<i32>>,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let flat = PrimitiveArray::from_option_iter(values.iter().copied()).into_array();
        let chunks = values
            .into_iter()
            .map(|value| PrimitiveArray::from_option_iter([value]).into_array())
            .collect::<Vec<_>>();
        let mut split = Accumulator::try_new(
            IsSorted,
            IsSortedOptions::default().with_strict(true),
            flat.dtype().clone(),
        )?;
        for chunk in &chunks {
            split.accumulate(chunk, &mut ctx)?;
        }

        let expected_scalar = Scalar::bool(expected, Nullability::NonNullable);
        assert_eq!(split.finish()?, expected_scalar);

        let chunked = ChunkedArray::try_new(chunks, flat.dtype().clone())?.into_array();
        for array in [flat, chunked] {
            let mut acc = Accumulator::try_new(
                IsSorted,
                IsSortedOptions::default().with_strict(true),
                array.dtype().clone(),
            )?;
            acc.accumulate(&array, &mut ctx)?;

            assert_eq!(acc.finish()?, expected_scalar);
            assert_eq!(is_strict_sorted(&array, &mut ctx)?, expected);
        }

        Ok(())
    }

    #[rstest]
    #[case::non_strict(false, [true, false, true, true,  true])]
    #[case::strict(    true,  [true, false, true, false, true])]
    fn grouped_is_sorted(#[case] strict: bool, #[case] expected: [bool; 5]) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(1), None, None])
                .into_array();
        let dtype = elements.dtype().clone();

        // The groups are `[1, 2, 3]`, `[3, 1]`, `[]`, `[null, null]`, and a null group.
        let groups = ListViewArray::try_new(
            elements,
            buffer![0i32, 2, 0, 4, 0].into_array(),
            buffer![3i32, 2, 0, 2, 0].into_array(),
            Validity::from_iter([true, true, true, true, false]),
            &mut ctx,
        )?
        .into_array();

        let options = IsSortedOptions::default().with_strict(strict);
        let mut acc = GroupedAccumulator::try_new(IsSorted, options, dtype)?;
        acc.accumulate_list(&groups, &mut ctx)?;

        assert_arrays_eq!(acc.finish()?, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }

    /// A bool array must agree with the equivalent integer array: nulls sort first, so a null
    /// after a valid value is unsorted.
    #[rstest]
    #[case::null_between_values([Some(false), None, Some(true)])]
    #[case::null_first([None, Some(false), Some(true)])]
    #[case::null_last([Some(false), Some(true), None])]
    fn bool_nulls_sort_first_like_primitive(#[case] values: [Option<bool>; 3]) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let bools = BoolArray::from_iter(values).into_array();
        let ints = PrimitiveArray::from_option_iter(values.map(|v| v.map(i32::from))).into_array();

        let expected = values[1..].iter().all(|v| v.is_some());
        assert_eq!(is_sorted(&ints, &mut ctx)?, expected);
        assert_eq!(is_sorted(&bools, &mut ctx)?, expected);
        assert_eq!(
            is_strict_sorted(&bools, &mut ctx)?,
            is_strict_sorted(&ints, &mut ctx)?
        );
        Ok(())
    }

    /// Splitting a bool array into chunks must not change the non-strict verdict.
    #[test]
    fn bool_valid_then_null_matches_chunked() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let flat = BoolArray::from_iter([Some(true), None]).into_array();
        let chunked = ChunkedArray::try_new(
            vec![
                BoolArray::from_iter([Some(true)]).into_array(),
                BoolArray::from_iter([None::<bool>]).into_array(),
            ],
            flat.dtype().clone(),
        )?
        .into_array();

        assert!(!is_sorted(&flat, &mut ctx)?);
        assert!(!is_sorted(&chunked, &mut ctx)?);
        Ok(())
    }

    /// Nullable primitive arrays are covered by `null_order_batch_boundaries`.
    #[rstest]
    #[case::float(buffer![f64::INFINITY, 2.0, -0.0, f64::NEG_INFINITY].into_array())]
    #[case::bool(BoolArray::from_iter([None, Some(true), Some(false)]).into_array())]
    #[case::decimal(DecimalArray::new(
        buffer![3i128, 2, 1], DecimalDType::new(19, 2), Validity::NonNullable,
    ).into_array())]
    #[case::utf8(VarBinViewArray::from_iter_nullable_str([None, Some("zz"), Some("aa")]).into_array())]
    #[case::binary(VarBinViewArray::from_iter_bin([b"zz", b"aa"]).into_array())]
    #[case::fixed_size_bytes(FixedSizeListArray::new(
        buffer![3u8, 0, 2, 0].into_array(), 2, Validity::NonNullable, 2,
    ).into_array())]
    fn descending_supported_types(
        #[case] array: ArrayRef,
        #[values(false, true)] strict: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let options = IsSortedOptions::default()
            .with_strict(strict)
            .with_direction(SortDirection::Desc);

        assert!(is_sorted_with_options(&array, options, &mut ctx)?);
        assert!(!is_sorted(&array, &mut ctx)?);
        Ok(())
    }

    /// Primitive arrays are covered by `null_order_batch_boundaries`.
    #[rstest]
    #[case::bool(BoolArray::from_iter([Some(false), Some(true), None]).into_array())]
    #[case::decimal(DecimalArray::new(
        buffer![1i128, 2, 0],
        DecimalDType::new(19, 2),
        Validity::from_iter([true, true, false]),
    ).into_array())]
    #[case::utf8(VarBinViewArray::from_iter_nullable_str([Some("aa"), Some("zz"), None]).into_array())]
    #[case::fixed_size_bytes(FixedSizeListArray::new(
        buffer![0u8, 2, 0, 3, 0, 0].into_array(), 2, Validity::from_iter([true, true, false]), 3,
    ).into_array())]
    fn nulls_last_supported_types(
        #[case] array: ArrayRef,
        #[values(false, true)] strict: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let options = IsSortedOptions::default()
            .with_strict(strict)
            .with_null_order(NullOrder::Last);

        assert!(is_sorted_with_options(&array, options, &mut ctx)?);
        assert!(!is_sorted_with_options(
            &array,
            IsSortedOptions::default().with_strict(true),
            &mut ctx
        )?);
        Ok(())
    }

    /// Expected verdicts are `[nulls first, nulls last]`, each as `[non-strict, strict]`.
    ///
    /// Empty and single-element arrays are covered by `strict_sorted_batch_boundaries` and
    /// `singleton_null_is_strict_sorted`; their verdict doesn't depend on the options.
    #[rstest]
    #[case::leading_null(
        SortDirection::Desc, vec![None, Some(3), Some(2)], [[true, true], [false, false]]
    )]
    #[case::trailing_null(
        SortDirection::Desc, vec![Some(3), Some(2), None], [[false, false], [true, true]]
    )]
    #[case::increasing(SortDirection::Desc, vec![Some(2), Some(3)], [[false, false], [false, false]])]
    #[case::repeated_trailing_nulls(
        SortDirection::Desc, vec![Some(3), None, None], [[false, false], [true, false]]
    )]
    #[case::asc_leading_null(
        SortDirection::Asc, vec![None, Some(2), Some(3)], [[true, true], [false, false]]
    )]
    #[case::asc_trailing_null(
        SortDirection::Asc, vec![Some(2), Some(3), None], [[false, false], [true, true]]
    )]
    #[case::asc_interior_null(
        SortDirection::Asc, vec![Some(2), None, Some(3)], [[false, false], [false, false]]
    )]
    fn null_order_batch_boundaries(
        #[case] direction: SortDirection,
        #[case] values: Vec<Option<i32>>,
        #[case] expected: [[bool; 2]; 2],
        #[values(NullOrder::First, NullOrder::Last)] null_order: NullOrder,
        #[values(false, true)] strict: bool,
    ) -> VortexResult<()> {
        let expected = expected[usize::from(null_order == NullOrder::Last)][usize::from(strict)];
        let mut ctx = SESSION.create_execution_ctx();
        let options = IsSortedOptions {
            strict,
            direction,
            null_order,
        };
        let flat = PrimitiveArray::from_option_iter(values.iter().copied()).into_array();
        assert_eq!(is_sorted_with_options(&flat, options, &mut ctx)?, expected);

        let mut split = Accumulator::try_new(IsSorted, options, flat.dtype().clone())?;
        let mut merged = Accumulator::try_new(IsSorted, options, flat.dtype().clone())?;
        for value in values {
            let chunk = PrimitiveArray::from_option_iter([value]).into_array();
            split.accumulate(&chunk, &mut ctx)?;

            let mut single = Accumulator::try_new(IsSorted, options, flat.dtype().clone())?;
            single.accumulate(&chunk, &mut ctx)?;
            merged.merge_from(&mut single)?;
        }

        let expected = Scalar::bool(expected, Nullability::NonNullable);
        assert_eq!(split.finish()?, expected);
        assert_eq!(merged.finish()?, expected);

        Ok(())
    }

    #[rstest]
    #[case::descending(
        buffer![1i32, 2, 3].into_array(), SortDirection::Desc, NullOrder::First, false
    )]
    #[case::descending_sorted(
        buffer![3i32, 2, 1].into_array(), SortDirection::Desc, NullOrder::First, true
    )]
    #[case::nulls_last(
        PrimitiveArray::from_option_iter([None, Some(1i32), Some(2)]).into_array(),
        SortDirection::Asc,
        NullOrder::Last,
        false,
    )]
    #[case::nulls_last_sorted(
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), None]).into_array(),
        SortDirection::Asc,
        NullOrder::Last,
        true,
    )]
    fn non_default_order_ignores_stats(
        #[case] array: ArrayRef,
        #[case] direction: SortDirection,
        #[case] null_order: NullOrder,
        #[case] sorted: bool,
        #[values(false, true)] strict: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let stats = [Stat::IsSorted, Stat::IsStrictSorted];
        for stat in stats {
            array
                .statistics()
                .set(stat, Precision::Exact((!sorted).into()));
        }
        let options = IsSortedOptions {
            strict,
            direction,
            null_order,
        };

        assert_eq!(is_sorted_with_options(&array, options, &mut ctx)?, sorted);

        for stat in stats {
            assert_eq!(
                array.statistics().get_as::<bool>(stat),
                Precision::Exact(!sorted),
            );
        }

        Ok(())
    }

    fn known_without_nulls() -> ArrayRef {
        let array = PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3)]).into_array();
        array
            .statistics()
            .set(Stat::NullCount, Precision::Exact(0u64.into()));
        array
    }

    /// Without nulls, nulls last has the same verdict as nulls first, so it reads the ascending
    /// statistics. The planted verdict is stale, so the result shows whether they were read.
    #[rstest]
    #[case::non_nullable(buffer![1i32, 2, 3].into_array(), true)]
    #[case::known_without_nulls(known_without_nulls(), true)]
    #[case::unknown_null_count(
        PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3)]).into_array(),
        false,
    )]
    fn nulls_last_reads_stats_without_nulls(
        #[case] array: ArrayRef,
        #[case] reads_stats: bool,
        #[values(false, true)] strict: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        for stat in [Stat::IsSorted, Stat::IsStrictSorted] {
            array.statistics().set(stat, Precision::Exact(false.into()));
        }
        let options = IsSortedOptions::default()
            .with_strict(strict)
            .with_null_order(NullOrder::Last);

        assert_eq!(
            is_sorted_with_options(&array, options, &mut ctx)?,
            !reads_stats
        );
        Ok(())
    }

    #[rstest]
    #[case::ascending(false, SortDirection::Asc, NullOrder::First, 0)]
    #[case::strict_ascending(true, SortDirection::Asc, NullOrder::First, 1)]
    #[case::descending(false, SortDirection::Desc, NullOrder::First, 2)]
    #[case::strict_descending(true, SortDirection::Desc, NullOrder::First, 3)]
    #[case::ascending_nulls_last(false, SortDirection::Asc, NullOrder::Last, 4)]
    #[case::strict_ascending_nulls_last(true, SortDirection::Asc, NullOrder::Last, 5)]
    #[case::descending_nulls_last(false, SortDirection::Desc, NullOrder::Last, 6)]
    #[case::strict_descending_nulls_last(true, SortDirection::Desc, NullOrder::Last, 7)]
    fn options_metadata(
        #[case] strict: bool,
        #[case] direction: SortDirection,
        #[case] null_order: NullOrder,
        #[case] byte: u8,
    ) -> VortexResult<()> {
        let options = IsSortedOptions {
            strict,
            direction,
            null_order,
        };
        assert_eq!(IsSorted.serialize(&options)?, Some(vec![byte]));
        assert_eq!(IsSorted.deserialize(&[byte], &SESSION)?, options);
        Ok(())
    }
}

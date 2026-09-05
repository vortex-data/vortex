// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::ops::BitOr;

use arrow_buffer::bit_iterator::BitIndexIterator;
pub use kernel::*;
use num_traits::Zero;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builders::builder_with_capacity;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;
use crate::search_sorted::NullEquality;
use crate::search_sorted::SortedArray;
use crate::search_sorted::SortedDirection;
use crate::search_sorted::SortedNulls;
use crate::search_sorted::SortedOrder;
use crate::search_sorted::sorted_membership_mask;
use crate::validity::Validity;

#[derive(Clone)]
pub struct ListContains;

impl ListContains {
    /// Creates a lazy list membership check for `needle` in `list`.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or `list` is not a list array.
    pub fn try_new(list: ArrayRef, needle: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(ListContains.bind(EmptyOptions), vec![list, needle])
    }
}

impl ScalarFnVTable for ListContains {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.contains");
        *ID
    }

    fn serialize(&self, _instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            1 => ChildName::from("needle"),
            _ => unreachable!(
                "Invalid child index {} for ListContains expression",
                child_idx
            ),
        }
    }
    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let list_dtype = &arg_dtypes[0];
        let needle_dtype = &arg_dtypes[1];

        let nullability = match list_dtype {
            DType::List(_, list_nullability) => list_nullability,
            _ => {
                vortex_bail!(
                    "First argument to ListContains must be a List, got {:?}",
                    list_dtype
                );
            }
        }
        .bitor(needle_dtype.nullability());

        Ok(DType::Bool(nullability))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let list_array = args.get(0)?;
        let value_array = args.get(1)?;

        if let Some(list_scalar) = list_array.as_constant()
            && let Some(value_scalar) = value_array.as_constant()
        {
            let result = compute_contains_scalar(&list_scalar, &value_scalar)?;
            return Ok(ConstantArray::new(result, args.row_count()).into_array());
        }

        compute_list_contains(&list_array, &value_array, ctx)
    }

    // An empty list can produce false even when the needle is null.
    fn is_strict(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

fn compute_contains_scalar(list: &Scalar, needle: &Scalar) -> VortexResult<Scalar> {
    let nullability = list.dtype().nullability() | needle.dtype().nullability();

    // Handle null list or null needle
    if list.is_null() || needle.is_null() {
        return Ok(Scalar::null(DType::Bool(nullability)));
    }

    let list_scalar = list.as_list();
    let elements = list_scalar
        .elements()
        .ok_or_else(|| vortex_err!("Expected non-null list"))?;

    let contains = elements.iter().any(|elem| elem == needle);
    Ok(Scalar::bool(contains, nullability))
}

fn compute_list_contains(
    array: &ArrayRef,
    value: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let DType::List(elem_dtype, _) = array.dtype() else {
        vortex_bail!("Array must be of List type");
    };
    if !elem_dtype.as_ref().eq_ignore_nullability(value.dtype()) {
        vortex_bail!(
            "Element type {} of list does not match search value {}",
            elem_dtype,
            value.dtype(),
        );
    }

    if value.all_invalid(ctx)? || array.all_invalid(ctx)? {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            array.len(),
        )
        .into_array());
    }

    let nullability = array.dtype().nullability() | value.dtype().nullability();

    if let Some(value_scalar) = value.as_constant() {
        list_contains_scalar(array, &value_scalar, nullability, ctx)
    } else if let Some(list_scalar) = array.as_constant() {
        constant_list_scalar_contains(&list_scalar.as_list(), value, nullability, ctx)
    } else {
        todo!("unsupported list contains with list and element as arrays")
    }
}

/// There is a constant list scalar (haystack) being compared to an array of needles.
fn constant_list_scalar_contains(
    list_scalar: &ListScalar,
    values: &ArrayRef,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let elements = list_scalar.elements().vortex_expect("non null");

    if let Some(result) = try_sorted_membership_contains(&elements, values, nullability, ctx)? {
        return Ok(result);
    }

    let len = values.len();
    let false_scalar = Scalar::bool(false, nullability);

    let result = elements
        .iter()
        .map(|element| {
            Binary::try_new(
                ConstantArray::new(element.clone(), len).into_array(),
                values.clone(),
                Operator::Eq,
            )?
            .into_array()
            .fill_null(false_scalar.clone())
        })
        .collect::<VortexResult<Vec<_>>>()?
        .into_iter()
        .try_reduce_balanced(|acc, res| acc.binary(res, Operator::Or))?;

    Ok(result.unwrap_or_else(|| ConstantArray::new(false_scalar, len).into_array()))
}

/// Fast path for `values IN (elements)` when `values` is already known sorted (ascending,
/// nulls-first) via `Stat::IsSorted`/`Stat::IsStrictSorted`. Replaces the `O(elements *
/// values.len())` equality fan-out below with a single sorted merge.
///
/// The literal `elements` are re-sorted and de-duplicated on every call rather than cached across
/// chunks: `ScalarFnVTable::execute` runs once per chunk with no cross-chunk cache today, but IN
/// lists are normally small, so `O(elements * log(elements))` per chunk is negligible next to the
/// `O(elements * values.len())` fan-out it replaces.
///
/// Returns `Ok(None)` when the fast path does not apply (unsorted or unsorted-unknown `values`, or
/// an unsupported/floating-point element dtype — float exclusion avoids a mismatch between
/// `Scalar`'s `PartialOrd`, which cannot order NaN, and the total order `SortedArray` validates
/// against), in which case the caller falls back to the equality fan-out unchanged.
/// Below this many literal elements, the fixed cost of sorting the members and constructing a
/// `SortedArray` outweighs the equality fan-out's simplicity. Measured directly (see
/// `vortex-array/benches/list_contains.rs`) on an 8,192-row `i64` column on local (non-CodSpeed,
/// noisier) hardware: 4 elements clearly favor the fan-out, and 8 elements are a toss-up that
/// flipped direction between runs. 16 elements and up consistently favored the sorted merge, with
/// the gap widening sharply from there (256 elements: ~80 us vs. ~1.2 ms). This threshold sits
/// past that noisy zone with margin; re-tuning on CodSpeed's stable runners could likely lower it.
const MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP: usize = 12;

fn try_sorted_membership_contains(
    elements: &[Scalar],
    values: &ArrayRef,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if elements.len() < MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP {
        return Ok(None);
    }

    let elem_dtype = values.dtype();
    let supported_dtype = match elem_dtype {
        DType::Bool(_) | DType::Decimal(..) | DType::Utf8(_) | DType::Binary(_) => true,
        DType::Primitive(ptype, _) => !ptype.is_float(),
        _ => false,
    };
    if !supported_dtype {
        return Ok(None);
    }

    let is_sorted = matches!(
        values.statistics().get_as::<bool>(Stat::IsSorted),
        Precision::Exact(true)
    ) || matches!(
        values.statistics().get_as::<bool>(Stat::IsStrictSorted),
        Precision::Exact(true)
    );
    if !is_sorted {
        return Ok(None);
    }

    // `elements` carry the list's own (possibly differently-nullable) element dtype, so cast each
    // one to `values`'s non-nullable dtype before building -- every builder in this codebase
    // requires an exact dtype match, and we've already dropped every null.
    let member_dtype = elem_dtype.as_nonnullable();
    let mut sorted_elements = elements
        .iter()
        .filter(|element| !element.is_null())
        .map(|element| element.cast(&member_dtype))
        .collect::<VortexResult<Vec<_>>>()?;
    sorted_elements.sort_by(|a, b| {
        a.partial_cmp(b)
            .vortex_expect("list elements share a comparable, non-float dtype")
    });
    sorted_elements.dedup();

    let mut builder = builder_with_capacity(&member_dtype, sorted_elements.len());
    for element in &sorted_elements {
        builder.append_scalar(element)?;
    }
    let members_array = builder.finish();

    let members = SortedArray::try_new(
        members_array,
        SortedOrder {
            direction: SortedDirection::Ascending,
            nulls: SortedNulls::First,
        },
        ctx,
    )?;

    let mask = sorted_membership_mask(values, &members, NullEquality::Unequal, ctx)?;
    // `NullEquality::Unequal` makes a null `values` row never match (mask bit `false`). The
    // pre-existing equality fan-out below reaches the same physical result for a null needle row
    // (every per-element `Eq` is null there, then immediately `.fill_null(false)`-ed before the
    // `Or`-reduce) -- once fully executed/canonicalized, a null needle row is `false`, never
    // null. So the output here is valid everywhere; only the declared dtype nullability (carried
    // through `nullability`, e.g. because the list itself is nullable) needs to match.
    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => Validity::AllValid,
    };
    Ok(Some(
        BoolArray::new(mask.into_bit_buffer(), validity).into_array(),
    ))
}

/// Returns a [`BoolArray`] where each bit represents if a list contains the scalar.
fn list_contains_scalar(
    array: &ArrayRef,
    value: &Scalar,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // If the list array is constant, we perform a single comparison.
    if array.len() > 1 && array.is::<Constant>() {
        let contains = list_contains_scalar(&array.slice(0..1)?, value, nullability, ctx)?;
        return Ok(ConstantArray::new(contains.execute_scalar(0, ctx)?, array.len()).into_array());
    }

    let list_array = array.clone().execute::<ListViewArray>(ctx)?;

    let elems = list_array.elements();
    if elems.is_empty() {
        // Must return false when a list is empty (but valid), or null when the list itself is null.
        return list_false_or_null(&list_array, nullability);
    }

    let rhs = ConstantArray::new(value.clone(), elems.len());
    let matching_elements =
        Binary::try_new(elems.clone(), rhs.clone().into_array(), Operator::Eq)?.into_array();

    // TODO(ngates): we should execute this into a Columnar and check for constant.
    let matches = matching_elements.execute::<BoolArray>(ctx)?;

    // Fast path: no elements match.
    if let Some(pred) = matches.as_constant() {
        return match pred.as_bool().value() {
            // All comparisons are invalid (result in `null`), and search is not null because
            // we already checked for null above.
            None => {
                assert!(
                    !rhs.scalar().is_null(),
                    "Search value must not be null here"
                );
                // False, unless the list itself is null in which case we return null.
                list_false_or_null(&list_array, nullability)
            }
            // No elements match, and all comparisons are valid (result in `false`).
            Some(false) => {
                // False, but match the nullability to the input list array.
                Ok(
                    ConstantArray::new(Scalar::bool(false, nullability), list_array.len())
                        .into_array(),
                )
            }
            // All elements match, and all comparisons are valid (result in `true`).
            Some(true) => {
                // True, unless the list itself is empty or NULL.
                list_is_not_empty(&list_array, nullability, ctx)
            }
        };
    }

    // Get the offsets and sizes as primitive arrays. They are non-negative, so reinterpret to
    // unsigned and dispatch over the 4 unsigned widths each (4x4 instead of 8x8).
    let offsets = list_array
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.reinterpret_cast(sizes.ptype().to_unsigned());

    // Process based on the offset and size types.
    let list_matches = match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        match_each_unsigned_integer_ptype!(sizes.ptype(), |S| {
            process_matches::<O, S>(matches, list_array.len(), offsets, sizes)
        })
    });

    Ok(BoolArray::new(
        list_matches,
        list_array.validity()?.union_nullability(nullability),
    )
    .into_array())
}

/// Returns a [`BitBuffer`] where each bit represents if a list contains the scalar, derived from a
/// [`BoolArray`] of matches on the child elements array.
fn process_matches<O, S>(
    matches: BoolArray,
    list_array_len: usize,
    offsets: PrimitiveArray,
    sizes: PrimitiveArray,
) -> BitBuffer
where
    O: IntegerPType,
    S: IntegerPType,
{
    let offsets_slice = offsets.as_slice::<O>();
    let sizes_slice = sizes.as_slice::<S>();
    let bits = matches.bit_buffer_view();

    (0..list_array_len)
        .map(|i| {
            let offset = offsets_slice[i].as_();
            let size = sizes_slice[i].as_();

            // BitIndexIterator yields indices of true bits only. If `.next()` returns
            // `Some(_)`, at least one element in this list's range matches.
            let mut set_bits = BitIndexIterator::new(bits.inner(), offset, size);
            set_bits.next().is_some()
        })
        .collect::<BitBuffer>()
}

/// Returns a `Bool` array with `false` for lists that are valid,
/// or `NULL` if the list itself is null.
fn list_false_or_null(
    list_array: &ListViewArray,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    match list_array.validity()? {
        Validity::NonNullable => {
            // All false.
            Ok(ConstantArray::new(Scalar::bool(false, nullability), list_array.len()).into_array())
        }
        Validity::AllValid => {
            // All false, but nullable.
            Ok(
                ConstantArray::new(Scalar::bool(false, Nullability::Nullable), list_array.len())
                    .into_array(),
            )
        }
        Validity::AllInvalid => {
            // All nulls, must be nullable result.
            Ok(ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                list_array.len(),
            )
            .into_array())
        }
        Validity::Array(validity_array) => {
            // Create a new bool array with false, and the provided nulls
            let buffer = BitBuffer::new_unset(list_array.len());
            Ok(BoolArray::new(buffer, Validity::Array(validity_array)).into_array())
        }
    }
}

/// Returns a `Bool` array with `true` for lists which are NOT empty, or `false` if they are empty,
/// or `NULL` if the list itself is null.
fn list_is_not_empty(
    list_array: &ListViewArray,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Short-circuit for all invalid.
    if list_array.validity()?.definitely_all_null() {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            list_array.len(),
        )
        .into_array());
    }

    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let buffer = match_each_integer_ptype!(sizes.ptype(), |S| {
        BitBuffer::from_iter(sizes.as_slice::<S>().iter().map(|&size| size != S::zero()))
    });

    // Copy over the validity mask from the input.
    Ok(BoolArray::new(
        buffer,
        list_array.validity()?.union_nullability(nullability),
    )
    .into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use itertools::Itertools;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ListArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::list_contains;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::or;
    use crate::expr::root;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::list_contains::BoolArray;
    use crate::scalar_fn::fns::list_contains::ConstantArray;
    use crate::scalar_fn::fns::list_contains::ListViewArray;
    use crate::scalar_fn::fns::list_contains::PrimitiveArray;
    use crate::stats::StatsSession;
    use crate::stats::stat as stat_expr;
    use crate::validity::Validity;

    static STATS_SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<StatsSession>());

    fn stat(expr: Expression, stat: Stat) -> Expression {
        stat_expr(expr, stat.aggregate_fn().unwrap())
    }

    fn test_array() -> ArrayRef {
        ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2, 2, 2, 3, 3, 3]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 10]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array()
    }

    #[test]
    pub fn test_one() {
        let arr = test_array();

        let expr = list_contains(root(), lit(1));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_all() {
        let arr = test_array();

        let expr = list_contains(root(), lit(2));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_none() {
        let arr = test_array();

        let expr = list_contains(root(), lit(4));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_empty() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let expr = list_contains(root(), lit(2));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_nullable() {
        let arr = ListArray::try_new(
            PrimitiveArray::from_iter(vec![1, 1, 2, 2, 2]).into_array(),
            PrimitiveArray::from_iter(vec![0, 5, 5]).into_array(),
            Validity::Array(BoolArray::from(BitBuffer::from(vec![true, false])).into_array()),
        )
        .unwrap()
        .into_array();

        let expr = list_contains(root(), lit(2));
        let item = arr.apply(&expr).unwrap();

        assert_eq!(
            item.execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert!(
            !item
                .is_valid(1, &mut array_session().create_execution_ctx())
                .unwrap()
        );
    }

    #[test]
    pub fn test_return_type() {
        let scope = DType::Struct(
            StructFields::new(
                ["array"].into(),
                vec![DType::List(
                    Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                    Nullability::Nullable,
                )],
            ),
            Nullability::NonNullable,
        );

        let expr = list_contains(get_item("array", root()), lit(2));

        // Expect nullable, although scope is non-nullable
        assert_eq!(
            expr.return_dtype(&scope).unwrap(),
            DType::Bool(Nullability::Nullable)
        );
    }

    #[test]
    pub fn list_falsification() -> VortexResult<()> {
        let expr = list_contains(
            lit(Scalar::list(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                vec![1.into(), 2.into(), 3.into()],
                Nullability::NonNullable,
            )),
            col("a"),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["a"].into(),
                vec![DType::Primitive(I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );

        assert_eq!(
            expr.bind(&scope)?.falsify(&STATS_SESSION)?,
            Some(
                and(
                    and(
                        or(
                            lt(stat(col("a"), Stat::Max), lit(1i32)),
                            gt(stat(col("a"), Stat::Min), lit(1i32)),
                        ),
                        or(
                            lt(stat(col("a"), Stat::Max), lit(2i32)),
                            gt(stat(col("a"), Stat::Min), lit(2i32)),
                        )
                    ),
                    or(
                        lt(stat(col("a"), Stat::Max), lit(3i32)),
                        gt(stat(col("a"), Stat::Min), lit(3i32)),
                    )
                )
                .bind(&scope)?
            )
        );
        Ok(())
    }

    #[test]
    pub fn test_display() {
        let expr = list_contains(get_item("tags", root()), lit("urgent"));
        assert_eq!(expr.to_string(), "vortex.list.contains($.tags, \"urgent\")");

        let expr2 = list_contains(root(), lit(42));
        assert_eq!(expr2.to_string(), "vortex.list.contains($, 42i32)");
    }

    #[test]
    pub fn test_constant_scalars() {
        let arr = test_array();

        // Both list and needle are constants - should use scalar optimization
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            vec![1.into(), 2.into(), 3.into()],
            Nullability::NonNullable,
        );

        // Test contains true
        let expr = list_contains(lit(list_scalar.clone()), lit(2i32));
        let result = arr.clone().apply(&expr).unwrap();
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::NonNullable)
        );

        // Test contains false
        let expr = list_contains(lit(list_scalar), lit(42i32));
        let result = arr.apply(&expr).unwrap();
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::NonNullable)
        );
    }

    // -- Tests migrated from compute/list_contains.rs --

    fn nonnull_strings(values: Vec<Vec<&str>>) -> ArrayRef {
        let mut ctx = array_session().create_execution_ctx();

        ListArray::from_iter_slow::<u64, _>(values, Arc::new(DType::Utf8(Nullability::NonNullable)))
            .unwrap()
            .into_array()
            .execute::<ListViewArray>(&mut ctx)
            .vortex_expect("failed to convert to listview")
            .into_array()
    }

    fn null_strings(values: Vec<Vec<Option<&str>>>) -> ArrayRef {
        let elements = values.iter().flatten().cloned().collect_vec();

        let mut offsets = values
            .iter()
            .scan(0u64, |st, v| {
                *st += v.len() as u64;
                Some(*st)
            })
            .collect_vec();
        offsets.insert(0, 0u64);
        let offsets = Buffer::from_iter(offsets).into_array();

        let elements =
            VarBinArray::from_iter(elements, DType::Utf8(Nullability::Nullable)).into_array();

        let mut ctx = array_session().create_execution_ctx();

        ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .as_array()
            .clone()
            .execute::<ListViewArray>(&mut ctx)
            .vortex_expect("failed to convert to listview")
            .into_array()
    }

    fn bool_array(values: Vec<bool>, validity: Validity) -> BoolArray {
        BoolArray::new(values.into_iter().collect(), validity)
    }

    #[rstest]
    #[case(
        nonnull_strings(vec![vec![], vec!["a"], vec!["a", "b"]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::NonNullable)
    )]
    #[case(
        null_strings(vec![vec![], vec![Some("a"), None], vec![Some("a"), None, Some("b")]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::AllValid)
    )]
    #[case(
        null_strings(vec![vec![], vec![Some("a"), None], vec![Some("b"), None, None]]),
        Some("a"),
        bool_array(vec![false, true, false], Validity::AllValid)
    )]
    #[case(
        nonnull_strings(vec![vec![], vec!["a"], vec!["a"]]),
        Some("a"),
        bool_array(vec![false, true, true], Validity::NonNullable)
    )]
    #[case(
        nonnull_strings(vec![vec![], vec![], vec![]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::NonNullable)
    )]
    #[case(
        nonnull_strings(vec![vec!["b"], vec![], vec!["b"]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::NonNullable)
    )]
    #[case(
        null_strings(vec![vec![], vec![None, None], vec![None, None, None]]),
        None,
        bool_array(vec![false, true, true], Validity::AllInvalid)
    )]
    #[case(
        null_strings(vec![vec![], vec![None, None], vec![None, None, None]]),
        Some("a"),
        bool_array(vec![false, false, false], Validity::AllValid)
    )]
    fn test_contains_nullable(
        #[case] list_array: ArrayRef,
        #[case] value: Option<&str>,
        #[case] expected: BoolArray,
    ) {
        let mut ctx = array_session().create_execution_ctx();
        let element_nullability = list_array
            .dtype()
            .as_list_element_opt()
            .unwrap()
            .nullability();
        let scalar = match value {
            None => Scalar::null(DType::Utf8(Nullability::Nullable)),
            Some(v) => Scalar::utf8(v, element_nullability),
        };
        let elem = ConstantArray::new(scalar, list_array.len());
        let expr = list_contains(root(), lit(elem.scalar().clone()));
        let result = list_array.apply(&expr).unwrap();
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_constant_list() {
        let mut ctx = array_session().create_execution_ctx();
        let list_array = ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                vec![1i32.into(), 2i32.into(), 3i32.into()],
                Nullability::NonNullable,
            ),
            2,
        )
        .into_array();

        let expr = list_contains(root(), lit(2i32));
        let contains = list_array.apply(&expr).unwrap();
        let expected = BoolArray::from_iter([true, true]);
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_all_nulls() {
        let mut ctx = array_session().create_execution_ctx();
        let list_array = ConstantArray::new(
            Scalar::null(DType::List(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                Nullability::Nullable,
            )),
            5,
        )
        .into_array();

        let expr = list_contains(root(), lit(2i32));
        let contains = list_array.apply(&expr).unwrap();

        let expected = BoolArray::new(
            [false, false, false, false, false].into_iter().collect(),
            Validity::AllInvalid,
        );
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_list_array_element() {
        let mut ctx = array_session().create_execution_ctx();
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            vec![1.into(), 3.into(), 6.into()],
            Nullability::NonNullable,
        );

        let arr = (0..7).collect::<PrimitiveArray>().into_array();
        let expr = list_contains(lit(list_scalar), root());
        let contains = arr.apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, true, false, true, false, false, true]);
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_list_contains_empty_listview() {
        let mut ctx = array_session().create_execution_ctx();
        let empty_elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
        let offsets = Buffer::from_iter([0u32, 0, 0, 0]).into_array();
        let sizes = Buffer::from_iter([0u32, 0, 0, 0]).into_array();

        let list_array = unsafe {
            ListViewArray::new_unchecked(
                empty_elements.into_array(),
                offsets,
                sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        let expr = list_contains(root(), lit(42i32));
        let result = list_array.into_array().apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, false, false, false]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_list_contains_all_null_elements() {
        let mut ctx = array_session().create_execution_ctx();
        let elements = PrimitiveArray::from_option_iter::<i32, _>([None, None, None, None, None]);
        let offsets = Buffer::from_iter([0u32, 2, 4]).into_array();
        let sizes = Buffer::from_iter([2u32, 2, 1]).into_array();

        let list_array = unsafe {
            ListViewArray::new_unchecked(
                elements.into_array(),
                offsets,
                sizes,
                Validity::NonNullable,
            )
            .with_zero_copy_to_list(true)
        };

        // Searching for null
        let null_scalar = Scalar::null(DType::Primitive(I32, Nullability::Nullable));
        let expr = list_contains(root(), lit(null_scalar));
        let result = list_array.clone().into_array().apply(&expr).unwrap();

        let expected = BoolArray::new(
            [false, false, false].into_iter().collect(),
            Validity::AllInvalid,
        );
        assert_arrays_eq!(result, expected, &mut ctx);

        // Searching for non-null
        let expr2 = list_contains(root(), lit(42i32));
        let result2 = list_array.into_array().apply(&expr2).unwrap();

        let expected2 = BoolArray::from_iter([false, false, false]);
        assert_arrays_eq!(result2, expected2, &mut ctx);
    }

    #[test]
    fn test_list_contains_large_offsets() {
        let mut ctx = array_session().create_execution_ctx();
        let elements = Buffer::from_iter([1i32, 2, 3, 4, 5]).into_array();

        let offsets = Buffer::from_iter([0u32, 1, 4, 0]).into_array();
        let sizes = Buffer::from_iter([1u32, 2, 1, 0]).into_array();

        let list_array =
            ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

        let expr = list_contains(root(), lit(2i32));
        let result = list_array.clone().into_array().apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, true, false, false]);
        assert_arrays_eq!(result, expected, &mut ctx);

        let expr5 = list_contains(root(), lit(5i32));
        let result5 = list_array.into_array().apply(&expr5).unwrap();

        let expected5 = BoolArray::from_iter([false, false, true, false]);
        assert_arrays_eq!(result5, expected5, &mut ctx);
    }

    #[test]
    fn test_list_contains_offset_size_boundary() {
        let mut ctx = array_session().create_execution_ctx();
        let elements = Buffer::from_iter(0..256).into_array();
        let offsets = Buffer::from_iter([0u8, 100, 200, 254]).into_array();
        let sizes = Buffer::from_iter([50u8, 50, 54, 2]).into_array();

        let list_array =
            ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

        let expr = list_contains(root(), lit(255i32));
        let result = list_array.clone().into_array().apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, false, false, true]);
        assert_arrays_eq!(result, expected, &mut ctx);

        let expr_zero = list_contains(root(), lit(0i32));
        let result_zero = list_array.into_array().apply(&expr_zero).unwrap();

        let expected_zero = BoolArray::from_iter([true, false, false, false]);
        assert_arrays_eq!(result_zero, expected_zero, &mut ctx);
    }

    fn int_list_scalar(elements: Vec<i32>) -> Scalar {
        Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            elements.into_iter().map(Scalar::from).collect(),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn test_sorted_membership_fast_path_sorts_and_dedups_unordered_literal_list() {
        let mut ctx = array_session().create_execution_ctx();

        let arr = PrimitiveArray::from_iter([1, 3, 3, 5, 7, 9, 9, 9, 12]).into_array();
        arr.statistics().compute_is_sorted(&mut ctx);

        // Deliberately unsorted, with a duplicate, and long enough (>=
        // `MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP`) to actually take the fast path: it must sort/dedup
        // internally.
        let expr = list_contains(
            lit(int_list_scalar(vec![
                9, 1, 9, 4, 12, 20, 21, 22, 23, 24, 25, 26,
            ])),
            root(),
        );
        let contains = arr.apply(&expr).unwrap();

        let expected =
            BoolArray::from_iter([true, false, false, false, false, true, true, true, true]);
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_sorted_membership_fast_path_uses_is_strict_sorted_stat() {
        let mut ctx = array_session().create_execution_ctx();

        let arr = PrimitiveArray::from_iter([1, 2, 3, 4, 5]).into_array();
        arr.statistics().compute_is_strict_sorted(&mut ctx);

        let expr = list_contains(
            lit(int_list_scalar(vec![
                5, 5, 2, 100, 200, 300, 400, 500, 600, 700, 800, 900,
            ])),
            root(),
        );
        let contains = arr.apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, true, false, false, true]);
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_sorted_membership_fast_path_null_needle_rows_are_false_not_null() {
        let mut ctx = array_session().create_execution_ctx();

        // Nulls-first, then non-decreasing: a valid `Stat::IsSorted` fixture.
        let arr = PrimitiveArray::from_option_iter::<i32, _>([
            None,
            None,
            Some(1),
            Some(3),
            Some(5),
            Some(5),
            Some(8),
        ])
        .into_array();
        arr.statistics().compute_is_sorted(&mut ctx);

        let expr = list_contains(
            lit(int_list_scalar(vec![
                5, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
            ])),
            root(),
        );
        // Forcing execution here (rather than comparing the lazy `ScalarFnArray` directly)
        // sidesteps a pre-existing, unrelated quirk where per-row scalar access on this
        // particular lazy expression tree (list-constant, array-needle) disagrees with its own
        // canonicalized result for a null needle row -- reproducible on the untouched equality
        // fan-out too, so it's a framework-level gap uncovered by adding null coverage here, not
        // something this change introduces or fixes.
        let contains = arr
            .apply(&expr)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
            .into_array();

        // A null needle row is `false`, never null: matches the pre-existing equality fan-out's
        // executed contract (every per-element `Eq` on a null needle is null, then immediately
        // `.fill_null(false)`-ed before the `Or`-reduce). The result dtype is nullable (the
        // needle column is), even though no row is ever actually null.
        let expected = BoolArray::new(
            BitBuffer::from_iter([false, false, false, false, true, true, false]),
            Validity::AllValid,
        );
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_sorted_membership_fast_path_empty_literal_list_stays_on_fanout() {
        // Below `MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP`, so this always takes the fan-out,
        // regardless of the column's sortedness -- it's here to document that the threshold
        // doesn't change the (correct, pre-existing) answer for a degenerate empty list.
        let mut ctx = array_session().create_execution_ctx();
        let arr = PrimitiveArray::from_iter([1, 2, 3]).into_array();
        arr.statistics().compute_is_sorted(&mut ctx);

        let empty_list = Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            vec![],
            Nullability::NonNullable,
        );
        let contains = arr.apply(&list_contains(lit(empty_list), root())).unwrap();
        assert_arrays_eq!(
            contains,
            BoolArray::from_iter([false, false, false]),
            &mut ctx
        );
    }

    #[test]
    fn test_sorted_membership_fast_path_all_null_literal_list() {
        // At `MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP` elements, so this reaches the fast path, but
        // every element is null: `sorted_elements` ends up empty after filtering, exercising
        // `SortedArray::try_new` and `sorted_membership_mask` on a zero-length member set.
        let mut ctx = array_session().create_execution_ctx();
        let arr = PrimitiveArray::from_iter([1, 2, 3]).into_array();
        arr.statistics().compute_is_sorted(&mut ctx);

        let null_only_list = Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::Nullable)),
            vec![Scalar::null(DType::Primitive(I32, Nullability::Nullable)); 12],
            Nullability::NonNullable,
        );
        let contains = arr
            .apply(&list_contains(lit(null_only_list), root()))
            .unwrap();
        assert_arrays_eq!(
            contains,
            BoolArray::from_iter([false, false, false]),
            &mut ctx
        );
    }

    #[test]
    fn test_sorted_membership_fast_path_utf8() {
        let mut ctx = array_session().create_execution_ctx();

        let arr = VarBinArray::from_iter(
            ["ant", "bee", "cat", "dog", "eel"].map(Some),
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        arr.statistics().compute_is_sorted(&mut ctx);

        let list_scalar = Scalar::list(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            vec![
                Scalar::from("dog"),
                Scalar::from("ant"),
                Scalar::from("dog"),
                Scalar::from("bee"),
                Scalar::from("xyz"),
                Scalar::from("fff"),
                Scalar::from("ggg"),
                Scalar::from("hhh"),
                Scalar::from("iii"),
                Scalar::from("jjj"),
                Scalar::from("kkk"),
                Scalar::from("lll"),
            ],
            Nullability::NonNullable,
        );
        let contains = arr.apply(&list_contains(lit(list_scalar), root())).unwrap();

        let expected = BoolArray::from_iter([true, true, false, true, false]);
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[test]
    fn test_sorted_membership_fast_path_skips_float_dtype() {
        let mut ctx = array_session().create_execution_ctx();

        // Sorted float column with a literal list past `MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP`:
        // `Stat::IsSorted` is true, but the fast path deliberately excludes floats (see
        // `try_sorted_membership_contains`), so this exercises the fan-out fallback specifically
        // via the dtype check, not the length threshold.
        let arr = PrimitiveArray::from_iter([1.0f64, 2.0, 3.0, 4.0]).into_array();
        arr.statistics().compute_is_sorted(&mut ctx);

        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(
                crate::dtype::PType::F64,
                Nullability::NonNullable,
            )),
            vec![
                Scalar::from(3.0f64),
                Scalar::from(1.0f64),
                Scalar::from(50.0f64),
                Scalar::from(60.0f64),
                Scalar::from(70.0f64),
                Scalar::from(80.0f64),
                Scalar::from(90.0f64),
                Scalar::from(100.0f64),
                Scalar::from(110.0f64),
                Scalar::from(120.0f64),
                Scalar::from(130.0f64),
                Scalar::from(140.0f64),
            ],
            Nullability::NonNullable,
        );
        let contains = arr.apply(&list_contains(lit(list_scalar), root())).unwrap();

        let expected = BoolArray::from_iter([true, false, true, false]);
        assert_arrays_eq!(contains, expected, &mut ctx);
    }

    #[rstest]
    // Below `MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP`: both sides take the fan-out.
    #[case(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], vec![10, 1, 1, 5])]
    #[case(vec![-5, -3, -3, 0, 2, 2, 2, 9], vec![-3, 9, 9, 100])]
    #[case(vec![1], vec![1])]
    #[case(vec![1, 2, 3], vec![])]
    #[case(vec![1, 2, 3], vec![100])]
    // At or past the threshold: the sorted column genuinely exercises the fast path.
    #[case(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], vec![10, 1, 1, 5, 7, 7, 3, 2, 4, 6, 8, 9])]
    #[case(vec![-5, -3, -3, 0, 2, 2, 2, 9], vec![-3, 9, 9, 100, 0, -5, -5, 2, 2, 50, 51, 52])]
    fn test_sorted_membership_fast_path_matches_fanout(
        #[case] data: Vec<i32>,
        #[case] list_values: Vec<i32>,
    ) {
        let mut ctx = array_session().create_execution_ctx();
        let expr = list_contains(lit(int_list_scalar(list_values)), root());

        // Same logical data, constructed independently so each copy owns its own stats: one
        // exercises the sorted fast path, the other (stat never computed) exercises the
        // pre-existing equality fan-out. They must agree.
        let sorted_column = PrimitiveArray::from_iter(data.clone()).into_array();
        sorted_column.statistics().compute_is_sorted(&mut ctx);
        let fast_path_result = sorted_column
            .apply(&expr)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
            .into_array();

        let plain_column = PrimitiveArray::from_iter(data).into_array();
        let fanout_result = plain_column
            .apply(&expr)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
            .into_array();

        assert_arrays_eq!(fast_path_result, fanout_result, &mut ctx);
    }

    #[rstest]
    // Below `MIN_ELEMENTS_FOR_SORTED_MEMBERSHIP`: both sides take the fan-out.
    #[case(vec![None, None, Some(1), Some(3), Some(5), Some(5), Some(8)], vec![5, 100])]
    #[case(vec![None, Some(-2), Some(-2), Some(0), Some(4)], vec![-2, 4, 4, 7])]
    #[case(vec![None, None, None], vec![1, 2])]
    // At or past the threshold: the sorted column genuinely exercises the fast path.
    #[case(
        vec![None, None, Some(1), Some(3), Some(5), Some(5), Some(8)],
        vec![5, 100, 200, 300, 8, 301, 302, 303, 304, 305, 306, 307]
    )]
    #[case(
        vec![None, Some(-2), Some(-2), Some(0), Some(4)],
        vec![-2, 4, 4, 7, 99, 100, 101, 102, 103, 104, 105, 106]
    )]
    fn test_sorted_membership_fast_path_matches_fanout_with_nulls(
        #[case] data: Vec<Option<i32>>,
        #[case] list_values: Vec<i32>,
    ) {
        let mut ctx = array_session().create_execution_ctx();
        let expr = list_contains(lit(int_list_scalar(list_values)), root());

        let sorted_column = PrimitiveArray::from_option_iter::<i32, _>(data.clone()).into_array();
        sorted_column.statistics().compute_is_sorted(&mut ctx);
        let fast_path_result = sorted_column
            .apply(&expr)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
            .into_array();

        let plain_column = PrimitiveArray::from_option_iter::<i32, _>(data).into_array();
        let fanout_result = plain_column
            .apply(&expr)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
            .into_array();

        assert_arrays_eq!(fast_path_result, fanout_result, &mut ctx);
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;
mod prepared;

use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::iter;
use std::ops::BitOr;
use std::sync::Arc;

use arrow_buffer::bit_iterator::BitIndexIterator;
pub use kernel::*;
use num_traits::Zero;
pub use prepared::PreparedSet;
use prost::Message;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::lit;
use crate::match_each_integer_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

#[derive(Clone)]
pub struct ListContains;

/// Options for [`ListContains`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ListContainsOptions {
    /// Whether a null list element is SQL's unknown value.
    ///
    /// Off, the default, a null element is never equal to anything: a needle that matches no
    /// element yields `false`, and a needle that matches some element yields `true`. On, the
    /// comparison against a null element is unknown, so a needle that matches no element yields
    /// `null` when the list holds a null — the three-valued semantics of SQL `IN`, under which
    /// `x NOT IN (1, NULL)` is never true.
    ///
    /// A null needle yields `null` against a list with elements either way. Against an empty list
    /// there is nothing to compare it to: off, that is `false`; on, a null needle is `null` against
    /// any list, which makes [`ListContains`] strict.
    pub sql_null_semantics: bool,
}

impl Display for ListContainsOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.sql_null_semantics {
            write!(f, "sql_null_semantics")?;
        }
        Ok(())
    }
}

impl ListContainsOptions {
    /// The declared nullability of a `list_contains` result: a null list or needle can yield
    /// null, and under [`sql_null_semantics`](Self::sql_null_semantics) so can a null element.
    pub fn result_nullability(&self, list_dtype: &DType, needle_dtype: &DType) -> Nullability {
        let mut nullability = list_dtype.nullability().bitor(needle_dtype.nullability());
        if self.sql_null_semantics
            && let Some(element_dtype) = list_dtype.as_any_size_list_element_opt()
        {
            nullability |= element_dtype.nullability();
        }
        nullability
    }
}

impl ListContains {
    /// Creates a lazy list membership check for `needle` in `list`, with a null element that
    /// never matches.
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths, `list` is not a list array,
    /// or its element dtype differs from the needle's dtype ignoring nullability.
    pub fn try_new(list: ArrayRef, needle: ArrayRef) -> VortexResult<ScalarFnArray> {
        Self::try_new_opts(list, needle, ListContainsOptions::default())
    }

    /// Creates a lazy list membership check for `needle` in `list` with explicit
    /// [`ListContainsOptions`].
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths, `list` is not a list array,
    /// or its element dtype differs from the needle's dtype ignoring nullability.
    pub fn try_new_opts(
        list: ArrayRef,
        needle: ArrayRef,
        options: ListContainsOptions,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(ListContains.bind(options), vec![list, needle])
    }
}

impl ScalarFnVTable for ListContains {
    type Options = ListContainsOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.contains");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        // The default encodes to no bytes, which is also what every file written before the
        // option existed carries.
        Ok(Some(
            pb::ListContainsOpts {
                sql_null_semantics: instance.sql_null_semantics,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::ListContainsOpts::decode(metadata)?;
        Ok(ListContainsOptions {
            sql_null_semantics: opts.sql_null_semantics,
        })
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
    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let list_dtype = &arg_dtypes[0];
        let needle_dtype = &arg_dtypes[1];

        let DType::List(element_dtype, _) = list_dtype else {
            vortex_bail!(
                "First argument to ListContains must be a List, got {:?}",
                list_dtype
            );
        };
        if !element_dtype.eq_ignore_nullability(needle_dtype) {
            vortex_bail!(
                "Element type {} of list does not match search value {}",
                element_dtype,
                needle_dtype,
            );
        }

        Ok(DType::Bool(
            options.result_nullability(list_dtype, needle_dtype),
        ))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let list_array = args.get(0)?;
        let value_array = args.get(1)?;

        // Borrow the list: a constant list scalar owns every element, so cloning it is not free.
        if let Some(list) = list_array.as_opt::<Constant>()
            && let Some(value_scalar) = value_array.as_constant()
        {
            let result = compute_contains_scalar(list.scalar(), &value_scalar, options)?;
            return Ok(ConstantArray::new(result, args.row_count()).into_array());
        }

        compute_list_contains(&list_array, &value_array, options, ctx)
    }

    fn simplify_untyped(
        &self,
        options: &Self::Options,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let Some(list) = expr.child(0).as_opt::<Literal>() else {
            return Ok(None);
        };
        Ok(normalized_set(list, options)
            .map(|list| ListContains.new_expr(*options, [lit(list), expr.child(1).clone()])))
    }

    // Off SQL null semantics an empty list answers `false` even for a null needle; on them a null
    // needle is null against any list, as a null list always is.
    fn is_strict(&self, options: &Self::Options) -> bool {
        options.sql_null_semantics
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// A constant list rewritten into the form a set probe wants — its elements sorted, without
/// duplicates, and with at most one null — or `None` when it is in that form already or its
/// elements have no total order.
///
/// Neither the order of the elements nor their repetition changes a membership test, and one null
/// element decides as much as many. Whether a null survives does matter: under SQL null semantics
/// it makes a non-match unknown, and off them a list of nothing but nulls is still not empty, which
/// a null needle tells apart.
///
/// Normalizing once, while the expression is optimized, spares every batch the sort.
fn normalized_set(list: &Scalar, options: &ListContainsOptions) -> Option<Scalar> {
    let DType::List(element_dtype, nullability) = list.dtype() else {
        return None;
    };
    if !matches!(
        element_dtype.as_ref(),
        DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_)
    ) {
        return None;
    }
    let elements = list.as_list().elements()?;

    let had_null = elements.iter().any(Scalar::is_null);
    let mut set: Vec<Scalar> = elements
        .iter()
        .filter(|element| !element.is_null())
        .cloned()
        .collect();
    let mut incomparable = false;
    set.sort_by(|a, b| {
        a.partial_cmp(b).unwrap_or_else(|| {
            incomparable = true;
            Ordering::Equal
        })
    });
    if incomparable {
        return None;
    }
    set.dedup();
    if had_null && (options.sql_null_semantics || set.is_empty()) {
        set.push(Scalar::null(element_dtype.as_ref().clone()));
    }

    (set != elements).then(|| Scalar::list(Arc::clone(element_dtype), set, *nullability))
}

fn compute_contains_scalar(
    list: &Scalar,
    needle: &Scalar,
    options: &ListContainsOptions,
) -> VortexResult<Scalar> {
    if !matches!(list.dtype(), DType::List(..)) {
        vortex_bail!(
            "First argument to ListContains must be a List, got {}",
            list.dtype()
        );
    }
    let nullability = options.result_nullability(list.dtype(), needle.dtype());

    if list.is_null() {
        return Ok(Scalar::null(DType::Bool(nullability)));
    }

    let list_scalar = list.as_list();
    let elements = list_scalar
        .elements()
        .ok_or_else(|| vortex_err!("Expected non-null list"))?;

    if needle.is_null() {
        return Ok(if elements.is_empty() && !options.sql_null_semantics {
            Scalar::bool(false, nullability)
        } else {
            Scalar::null(DType::Bool(nullability))
        });
    }

    let contains = elements.iter().any(|elem| elem == needle);
    if !contains && options.sql_null_semantics && elements.iter().any(Scalar::is_null) {
        return Ok(Scalar::null(DType::Bool(nullability)));
    }
    Ok(Scalar::bool(contains, nullability))
}

fn compute_list_contains(
    array: &ArrayRef,
    value: &ArrayRef,
    options: &ListContainsOptions,
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

    // A null needle is null against every list only when the function is strict: otherwise an
    // empty list still answers `false`, which the paths below work out per list.
    if array.all_invalid(ctx)? || (options.sql_null_semantics && value.all_invalid(ctx)?) {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            array.len(),
        )
        .into_array());
    }

    let nullability = options.result_nullability(array.dtype(), value.dtype());

    if let Some(value_scalar) = value.as_constant() {
        return list_contains_scalar(array, &value_scalar, nullability, options, ctx);
    }

    if !array.is::<Constant>() {
        return lists_contain_needles(array, value, nullability, options, ctx);
    }

    let set = ListContainsSet::try_new(array, value.dtype(), options, ctx)?
        .ok_or_else(|| vortex_err!("A non-null constant list of {} has a set", value.dtype()))?;
    set.prepare(ctx)?.contains(value, ctx)
}

/// Returns a [`BoolArray`] where each bit represents if a list contains the scalar.
///
/// This is the canonical implementation, over an executed [`ListViewArray`].
fn list_contains_scalar(
    array: &ArrayRef,
    value: &Scalar,
    nullability: Nullability,
    options: &ListContainsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // If the list array is constant, we perform a single comparison.
    if array.len() > 1 && array.is::<Constant>() {
        let contains = list_contains_scalar(&array.slice(0..1)?, value, nullability, options, ctx)?;
        return Ok(ConstantArray::new(contains.execute_scalar(0, ctx)?, array.len()).into_array());
    }

    let list_array = array.clone().execute::<ListViewArray>(ctx)?;

    if value.is_null() {
        return null_needle_in_lists(&list_array, nullability, ctx);
    }

    let elems = list_array.elements();
    if elems.is_empty() {
        // Must return false when a list is empty (but valid), or null when the list itself is null.
        return list_false_or_null(&list_array, nullability, ctx);
    }

    let rhs = ConstantArray::new(value.clone(), elems.len());
    let matching_elements =
        Binary::try_new(elems.clone(), rhs.clone().into_array(), Operator::Eq)?.into_array();

    // TODO(ngates): we should execute this into a Columnar and check for constant.
    let mut matches = matching_elements.execute::<BoolArray>(ctx)?;
    let valid = matches.validity()?.execute_mask(matches.len(), ctx)?;

    // Under SQL null semantics a list holding a null element answers `null` for a needle that
    // matches none of its other elements. The needle is non-null here, so a null comparison is a
    // null element; fold "any match" and "any null" per list and keep only the decided rows.
    if options.sql_null_semantics && !valid.all_true() {
        let valid = valid.to_bit_buffer();
        let any_true = fold_lists(
            BoolArray::new(&matches.to_bit_buffer() & &valid, Validity::NonNullable),
            &list_array,
            ctx,
        )?;
        let any_null = fold_lists(
            BoolArray::new(!valid, Validity::NonNullable),
            &list_array,
            ctx,
        )?;
        let decided = &any_true | &!any_null;
        let validity = list_array
            .validity()?
            .and(Validity::from(decided))?
            .union_nullability(nullability);
        return Ok(BoolArray::new(any_true, validity).into_array());
    }

    // Null comparisons carry unspecified value bits, which must not contribute a match.
    if !valid.all_true() {
        matches = BoolArray::new(
            &matches.to_bit_buffer() & &valid.to_bit_buffer(),
            Validity::NonNullable,
        );
    }

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
                list_false_or_null(&list_array, nullability, ctx)
            }
            // No elements match, and all comparisons are valid (result in `false`).
            Some(false) => list_false_or_null(&list_array, nullability, ctx),
            // All elements match, and all comparisons are valid (result in `true`).
            Some(true) => {
                // True, unless the list itself is empty or NULL.
                list_is_not_empty(&list_array, nullability, ctx)
            }
        };
    }

    let list_matches = fold_lists(matches, &list_array, ctx)?;

    Ok(BoolArray::new(
        list_matches,
        list_array.validity()?.union_nullability(nullability),
    )
    .into_array())
}

/// Neither side is constant: row `i` asks whether list `i` holds needle `i`.
///
/// Each list's elements are gathered next to one copy of its row's needle, so a single equality
/// answers every comparison, and a fold over each list's contiguous range answers each row.
fn lists_contain_needles(
    array: &ArrayRef,
    values: &ArrayRef,
    nullability: Nullability,
    options: &ListContainsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list_array = array.clone().execute::<ListViewArray>(ctx)?;
    let len = list_array.len();
    let list_valid = list_array
        .validity()?
        .execute_mask(len, ctx)?
        .to_bit_buffer();

    let offsets = list_array
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.reinterpret_cast(sizes.ptype().to_unsigned());
    let gathered = match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        match_each_unsigned_integer_ptype!(sizes.ptype(), |S| {
            GatheredLists::new(
                offsets.as_slice::<O>(),
                sizes.as_slice::<S>(),
                &list_valid,
                ctx,
            )
        })
    });

    let elements = list_array
        .elements()
        .take(PrimitiveArray::new(gathered.element_idx, Validity::NonNullable).into_array())?;
    let needles =
        values.take(PrimitiveArray::new(gathered.row_idx, Validity::NonNullable).into_array())?;
    let matches = Binary::try_new(elements, needles, Operator::Eq)?
        .into_array()
        .execute::<BoolArray>(ctx)?;
    let compared = matches
        .validity()?
        .execute_mask(matches.len(), ctx)?
        .to_bit_buffer();

    let starts = PrimitiveArray::new(gathered.starts, Validity::NonNullable);
    let lens = PrimitiveArray::new(gathered.lens, Validity::NonNullable);
    let any_true = process_matches::<u64, u64>(
        BoolArray::new(&matches.to_bit_buffer() & &compared, Validity::NonNullable),
        len,
        starts.clone(),
        lens.clone(),
        ctx,
    );

    // A null needle is null against a list with elements, but an empty list holds nothing to
    // compare it to unless the function is strict.
    let needle_valid = values.validity()?.execute_mask(len, ctx)?.to_bit_buffer();
    let mut decided = if options.sql_null_semantics {
        needle_valid
    } else {
        &needle_valid | &!&non_empty_lists(&list_array, ctx)?
    };
    // Under SQL null semantics a comparison with a null element leaves a non-match unknown.
    if options.sql_null_semantics {
        let any_null = process_matches::<u64, u64>(
            BoolArray::new(!&compared, Validity::NonNullable),
            len,
            starts,
            lens,
            ctx,
        );
        decided = &decided & &(&any_true | &!&any_null);
    }

    // A non-nullable result has non-null lists and needles, and no null element that could leave
    // a row undecided.
    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => list_array.validity()?.and(Validity::from(decided))?,
    };
    Ok(BoolArray::new(any_true, validity).into_array())
}

/// The elements of every valid list laid out contiguously, each next to the row it belongs to.
struct GatheredLists {
    /// The position of each gathered element in the list view's elements.
    element_idx: Buffer<u64>,
    /// The row, and so the needle, each gathered element is compared to.
    row_idx: Buffer<u64>,
    /// Where each row's run of gathered elements starts.
    starts: Buffer<u64>,
    /// How long each row's run is: the list's size, or zero for a null list.
    lens: Buffer<u64>,
}

impl GatheredLists {
    fn new<O: IntegerPType, S: IntegerPType>(
        offsets: &[O],
        sizes: &[S],
        list_valid: &BitBuffer,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        let rows = sizes.len();
        let total: usize = (0..rows)
            .filter(|&row| list_valid.value(row))
            .map(|row| sizes[row].as_())
            .sum();
        let allocator = ctx.allocator();
        let mut element_idx = BufferMut::<u64>::with_capacity_in(total, allocator.clone());
        let mut row_idx = BufferMut::<u64>::with_capacity_in(total, allocator.clone());
        let mut starts = BufferMut::<u64>::with_capacity_in(rows, allocator.clone());
        let mut lens = BufferMut::<u64>::with_capacity_in(rows, allocator.clone());

        for row in 0..rows {
            starts.push(element_idx.len() as u64);
            // A null list's offset and size need not point at anything.
            if !list_valid.value(row) {
                lens.push(0);
                continue;
            }
            let offset: usize = offsets[row].as_();
            let size: usize = sizes[row].as_();
            element_idx.extend((offset..offset + size).map(|idx| idx as u64));
            row_idx.extend(iter::repeat_n(row as u64, size));
            lens.push(size as u64);
        }

        Self {
            element_idx: element_idx.freeze(),
            row_idx: row_idx.freeze(),
            starts: starts.freeze(),
            lens: lens.freeze(),
        }
    }
}

/// For each list, whether any set bit of `matches` falls in the list's element range.
fn fold_lists(
    matches: BoolArray,
    list_array: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    // Get the offsets and sizes as primitive arrays. They are non-negative, so reinterpret to
    // unsigned and dispatch over the 4 unsigned widths each (4x4 instead of 8x8).
    let offsets = list_array
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.reinterpret_cast(sizes.ptype().to_unsigned());

    Ok(match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        match_each_unsigned_integer_ptype!(sizes.ptype(), |S| {
            process_matches::<O, S>(matches, list_array.len(), offsets, sizes, ctx)
        })
    }))
}

/// Returns a [`BitBuffer`] where each bit represents if a list contains the scalar, derived from a
/// [`BoolArray`] of matches on the child elements array.
fn process_matches<O, S>(
    matches: BoolArray,
    list_array_len: usize,
    offsets: PrimitiveArray,
    sizes: PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> BitBuffer
where
    O: IntegerPType,
    S: IntegerPType,
{
    let offsets_slice = offsets.as_slice::<O>();
    let sizes_slice = sizes.as_slice::<S>();
    let bits = matches.bit_buffer_view();

    BitBuffer::collect_bool_in(
        list_array_len,
        |i| {
            let offset = offsets_slice[i].as_();
            let size = sizes_slice[i].as_();

            // BitIndexIterator yields indices of true bits only. If `.next()` returns
            // `Some(_)`, at least one element in this list's range matches.
            let mut set_bits = BitIndexIterator::new(bits.inner(), bits.offset() + offset, size);
            set_bits.next().is_some()
        },
        ctx.allocator().clone(),
    )
}

/// Returns a `Bool` array with `false` for lists that are valid,
/// or `NULL` if the list itself is null.
fn list_false_or_null(
    list_array: &ListViewArray,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
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
            let buffer = BitBuffer::new_unset_in(list_array.len(), ctx.allocator().clone());
            Ok(BoolArray::new(buffer, Validity::Array(validity_array)).into_array())
        }
    }
}

/// A null needle against each list, off SQL null semantics: `false` for an empty list, which holds
/// nothing to compare it to, and `null` for any other list.
fn null_needle_in_lists(
    list_array: &ListViewArray,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let empty = !&non_empty_lists(list_array, ctx)?;
    let validity = list_array
        .validity()?
        .and(Validity::from(empty))?
        .union_nullability(nullability);
    Ok(BoolArray::new(
        BitBuffer::new_unset_in(list_array.len(), ctx.allocator().clone()),
        validity,
    )
    .into_array())
}

/// One bit per list, set when the list holds at least one element.
fn non_empty_lists(list_array: &ListViewArray, ctx: &mut ExecutionCtx) -> VortexResult<BitBuffer> {
    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_integer_ptype!(sizes.ptype(), |S| {
        let sizes = sizes.as_slice::<S>();
        BitBuffer::collect_bool_in(
            sizes.len(),
            |idx| sizes[idx] != S::zero(),
            ctx.allocator().clone(),
        )
    }))
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

    // Copy over the validity mask from the input.
    Ok(BoolArray::new(
        non_empty_lists(list_array, ctx)?,
        list_array.validity()?.union_nullability(nullability),
    )
    .into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use itertools::Itertools;
    use num_traits::PrimInt;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::process_matches;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::NativePType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::in_list;
    use crate::expr::list_contains;
    use crate::expr::list_contains_opts;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::or;
    use crate::expr::root;
    use crate::expr::stats::Stat;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::list_contains::ListContains;
    use crate::scalar_fn::fns::list_contains::ListContainsOptions;
    use crate::scalar_fn::fns::literal::Literal;
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

    #[test]
    fn test_null_in_empty_list() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = test_array();
        let dtype = Arc::new(DType::Primitive(I32, Nullability::NonNullable));

        let list = Scalar::list(Arc::clone(&dtype), vec![], Nullability::NonNullable);
        let needle = Scalar::null(DType::Primitive(I32, Nullability::Nullable));

        let expr = list_contains(lit(list.clone()), lit(needle.clone()));
        let result = array.clone().apply(&expr)?;
        assert_eq!(
            result.execute_scalar(0, &mut ctx)?,
            Scalar::bool(false, Nullability::Nullable)
        );

        let expr = list_contains(lit(list), lit(2i32));
        let result = array.clone().apply(&expr)?;
        assert_eq!(
            result.execute_scalar(0, &mut ctx)?,
            Scalar::bool(false, Nullability::NonNullable)
        );

        let list = Scalar::null(DType::List(dtype, Nullability::Nullable));
        let expr = list_contains(lit(list), lit(needle));
        let result = array.apply(&expr)?;
        assert!(result.execute_scalar(0, &mut ctx)?.is_null());
        Ok(())
    }

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
    // A null needle is null against a list with elements, but an empty list holds nothing to
    // compare it to.
    #[case(
        null_strings(vec![vec![], vec![None, None], vec![None, None, None]]),
        None,
        bool_array(vec![false, false, false], Validity::from_iter([true, false, false]))
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

    const SQL: ListContainsOptions = ListContainsOptions {
        sql_null_semantics: true,
    };

    #[rstest]
    #[case::default(ListContainsOptions::default(), Some(false))]
    #[case::sql(SQL, None)]
    fn test_constant_needle_ignores_null_value_bits(
        #[case] options: ListContainsOptions,
        #[case] null_element_result: Option<bool>,
    ) -> VortexResult<()> {
        // Both the null element and the valid match have the same physical value.
        let elements = PrimitiveArray::new(
            buffer![7i32, 7, 8],
            Validity::from(BitBuffer::from_iter([false, true, true])),
        );
        let lists = ListArray::try_new(
            elements.into_array(),
            buffer![0u32, 1, 2, 3, 3].into_array(),
            Validity::AllValid,
        )?
        .into_array();
        assert_result(
            lists.apply(&list_contains_opts(root(), lit(7i32), options)),
            [null_element_result, Some(true), Some(false), Some(false)],
        )
    }

    #[rstest]
    #[case::default(ListContainsOptions::default())]
    #[case::sql(SQL)]
    fn test_no_matches_preserves_null_lists(
        #[case] options: ListContainsOptions,
    ) -> VortexResult<()> {
        let lists = ListArray::try_new(
            buffer![1i32, 2].into_array(),
            buffer![0u32, 1, 2].into_array(),
            Validity::from(BitBuffer::from_iter([true, false])),
        )?
        .into_array();
        assert_result(
            lists.apply(&list_contains_opts(root(), lit(9i32), options)),
            [Some(false), None],
        )
    }

    #[test]
    fn test_fold_sliced_match_bitmap() {
        let mut ctx = array_session().create_execution_ctx();
        let bits = BitBuffer::from_iter([true, false, true, false]).slice(1..);
        let result = process_matches::<u32, u32>(
            BoolArray::new(bits, Validity::NonNullable),
            3,
            PrimitiveArray::from_iter([0u32, 1, 2]),
            PrimitiveArray::from_iter([1u32, 1, 1]),
            &mut ctx,
        );
        assert_eq!(result, BitBuffer::from_iter([false, true, false]));
    }

    #[rstest]
    #[case::default(ListContainsOptions::default())]
    #[case::sql(SQL)]
    fn test_reject_mismatched_needle_type(#[case] options: ListContainsOptions) {
        let list = lit(i32_set(vec![Some(1)]));
        let dtype = DType::Utf8(Nullability::NonNullable);
        assert!(
            list_contains_opts(list.clone(), root(), options)
                .bind(&dtype)
                .is_err()
        );
        assert!(
            list_contains_opts(list, lit("1"), options)
                .bind(&dtype)
                .is_err()
        );
    }

    /// A constant `List<i32?>` set, so that it can hold a null element.
    fn i32_set(values: Vec<Option<i32>>) -> Scalar {
        let element = DType::Primitive(I32, Nullability::Nullable);
        Scalar::list(
            Arc::new(element.clone()),
            values
                .into_iter()
                .map(|v| match v {
                    Some(v) => Scalar::primitive(v, Nullability::Nullable),
                    None => Scalar::null(element.clone()),
                })
                .collect(),
            Nullability::NonNullable,
        )
    }

    fn f64_set(values: Vec<f64>) -> Scalar {
        Scalar::list(
            Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
            values.into_iter().map(Scalar::from).collect(),
            Nullability::NonNullable,
        )
    }

    fn utf8_set(values: Vec<Option<&str>>) -> Scalar {
        let element = DType::Utf8(Nullability::Nullable);
        Scalar::list(
            Arc::new(element.clone()),
            values
                .into_iter()
                .map(|v| match v {
                    Some(v) => Scalar::utf8(v, Nullability::Nullable),
                    None => Scalar::null(element.clone()),
                })
                .collect(),
            Nullability::NonNullable,
        )
    }

    fn assert_result(
        result: VortexResult<ArrayRef>,
        expected: impl IntoIterator<Item = Option<bool>>,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result?, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }

    #[test]
    fn constant_set_of_floats_is_bitwise() -> VortexResult<()> {
        // Membership has to agree with the compare kernel: `-0.0` and `0.0` are different
        // members, and NaN is a member of a set that holds NaN.
        let needles = PrimitiveArray::from_option_iter([
            Some(1.0f64),
            Some(f64::NAN),
            Some(0.0),
            Some(-0.0),
            Some(2.0),
            None,
        ])
        .into_array();
        let set = f64_set(vec![1.0, f64::NAN, -0.0]);
        assert_result(
            needles.apply(&list_contains(lit(set), root())),
            [
                Some(true),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                None,
            ],
        )
    }

    #[test]
    fn constant_set_null_element_never_matches_by_default() -> VortexResult<()> {
        let needles =
            PrimitiveArray::from_option_iter([Some(1), Some(2), None, Some(4)]).into_array();
        let set = i32_set(vec![Some(2), None, Some(4)]);
        assert_result(
            needles.apply(&list_contains(lit(set), root())),
            [Some(false), Some(true), None, Some(true)],
        )
    }

    #[test]
    fn constant_set_null_element_is_unknown_under_sql_semantics() -> VortexResult<()> {
        // `x IN (2, NULL, 4)`: a match is still true, but a non-match is null, so `NOT IN` never
        // admits it. A null needle stays null either way.
        let needles =
            PrimitiveArray::from_option_iter([Some(1), Some(2), None, Some(4)]).into_array();
        let set = i32_set(vec![Some(2), None, Some(4)]);
        assert_result(
            needles.clone().apply(&in_list(root(), lit(set.clone()))),
            [None, Some(true), None, Some(true)],
        )?;
        assert_result(
            needles.apply(&crate::expr::not(in_list(root(), lit(set)))),
            [None, Some(false), None, Some(false)],
        )
    }

    #[test]
    fn sql_semantics_without_a_null_element_match_the_default() -> VortexResult<()> {
        let needles = PrimitiveArray::from_option_iter([Some(1), Some(2), None]).into_array();
        let set = i32_set(vec![Some(2)]);
        assert_result(
            needles.apply(&in_list(root(), lit(set))),
            [Some(false), Some(true), None],
        )
    }

    fn empty_i32_list() -> Scalar {
        Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::Nullable)),
            vec![],
            Nullability::NonNullable,
        )
    }

    #[rstest]
    #[case::default(ListContainsOptions::default(), Some(false))]
    #[case::sql(SQL, None)]
    fn null_needle_in_an_empty_list(
        #[case] options: ListContainsOptions,
        #[case] expected: Option<bool>,
    ) -> VortexResult<()> {
        // Off SQL null semantics an empty list holds nothing to compare a null needle to; on them
        // the function is strict. Every evaluation path has to agree, a single row included.
        let mut ctx = array_session().create_execution_ctx();
        let null = Scalar::null(DType::Primitive(I32, Nullability::Nullable));

        let scalar = ConstantArray::new(null, 2)
            .into_array()
            .apply(&list_contains_opts(lit(empty_i32_list()), root(), options))?;
        assert_eq!(
            scalar.execute_scalar(0, &mut ctx)?.as_bool().value(),
            expected
        );

        let needles = PrimitiveArray::from_option_iter([Some(1), None]).into_array();
        let column = needles.apply(&list_contains_opts(lit(empty_i32_list()), root(), options))?;
        assert_eq!(
            column.execute_scalar(1, &mut ctx)?.as_bool().value(),
            expected
        );
        assert_arrays_eq!(
            column,
            BoolArray::from_iter([Some(false), expected]),
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    #[case::default(ListContainsOptions::default(), [None, Some(false), None])]
    #[case::sql(SQL, [None, None, None])]
    fn null_needle_in_a_list_column(
        #[case] options: ListContainsOptions,
        #[case] expected: [Option<bool>; 3],
    ) -> VortexResult<()> {
        // Lists `[1]`, `[]` and a null list against a null needle.
        let lists = ListArray::try_new(
            PrimitiveArray::from_option_iter([Some(1i32)]).into_array(),
            PrimitiveArray::from_iter(vec![0, 1, 1, 1]).into_array(),
            Validity::from_iter([true, true, false]),
        )?
        .into_array();
        let null = Scalar::null(DType::Primitive(I32, Nullability::Nullable));
        assert_result(
            lists.apply(&list_contains_opts(root(), lit(null), options)),
            expected,
        )
    }

    /// Every row of `result` agrees with that row evaluated on its own, then matches `expected`.
    fn assert_rows_agree(result: ArrayRef, expected: BoolArray) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let whole = result.clone().execute::<BoolArray>(&mut ctx)?.into_array();
        for row in 0..result.len() {
            assert_eq!(
                result.execute_scalar(row, &mut ctx)?,
                whole.execute_scalar(row, &mut ctx)?,
                "row {row}"
            );
        }
        assert_arrays_eq!(whole, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::default(
        ListContainsOptions::default(),
        [Some(true), Some(false), None, Some(false), None, Some(true)]
    )]
    #[case::sql(SQL, [Some(true), None, None, None, None, Some(true)])]
    fn list_column_against_needle_column(
        #[case] options: ListContainsOptions,
        #[case] expected: [Option<bool>; 6],
    ) -> VortexResult<()> {
        // Lists `[1, 2]`, `[]`, null, `[3, null]`, `[4]`, `[5]` against needles 2, null, 1, 7,
        // null, 5.
        let lists = ListArray::try_new(
            PrimitiveArray::from_option_iter([
                Some(1i32),
                Some(2),
                Some(3),
                None,
                Some(4),
                Some(5),
            ])
            .into_array(),
            PrimitiveArray::from_iter(vec![0, 2, 2, 2, 4, 5, 6]).into_array(),
            Validity::from_iter([true, true, false, true, true, true]),
        )?
        .into_array();
        let needles =
            PrimitiveArray::from_option_iter([Some(2i32), None, Some(1), Some(7), None, Some(5)])
                .into_array();
        let result = ListContains::try_new_opts(lists, needles, options)?.into_array();
        assert_rows_agree(result, BoolArray::from_iter(expected))
    }

    #[test]
    fn list_view_column_with_overlapping_views() -> VortexResult<()> {
        // Views out of order and overlapping in `[1, 2, 3, 4]`: `[3, 4]` then `[1, 2, 3]`.
        let lists = ListViewArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![2u32, 0].into_array(),
            buffer![2u32, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let needles = buffer![4i32, 4].into_array();
        let result = ListContains::try_new(lists, needles)?.into_array();
        assert_rows_agree(result, BoolArray::from_iter([true, false]))
    }

    fn optimized_set(expr: &Expression) -> VortexResult<Scalar> {
        let optimized = expr.optimize_recursive(&DType::Primitive(I32, Nullability::Nullable))?;
        Ok(optimized
            .child(0)
            .as_opt::<Literal>()
            .vortex_expect("the set stays a literal")
            .clone())
    }

    #[rstest]
    // Off SQL null semantics a null element next to others decides nothing.
    #[case::default(ListContainsOptions::default(), vec![Some(3), Some(1), None, Some(3), Some(2)], vec![Some(1), Some(2), Some(3)])]
    // On them one null is kept, last, to leave a non-match unknown.
    #[case::sql(SQL, vec![Some(3), None, Some(1), None, Some(3)], vec![Some(1), Some(3), None])]
    // A list of nothing but nulls keeps one, since an empty list answers a null needle apart.
    #[case::only_nulls(ListContainsOptions::default(), vec![None, None], vec![None])]
    fn optimize_normalizes_the_set(
        #[case] options: ListContainsOptions,
        #[case] set: Vec<Option<i32>>,
        #[case] expected: Vec<Option<i32>>,
    ) -> VortexResult<()> {
        let expr = list_contains_opts(lit(i32_set(set)), root(), options);
        assert_eq!(optimized_set(&expr)?, i32_set(expected.clone()));
        // A normalized set is left alone, so optimization reaches a fixed point.
        let normalized = list_contains_opts(lit(i32_set(expected.clone())), root(), options);
        assert_eq!(optimized_set(&normalized)?, i32_set(expected));
        Ok(())
    }

    #[rstest]
    #[case::default(ListContainsOptions::default())]
    #[case::sql(SQL)]
    fn optimize_keeps_the_answer(#[case] options: ListContainsOptions) -> VortexResult<()> {
        // Floats compare bitwise: `-0.0` and `0.0` stay distinct members, and NaN is one member.
        let element = DType::Primitive(PType::F64, Nullability::Nullable);
        let set = Scalar::list(
            Arc::new(element.clone()),
            [
                Some(2.0),
                Some(f64::NAN),
                None,
                Some(-0.0),
                Some(2.0),
                Some(f64::NAN),
            ]
            .into_iter()
            .map(|v| match v {
                Some(v) => Scalar::primitive(v, Nullability::Nullable),
                None => Scalar::null(element.clone()),
            })
            .collect(),
            Nullability::NonNullable,
        );
        let needles = PrimitiveArray::from_option_iter([
            Some(2.0f64),
            Some(0.0),
            Some(-0.0),
            Some(f64::NAN),
            Some(5.0),
            None,
        ])
        .into_array();
        let expr = list_contains_opts(lit(set), root(), options);
        let optimized = expr.optimize_recursive(needles.dtype())?;
        assert_ne!(optimized, expr, "the set is normalized");

        let mut ctx = array_session().create_execution_ctx();
        let expected = needles
            .clone()
            .apply(&expr)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(needles.apply(&optimized)?, expected, &mut ctx);
        Ok(())
    }

    /// Probes `set` with each element, its neighbours and the type's extremes, against a naive
    /// oracle.
    fn assert_integer_membership<T>(set: Vec<T>) -> VortexResult<()>
    where
        T: NativePType + PrimInt + Into<PValue>,
    {
        let mut needles = vec![T::min_value(), T::max_value()];
        for &element in &set {
            needles.push(element);
            needles.extend(element.checked_sub(&T::one()));
            needles.extend(element.checked_add(&T::one()));
        }
        let expected = BoolArray::from_iter(needles.iter().map(|needle| set.contains(needle)));
        let list = Scalar::list(
            Arc::new(DType::Primitive(T::PTYPE, Nullability::NonNullable)),
            set.into_iter()
                .map(|v| Scalar::primitive(v, Nullability::NonNullable))
                .collect(),
            Nullability::NonNullable,
        );
        let mut ctx = array_session().create_execution_ctx();
        let result = PrimitiveArray::from_iter(needles)
            .into_array()
            .apply(&list_contains(lit(list), root()))?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn integer_sets_through_every_probe() -> VortexResult<()> {
        // Dense spans probe a bitmap, including spans straddling zero and whole types.
        assert_integer_membership(vec![-3i32, -1, 0, 2])?;
        assert_integer_membership(vec![i8::MIN, 0, i8::MAX])?;
        assert_integer_membership(vec![u64::MAX, u64::MAX - 2])?;
        assert_integer_membership(vec![i64::MIN, i64::MIN + 5])?;
        // A sparse set probes a binary search when small and a hash set when large.
        assert_integer_membership(vec![-(1i64 << 40), 1, 1 << 40])?;
        assert_integer_membership((0..100u64).map(|v| v << 30).collect())?;
        assert_integer_membership(vec![i64::MIN, i64::MAX])
    }

    #[rstest]
    #[case::default(ListContainsOptions::default())]
    #[case::sql(SQL)]
    fn chunked_needles_agree_with_flat_needles(
        #[case] options: ListContainsOptions,
    ) -> VortexResult<()> {
        // Chunks in different encodings, one of them empty, against a set with a null element.
        let chunks = vec![
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(4)]).into_array(),
            PrimitiveArray::from_option_iter::<i32, _>([]).into_array(),
            DictArray::try_new(
                PrimitiveArray::from_option_iter([Some(0u32), None, Some(1)]).into_array(),
                PrimitiveArray::from_option_iter([Some(2i32), Some(9)]).into_array(),
            )?
            .into_array(),
        ];
        let dtype = chunks[0].dtype().clone();
        let chunked = ChunkedArray::try_new(chunks, dtype)?.into_array();
        let expr = list_contains_opts(lit(i32_set(vec![Some(2), None, Some(4)])), root(), options);

        let mut ctx = array_session().create_execution_ctx();
        let flat =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(4), Some(2), None, Some(9)])
                .into_array()
                .apply(&expr)?
                .execute::<BoolArray>(&mut ctx)?;
        assert_rows_agree(chunked.apply(&expr)?, flat)
    }

    #[test]
    fn chunked_string_needles() -> VortexResult<()> {
        let chunks = vec![
            VarBinViewArray::from_iter_nullable_str([Some("a"), None]).into_array(),
            VarBinViewArray::from_iter_nullable_str([
                Some("a value longer than twelve bytes"),
                Some("b"),
            ])
            .into_array(),
        ];
        let dtype = chunks[0].dtype().clone();
        let chunked = ChunkedArray::try_new(chunks, dtype)?.into_array();
        let set = utf8_set(vec![Some("a"), Some("a value longer than twelve bytes")]);
        assert_rows_agree(
            chunked.apply(&list_contains(lit(set), root()))?,
            BoolArray::from_iter([Some(true), None, Some(true), Some(false)]),
        )
    }

    #[test]
    fn strict_only_under_sql_semantics() {
        let default = list_contains(lit(empty_i32_list()), root());
        let sql = in_list(root(), lit(empty_i32_list()));
        assert!(
            !default
                .as_scalar()
                .is_some_and(|f| f.signature().is_strict())
        );
        assert!(sql.as_scalar().is_some_and(|f| f.signature().is_strict()));
    }

    #[test]
    fn constant_set_of_only_nulls() -> VortexResult<()> {
        // Every element is dropped from the set, so nothing matches; under SQL null semantics no
        // answer is known instead.
        let needles = PrimitiveArray::from_option_iter([Some(1), None]).into_array();
        let set = i32_set(vec![None, None]);
        assert_result(
            needles
                .clone()
                .apply(&list_contains(lit(set.clone()), root())),
            [Some(false), None],
        )?;
        assert_result(needles.apply(&in_list(root(), lit(set))), [None, None])
    }

    #[test]
    fn constant_set_row_probe_keeps_the_declared_nullability() -> VortexResult<()> {
        // A nullable element dtype decides nothing off SQL null semantics, so non-null needles give
        // a non-nullable result, though each comparison is against a nullable element.
        let needles = BoolArray::from_iter([true, false]).into_array();
        let element = DType::Bool(Nullability::Nullable);
        let set = Scalar::list(
            Arc::new(element.clone()),
            vec![
                Scalar::bool(true, Nullability::Nullable),
                Scalar::null(element),
            ],
            Nullability::NonNullable,
        );
        let result = needles.apply(&list_contains(lit(set), root()))?;
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));
        assert_rows_agree(result, BoolArray::from_iter([true, false]))
    }

    #[test]
    fn constant_set_row_probe_of_only_nulls() -> VortexResult<()> {
        // The row probe must also preserve the distinction between an empty set and null elements.
        let needles = BoolArray::from_iter([Some(true), None]).into_array();
        let element = DType::Bool(Nullability::Nullable);
        let set = Scalar::list(
            Arc::new(element.clone()),
            vec![Scalar::null(element.clone()), Scalar::null(element)],
            Nullability::NonNullable,
        );
        assert_result(
            needles
                .clone()
                .apply(&list_contains(lit(set.clone()), root())),
            [Some(false), None],
        )?;
        assert_result(needles.apply(&in_list(root(), lit(set))), [None, None])
    }

    #[test]
    fn constant_set_of_strings_reaches_out_of_line_values() -> VortexResult<()> {
        // Values longer than 12 bytes live outside the view; both kinds must be looked up.
        let long = "a value longer than twelve bytes";
        let needles = VarBinViewArray::from_iter_nullable_str([
            Some("a"),
            Some(long),
            Some("a value longer than twelve byteX"),
            Some("b"),
            None,
        ])
        .into_array();
        let set = utf8_set(vec![Some("a"), Some(long)]);
        assert_result(
            needles.clone().apply(&list_contains(lit(set), root())),
            [Some(true), Some(true), Some(false), Some(false), None],
        )?;
        let set_with_null = utf8_set(vec![Some("a"), None]);
        assert_result(
            needles.apply(&in_list(root(), lit(set_with_null))),
            [Some(true), None, None, None, None],
        )
    }

    #[rstest]
    // Non-strict, so a null code keeps the dictionary rewrite from pushing the function into the
    // values: the needle is executed into its canonical encoding and probed against the set.
    #[case::executed(ListContainsOptions::default(), [Some(false), Some(true), None, Some(true)])]
    // Strict, so the rewrite probes the dictionary's two values instead of its four rows.
    #[case::pushed_into_values(SQL, [Some(false), Some(true), None, Some(true)])]
    fn constant_set_probes_a_dictionary_needle(
        #[case] options: ListContainsOptions,
        #[case] expected: [Option<bool>; 4],
    ) -> VortexResult<()> {
        let needles = DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u32), Some(1), None, Some(1)]).into_array(),
            PrimitiveArray::from_iter([1i32, 2]).into_array(),
        )?
        .into_array();
        let set = i32_set(vec![Some(2), Some(4)]);
        assert_rows_agree(
            needles.apply(&list_contains_opts(lit(set), root(), options))?,
            BoolArray::from_iter(expected),
        )
    }

    #[test]
    fn constant_set_of_strings_probes_an_encoded_needle() -> VortexResult<()> {
        let needles = VarBinArray::from_iter(
            [Some("a"), Some("b"), None],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let set = utf8_set(vec![Some("a"), Some("c")]);
        assert_result(
            needles.apply(&list_contains(lit(set), root())),
            [Some(true), Some(false), None],
        )
    }

    #[test]
    fn constant_set_row_probe_honours_both_semantics() -> VortexResult<()> {
        // Boolean sets use the general sorted row probe.
        let needles = BoolArray::from_iter([Some(true), Some(false), None]).into_array();
        let element = DType::Bool(Nullability::Nullable);
        let set = Scalar::list(
            Arc::new(element.clone()),
            vec![
                Scalar::bool(true, Nullability::Nullable),
                Scalar::null(element),
            ],
            Nullability::NonNullable,
        );
        assert_result(
            needles
                .clone()
                .apply(&list_contains(lit(set.clone()), root())),
            [Some(true), Some(false), None],
        )?;
        assert_result(
            needles.apply(&in_list(root(), lit(set))),
            [Some(true), None, None],
        )
    }

    #[test]
    fn list_array_null_elements_under_sql_semantics() -> VortexResult<()> {
        // Lists `[1, null]`, `[2]`, `[3]`, `[]` against a constant needle.
        let lists = ListArray::try_new(
            PrimitiveArray::from_option_iter([Some(1), None, Some(2), Some(3)]).into_array(),
            PrimitiveArray::from_iter(vec![0, 2, 3, 4, 4]).into_array(),
            Validity::AllValid,
        )?
        .into_array();
        assert_result(
            lists.clone().apply(&list_contains(root(), lit(5))),
            [Some(false), Some(false), Some(false), Some(false)],
        )?;
        assert_result(
            lists
                .clone()
                .apply(&list_contains_opts(root(), lit(5), SQL)),
            [None, Some(false), Some(false), Some(false)],
        )?;
        assert_result(
            lists.apply(&list_contains_opts(root(), lit(1), SQL)),
            [Some(true), Some(false), Some(false), Some(false)],
        )
    }

    #[test]
    fn scalar_needle_in_scalar_set_under_sql_semantics() -> VortexResult<()> {
        let set = i32_set(vec![Some(2), None]);
        let mut ctx = array_session().create_execution_ctx();
        let hit = ConstantArray::new(Scalar::primitive(2i32, Nullability::Nullable), 1)
            .into_array()
            .apply(&in_list(root(), lit(set.clone())))?;
        assert_eq!(
            hit.execute_scalar(0, &mut ctx)?.as_bool().value(),
            Some(true)
        );
        let miss = ConstantArray::new(Scalar::primitive(3i32, Nullability::Nullable), 1)
            .into_array()
            .apply(&in_list(root(), lit(set)))?;
        assert!(miss.execute_scalar(0, &mut ctx)?.is_null());
        Ok(())
    }

    #[test]
    fn sql_semantics_widen_the_return_dtype_to_the_element_nullability() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::new(
                ["a"].into(),
                vec![DType::Primitive(I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let set = lit(i32_set(vec![Some(1)]));
        assert_eq!(
            list_contains(set.clone(), col("a")).return_dtype(&dtype)?,
            DType::Bool(Nullability::NonNullable)
        );
        assert_eq!(
            in_list(col("a"), set).return_dtype(&dtype)?,
            DType::Bool(Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn options_round_trip_through_serde() -> VortexResult<()> {
        use crate::scalar_fn::ScalarFnVTable;
        let session = array_session();
        for options in [ListContainsOptions::default(), SQL] {
            let bytes = ListContains
                .serialize(&options)?
                .vortex_expect("serialized");
            assert_eq!(ListContains.deserialize(&bytes, &session)?, options);
        }
        // Files written before the option existed carry no bytes at all.
        assert_eq!(
            ListContains.deserialize(&[], &session)?,
            ListContainsOptions::default()
        );
        Ok(())
    }
}

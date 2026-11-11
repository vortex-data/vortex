// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-wise comparison of nested (struct, list, fixed-size list, map) arrays.
//!
//! Nested comparisons canonicalize both operands recursively and build a tree of row
//! comparators, one per nested level. The ordering semantics match [`Scalar`] comparison:
//! structs compare field-by-field in declaration order, lists compare element-by-element and
//! then by length, maps compare as the list of their `{key, value}` entries, and a null value
//! (at any nesting depth) orders before every non-null value by default. Only top-level nulls make the
//! comparison result null.
//!
//! A [`CompareMode`] changes how the floats at each level compare, and whether a nested null
//! orders before or after all values. In a [`NanOrder::Ieee754`]
//! mode, two floats can be unordered. The first pair of elements that is not equal decides the
//! order of the rows, so an unordered pair makes the rows unordered: only `!=` is then true.
//!
//! [`Scalar`]: crate::scalar::Scalar

use std::cmp::Ordering;

use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::MapArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::decimal::widened_buffer;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_float_ptype;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::binary::compare::primitive::ModePredicates;
use crate::scalar_fn::fns::binary::compare::primitive::NAN_CANONICAL;
use crate::scalar_fn::fns::binary::compare::primitive::NAN_IEEE754;
use crate::scalar_fn::fns::binary::compare::primitive::NAN_TOTAL_ORDER;
use crate::scalar_fn::fns::operators::CompareMode;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::NanOrder;
use crate::scalar_fn::fns::operators::NestedNulls;
use crate::scalar_fn::fns::operators::SignedZeros;

/// Compares row `i` of the left operand against row `j` of the right operand.
type Comparator<O> = Box<dyn Fn(usize, usize) -> O>;

/// A row comparator in total order.
pub(crate) type RowComparator = Comparator<Ordering>;

/// The order of two rows: [`Ordering`] for a total order, or `Option<Ordering>` for a partial
/// order, where `None` means that the rows are unordered.
trait RowOrder: Copy + 'static {
    const EQUAL: Self;

    fn total(ordering: Ordering) -> Self;

    fn partial(ordering: Option<Ordering>) -> Self;

    fn is_equal(self) -> bool;
}

impl RowOrder for Ordering {
    const EQUAL: Self = Ordering::Equal;

    #[inline]
    fn total(ordering: Ordering) -> Self {
        ordering
    }

    #[inline]
    fn partial(ordering: Option<Ordering>) -> Self {
        ordering.vortex_expect("a total compare mode orders all values")
    }

    #[inline]
    fn is_equal(self) -> bool {
        self.is_eq()
    }
}

impl RowOrder for Option<Ordering> {
    const EQUAL: Self = Some(Ordering::Equal);

    #[inline]
    fn total(ordering: Ordering) -> Self {
        Some(ordering)
    }

    #[inline]
    fn partial(ordering: Option<Ordering>) -> Self {
        ordering
    }

    #[inline]
    fn is_equal(self) -> bool {
        self == Some(Ordering::Equal)
    }
}

/// Compare two nested arrays row by row.
pub(super) fn compare_nested(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    mode: CompareMode,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs = lhs
        .clone()
        .execute::<RecursiveCanonical>(ctx)?
        .0
        .into_array();
    let rhs = rhs
        .clone()
        .execute::<RecursiveCanonical>(ctx)?
        .0
        .into_array();

    let validity = compare_validity(lhs.validity()?, rhs.validity()?, nullability)?;

    let bits = if mode.nan_order() == NanOrder::Ieee754 {
        let comparator = build_mode_comparator::<Option<Ordering>>(&lhs, &rhs, mode, ctx)?;
        partial_ordering_bits(len, &comparator, op, ctx.allocator())
    } else {
        let comparator = build_mode_comparator::<Ordering>(&lhs, &rhs, mode, ctx)?;
        total_ordering_bits(len, &comparator, op, ctx.allocator())
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

/// Bit-pack `op` over `len` rows that a total order compares.
///
/// Dispatch the operator outside the row loop so the predicate inlines into each loop; the
/// comparator call itself stays virtual.
fn total_ordering_bits(
    len: usize,
    comparator: &RowComparator,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_ordering_bits(len, comparator, Ordering::is_eq, allocator),
        CompareOperator::NotEq => {
            collect_ordering_bits(len, comparator, Ordering::is_ne, allocator)
        }
        CompareOperator::Gt => collect_ordering_bits(len, comparator, Ordering::is_gt, allocator),
        CompareOperator::Gte => collect_ordering_bits(len, comparator, Ordering::is_ge, allocator),
        CompareOperator::Lt => collect_ordering_bits(len, comparator, Ordering::is_lt, allocator),
        CompareOperator::Lte => collect_ordering_bits(len, comparator, Ordering::is_le, allocator),
    }
}

/// Bit-pack `op` over `len` rows that a partial order compares. Unordered rows satisfy only
/// [`CompareOperator::NotEq`].
fn partial_ordering_bits(
    len: usize,
    comparator: &Comparator<Option<Ordering>>,
    op: CompareOperator,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_ordering_bits(
            len,
            comparator,
            |o| o.is_some_and(Ordering::is_eq),
            allocator,
        ),
        CompareOperator::NotEq => collect_ordering_bits(
            len,
            comparator,
            |o| o.is_none_or(Ordering::is_ne),
            allocator,
        ),
        CompareOperator::Gt => collect_ordering_bits(
            len,
            comparator,
            |o| o.is_some_and(Ordering::is_gt),
            allocator,
        ),
        CompareOperator::Gte => collect_ordering_bits(
            len,
            comparator,
            |o| o.is_some_and(Ordering::is_ge),
            allocator,
        ),
        CompareOperator::Lt => collect_ordering_bits(
            len,
            comparator,
            |o| o.is_some_and(Ordering::is_lt),
            allocator,
        ),
        CompareOperator::Lte => collect_ordering_bits(
            len,
            comparator,
            |o| o.is_some_and(Ordering::is_le),
            allocator,
        ),
    }
}

/// Bit-pack `predicate(comparator(i, i))` over `len` rows.
fn collect_ordering_bits<O>(
    len: usize,
    comparator: &Comparator<O>,
    predicate: impl Fn(O) -> bool,
    allocator: &BufferAllocatorRef,
) -> BitBuffer {
    BitBuffer::collect_bool_in(len, |i| predicate(comparator(i, i)), allocator.clone())
}

/// The validity mask of a recursively canonical array.
fn validity_mask(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
    array.validity()?.execute_mask(array.len(), ctx)
}

/// Build a row comparator over two recursively canonical arrays of the same logical dtype
/// (ignoring nullability). Null values order before all non-null values at every level.
pub(crate) fn build_comparator(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RowComparator> {
    build_mode_comparator(lhs, rhs, CompareMode::TOTAL_ORDER, ctx)
}

/// Build a row comparator that compares floats in `mode`. `O` must be `Option<Ordering>` when
/// `mode` can leave two floats unordered.
fn build_mode_comparator<O: RowOrder>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    mode: CompareMode,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Comparator<O>> {
    let lhs_mask = validity_mask(lhs, ctx)?;
    let rhs_mask = validity_mask(rhs, ctx)?;
    let values = build_values_comparator::<O>(lhs, rhs, mode, ctx)?;

    if lhs_mask.all_true() && rhs_mask.all_true() {
        return Ok(values);
    }

    // The order of a value against a null.
    let value_to_null = match mode.nested_nulls() {
        NestedNulls::First => Ordering::Greater,
        NestedNulls::Last => Ordering::Less,
    };

    Ok(Box::new(move |i, j| {
        match (lhs_mask.value(i), rhs_mask.value(j)) {
            (true, true) => values(i, j),
            (true, false) => O::total(value_to_null),
            (false, true) => O::total(value_to_null.reverse()),
            (false, false) => O::EQUAL,
        }
    }))
}

/// Build a comparator over the non-null values of two recursively canonical arrays.
fn build_values_comparator<O: RowOrder>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    mode: CompareMode,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Comparator<O>> {
    if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
        vortex_bail!(
            "Cannot compare different DTypes {} and {}",
            lhs.dtype(),
            rhs.dtype()
        );
    }

    Ok(match lhs.dtype() {
        // Null values compare through the validity wrapper; the value comparator is trivial.
        DType::Null => Box::new(|_, _| O::EQUAL),
        DType::Bool(_) => {
            let lhs = lhs.clone().execute::<BoolArray>(ctx)?.into_bit_buffer();
            let rhs = rhs.clone().execute::<BoolArray>(ctx)?.into_bit_buffer();
            Box::new(move |i, j| O::total(lhs.value(i).cmp(&rhs.value(j))))
        }
        DType::Primitive(ptype, _) if ptype.is_float() && !mode.is_total_order() => {
            match_each_float_ptype!(*ptype, |T| {
                float_comparator::<T, O>(lhs, rhs, mode, ctx)?
            })
        }
        DType::Primitive(ptype, _) => {
            match_each_native_ptype!(*ptype, |T| { primitive_comparator::<T, O>(lhs, rhs, ctx)? })
        }
        DType::Decimal(..) => {
            let lhs = lhs.clone().execute::<DecimalArray>(ctx)?;
            let rhs = rhs.clone().execute::<DecimalArray>(ctx)?;
            let common = lhs.values_type().max(rhs.values_type());
            crate::match_each_decimal_value_type!(common, |W| {
                let lhs = widened_buffer::<W>(&lhs);
                let rhs = widened_buffer::<W>(&rhs);
                Box::new(move |i: usize, j: usize| O::total(lhs[i].cmp(&rhs[j]))) as Comparator<O>
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let lhs = lhs.clone().execute::<VarBinViewArray>(ctx)?;
            let rhs = rhs.clone().execute::<VarBinViewArray>(ctx)?;
            Box::new(move |i, j| O::total(view_bytes(&lhs, i).cmp(view_bytes(&rhs, j))))
        }
        DType::Struct(..) => {
            let lhs = lhs.clone().execute::<StructArray>(ctx)?;
            let rhs = rhs.clone().execute::<StructArray>(ctx)?;
            let fields = lhs
                .iter_unmasked_fields()
                .zip(rhs.iter_unmasked_fields())
                .map(|(lhs_field, rhs_field)| {
                    build_mode_comparator::<O>(lhs_field, rhs_field, mode, ctx)
                })
                .collect::<VortexResult<Vec<_>>>()?;
            Box::new(move |i, j| {
                fields
                    .iter()
                    .map(|cmp| cmp(i, j))
                    .find(|ordering| !ordering.is_equal())
                    .unwrap_or(O::EQUAL)
            })
        }
        DType::List(..) => {
            let lhs = lhs.clone().execute::<ListViewArray>(ctx)?;
            let rhs = rhs.clone().execute::<ListViewArray>(ctx)?;
            let elements = build_mode_comparator::<O>(lhs.elements(), rhs.elements(), mode, ctx)?;
            // Materialize offsets and sizes once instead of paying `offset_at`/`size_at`'s
            // per-call encoding dispatch inside the row loop.
            let lhs_offsets = usize_values(lhs.offsets(), ctx)?;
            let lhs_sizes = usize_values(lhs.sizes(), ctx)?;
            let rhs_offsets = usize_values(rhs.offsets(), ctx)?;
            let rhs_sizes = usize_values(rhs.sizes(), ctx)?;
            Box::new(move |i, j| {
                let lhs_len = lhs_sizes[i];
                let rhs_len = rhs_sizes[j];
                (0..lhs_len.min(rhs_len))
                    .map(|el| elements(lhs_offsets[i] + el, rhs_offsets[j] + el))
                    .find(|ordering| !ordering.is_equal())
                    .unwrap_or_else(|| O::total(lhs_len.cmp(&rhs_len)))
            })
        }
        DType::FixedSizeList(_, size, _) => {
            let size = *size as usize;
            let lhs = lhs.clone().execute::<FixedSizeListArray>(ctx)?;
            let rhs = rhs.clone().execute::<FixedSizeListArray>(ctx)?;
            let elements = build_mode_comparator::<O>(lhs.elements(), rhs.elements(), mode, ctx)?;
            Box::new(move |i, j| {
                (0..size)
                    .map(|el| elements(i * size + el, j * size + el))
                    .find(|ordering| !ordering.is_equal())
                    .unwrap_or(O::EQUAL)
            })
        }
        DType::Extension(_) => {
            let lhs = lhs.clone().execute::<ExtensionArray>(ctx)?;
            let rhs = rhs.clone().execute::<ExtensionArray>(ctx)?;
            build_mode_comparator::<O>(lhs.storage_array(), rhs.storage_array(), mode, ctx)?
        }
        DType::Map(..) => {
            let lhs = lhs.clone().execute::<MapArray>(ctx)?;
            let rhs = rhs.clone().execute::<MapArray>(ctx)?;
            build_values_comparator::<O>(lhs.entries(), rhs.entries(), mode, ctx)?
        }
        DType::Union(..) | DType::Variant(_) => {
            vortex_bail!("compare is not supported for dtype {}", lhs.dtype())
        }
    })
}

fn primitive_comparator<T: NativePType, O: RowOrder>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Comparator<O>> {
    let lhs = lhs
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<T>();
    let rhs = rhs
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<T>();
    Ok(Box::new(move |i, j| O::total(lhs[i].total_compare(rhs[j]))))
}

/// A comparator over floats that orders signed zeros and `NaN` values as `mode` requires.
fn float_comparator<T: NativePType, O: RowOrder>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    mode: CompareMode,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Comparator<O>> {
    let lhs = lhs
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<T>();
    let rhs = rhs
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<T>();

    macro_rules! comparator {
        ($zeros_equal:expr, $nan:expr) => {
            Box::new(move |i, j| {
                O::partial(ModePredicates::<$zeros_equal, $nan>::compare(
                    lhs[i], rhs[j],
                ))
            })
        };
    }

    Ok(match (mode.signed_zeros(), mode.nan_order()) {
        (SignedZeros::Distinct, NanOrder::TotalOrder) => comparator!(false, NAN_TOTAL_ORDER),
        (SignedZeros::Distinct, NanOrder::Canonical) => comparator!(false, NAN_CANONICAL),
        (SignedZeros::Distinct, NanOrder::Ieee754) => comparator!(false, NAN_IEEE754),
        (SignedZeros::Equal, NanOrder::TotalOrder) => comparator!(true, NAN_TOTAL_ORDER),
        (SignedZeros::Equal, NanOrder::Canonical) => comparator!(true, NAN_CANONICAL),
        (SignedZeros::Equal, NanOrder::Ieee754) => comparator!(true, NAN_IEEE754),
    })
}

/// Materialize an integer array as `usize` values for `O(1)` row access.
fn usize_values(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<usize>> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(primitive.ptype(), |P| {
        Ok(primitive.as_slice::<P>().iter().map(|v| v.as_()).collect())
    })
}

/// The bytes of row `index`, resolved directly from the view without cloning buffer handles the
/// way `VarBinViewArray::bytes_at` does.
fn view_bytes(array: &VarBinViewArray, index: usize) -> &[u8] {
    let view = &array.views()[index];
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view = view.as_view();
        &array.buffer(view.buffer_index as usize).as_slice()[view.as_range()]
    }
}

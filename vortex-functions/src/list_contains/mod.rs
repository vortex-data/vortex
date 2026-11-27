// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod chunked;
mod dict;
mod haystack;
mod kernel;
mod normalize;
mod prepared;
mod runend;

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::iter;
use std::ops::BitOr;

pub use kernel::*;
use num_traits::Zero;
pub use prepared::PreparedSet;
pub use prepared::PreparedSetData;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::proto::expr as pb;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ReduceNode;
use vortex_array::scalar_fn::ReduceNodeValidity;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::compare::Compare;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::builtin::operators::CompareMode;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::scalar_fn::is_not_null_node;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::list_contains::normalize::FloatNormalization;

/// Registers [`ListContains`], [`PreparedSet`] and the element rules of [`ListContains`] for
/// chunked, dictionary and run-end needles.
pub(crate) fn initialize(session: &VortexSession) {
    let scalar_fns = session.scalar_fns();
    scalar_fns.register(ListContains);
    scalar_fns.register(PreparedSet);

    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(
        ListContains,
        Chunked,
        const { &ListContainsElementReduceAdaptor(Chunked) },
    );
    kernels.register_reduce_parent_rule(
        ListContains,
        Dict,
        const { &ListContainsElementReduceAdaptor(Dict) },
    );
    kernels.register_reduce_parent_rule(
        ListContains,
        RunEnd,
        const { &ListContainsElementReduceAdaptor(RunEnd) },
    );
}

#[derive(Clone)]
pub struct ListContains;

/// Options for [`ListContains`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// How the needle and the elements compare floats, at the top level and in nested values.
    ///
    /// The default options compare in [`CompareMode::TOTAL_ORDER`], which compares the bits of
    /// floats.
    pub compare_mode: CompareMode,
}

impl Default for ListContainsOptions {
    fn default() -> Self {
        Self {
            sql_null_semantics: false,
            compare_mode: CompareMode::TOTAL_ORDER,
        }
    }
}

impl Display for ListContainsOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.sql_null_semantics {
            write!(f, "sql_null_semantics")?;
        }
        if !self.compare_mode.is_total_order() {
            if self.sql_null_semantics {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.compare_mode)?;
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

    /// Builds a `list_contains` result from precomputed membership `bits` and needle validity,
    /// applying the null semantics of a constant list with `is_empty` and `has_null_element`.
    pub fn result_from_bits(
        &self,
        bits: BitBuffer,
        needle_validity: Validity,
        list_dtype: &DType,
        needle_dtype: &DType,
        is_empty: bool,
        has_null_element: bool,
    ) -> VortexResult<ArrayRef> {
        let nullability = self.result_nullability(list_dtype, needle_dtype);
        let validity = if is_empty && !self.sql_null_semantics {
            Validity::NonNullable
        } else if self.sql_null_semantics && has_null_element {
            // Only a match is known. A comparison with the null element makes a non-match unknown.
            needle_validity.and(Validity::from(bits.clone()))?
        } else {
            needle_validity
        };

        Ok(BoolArray::new(bits, validity.union_nullability(nullability)).into_array())
    }

    /// Builds a `list_contains` result when no valid needle row can match any list element.
    pub fn all_false_result(
        &self,
        len: usize,
        needle_validity: Validity,
        list_dtype: &DType,
        needle_dtype: &DType,
        is_empty: bool,
        has_null_element: bool,
    ) -> VortexResult<ArrayRef> {
        let nullability = self.result_nullability(list_dtype, needle_dtype);
        if self.sql_null_semantics && has_null_element {
            return Ok(
                ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array(),
            );
        }
        if is_empty && !self.sql_null_semantics {
            return Ok(ConstantArray::new(Scalar::bool(false, nullability), len).into_array());
        }

        match needle_validity.union_nullability(nullability) {
            Validity::NonNullable | Validity::AllValid => {
                Ok(ConstantArray::new(Scalar::bool(false, nullability), len).into_array())
            }
            Validity::AllInvalid => {
                Ok(ConstantArray::new(Scalar::null(DType::Bool(nullability)), len).into_array())
            }
            Validity::Array(validity) => ConstantArray::new(Scalar::bool(false, nullability), len)
                .into_array()
                .mask(validity),
        }
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
        ScalarFnArray::try_new(ListContains.bind(options), [list, needle])
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
                compare_mode: Some(instance.compare_mode.into()),
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

        // Options written before the mode existed have no mode. They compare in total order.
        let compare_mode = opts
            .compare_mode
            .map(CompareMode::try_from)
            .transpose()?
            .unwrap_or(CompareMode::TOTAL_ORDER);

        Ok(ListContainsOptions {
            sql_null_semantics: opts.sql_null_semantics,
            compare_mode,
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
        // Scalar equality compares the bits of floats, so a compare mode that normalizes them
        // takes the lookup below.
        if let Some(value_scalar) = value_array.as_constant()
            && let Some(list) = constant_list(&list_array)
            && FloatNormalization::try_new(options.compare_mode, value_scalar.dtype())?
                .is_identity()
        {
            let result = compute_contains_scalar(list, value_scalar, options)?;
            return Ok(ConstantArray::new(result, args.row_count()).into_array());
        }

        compute_list_contains(&list_array, &value_array, options, ctx)
    }

    /// A constant list and a constant needle fold to their constant answer, in expression and
    /// array trees alike. Otherwise a constant list becomes a [`PreparedSet`], so that every
    /// batch, and every part of the needle that the set is pushed into, looks up its needles in
    /// one shared set rather than preparing the list again.
    fn reduce<T: ReduceNode>(&self, options: &Self::Options, node: &T) -> VortexResult<Option<T>> {
        let list = node.child(0);
        let needle = node.child(1);

        if let Some(needle) = needle.as_constant()
            && let Some(list) = constant_list_node(&list)
            && FloatNormalization::try_new(options.compare_mode, needle.dtype())?.is_identity()
        {
            let result = compute_contains_scalar(list, needle, options)?;
            return Ok(Some(node.new_constant(result)));
        }

        let Some(set) = prepare_constant_list(&list, &needle)? else {
            return Ok(None);
        };

        let set = node.new_node(PreparedSet.bind(set), &[])?;
        Ok(Some(
            node.new_node(ListContains.bind(*options), &[set, needle])?,
        ))
    }

    /// The validity of `needle IN list` for a literal list, decided without a lookup.
    ///
    /// Off SQL null semantics only a null needle gives null, unless the list is empty, which
    /// answers `false` for every needle. Under them a null element can give null too, by leaving
    /// a non-match unknown, so only a list without one is decided from the needle alone. A null
    /// list gives null for every needle.
    ///
    /// Without this the validity of a node over a constant list would execute the node, which
    /// prepares the list as a set only to read the validity off the result.
    fn validity<T: ReduceNode>(
        &self,
        options: &Self::Options,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        let list = node.child(0);
        let Some(list) = constant_list_node(&list) else {
            return Ok(ReduceNodeValidity::Irreducible);
        };
        if !matches!(list.dtype(), DType::List(..)) {
            return Ok(ReduceNodeValidity::Irreducible);
        }
        let Some(elements) = list.as_list().element_values() else {
            return Ok(ReduceNodeValidity::Reduced(node.new_constant(false.into())));
        };

        if options.sql_null_semantics && elements.iter().any(Option::is_none) {
            return Ok(ReduceNodeValidity::Irreducible);
        }
        if elements.is_empty() && !options.sql_null_semantics {
            return Ok(ReduceNodeValidity::Reduced(node.new_constant(true.into())));
        }
        Ok(ReduceNodeValidity::Reduced(is_not_null_node(
            &node.child(1),
        )?))
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

    let nullability = options.result_nullability(array.dtype(), value.dtype());
    if array.is_empty() {
        return Ok(BoolArray::new(BitBuffer::empty(), nullability.into()).into_array());
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

    // The reduce step has already replaced a constant list with a prepared set.
    if let Some(set) = PreparedSet::set_of(array) {
        return set.contains(value, options, ctx);
    }

    if let Some(value_scalar) = value.as_constant() {
        return list_contains_scalar(array, value_scalar, nullability, options, ctx);
    }
    lists_contain_needles(array, value, nullability, options, ctx)
}

/// The list that every row of `array` holds: a constant, or a set prepared from one.
fn constant_list(array: &ArrayRef) -> Option<&Scalar> {
    if let Some(constant) = array.as_opt::<Constant>() {
        return Some(constant.data().scalar());
    }
    PreparedSet::set_of(array).map(PreparedSetData::list)
}

/// The set of `list` when it is a non-null constant list, a literal in an expression tree or a
/// constant array in an array tree, whose elements have the dtype of `needle`.
///
/// A null list has no set, and a needle of another dtype is an error for execution to report.
fn prepare_constant_list<T: ReduceNode>(
    list: &T,
    needle: &T,
) -> VortexResult<Option<PreparedSetData>> {
    let DType::List(element_dtype, _) = list.node_dtype()? else {
        return Ok(None);
    };
    if !element_dtype.eq_ignore_nullability(&needle.node_dtype()?) {
        return Ok(None);
    }

    // A literal shares its list with the set, so the list is not cloned. Any other scalar
    // function, a prepared set included, is not a constant list.
    if let Some(scalar_fn) = list.scalar_fn() {
        return match scalar_fn.as_opt::<Literal>() {
            Some(literal) if !literal.is_null() => {
                PreparedSetData::try_from_literal(scalar_fn.clone()).map(Some)
            }
            _ => Ok(None),
        };
    }

    match list.as_constant() {
        Some(list) if !list.is_null() => PreparedSetData::try_new(list.clone()).map(Some),
        _ => Ok(None),
    }
}

/// The list a reduce node holds when it is a constant or a prepared set, in an expression or an
/// array tree.
fn constant_list_node<T: ReduceNode>(node: &T) -> Option<&Scalar> {
    if let Some(list) = node.as_constant() {
        return Some(list);
    }
    node.scalar_fn()?
        .as_opt::<PreparedSet>()
        .map(|set| set.list())
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
    let matching_elements = Compare::try_new_with_mode(
        elems.clone(),
        rhs.into_array(),
        CompareOperator::Eq,
        options.compare_mode,
    )?
    .into_array();

    let mut matches = match matching_elements.execute::<Columnar>(ctx)? {
        Columnar::Constant(constant) => {
            return match constant.scalar().as_bool().value() {
                // The needle is not null, so every comparison is null only if every element is.
                None if options.sql_null_semantics => {
                    null_needle_in_lists(&list_array, nullability, ctx)
                }
                // No element matches: false, unless the list itself is null.
                None | Some(false) => list_false_or_null(&list_array, nullability, ctx),
                // Every element matches: true, unless the list itself is empty or null.
                Some(true) => list_is_not_empty(&list_array, nullability, ctx),
            };
        }
        Columnar::Canonical(canonical) => canonical.into_bool(),
    };
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
    let list_valid = list_array.validity()?.execute_mask(len, ctx)?;

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
    // A valid list with elements gathered at least one comparison. A null list gathered none,
    // and its row is null whatever its bit here.
    let non_empty =
        BitBuffer::collect_bool_in(len, |row| gathered.lens[row] != 0, ctx.allocator().clone());

    // `any_null` is only read under SQL null semantics.
    let comparisons =
        Compare::try_new_with_mode(elements, needles, CompareOperator::Eq, options.compare_mode)?
            .into_array();
    let (any_true, any_null) = match comparisons.execute::<Columnar>(ctx)? {
        // One answer for every comparison decides each row from whether it compared anything.
        Columnar::Constant(constant) => match constant.scalar().as_bool().value() {
            Some(true) => (non_empty.clone(), Mask::new_false(len)),
            Some(false) => (BitBuffer::new_unset(len), Mask::new_false(len)),
            None => (BitBuffer::new_unset(len), Mask::from(non_empty.clone())),
        },
        Columnar::Canonical(canonical) => {
            let matches = canonical.into_bool();
            let compared = matches.validity()?.execute_mask(matches.len(), ctx)?;
            let starts = PrimitiveArray::new(gathered.starts, Validity::NonNullable);
            let lens = PrimitiveArray::new(gathered.lens, Validity::NonNullable);

            let matched = Mask::from(matches.to_bit_buffer()) & &compared;
            let any_true = process_matches::<u64, u64>(
                BoolArray::new(matched.into_bit_buffer(), Validity::NonNullable),
                len,
                starts.clone(),
                lens.clone(),
                ctx,
            );

            let any_null = if options.sql_null_semantics && !compared.all_true() {
                Mask::from(process_matches::<u64, u64>(
                    BoolArray::new((!&compared).into_bit_buffer(), Validity::NonNullable),
                    len,
                    starts,
                    lens,
                    ctx,
                ))
            } else {
                Mask::new_false(len)
            };
            (any_true, any_null)
        }
    };

    // A null needle is null against a list with elements, but an empty list holds nothing to
    // compare it to unless the function is strict.
    let needle_valid = values.validity()?.execute_mask(len, ctx)?;
    let decided = if options.sql_null_semantics {
        // A comparison with a null element leaves a non-match unknown.
        needle_valid & &(Mask::from(any_true.clone()) | &!any_null)
    } else {
        &needle_valid | &!Mask::from(non_empty)
    };

    // A non-nullable result has non-null lists and needles, and no null element that could leave
    // a row undecided.
    let validity = match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable => list_array
            .validity()?
            .and(Validity::from_mask(decided, Nullability::Nullable))?,
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
        list_valid: &Mask,
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

            // A set bit means at least one element in this list's range matches.
            bits.slice(offset..offset + size)
                .set_indices()
                .next()
                .is_some()
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

/// `false` for an empty list, which holds nothing to compare against, and `null` for any other list.
///
/// This is a null needle off SQL null semantics, or under them a needle against lists that hold
/// only nulls.
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
mod tests;

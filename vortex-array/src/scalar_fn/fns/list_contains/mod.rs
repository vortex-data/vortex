// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Formatter;
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
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::and_collect;
use crate::expr::gt;
use crate::expr::lit;
use crate::expr::lt;
use crate::expr::or;
use crate::match_each_integer_ptype;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

#[derive(Clone)]
pub struct ListContains;

impl ScalarFnVTable for ListContains {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.list.contains")
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
    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "contains(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
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

    fn stat_falsification(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        let list = expr.child(0);
        let needle = expr.child(1);

        // falsification(contains([1,2,5], x)) =>
        //   falsification(x != 1) and falsification(x != 2) and falsification(x != 5)
        let min = list.stat_min(catalog)?;
        let max = list.stat_max(catalog)?;
        // If the list is constant when we can compare each element to the value
        if min == max {
            let list_ = min
                .as_opt::<Literal>()
                .and_then(|l| l.as_list_opt())
                .and_then(|l| l.elements())?;
            if list_.is_empty() {
                // contains([], x) is always false.
                return Some(lit(true));
            }
            let value_max = needle.stat_max(catalog)?;
            let value_min = needle.stat_min(catalog)?;

            return and_collect(list_.iter().map(move |v| {
                or(
                    lt(value_max.clone(), lit(v.clone())),
                    gt(value_min.clone(), lit(v.clone())),
                )
            }));
        }

        None
    }

    // Nullability matters for contains([], x) where x is false.
    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
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
        constant_list_scalar_contains(&list_scalar.as_list(), value, nullability)
    } else {
        todo!("unsupported list contains with list and element as arrays")
    }
}

/// There is a constant list scalar (haystack) being compared to an array of needles.
fn constant_list_scalar_contains(
    list_scalar: &ListScalar,
    values: &ArrayRef,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let elements = list_scalar.elements().vortex_expect("non null");

    let len = values.len();
    let false_scalar = Scalar::bool(false, nullability);

    let result = elements
        .iter()
        .map(|element| {
            Binary
                .try_new_array(
                    len,
                    Operator::Eq,
                    [
                        ConstantArray::new(element.clone(), len).into_array(),
                        values.clone(),
                    ],
                )?
                .fill_null(false_scalar.clone())
        })
        .collect::<VortexResult<Vec<_>>>()?
        .into_iter()
        .try_reduce_balanced(|acc, res| acc.binary(res, Operator::Or))?;

    Ok(result.unwrap_or_else(|| ConstantArray::new(false_scalar, len).into_array()))
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
    let matching_elements = Binary.try_new_array(
        elems.len(),
        Operator::Eq,
        &[elems.clone(), rhs.clone().into_array()],
    )?;

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

    // Get the offsets and sizes as primitive arrays.
    let offsets = list_array
        .offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let sizes = list_array.sizes().clone().execute::<PrimitiveArray>(ctx)?;

    // Process based on the offset and size types.
    let list_matches = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
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
    let bits = matches.to_bit_buffer();

    (0..list_array_len)
        .map(|i| {
            let offset = offsets_slice[i].as_();
            let size = sizes_slice[i].as_();

            // BitIndexIterator yields indices of true bits only. If `.next()` returns
            // `Some(_)`, at least one element in this list's range matches.
            let mut set_bits = BitIndexIterator::new(bits.inner().as_ref(), offset, size);
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
    if matches!(list_array.validity()?, Validity::AllInvalid) {
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

    use itertools::Itertools;
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_utils::aliases::hash_map::HashMap;
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ListArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
    use crate::dtype::DType;
    use crate::dtype::Field;
    use crate::dtype::FieldPath;
    use crate::dtype::FieldPathSet;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::gt;
    use crate::expr::list_contains;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::or;
    use crate::expr::pruning::checked_pruning_expr;
    use crate::expr::root;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::list_contains::BoolArray;
    use crate::scalar_fn::fns::list_contains::Constant;
    use crate::scalar_fn::fns::list_contains::ConstantArray;
    use crate::scalar_fn::fns::list_contains::ListViewArray;
    use crate::scalar_fn::fns::list_contains::PrimitiveArray;
    use crate::validity::Validity;

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
            item.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
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
            item.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
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
            item.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
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
            item.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert_eq!(
            item.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
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
            item.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::Nullable)
        );
        assert!(
            !item
                .is_valid(1, &mut LEGACY_SESSION.create_execution_ctx())
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
    pub fn list_falsification() {
        let expr = list_contains(
            lit(Scalar::list(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                vec![1.into(), 2.into(), 3.into()],
                Nullability::NonNullable,
            )),
            col("a"),
        );

        let (expr, st) = checked_pruning_expr(
            &expr,
            &FieldPathSet::from_iter([
                FieldPath::from_iter([Field::Name("a".into()), Field::Name("max".into())]),
                FieldPath::from_iter([Field::Name("a".into()), Field::Name("min".into())]),
            ]),
        )
        .unwrap();

        assert_eq!(
            &expr,
            &and(
                and(
                    or(lt(col("a_max"), lit(1i32)), gt(col("a_min"), lit(1i32)),),
                    or(lt(col("a_max"), lit(2i32)), gt(col("a_min"), lit(2i32)),)
                ),
                or(lt(col("a_max"), lit(3i32)), gt(col("a_min"), lit(3i32)),)
            )
        );

        assert_eq!(
            st.map(),
            &HashMap::from_iter([(
                FieldPath::from_name("a"),
                HashSet::from([Stat::Min, Stat::Max])
            )])
        );
    }

    #[test]
    pub fn test_display() {
        let expr = list_contains(get_item("tags", root()), lit("urgent"));
        assert_eq!(expr.to_string(), "contains($.tags, \"urgent\")");

        let expr2 = list_contains(root(), lit(42));
        assert_eq!(expr2.to_string(), "contains($, 42i32)");
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
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(true, Nullability::NonNullable)
        );

        // Test contains false
        let expr = list_contains(lit(list_scalar), lit(42i32));
        let result = arr.apply(&expr).unwrap();
        assert_eq!(
            result
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::bool(false, Nullability::NonNullable)
        );
    }

    // -- Tests migrated from compute/list_contains.rs --

    fn nonnull_strings(values: Vec<Vec<&str>>) -> ArrayRef {
        #[expect(deprecated)]
        let result = ListArray::from_iter_slow::<u64, _>(
            values,
            Arc::new(DType::Utf8(Nullability::NonNullable)),
        )
        .unwrap()
        .to_listview()
        .into_array();
        result
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

        #[expect(deprecated)]
        let result = ListArray::try_new(elements, offsets, Validity::NonNullable)
            .unwrap()
            .as_array()
            .to_listview()
            .into_array();
        result
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
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_constant_list() {
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
        assert!(contains.is::<Constant>(), "Expected constant result");
        let expected = BoolArray::from_iter([true, true]);
        assert_arrays_eq!(contains, expected);
    }

    #[test]
    fn test_all_nulls() {
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
        assert!(contains.is::<Constant>(), "Expected constant result");

        let expected = BoolArray::new(
            [false, false, false, false, false].into_iter().collect(),
            Validity::AllInvalid,
        );
        assert_arrays_eq!(contains, expected);
    }

    #[test]
    fn test_list_array_element() {
        let list_scalar = Scalar::list(
            Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
            vec![1.into(), 3.into(), 6.into()],
            Nullability::NonNullable,
        );

        let arr = (0..7).collect::<PrimitiveArray>().into_array();
        let expr = list_contains(lit(list_scalar), root());
        let contains = arr.apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, true, false, true, false, false, true]);
        assert_arrays_eq!(contains, expected);
    }

    #[test]
    fn test_list_contains_empty_listview() {
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
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_list_contains_all_null_elements() {
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
        assert_arrays_eq!(result, expected);

        // Searching for non-null
        let expr2 = list_contains(root(), lit(42i32));
        let result2 = list_array.into_array().apply(&expr2).unwrap();

        let expected2 = BoolArray::from_iter([false, false, false]);
        assert_arrays_eq!(result2, expected2);
    }

    #[test]
    fn test_list_contains_large_offsets() {
        let elements = Buffer::from_iter([1i32, 2, 3, 4, 5]).into_array();

        let offsets = Buffer::from_iter([0u32, 1, 4, 0]).into_array();
        let sizes = Buffer::from_iter([1u32, 2, 1, 0]).into_array();

        let list_array =
            ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

        let expr = list_contains(root(), lit(2i32));
        let result = list_array.clone().into_array().apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, true, false, false]);
        assert_arrays_eq!(result, expected);

        let expr5 = list_contains(root(), lit(5i32));
        let result5 = list_array.into_array().apply(&expr5).unwrap();

        let expected5 = BoolArray::from_iter([false, false, true, false]);
        assert_arrays_eq!(result5, expected5);
    }

    #[test]
    fn test_list_contains_offset_size_boundary() {
        let elements = Buffer::from_iter(0..256).into_array();
        let offsets = Buffer::from_iter([0u8, 100, 200, 254]).into_array();
        let sizes = Buffer::from_iter([50u8, 50, 54, 2]).into_array();

        let list_array =
            ListViewArray::new(elements.into_array(), offsets, sizes, Validity::NonNullable);

        let expr = list_contains(root(), lit(255i32));
        let result = list_array.clone().into_array().apply(&expr).unwrap();

        let expected = BoolArray::from_iter([false, false, false, true]);
        assert_arrays_eq!(result, expected);

        let expr_zero = list_contains(root(), lit(0i32));
        let result_zero = list_array.into_array().apply(&expr_zero).unwrap();

        let expected_zero = BoolArray::from_iter([true, false, false, false]);
        assert_arrays_eq!(result_zero, expected_zero);
    }
}

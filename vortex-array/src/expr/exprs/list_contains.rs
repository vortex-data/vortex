// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::ops::BitOr;
use std::ops::Deref;

use arrow_buffer::bit_iterator::BitIndexIterator;
use vortex_buffer::BitBuffer;
use vortex_compute::logical::LogicalOr;
use vortex_dtype::DType;
use vortex_dtype::IntegerPType;
use vortex_dtype::Nullability;
use vortex_dtype::PTypeDowncastExt;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_vector::BoolDatum;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::Vector;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;
use vortex_vector::listview::ListViewScalar;
use vortex_vector::listview::ListViewVector;
use vortex_vector::primitive::PVector;

use crate::ArrayRef;
use crate::compute::list_contains as compute_list_contains;
use crate::expr::Arity;
use crate::expr::Binary;
use crate::expr::ChildName;
use crate::expr::EmptyOptions;
use crate::expr::ExecutionArgs;
use crate::expr::ExprId;
use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::VTable;
use crate::expr::VTableExt;
use crate::expr::exprs::binary::and;
use crate::expr::exprs::binary::gt;
use crate::expr::exprs::binary::lt;
use crate::expr::exprs::binary::or;
use crate::expr::exprs::literal::Literal;
use crate::expr::exprs::literal::lit;
use crate::expr::operators;

pub struct ListContains;

impl VTable for ListContains {
    type Options = EmptyOptions;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.list.contains")
    }

    fn serialize(&self, _instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Self::Options> {
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
        let needle_dtype = &arg_dtypes[0];

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

    fn evaluate(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        scope: &ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let list_array = expr.child(0).evaluate(scope)?;
        let value_array = expr.child(1).evaluate(scope)?;
        compute_list_contains(list_array.as_ref(), value_array.as_ref())
    }

    fn execute(&self, _options: &Self::Options, args: ExecutionArgs) -> VortexResult<Datum> {
        let [lhs, rhs]: [Datum; _] = args
            .datums
            .try_into()
            .map_err(|_| vortex_err!("Wrong number of arguments for ListContains expression"))?;

        let matches = match (lhs.as_scalar().is_some(), rhs.as_scalar().is_some()) {
            (true, true) => {
                let list = lhs.into_scalar().vortex_expect("scalar").into_list();
                let needle = rhs.into_scalar().vortex_expect("scalar");
                // Convert the needle scalar to a vector with row_count
                // elements and reuse constant_list_scalar_contains
                let needle_vector = needle.repeat(args.row_count).freeze();
                constant_list_scalar_contains(list, needle_vector)?
            }
            (true, false) => constant_list_scalar_contains(
                lhs.into_scalar().vortex_expect("scalar").into_list(),
                rhs.into_vector().vortex_expect("vector"),
            )?,
            (false, true) => list_contains_scalar(
                lhs.unwrap_into_vector(args.row_count).into_list(),
                rhs.into_scalar().vortex_expect("scalar").into_list(),
            )?,
            (false, false) => {
                vortex_bail!(
                    "ListContains currently only supports constant needle (RHS) or constant list (LHS)"
                )
            }
        };
        Ok(Datum::Vector(matches.into()))
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

            return list_
                .iter()
                .map(move |v| {
                    or(
                        lt(value_max.clone(), lit(v.clone())),
                        gt(value_min.clone(), lit(v.clone())),
                    )
                })
                .reduce(and);
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

/// Creates an expression that checks if a value is contained in a list.
///
/// Returns a boolean array indicating whether the value appears in each list.
///
/// ```rust
/// # use vortex_array::expr::{list_contains, lit, root};
/// let expr = list_contains(root(), lit(42));
/// ```
pub fn list_contains(list: Expression, value: Expression) -> Expression {
    ListContains.new_expr(EmptyOptions, [list, value])
}

/// Returns a [`BoolVector`] where each bit represents if a list contains the scalar.
// FIXME(ngates): test implementation and move to vortex-compute
fn list_contains_scalar(list: ListViewVector, value: ListViewScalar) -> VortexResult<BoolVector> {
    // If the list array is constant, we perform a single comparison.
    // if list.len() > 1 && list.is_constant() {
    //     let contains = list_contains_scalar(&array.slice(0..1), value, nullability)?;
    //     return Ok(ConstantArray::new(contains.scalar_at(0), array.len()).into_array());
    // }

    let elems = list.elements();
    if elems.is_empty() {
        // Must return false when a list is empty (but valid), or null when the list itself is null.
        // return crate::compute::list_contains::list_false_or_null(&list_array, nullability);
        todo!()
    }

    let matches = Binary
        .bind(operators::Operator::Eq)
        .execute(ExecutionArgs {
            datums: vec![
                Datum::Vector(elems.deref().clone()),
                Datum::Scalar(value.into()),
            ],
            // FIXME(ngates): dtypes
            dtypes: vec![],
            row_count: elems.len(),
            return_dtype: DType::Bool(Nullability::Nullable),
        })?
        .unwrap_into_vector(elems.len())
        .into_bool()
        .into_bits();

    // // Fast path: no elements match.
    // if let Some(pred) = matches.as_constant() {
    //     return match pred.as_bool().value() {
    //         // All comparisons are invalid (result in `null`), and search is not null because
    //         // we already checked for null above.
    //         None => {
    //             assert!(
    //                 !rhs.scalar().is_null(),
    //                 "Search value must not be null here"
    //             );
    //             // False, unless the list itself is null in which case we return null.
    //             crate::compute::list_contains::list_false_or_null(&list_array, nullability)
    //         }
    //         // No elements match, and all comparisons are valid (result in `false`).
    //         Some(false) => {
    //             // False, but match the nullability to the input list array.
    //             Ok(
    //                 ConstantArray::new(Scalar::bool(false, nullability), list_array.len())
    //                     .into_array(),
    //             )
    //         }
    //         // All elements match, and all comparisons are valid (result in `true`).
    //         Some(true) => {
    //             // True, unless the list itself is empty or NULL.
    //             crate::compute::list_contains::list_is_not_empty(&list_array, nullability)
    //         }
    //     };
    // }

    // Get the offsets and sizes as primitive arrays.
    let offsets = list.offsets();
    let sizes = list.sizes();

    // Process based on the offset and size types.
    let list_matches = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            process_matches::<O, S>(
                matches,
                list.len(),
                offsets.downcast::<O>(),
                sizes.downcast::<S>(),
            )
        })
    });

    Ok(BoolVector::new(list_matches, list.validity().clone()))
}

// Then there is a constant list scalar (haystack) being compared to an array of needles.
// FIXME(ngates): test implementation and move to vortex-compute
fn constant_list_scalar_contains(list: ListViewScalar, values: Vector) -> VortexResult<BoolVector> {
    let elements = list.value().elements();

    // For each element in the list, we perform a full comparison over the values and OR
    // the results together.
    let mut result: BoolVector = BoolVector::new(
        BitBuffer::new_unset(values.len()),
        Mask::new(values.len(), true),
    );
    for i in 0..elements.len() {
        let element = Datum::Scalar(elements.scalar_at(i));
        let compared: BoolDatum = Binary
            .bind(operators::Operator::Eq)
            .execute(ExecutionArgs {
                datums: vec![Datum::Vector(values.clone()), element],
                dtypes: vec![
                    // FIXME(ngates): call compute function directly!
                ],
                row_count: values.len(),
                return_dtype: DType::Bool(Nullability::Nullable),
            })?
            .into_bool();
        let compared = Datum::from(compared)
            .unwrap_into_vector(values.len())
            .into_bool();

        result = LogicalOr::or(&result, &compared);
    }

    Ok(result)
}

/// Returns a [`BitBuffer`] where each bit represents if a list contains the scalar, derived from a
/// [`BoolArray`] of matches on the child elements array.
///
/// TODO(ngates): replace this for aggregation function.
fn process_matches<O, S>(
    matches: BitBuffer,
    list_array_len: usize,
    offsets: &PVector<O>,
    sizes: &PVector<S>,
) -> BitBuffer
where
    O: IntegerPType,
    S: IntegerPType,
{
    let offsets_slice = offsets.elements().as_slice();
    let sizes_slice = sizes.elements().as_slice();

    (0..list_array_len)
        .map(|i| {
            // TODO(ngates): does validity render this invalid?
            let offset = offsets_slice[i].as_();
            let size = sizes_slice[i].as_();

            // BitIndexIterator yields indices of true bits only. If `.next()` returns
            // `Some(_)`, at least one element in this list's range matches.
            let mut set_bits =
                BitIndexIterator::new(matches.inner().as_slice(), matches.offset() + offset, size);
            set_bits.next().is_some()
        })
        .collect::<BitBuffer>()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::BitBuffer;
    use vortex_dtype::DType;
    use vortex_dtype::Field;
    use vortex_dtype::FieldPath;
    use vortex_dtype::FieldPathSet;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType::I32;
    use vortex_dtype::StructFields;
    use vortex_scalar::Scalar;
    use vortex_utils::aliases::hash_map::HashMap;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::list_contains;
    use crate::Array;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::expr::exprs::binary::and;
    use crate::expr::exprs::binary::gt;
    use crate::expr::exprs::binary::lt;
    use crate::expr::exprs::binary::or;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;
    use crate::expr::pruning::checked_pruning_expr;
    use crate::expr::stats::Stat;
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
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(item.scalar_at(0), Scalar::bool(true, Nullability::Nullable));
        assert_eq!(
            item.scalar_at(1),
            Scalar::bool(false, Nullability::Nullable)
        );
    }

    #[test]
    pub fn test_all() {
        let arr = test_array();

        let expr = list_contains(root(), lit(2));
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(item.scalar_at(0), Scalar::bool(true, Nullability::Nullable));
        assert_eq!(item.scalar_at(1), Scalar::bool(true, Nullability::Nullable));
    }

    #[test]
    pub fn test_none() {
        let arr = test_array();

        let expr = list_contains(root(), lit(4));
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(
            item.scalar_at(0),
            Scalar::bool(false, Nullability::Nullable)
        );
        assert_eq!(
            item.scalar_at(1),
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
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(item.scalar_at(0), Scalar::bool(true, Nullability::Nullable));
        assert_eq!(
            item.scalar_at(1),
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
        let item = expr.evaluate(&arr).unwrap();

        assert_eq!(item.scalar_at(0), Scalar::bool(true, Nullability::Nullable));
        assert!(!item.is_valid(1));
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
}

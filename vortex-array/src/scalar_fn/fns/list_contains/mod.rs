// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::BitOr;

pub use kernel::*;
use prost::Message;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::ScalarFnArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::proto::expr as pb;
use crate::scalar::ListScalar;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;

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
    /// `x NOT IN (1, NULL)` is never true. A null needle yields `null` either way.
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
    /// The nullability of a `list_contains` result: a null list or a null needle always yields
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
    /// Returns an error if the children have different lengths or `list` is not a list array.
    pub fn try_new(list: ArrayRef, needle: ArrayRef) -> VortexResult<ScalarFnArray> {
        Self::try_new_opts(list, needle, ListContainsOptions::default())
    }

    /// Creates a lazy list membership check for `needle` in `list` with explicit
    /// [`ListContainsOptions`].
    ///
    /// # Errors
    ///
    /// Returns an error if the children have different lengths or `list` is not a list array.
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

        if !matches!(list_dtype, DType::List(..)) {
            vortex_bail!(
                "First argument to ListContains must be a List, got {:?}",
                list_dtype
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

        if let Some(list_scalar) = list_array.as_constant()
            && let Some(value_scalar) = value_array.as_constant()
        {
            let result = compute_contains_scalar(&list_scalar, &value_scalar, options)?;
            return Ok(ConstantArray::new(result, args.row_count()).into_array());
        }

        compute_list_contains(&list_array, &value_array, options, ctx)
    }

    // An empty list can produce false even when the needle is null.
    fn is_strict(&self, _options: &Self::Options) -> bool {
        false
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

    // Handle null list or null needle
    if list.is_null() || needle.is_null() {
        return Ok(Scalar::null(DType::Bool(nullability)));
    }

    let list_scalar = list.as_list();
    let elements = list_scalar
        .elements()
        .ok_or_else(|| vortex_err!("Expected non-null list"))?;

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

    if value.all_invalid(ctx)? || array.all_invalid(ctx)? {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Bool(Nullability::Nullable)),
            array.len(),
        )
        .into_array());
    }

    if value.as_constant().is_some() {
        // A constant needle is answered by the list encoding's [`ListContainsListKernel`].
        // Reaching here means no encoding claimed the expression, so execute the list into the
        // canonical list encoding and run its kernel.
        let list = array.clone().execute::<ListViewArray>(ctx)?;
        return ListView::list_contains(list.as_view(), value, options, ctx)?.ok_or_else(|| {
            vortex_err!(
                "No list contains kernel for a constant needle of {}",
                value.dtype()
            )
        });
    }

    if let Some(list_scalar) = array.as_constant() {
        // A constant list is probed by the needle encoding's [`ListContainsElementKernel`]. An
        // encoded needle gets one pass into canonical form so that the kernels of the canonical
        // encodings apply; the fold below serves whatever encoding has no kernel.
        if !value.is_canonical() {
            let value = value.clone().execute_until::<AnyCanonical>(ctx)?;
            return Ok(ListContains::try_new_opts(array.clone(), value, *options)?.into_array());
        }
        let nullability = options.result_nullability(array.dtype(), value.dtype());
        return constant_list_scalar_contains(
            &list_scalar.as_list(),
            value,
            nullability,
            options,
            ctx,
        );
    }

    todo!("unsupported list contains with list and element as arrays")
}

/// There is a constant list scalar (haystack) being compared to an array of needles: the `IN`
/// shape, `list_contains(lit([...]), column)`.
///
/// This is the fallback for a needle encoding with no [`ListContainsElementKernel`]: one equality
/// comparison per element, OR-ed together. A null element is dropped — it never equals anything —
/// and, under SQL null semantics, makes every non-matching row `null` instead of `false`.
fn constant_list_scalar_contains(
    list_scalar: &ListScalar,
    values: &ArrayRef,
    nullability: Nullability,
    options: &ListContainsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let elements = list_scalar.elements().vortex_expect("non null");

    let len = values.len();
    if elements.is_empty() {
        return Ok(ConstantArray::new(Scalar::bool(false, nullability), len).into_array());
    }
    let unknown = options.sql_null_semantics && elements.iter().any(Scalar::is_null);

    // One equality per element, folded with Kleene OR. A null needle compares null to every
    // element and so stays null. A null element compares null to every needle: under SQL null
    // semantics that is the answer wanted for a non-match, otherwise the element is dropped so
    // that it contributes nothing rather than turning a `false` into `null`.
    let result = elements
        .iter()
        .filter(|element| unknown || !element.is_null())
        .map(|element| {
            Ok(Binary::try_new(
                ConstantArray::new(element.clone(), len).into_array(),
                values.clone(),
                Operator::Eq,
            )?
            .into_array())
        })
        .collect::<VortexResult<Vec<_>>>()?
        .into_iter()
        .try_reduce_balanced(|acc, res| acc.binary(res, Operator::Or))?;

    match result {
        Some(result) => Ok(result),
        // Every element was null and dropped: nothing matches, but a null needle is still null.
        None => Ok(BoolArray::new(
            BitBuffer::full_in(false, len, ctx.allocator().clone()),
            values.validity()?.union_nullability(nullability),
        )
        .into_array()),
    }
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
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DictArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
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
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::list_contains::ListContains;
    use crate::scalar_fn::fns::list_contains::ListContainsOptions;
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

    const SQL: ListContainsOptions = ListContainsOptions {
        sql_null_semantics: true,
    };

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

    #[test]
    fn constant_set_probes_an_encoded_needle() -> VortexResult<()> {
        // A dict-encoded needle has no kernel of its own, so it is executed into its canonical
        // encoding for the primitive kernel to probe the set, rather than folding one comparison
        // per element.
        let needles = DictArray::try_new(
            PrimitiveArray::from_iter([0u32, 1, 2, 1]).into_array(),
            PrimitiveArray::from_option_iter([Some(1), Some(2), None]).into_array(),
        )?
        .into_array();
        let set = i32_set(vec![Some(2), Some(4)]);
        assert_result(
            needles.apply(&list_contains(lit(set), root())),
            [Some(false), Some(true), None, Some(true)],
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
    fn constant_set_fallback_honours_both_semantics() -> VortexResult<()> {
        // Bool has no set kernel, so this exercises the per-element OR chain.
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

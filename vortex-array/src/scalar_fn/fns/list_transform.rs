// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::InterleaveArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::BoundLambda;
use crate::matcher::Matcher;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

/// Lazily transform every element of a list with a bound lambda.
///
/// The typed lambda is stored in scalar-function options. The first child is the list and all
/// remaining children are its captures in the outer row domain. The first lambda parameter is the
/// element; an optional second parameter is its zero-based index within the containing list.
#[derive(Clone)]
pub struct ListTransform;

impl ListTransform {
    /// Create a lazy list transformation.
    pub fn try_new(
        list: ArrayRef,
        lambda: BoundLambda,
        captures: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<ScalarFnArray> {
        let children = std::iter::once(list).chain(captures).collect();
        ScalarFnArray::try_new(ListTransform.bind(lambda), children)
    }
}

impl ScalarFnVTable for ListTransform {
    type Options = BoundLambda;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.transform");
        *ID
    }

    fn arity(&self, lambda: &Self::Options) -> Arity {
        Arity::Exact(1 + lambda.captures().len())
    }

    fn child_name(&self, _lambda: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            index => ChildName::from(Arc::from(format!("capture[{}]", index - 1))),
        }
    }

    fn return_dtype(&self, lambda: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let Some((list_dtype, capture_dtypes)) = arg_dtypes.split_first() else {
            vortex_bail!("list_transform() requires a list argument");
        };
        match list_dtype {
            DType::List(element_dtype, list_nullability) => {
                validate_lambda(lambda, element_dtype, capture_dtypes)?;
                Ok(DType::List(
                    Arc::new(lambda.body_dtype().clone()),
                    *list_nullability,
                ))
            }
            DType::FixedSizeList(element_dtype, list_size, list_nullability) => {
                validate_lambda(lambda, element_dtype, capture_dtypes)?;
                Ok(DType::FixedSizeList(
                    Arc::new(lambda.body_dtype().clone()),
                    *list_size,
                    *list_nullability,
                ))
            }
            _ => vortex_bail!("list_transform() requires List or FixedSizeList, got {list_dtype}"),
        }
    }

    fn execute(
        &self,
        lambda: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let captures = (1..args.num_inputs())
            .map(|index| args.get(index))
            .collect::<VortexResult<Vec<_>>>()?;
        let input = input.execute_until::<AnyList>(ctx)?;
        if let Some(list) = input.as_opt::<List>() {
            execute_list(lambda, list.into_owned(), captures, ctx)
        } else if let Some(list) = input.as_opt::<FixedSizeList>() {
            execute_fixed_size_list(lambda, list.into_owned(), captures, ctx)
        } else if let Some(list) = input.as_opt::<ListView>() {
            execute_list_view(lambda, list.into_owned(), captures, ctx)
        } else {
            unreachable!("AnyList matcher returned a non-list array")
        }
    }
}

fn execute_list(
    lambda: &BoundLambda,
    list: ListArray,
    captures: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list = list.reset_offsets(false, ctx)?;
    let offsets = list.offsets();
    let sizes = offsets
        .slice(1..offsets.len())?
        .binary(offsets.slice(0..list.len())?, Operator::Sub)?;
    let transformed = transform_elements(
        lambda,
        list.elements().clone(),
        sizes,
        list.list_validity(),
        captures,
        lambda.body_dtype(),
        ctx,
    )?;

    Ok(ListArray::try_new(transformed, list.offsets().clone(), list.list_validity())?.into_array())
}

fn execute_fixed_size_list(
    lambda: &BoundLambda,
    list: FixedSizeListArray,
    captures: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list_size = list.list_size();
    let sizes = ConstantArray::new(u64::from(list_size), list.len()).into_array();
    let transformed = transform_elements(
        lambda,
        list.elements().clone(),
        sizes,
        list.fixed_size_list_validity(),
        captures,
        lambda.body_dtype(),
        ctx,
    )?;

    Ok(FixedSizeListArray::try_new(
        transformed,
        list_size,
        list.fixed_size_list_validity(),
        list.len(),
    )?
    .into_array())
}

fn execute_list_view(
    lambda: &BoundLambda,
    list: ListViewArray,
    captures: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Logical flattening removes overlaps and gives null lists empty element ranges, so each
    // invocation has one unambiguous parent row.
    let list = list.rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)?;
    let transformed = transform_elements(
        lambda,
        list.elements().clone(),
        list.sizes().clone(),
        list.listview_validity(),
        captures,
        lambda.body_dtype(),
        ctx,
    )?;

    // SAFETY: the rebuilt view is zero-copyable to List and `transform_elements` preserves its
    // element domain, so its offsets, sizes, and validity remain valid and zero-copyable.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            transformed,
            list.offsets().clone(),
            list.sizes().clone(),
            list.listview_validity(),
        )
        .with_zero_copy_to_list(true)
    }
    .into_array())
}

fn validate_lambda(
    lambda: &BoundLambda,
    element_dtype: &DType,
    capture_dtypes: &[DType],
) -> VortexResult<()> {
    vortex_ensure!(
        matches!(lambda.param_dtypes().len(), 1 | 2),
        "list_transform() lambda must take one or two parameters, got {}",
        lambda.param_dtypes().len()
    );
    vortex_ensure!(
        &lambda.param_dtypes()[0] == element_dtype,
        "list_transform() element parameter expects dtype {}, got {element_dtype}",
        lambda.param_dtypes()[0]
    );
    if lambda.param_dtypes().len() == 2 {
        let index_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        vortex_ensure!(
            lambda.param_dtypes()[1] == index_dtype,
            "list_transform() index parameter expects dtype {index_dtype}, got {}",
            lambda.param_dtypes()[1]
        );
    }
    vortex_ensure!(
        lambda.captures().len() == capture_dtypes.len(),
        "list_transform() lambda requires {} captures, got {}",
        lambda.captures().len(),
        capture_dtypes.len()
    );
    for (index, (capture, dtype)) in lambda.captures().iter().zip(capture_dtypes).enumerate() {
        vortex_ensure!(
            capture.dtype() == dtype,
            "list_transform() capture {index} expects dtype {}, got {dtype}",
            capture.dtype()
        );
    }
    vortex_ensure!(
        lambda.body().is_root_bound_to(element_dtype),
        "list_transform() lambda root expects a different dtype than {element_dtype}"
    );
    Ok(())
}

fn transform_elements(
    lambda: &BoundLambda,
    elements: ArrayRef,
    sizes: ArrayRef,
    validity: Validity,
    captures: Vec<ArrayRef>,
    body_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let invocation_count = elements.len();
    let parent_indices = parent_indices(sizes.clone(), invocation_count)?;
    let invocation_mask = if matches!(validity, Validity::NonNullable | Validity::AllValid) {
        None
    } else {
        let invocation_validity = validity.take(&parent_indices)?;
        let invocation_mask = invocation_validity.execute_mask(invocation_count, ctx)?;
        if invocation_mask.true_count() == invocation_count {
            None
        } else {
            Some(invocation_mask)
        }
    };

    let elements = if let Some(mask) = &invocation_mask {
        elements.filter(mask.clone())?
    } else {
        elements
    };
    let capture_indices = match &invocation_mask {
        Some(mask) => parent_indices.filter(mask.clone())?,
        None => parent_indices,
    };
    let captures = captures
        .into_iter()
        .map(|capture| capture.take(capture_indices.clone()))
        .collect::<VortexResult<Vec<_>>>()?;

    let mut parameters = vec![elements];
    if lambda.param_dtypes().len() > 1 {
        let local_indices = local_indices(sizes, invocation_count)?;
        let local_indices = match &invocation_mask {
            Some(mask) => local_indices.filter(mask.clone())?,
            None => local_indices,
        };
        parameters.push(local_indices);
    }
    let transformed = lambda.apply(parameters[0].clone(), &parameters, &captures)?;

    match invocation_mask {
        Some(mask) => scatter_valid_invocations(transformed, &mask, body_dtype),
        None => Ok(transformed),
    }
}

fn parent_indices(sizes: ArrayRef, element_count: usize) -> VortexResult<ArrayRef> {
    let parent_count = sizes.len();
    let sizes_dtype = DType::Primitive(
        sizes.dtype().as_ptype().to_unsigned(),
        Nullability::NonNullable,
    );
    let sizes = sizes.cast(sizes_dtype)?;
    let starts =
        PrimitiveArray::from_iter((0..parent_count).map(|index| index as u64)).into_array();
    let multipliers = ConstantArray::new(0_u64, parent_count).into_array();
    Ok(PiecewiseSequenceArray::try_new(starts, sizes, multipliers, element_count)?.into_array())
}

fn local_indices(sizes: ArrayRef, element_count: usize) -> VortexResult<ArrayRef> {
    let parent_count = sizes.len();
    let sizes_dtype = DType::Primitive(
        sizes.dtype().as_ptype().to_unsigned(),
        Nullability::NonNullable,
    );
    let sizes = sizes.cast(sizes_dtype)?;
    let starts = ConstantArray::new(0_u64, parent_count).into_array();
    let multipliers = ConstantArray::new(1_u64, parent_count).into_array();
    Ok(PiecewiseSequenceArray::try_new(starts, sizes, multipliers, element_count)?.into_array())
}

fn scatter_valid_invocations(
    transformed: ArrayRef,
    mask: &Mask,
    body_dtype: &DType,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        transformed.len() == mask.true_count(),
        "list_transform() produced {} valid invocations, expected {}",
        transformed.len(),
        mask.true_count()
    );

    let mut array_indices = BufferMut::<u8>::with_capacity(mask.len());
    let mut row_indices = BufferMut::<u64>::with_capacity(mask.len());
    let mut valid_index = 0_u64;
    for valid in mask.iter() {
        if valid {
            array_indices.push(0);
            row_indices.push(valid_index);
            valid_index += 1;
        } else {
            array_indices.push(1);
            row_indices.push(0);
        }
    }

    let placeholder = if body_dtype.is_nullable() {
        Scalar::null(body_dtype.clone())
    } else {
        Scalar::zero_value(body_dtype)
    };
    let default = ConstantArray::new(placeholder, 1).into_array();
    Ok(InterleaveArray::try_new(
        vec![transformed, default],
        array_indices.into_array(),
        row_indices.into_array(),
    )?
    .into_array())
}

/// Matches a `List`, `ListView`, or `FixedSizeList` physical array.
struct AnyList;

impl Matcher for AnyList {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.as_opt::<List>().is_some()
            || array.as_opt::<ListView>().is_some()
            || array.as_opt::<FixedSizeList>().is_some())
        .then_some(())
    }
}

#[cfg(test)]
mod tests {
    //! Execution examples for `list_transform`.
    //!
    //! Every test documents the logical input, lambda, and expected output. The physical inputs cover
    //! ordinary lists, overlapping list views, nullable lists and elements, captured arrays, lazy
    //! capture expressions, and nested list trees.

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::assert_arrays_eq;
    use crate::expr::Expression;
    use crate::expr::Lambda;
    use crate::expr::Scope;
    use crate::expr::Variable;
    use crate::expr::lit;
    use crate::expr::var;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::fns::list_length::ListLength;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    fn binary(operator: Operator, lhs: Expression, rhs: Expression) -> Expression {
        crate::expr::binary(operator, lhs, rhs)
    }

    fn lit_i32(value: i32) -> Expression {
        lit(value)
    }

    fn list_element_dtype(dtype: &DType) -> VortexResult<&DType> {
        match dtype {
            DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..) => {
                Ok(element_dtype)
            }
            dtype => vortex_bail!("test list lambda requires a list dtype, got {dtype}"),
        }
    }

    fn bind_lambda_for_list_dtype(
        list_dtype: &DType,
        params: &[&str],
        body: Expression,
        captures: &[(&str, DType)],
    ) -> VortexResult<BoundLambda> {
        vortex_ensure!(
            matches!(params.len(), 1 | 2),
            "test list lambda must have one or two parameters"
        );
        let element_dtype = list_element_dtype(list_dtype)?;
        let parameter_dtypes = std::iter::once(element_dtype.clone())
            .chain(
                (params.len() == 2)
                    .then_some(DType::Primitive(PType::U64, Nullability::NonNullable)),
            )
            .collect::<Vec<_>>();
        let scope = Scope::new(element_dtype.clone());
        let scope = if captures.is_empty() {
            scope
        } else {
            scope.with_bindings(
                captures
                    .iter()
                    .map(|(name, dtype)| (Variable::new(name), dtype.clone())),
            )?
        };
        let scope = scope.with_bindings(
            params
                .iter()
                .zip(parameter_dtypes)
                .map(|(name, dtype)| (Variable::new(name), dtype)),
        )?;
        BoundLambda::bind(&Lambda::try_new(params.iter().copied(), body)?, &scope)
    }

    fn bind_list_lambda(
        list: &ArrayRef,
        params: &[&str],
        body: Expression,
        captures: &[(&str, ArrayRef)],
    ) -> VortexResult<BoundLambda> {
        let capture_dtypes = captures
            .iter()
            .map(|(name, array)| (*name, array.dtype().clone()))
            .collect::<Vec<_>>();
        bind_lambda_for_list_dtype(list.dtype(), params, body, &capture_dtypes)
    }

    fn list_transform<'a>(
        list: ArrayRef,
        params: &[&str],
        body: Expression,
        captures: impl IntoIterator<Item = (&'a str, ArrayRef)>,
    ) -> VortexResult<ScalarFnArray> {
        let captures = captures.into_iter().collect::<Vec<_>>();
        let lambda = bind_list_lambda(&list, params, body, &captures)?;
        ListTransform::try_new(list, lambda, captures.into_iter().map(|(_, array)| array))
    }

    fn list(elements: ArrayRef, offsets: ArrayRef, validity: Validity) -> VortexResult<ArrayRef> {
        ListArray::try_new(elements, offsets, validity).map(IntoArray::into_array)
    }

    fn fixed_size_list(
        elements: ArrayRef,
        list_size: u32,
        validity: Validity,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        FixedSizeListArray::try_new(elements, list_size, validity, len).map(IntoArray::into_array)
    }

    fn nested_list(
        elements: ArrayRef,
        inner_offsets: ArrayRef,
        outer_offsets: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        let inner = list(elements, inner_offsets, Validity::NonNullable)?;
        list(inner, outer_offsets, Validity::NonNullable)
    }

    fn assert_transform(transform: ScalarFnArray, expected: ArrayRef) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(transform, expected, &mut ctx);
        Ok(())
    }

    fn assert_list_transform(transform: ScalarFnArray, expected: ArrayRef) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let actual = transform.into_array().execute_until::<AnyList>(&mut ctx)?;
        assert!(actual.as_opt::<List>().is_some());
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    fn assert_fixed_size_list_transform(
        transform: ScalarFnArray,
        expected: ArrayRef,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let actual = transform.into_array().execute_until::<AnyList>(&mut ctx)?;
        assert!(actual.as_opt::<FixedSizeList>().is_some());
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    fn assert_list_view_transform(
        transform: ScalarFnArray,
        expected: ArrayRef,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let actual = transform.into_array().execute_until::<AnyList>(&mut ctx)?;
        assert!(actual.as_opt::<ListView>().is_some());
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// Input: `[[1, 2], [], [3]]`
    ///
    /// Lambda: `x -> x + 1`
    ///
    /// Output: `[[2, 3], [], [4]]`
    #[test]
    fn scalar_call() -> VortexResult<()> {
        let body = binary(Operator::Add, var("x"), lit_i32(1));
        let input = list(
            buffer![1_i32, 2, 3].into_array(),
            buffer![0_u32, 2, 2, 3].into_array(),
            Validity::NonNullable,
        )?;
        let lambda = bind_list_lambda(&input, &["x"], body, &[])?;
        let transform = ListTransform::try_new(input, lambda.clone(), [])?;
        assert_eq!(transform.child_count(), 1);
        assert!(transform.as_ref().is::<ScalarFn>());
        assert_eq!(transform.scalar_fn().as_::<ListTransform>(), &lambda);

        let expected = list(
            buffer![2_i32, 3, 4].into_array(),
            buffer![0_u32, 2, 2, 3].into_array(),
            Validity::NonNullable,
        )?;
        assert_list_transform(transform, expected)
    }

    /// Input: `[[10, 10], [], [10, 10, 10]]`
    ///
    /// Lambda: `(x, i) -> x + i`, where `i` is zero-based within each list.
    ///
    /// Output: `[[10, 11], [], [10, 11, 12]]`
    #[test]
    fn local_index_parameter_on_list() -> VortexResult<()> {
        let body = binary(Operator::Add, var("x"), var("i"));
        let input = list(
            buffer![10_u64, 10, 10, 10, 10].into_array(),
            buffer![0_u32, 2, 2, 5].into_array(),
            Validity::NonNullable,
        )?;
        let expected = list(
            buffer![10_u64, 11, 10, 11, 12].into_array(),
            buffer![0_u32, 2, 2, 5].into_array(),
            Validity::NonNullable,
        )?;

        assert_list_transform(list_transform(input, &["x", "i"], body, [])?, expected)
    }

    /// Input: `[[-2, 0, 3], [5]]`
    ///
    /// Lambda: `x -> (x * 2) + 3`
    ///
    /// Output: `[[-1, 3, 9], [13]]`
    #[test]
    fn nested_scalar_calls() -> VortexResult<()> {
        let doubled = binary(Operator::Mul, var("x"), lit_i32(2));
        let body = binary(Operator::Add, doubled, lit_i32(3));
        let input = list(
            buffer![-2_i32, 0, 3, 5].into_array(),
            buffer![0_u32, 3, 4].into_array(),
            Validity::NonNullable,
        )?;
        let expected = list(
            buffer![-1_i32, 3, 9, 13].into_array(),
            buffer![0_u32, 3, 4].into_array(),
            Validity::NonNullable,
        )?;

        assert_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Input: `[[1, 2], [], [3]]`
    ///
    /// Lambda: `_ -> 7`
    ///
    /// Output: `[[7, 7], [], [7]]`
    #[test]
    fn unused_parameter_and_literal_body() -> VortexResult<()> {
        let input = list(
            buffer![1_i32, 2, 3].into_array(),
            buffer![0_u32, 2, 2, 3].into_array(),
            Validity::NonNullable,
        )?;
        let expected = list(
            buffer![7_i32, 7, 7].into_array(),
            buffer![0_u32, 2, 2, 3].into_array(),
            Validity::NonNullable,
        )?;

        assert_transform(list_transform(input, &["_"], lit_i32(7), [])?, expected)
    }

    /// Physical elements: `[1, 2, 3]`
    ///
    /// Input views: `[[1, 2], [2, 3]]`
    ///
    /// Lambda: `x -> x * 10`
    ///
    /// Output: `[[10, 20], [20, 30]]`
    #[test]
    fn overlapping_list_view_without_captures() -> VortexResult<()> {
        let input = ListViewArray::new(
            buffer![1_i32, 2, 3].into_array(),
            buffer![0_u32, 1].into_array(),
            buffer![2_u32, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let body = binary(Operator::Mul, var("x"), lit_i32(10));
        let expected = list(
            buffer![10_i32, 20, 20, 30].into_array(),
            buffer![0_u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?;

        assert_list_view_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Physical elements: `[10, 20, 30]`
    ///
    /// Input views: `[[10, 20], [20, 30]]`
    ///
    /// Lambda: `(x, i) -> x + i`, where each view receives local indices `[0, 1]`.
    ///
    /// Output: `[[10, 21], [20, 31]]`
    #[test]
    fn local_index_parameter_on_overlapping_list_view() -> VortexResult<()> {
        let input = ListViewArray::new(
            buffer![10_u64, 20, 30].into_array(),
            buffer![0_u32, 1].into_array(),
            buffer![2_u32, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let body = binary(Operator::Add, var("x"), var("i"));
        let expected = list(
            buffer![10_u64, 21, 20, 31].into_array(),
            buffer![0_u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?;

        assert_list_view_transform(list_transform(input, &["x", "i"], body, [])?, expected)
    }

    /// Physical elements: `[1, 2, 3]`
    ///
    /// Input views: `[[1, 2], [2, 3]]`; capture: `[10, 20]`
    ///
    /// Lambda: `x -> x + capture`
    ///
    /// Output: `[[11, 12], [22, 23]]`
    #[test]
    fn captures_are_spread_by_parent_occurrence() -> VortexResult<()> {
        let body = binary(Operator::Add, var("x"), var("capture"));
        let input = ListViewArray::new(
            buffer![1_i32, 2, 3].into_array(),
            buffer![0_u32, 1].into_array(),
            buffer![2_u32, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let capture = buffer![10_i32, 20].into_array();
        let transform = list_transform(input, &["x"], body, [("capture", capture)])?;
        assert_eq!(transform.child_count(), 2);

        let expected = list(
            buffer![11_i32, 12, 22, 23].into_array(),
            buffer![0_u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?;
        assert_list_view_transform(transform, expected)
    }

    /// Input: `[[0, 1, 2], [], [3, 4]]`
    ///
    /// Captured expression: `list_length(input) = [3, 0, 2]`
    ///
    /// Lambda: `x -> x + list_length(input)`
    ///
    /// Output: `[[3, 4, 5], [], [5, 6]]`
    #[test]
    fn lazy_scalar_function_capture() -> VortexResult<()> {
        let input = list(
            buffer![0_u64, 1, 2, 3, 4].into_array(),
            buffer![0_u32, 3, 3, 5].into_array(),
            Validity::NonNullable,
        )?;
        let lengths = ScalarFnArray::try_new(
            ListLength.bind(EmptyOptions),
            std::slice::from_ref(&input).to_vec(),
        )?
        .into_array();
        let body = binary(Operator::Add, var("x"), var("lengths"));
        let expected = list(
            buffer![3_u64, 4, 5, 5, 6].into_array(),
            buffer![0_u32, 3, 3, 5].into_array(),
            Validity::NonNullable,
        )?;

        assert_list_transform(
            list_transform(input, &["x"], body, [("lengths", lengths)])?,
            expected,
        )
    }

    /// Input: `[[1], null-containing-[0], [4]]`
    ///
    /// Lambda: `x -> 8 / x`
    ///
    /// Output: `[[8], null, [2]]`
    ///
    /// The physical zero belongs only to the null list and must not be evaluated.
    #[test]
    fn null_lists_do_not_evaluate_hidden_elements() -> VortexResult<()> {
        let body = binary(Operator::Div, lit_i32(8), var("x"));
        let validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let input = list(
            buffer![1_i32, 0, 4].into_array(),
            buffer![0_u32, 1, 2, 3].into_array(),
            validity.clone(),
        )?;
        let expected = list(
            buffer![8_i32, 2].into_array(),
            buffer![0_u32, 1, 1, 2].into_array(),
            validity,
        )?;

        assert_list_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Input: fixed-size lists `[[1, 2], [3, 4]]`; capture: `[10, 20]`
    ///
    /// Lambda: `x -> x + capture`
    ///
    /// Output: fixed-size lists `[[11, 12], [23, 24]]`
    #[test]
    fn fixed_size_list_with_capture_preserves_encoding() -> VortexResult<()> {
        let input = fixed_size_list(
            buffer![1_i32, 2, 3, 4].into_array(),
            2,
            Validity::NonNullable,
            2,
        )?;
        let body = binary(Operator::Add, var("x"), var("capture"));
        let expected = fixed_size_list(
            buffer![11_i32, 12, 23, 24].into_array(),
            2,
            Validity::NonNullable,
            2,
        )?;

        assert_fixed_size_list_transform(
            list_transform(
                input,
                &["x"],
                body,
                [("capture", buffer![10_i32, 20].into_array())],
            )?,
            expected,
        )
    }

    /// Input: fixed-size lists `[[10, 10], [10, 10]]`
    ///
    /// Lambda: `(x, i) -> x + i`, where each row receives local indices `[0, 1]`.
    ///
    /// Output: fixed-size lists `[[10, 11], [10, 11]]`
    #[test]
    fn local_index_parameter_on_fixed_size_list() -> VortexResult<()> {
        let input = fixed_size_list(
            buffer![10_u64, 10, 10, 10].into_array(),
            2,
            Validity::NonNullable,
            2,
        )?;
        let body = binary(Operator::Add, var("x"), var("i"));
        let expected = fixed_size_list(
            buffer![10_u64, 11, 10, 11].into_array(),
            2,
            Validity::NonNullable,
            2,
        )?;

        assert_fixed_size_list_transform(list_transform(input, &["x", "i"], body, [])?, expected)
    }

    /// Input: fixed-size lists `[[1], null-containing-[0], [4]]`
    ///
    /// Lambda: `x -> 8 / x`
    ///
    /// Output: fixed-size lists `[[8], null, [2]]`
    ///
    /// The lambda is evaluated only for valid outer rows, while the fixed-size physical shape is
    /// retained.
    #[test]
    fn nullable_fixed_size_list_skips_hidden_elements() -> VortexResult<()> {
        let body = binary(Operator::Div, lit_i32(8), var("x"));
        let validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let input = fixed_size_list(buffer![1_i32, 0, 4].into_array(), 1, validity.clone(), 3)?;
        let expected = fixed_size_list(buffer![8_i32, 0, 2].into_array(), 1, validity, 3)?;

        assert_fixed_size_list_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Input: two null fixed-size lists with physical zeros.
    ///
    /// Lambda: `x -> 8 / x`
    ///
    /// Output: `[null, null]`
    ///
    /// No lambda invocation is executed when every outer row is null.
    #[test]
    fn all_null_fixed_size_list_has_no_invocations() -> VortexResult<()> {
        let body = binary(Operator::Div, lit_i32(8), var("x"));
        let input = fixed_size_list(
            buffer![0_i32, 0, 0, 0].into_array(),
            2,
            Validity::AllInvalid,
            2,
        )?;
        let expected = fixed_size_list(
            buffer![0_i32, 0, 0, 0].into_array(),
            2,
            Validity::AllInvalid,
            2,
        )?;

        assert_fixed_size_list_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Input: three fixed-size lists of width zero: `[[], [], []]`
    ///
    /// Lambda: `x -> x + 1`
    ///
    /// Output: `[[], [], []]`
    #[test]
    fn degenerate_fixed_size_list_preserves_encoding() -> VortexResult<()> {
        let input = fixed_size_list(
            PrimitiveArray::from_iter([0_i32; 0]).into_array(),
            0,
            Validity::NonNullable,
            3,
        )?;
        let body = binary(Operator::Add, var("x"), lit_i32(1));
        let expected = fixed_size_list(
            PrimitiveArray::from_iter([0_i32; 0]).into_array(),
            0,
            Validity::NonNullable,
            3,
        )?;

        assert_fixed_size_list_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Input: `[[1, null, 3], []]`
    ///
    /// Lambda: `x -> x + 1`
    ///
    /// Output: `[[2, null, 4], []]`
    #[test]
    fn nullable_elements_propagate_through_scalar_functions() -> VortexResult<()> {
        let input = list(
            PrimitiveArray::from_option_iter([Some(1_i32), None, Some(3)]).into_array(),
            buffer![0_u32, 3, 3].into_array(),
            Validity::NonNullable,
        )?;
        let body = binary(Operator::Add, var("x"), lit_i32(1));
        let expected = list(
            PrimitiveArray::from_option_iter([Some(2_i32), None, Some(4)]).into_array(),
            buffer![0_u32, 3, 3].into_array(),
            Validity::NonNullable,
        )?;

        assert_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Input: an empty `List<i32>` with zero outer rows.
    ///
    /// Lambda: `x -> x + 1`
    ///
    /// Output: an empty `List<i32>` with zero outer rows.
    #[test]
    fn empty_outer_array() -> VortexResult<()> {
        let input = list(
            PrimitiveArray::from_iter([0_i32; 0]).into_array(),
            buffer![0_u32].into_array(),
            Validity::NonNullable,
        )?;
        let body = binary(Operator::Add, var("x"), lit_i32(1));
        let expected = list(
            PrimitiveArray::from_iter([0_i32; 0]).into_array(),
            buffer![0_u32].into_array(),
            Validity::NonNullable,
        )?;

        assert_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Input: `[[[1, 2], [3]], [[]]]`
    ///
    /// Lambda: `x -> list_length(x)`
    ///
    /// Output: `[[2, 1], [0]]`
    #[test]
    fn scalar_function_over_nested_list_elements() -> VortexResult<()> {
        let input = nested_list(
            buffer![1_i32, 2, 3].into_array(),
            buffer![0_u32, 2, 3, 3].into_array(),
            buffer![0_u32, 2, 3].into_array(),
        )?;
        let body = crate::expr::list_length(var("x"));
        let expected = list(
            buffer![2_u64, 1, 0].into_array(),
            buffer![0_u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?;

        assert_transform(list_transform(input, &["x"], body, [])?, expected)
    }

    /// Input: `[[[1, 2], [3]], [[], [4, 5, 6]]]`
    ///
    /// Lambda: `x -> list_transform(x, y -> y + 1)`
    ///
    /// Output: `[[[2, 3], [4]], [[], [5, 6, 7]]]`
    #[test]
    fn nested_list_transform() -> VortexResult<()> {
        let input = nested_list(
            buffer![1_i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0_u32, 2, 3, 3, 6].into_array(),
            buffer![0_u32, 2, 4].into_array(),
        )?;
        let inner_list_dtype = list_element_dtype(input.dtype())?;
        let inner_body = binary(Operator::Add, var("y"), lit_i32(1));
        let inner_lambda = bind_lambda_for_list_dtype(inner_list_dtype, &["y"], inner_body, &[])?;
        let outer_body = Expression::try_new(ListTransform.bind(inner_lambda), [var("x")])?;
        let expected = nested_list(
            buffer![2_i32, 3, 4, 5, 6, 7].into_array(),
            buffer![0_u32, 2, 3, 3, 6].into_array(),
            buffer![0_u32, 2, 4].into_array(),
        )?;

        assert_transform(list_transform(input, &["x"], outer_body, [])?, expected)
    }

    /// Input: `[[[1, 2], [3]], [[], [4, 5, 6]]]`
    ///
    /// Lambda: `x -> list_transform(x, y -> y + list_length(x))`
    ///
    /// Output: `[[[3, 4], [4]], [[], [7, 8, 9]]]`
    #[test]
    fn nested_lambda_captures_outer_parameter_expression() -> VortexResult<()> {
        let input = nested_list(
            buffer![1_u64, 2, 3, 4, 5, 6].into_array(),
            buffer![0_u32, 2, 3, 3, 6].into_array(),
            buffer![0_u32, 2, 4].into_array(),
        )?;
        let inner_list_dtype = list_element_dtype(input.dtype())?;
        let length_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let inner_body = binary(Operator::Add, var("y"), var("length"));
        let inner_lambda = bind_lambda_for_list_dtype(
            inner_list_dtype,
            &["y"],
            inner_body,
            &[("length", length_dtype)],
        )?;
        let outer_body = Expression::try_new(
            ListTransform.bind(inner_lambda),
            [var("x"), crate::expr::list_length(var("x"))],
        )?;
        let expected = nested_list(
            buffer![3_u64, 4, 4, 7, 8, 9].into_array(),
            buffer![0_u32, 2, 3, 3, 6].into_array(),
            buffer![0_u32, 2, 4].into_array(),
        )?;

        assert_transform(list_transform(input, &["x"], outer_body, [])?, expected)
    }

    /// Input: `[[[1, 2], [3]], [[], [4, 5, 6]]]`; outer capture: `[10, 20]`
    ///
    /// Lambda: `x -> list_transform(x, y -> y + outer_capture)`
    ///
    /// Output: `[[[11, 12], [13]], [[], [24, 25, 26]]]`
    #[test]
    fn capture_is_spread_across_two_nested_list_domains() -> VortexResult<()> {
        let input = nested_list(
            buffer![1_i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0_u32, 2, 3, 3, 6].into_array(),
            buffer![0_u32, 2, 4].into_array(),
        )?;
        let inner_list_dtype = list_element_dtype(input.dtype())?;
        let capture_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let inner_body = binary(Operator::Add, var("y"), var("outer_capture"));
        let inner_lambda = bind_lambda_for_list_dtype(
            inner_list_dtype,
            &["y"],
            inner_body,
            &[("outer_capture", capture_dtype)],
        )?;
        let outer_body = Expression::try_new(
            ListTransform.bind(inner_lambda),
            [var("x"), var("outer_capture")],
        )?;
        let expected = nested_list(
            buffer![11_i32, 12, 13, 24, 25, 26].into_array(),
            buffer![0_u32, 2, 3, 3, 6].into_array(),
            buffer![0_u32, 2, 4].into_array(),
        )?;

        assert_transform(
            list_transform(
                input,
                &["x"],
                outer_body,
                [("outer_capture", buffer![10_i32, 20].into_array())],
            )?,
            expected,
        )
    }
}

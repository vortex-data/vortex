// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
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
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::is_not_null_node;
use crate::validity::Validity;

/// The options of [`ListTransform`]: its lambda, once bound.
///
/// The bound node's children are the list followed by the lambda's captures.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ListLambdaOptions(Option<Arc<BoundLambda>>);

impl ListLambdaOptions {
    /// The bound lambda, or `None` in an unbound node.
    pub fn lambda(&self) -> Option<&BoundLambda> {
        self.0.as_deref()
    }

    fn bound(&self) -> VortexResult<&BoundLambda> {
        match self.lambda() {
            Some(lambda) => Ok(lambda),
            None => vortex_bail!("list_transform() is unbound; bind it first"),
        }
    }
}

impl From<BoundLambda> for ListLambdaOptions {
    fn from(lambda: BoundLambda) -> Self {
        Self(Some(Arc::new(lambda)))
    }
}

impl Display for ListLambdaOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.lambda() {
            None => Ok(()),
            Some(lambda) => Display::fmt(lambda, f),
        }
    }
}

/// Transform every element of a list with a lambda, preserving the list structure.
///
/// The lambda's first parameter is the element; an optional second parameter is its zero-based
/// `u64` index within its list. A null list stays null and the body never observably runs on its
/// elements.
#[derive(Clone)]
pub struct ListTransform;

fn element_dtype(list: &DType) -> VortexResult<&DType> {
    match list {
        DType::List(element, _) | DType::FixedSizeList(element, ..) => Ok(element),
        other => vortex_bail!("list_transform() requires List or FixedSizeList, got {other}"),
    }
}

fn index_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

impl ScalarFnVTable for ListTransform {
    type Options = ListLambdaOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.transform");
        *ID
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        Arity::Exact(
            1 + options
                .lambda()
                .map_or(0, |lambda| lambda.capture_dtypes().len()),
        )
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            index => ChildName::from(Arc::from(format!("capture[{}]", index - 1))),
        }
    }

    fn lambda_params(
        &self,
        _options: &Self::Options,
        lambda_index: usize,
        param_count: usize,
        arg_dtypes: &[DType],
    ) -> VortexResult<Vec<DType>> {
        vortex_ensure!(lambda_index == 0, "list_transform() takes one lambda");
        let Some(list) = arg_dtypes.first() else {
            vortex_bail!("list_transform() requires a list argument");
        };
        let element = element_dtype(list)?.clone();
        match param_count {
            1 => Ok(vec![element]),
            2 => Ok(vec![element, index_dtype()]),
            n => vortex_bail!("list_transform() lambda must take one or two parameters, got {n}"),
        }
    }

    fn with_lambdas(
        &self,
        _options: &Self::Options,
        mut lambdas: Vec<BoundLambda>,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(lambdas.len() == 1, "list_transform() takes one lambda");
        Ok(lambdas.pop().vortex_expect("checked length").into())
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lambda = options.bound()?;
        let Some((list, captures)) = arg_dtypes.split_first() else {
            vortex_bail!("list_transform() requires a list argument");
        };
        let params = self.lambda_params(
            options,
            0,
            lambda.params().len(),
            std::slice::from_ref(list),
        )?;
        lambda.check_inputs(params.iter().chain(captures))?;
        let body = Arc::new(lambda.body_dtype().clone());
        Ok(match list {
            DType::List(_, nullability) => DType::List(body, *nullability),
            DType::FixedSizeList(_, size, nullability) => {
                DType::FixedSizeList(body, *size, *nullability)
            }
            _ => unreachable!("lambda_params checked the list dtype"),
        })
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lambda = options.bound()?;
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

    fn validity<T: ReduceNode>(
        &self,
        _options: &Self::Options,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        Ok(ReduceNodeValidity::Reduced(is_not_null_node(
            &node.child(0),
        )?))
    }

    fn is_infallible(&self, options: &Self::Options) -> bool {
        options.lambda().is_some_and(BoundLambda::is_infallible)
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

    // An infallible body can run over element slots hidden under null lists: the results stay
    // hidden behind the list validity. Only a fallible body needs those slots filtered out.
    let may_hide = !matches!(validity, Validity::NonNullable | Validity::AllValid);
    let needs_parents = !captures.is_empty() || (may_hide && !lambda.is_infallible());
    let parent_indices = needs_parents
        .then(|| parent_indices(sizes.clone(), invocation_count))
        .transpose()?;

    let invocation_mask = match &parent_indices {
        Some(parent_indices) if may_hide && !lambda.is_infallible() => {
            let invocation_validity = validity.take(parent_indices)?;
            let invocation_mask = invocation_validity.execute_mask(invocation_count, ctx)?;
            (invocation_mask.true_count() != invocation_count).then_some(invocation_mask)
        }
        _ => None,
    };

    let filter = |array: ArrayRef| -> VortexResult<ArrayRef> {
        match &invocation_mask {
            Some(mask) => array.filter(mask.clone()),
            None => Ok(array),
        }
    };

    let mut env = vec![filter(elements)?];
    if lambda.params().len() > 1 {
        env.push(filter(local_indices(sizes, invocation_count)?)?);
    }
    if let Some(parent_indices) = parent_indices {
        let capture_indices = filter(parent_indices)?;
        for capture in captures {
            env.push(capture.take(capture_indices.clone())?);
        }
    }
    let transformed = lambda.evaluate(&env)?;

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
mod tests;

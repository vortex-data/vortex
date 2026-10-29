// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::iter;
use std::sync::Arc;

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrays::listview::list_from_list_view;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::expr::BoundLambda;
use crate::expr::Lambda;
use crate::expr::display::ExprDisplay;
use crate::match_each_integer_ptype;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Apply a lambda to every element of each list, keeping each list's length and validity.
///
/// The first child is the list, of `List` dtype in any encoding, and the result is a `List`. The
/// lambda's first parameter is the element. Each other child is a value the lambda reads from
/// outside, one per row, and is the lambda's next parameter: every element of a row's list sees
/// that row's value. A lambda that can fail runs only on the elements of valid lists.
#[derive(Clone)]
pub struct ListTransform;

/// The lambda of a [`ListTransform`], as written or bound to the element's dtype.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ListTransformOptions {
    /// The lambda as written, in an [`Expression`](crate::expr::Expression).
    Lambda(Lambda),
    /// The lambda bound to the element's dtype, in a
    /// [`BoundExpression`](crate::expr::BoundExpression) or an array.
    Bound(BoundLambda),
}

impl ListTransformOptions {
    /// The lambda's parameter names: the element, then one per value it reads from outside.
    fn params(&self) -> &[FieldName] {
        match self {
            Self::Lambda(lambda) => lambda.params(),
            Self::Bound(lambda) => lambda.params().names().as_ref(),
        }
    }
}

impl Display for ListTransformOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lambda(lambda) => lambda.fmt(f),
            Self::Bound(lambda) => lambda.fmt(f),
        }
    }
}

impl ScalarFnVTable for ListTransform {
    type Options = ListTransformOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.transform");
        *ID
    }

    fn arity(&self, _options: &ListTransformOptions) -> Arity {
        Arity::Variadic { min: 1, max: None }
    }

    fn child_name(&self, options: &ListTransformOptions, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            _ => ChildName::from(Arc::clone(options.params()[child_idx].inner())),
        }
    }

    fn fmt_sql(
        &self,
        options: &ListTransformOptions,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}(", self.id())?;
        for index in 0..expr.display_children_count() {
            write!(f, "{}, ", expr.display_child(index))?;
        }
        write!(f, "{options})")
    }

    fn return_dtype(&self, options: &ListTransformOptions, args: &[DType]) -> VortexResult<DType> {
        let params = param_dtypes(options.params(), args)?;
        let element = match options {
            ListTransformOptions::Lambda(lambda) => lambda.return_dtype(&params)?,
            ListTransformOptions::Bound(lambda) if lambda.is_bound_to(&params) => {
                lambda.dtype().clone()
            }
            ListTransformOptions::Bound(lambda) => lambda.rebind(&params)?.dtype().clone(),
        };
        Ok(DType::List(Arc::new(element), args[0].nullability()))
    }

    fn bind_options(
        &self,
        options: &ListTransformOptions,
        args: &[DType],
    ) -> VortexResult<Option<ListTransformOptions>> {
        let params = param_dtypes(options.params(), args)?;
        Ok(match options {
            ListTransformOptions::Lambda(lambda) => {
                Some(ListTransformOptions::Bound(lambda.bind(&params)?))
            }
            ListTransformOptions::Bound(lambda) if lambda.is_bound_to(&params) => None,
            // A rewrite changed the dtype of the list or of a value the lambda reads, such as by
            // making it non-nullable.
            ListTransformOptions::Bound(lambda) => {
                Some(ListTransformOptions::Bound(lambda.rebind(&params)?))
            }
        })
    }

    fn execute(
        &self,
        options: &ListTransformOptions,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let ListTransformOptions::Bound(lambda) = options else {
            vortex_bail!("list_transform()'s lambda is not bound");
        };
        let captures = (1..args.num_inputs())
            .map(|index| args.get(index))
            .collect::<VortexResult<Vec<_>>>()?;
        match args.get(0)?.try_downcast::<List>() {
            Ok(list) => transform_list(list, lambda, &captures, ctx),
            // Overlapping views share elements between rows, but each element must see the value
            // its own row reads from outside, so the views are copied apart into a list first.
            Err(list) if !captures.is_empty() => {
                let list = list_from_list_view(list.execute::<ListViewArray>(ctx)?, ctx)?;
                transform_list(list, lambda, &captures, ctx)
            }
            Err(list) => transform_list_view(list.execute::<ListViewArray>(ctx)?, lambda, ctx),
        }
    }

    /// Strict only when the lambda reads nothing from outside: a null value read from outside
    /// doesn't make a valid list null.
    fn is_strict(&self, options: &ListTransformOptions) -> bool {
        options.params().len() == 1
    }

    fn is_infallible(&self, options: &ListTransformOptions) -> bool {
        match options {
            ListTransformOptions::Lambda(lambda) => lambda.is_infallible(),
            ListTransformOptions::Bound(lambda) => lambda.is_infallible(),
        }
    }
}

/// The dtypes of the lambda's parameters for these children: the list's element, then the values
/// it reads from outside.
fn param_dtypes(params: &[FieldName], args: &[DType]) -> VortexResult<Vec<DType>> {
    let DType::List(element, _) = &args[0] else {
        vortex_bail!("list_transform() takes a List, got {}", args[0]);
    };
    // The parameters for values read from outside come last, so the rest are the written ones.
    let written = &params[..params.len().saturating_sub(args.len() - 1)];
    vortex_ensure!(
        params.len() == args.len(),
        "list_transform() takes a lambda of one parameter, the element, not ({})",
        written.iter().format(", ")
    );
    Ok(iter::once(element.as_ref().clone())
        .chain(args[1..].iter().cloned())
        .collect())
}

/// For each element of lists whose offsets start at 0 and end at the last element, the index of
/// the list that holds it: `[0, 0, 2]` for the offsets `[0, 2, 2, 3]`.
fn list_of_each_element(
    offsets: &ArrayRef,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let offsets = offsets.clone().execute::<PrimitiveArray>(ctx)?;
    let mut lists = BufferMut::<u64>::with_capacity(len);
    match_each_integer_ptype!(offsets.ptype(), |O| {
        for (list, bounds) in offsets.as_slice::<O>().windows(2).enumerate() {
            lists.push_n(list as u64, (bounds[1] - bounds[0]).as_());
        }
    });
    Ok(lists.freeze().into_array())
}

/// The lambda over a list's elements, inside the list's offsets and validity.
///
/// A lambda that can fail runs only on the elements valid lists reference. One that can't runs on
/// every element from the first list to the last: results under null lists are never read.
/// `captures` are the values the lambda reads from outside, one per row.
fn transform_list(
    list: ListArray,
    lambda: &BoundLambda,
    captures: &[ArrayRef],
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Either way, the offsets start at 0 and end at the last element.
    let list = if lambda.is_infallible() {
        trim_ends(list, ctx)?
    } else {
        list.compact(ctx)?
    };
    let elements = list.elements();
    let mut inputs = vec![elements.clone()];
    if !captures.is_empty() {
        // Each value read from outside is repeated for every element of its row's list.
        let lists = list_of_each_element(list.offsets(), elements.len(), ctx)?;
        for capture in captures {
            inputs.push(capture.take(lists.clone())?);
        }
    }
    let values = lambda.evaluate(inputs, elements.len())?;
    vortex_ensure_eq!(values.len(), elements.len());
    // SAFETY: the offsets and validity are the list's, over one value per element.
    Ok(
        unsafe { ListArray::new_unchecked(values, list.offsets().clone(), list.list_validity()) }
            .into_array(),
    )
}

/// The lambda over a list view's elements, inside the view's offsets, sizes and validity.
///
/// A lambda that can fail runs only on the elements valid lists reference.
fn transform_list_view(
    view: ListViewArray,
    lambda: &BoundLambda,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let view = if lambda.is_infallible() {
        view
    } else {
        view.rebuild(ListViewRebuildMode::MakeExactWithEmptyNulls, ctx)?
    };
    let values = lambda.evaluate(vec![view.elements().clone()], view.elements().len())?;
    vortex_ensure_eq!(values.len(), view.elements().len());
    // SAFETY: the offsets, sizes and validity are the view's, over one value per element, so the
    // views keep their bounds and their layout.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            values,
            view.offsets().clone(),
            view.sizes().clone(),
            view.listview_validity(),
        )
        .with_zero_copy_to_list(view.is_zero_copy_to_list())
    }
    .into_array())
}

/// The list without what comes before its first list and after its last.
fn trim_ends(list: ListArray, ctx: &mut ExecutionCtx) -> VortexResult<ListArray> {
    if list.offset_at(0, ctx)? == 0 && list.offset_at(list.len(), ctx)? == list.elements().len() {
        return Ok(list);
    }
    list.reset_offsets(false, ctx)
}

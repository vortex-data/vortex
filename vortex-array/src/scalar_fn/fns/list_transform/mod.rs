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
use crate::arrays::StructArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrays::listview::list_from_list_view;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::StructFields;
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
/// The lambda takes one parameter, the element. The first child is the list, of `List` dtype in any
/// encoding, and the result is a `List`. When the lambda reads values from outside it, the second
/// child is its [environment](Lambda::environment), one struct per row: every element of a row's
/// list sees that row's values. A lambda that can fail runs only on the elements of valid lists.
#[derive(Clone)]
pub struct ListTransform;

/// The lambda of a [`ListTransform`], as written or bound to the dtypes of its inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ListTransformOptions {
    /// The lambda as written, in an [`Expression`](crate::expr::Expression).
    Lambda(Lambda),
    /// The lambda bound to the dtypes of its inputs, in a
    /// [`BoundExpression`](crate::expr::BoundExpression) or an array.
    Bound(BoundLambda),
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
        Arity::Variadic {
            min: 1,
            max: Some(2),
        }
    }

    fn child_name(&self, _options: &ListTransformOptions, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            1 => ChildName::from("environment"),
            _ => unreachable!("Invalid child index {child_idx} for list_transform()"),
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
        let (element, environment) = input_dtypes(args)?;
        let element = match options {
            ListTransformOptions::Lambda(lambda) => {
                lambda.return_dtype(&element_param(lambda, element)?, &environment)?
            }
            ListTransformOptions::Bound(lambda) => {
                check_bound_to(lambda, element, &environment)?;
                lambda.dtype().clone()
            }
        };
        Ok(DType::List(Arc::new(element), args[0].nullability()))
    }

    fn bind_options(
        &self,
        options: &ListTransformOptions,
        args: &[DType],
    ) -> VortexResult<Option<ListTransformOptions>> {
        let (element, environment) = input_dtypes(args)?;
        match options {
            ListTransformOptions::Lambda(lambda) => Ok(Some(ListTransformOptions::Bound(
                lambda.bind(&element_param(lambda, element)?, &environment)?,
            ))),
            ListTransformOptions::Bound(lambda) => {
                check_bound_to(lambda, element, &environment)?;
                Ok(None)
            }
        }
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
        let environment = (args.num_inputs() > 1).then(|| args.get(1)).transpose()?;
        match args.get(0)?.try_downcast::<List>() {
            Ok(list) => transform_list(list, lambda, environment.as_ref(), ctx),
            // Overlapping views share elements between rows, but each element must see its own
            // row's environment, so the views are copied apart into a list first.
            Err(list) if environment.is_some() => {
                let list = list_from_list_view(list.execute::<ListViewArray>(ctx)?, ctx)?;
                transform_list(list, lambda, environment.as_ref(), ctx)
            }
            Err(list) => transform_list_view(list.execute::<ListViewArray>(ctx)?, lambda, ctx),
        }
    }

    fn is_strict(&self, _options: &ListTransformOptions) -> bool {
        true
    }

    fn is_infallible(&self, options: &ListTransformOptions) -> bool {
        match options {
            ListTransformOptions::Lambda(lambda) => lambda.is_infallible(),
            ListTransformOptions::Bound(lambda) => lambda.is_infallible(),
        }
    }
}

/// Check that a bound lambda's inputs have the dtypes of the element and of the environment's
/// fields.
fn check_bound_to(
    lambda: &BoundLambda,
    element: DType,
    environment: &StructFields,
) -> VortexResult<()> {
    let inputs = iter::once(element)
        .chain(environment.fields())
        .collect_vec();
    vortex_ensure!(
        lambda.is_bound_to(&inputs),
        "list_transform()'s lambda {lambda} is bound to {}, not {}",
        lambda.params().fields().format(", "),
        inputs.iter().format(", ")
    );
    Ok(())
}

/// The dtypes of the lambda's inputs for these children: the list's element, and the fields of the
/// environment, empty without one.
fn input_dtypes(args: &[DType]) -> VortexResult<(DType, StructFields)> {
    let DType::List(element, _) = &args[0] else {
        vortex_bail!("list_transform() takes a List, got {}", args[0]);
    };
    let environment = match args.get(1) {
        None => StructFields::empty(),
        Some(DType::Struct(fields, _)) => fields.clone(),
        Some(other) => vortex_bail!("list_transform()'s environment must be a struct, got {other}"),
    };
    Ok((element.as_ref().clone(), environment))
}

/// The dtype of the lambda's one parameter, the element.
fn element_param(lambda: &Lambda, element: DType) -> VortexResult<[DType; 1]> {
    vortex_ensure!(
        lambda.params().len() == 1,
        "list_transform() takes a lambda of one parameter, the element, not ({})",
        lambda.params().iter().format(", ")
    );
    Ok([element])
}

/// For each element of lists whose offsets start at 0 and end at the last element, the index of
/// the list that holds it: `[0, 0, 2]` for the offsets `[0, 2, 2, 3]`.
fn list_of_each_element(
    offsets: &ArrayRef,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let offsets = offsets.clone().execute::<PrimitiveArray>(ctx)?;
    let mut lists = BufferMut::<u64>::with_capacity_in(len, ctx.allocator().clone());
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
/// `environment` is the lambda's environment, one struct per row, when it reads from outside.
fn transform_list(
    list: ListArray,
    lambda: &BoundLambda,
    environment: Option<&ArrayRef>,
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
    if let Some(environment) = environment {
        // Each row's environment is repeated for every element of its list.
        let lists = list_of_each_element(list.offsets(), elements.len(), ctx)?;
        let environment = environment.clone().execute::<StructArray>(ctx)?;
        for field in environment.iter_unmasked_fields() {
            inputs.push(field.take(lists.clone())?);
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

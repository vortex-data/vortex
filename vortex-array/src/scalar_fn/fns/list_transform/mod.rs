// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
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
use crate::validity::Validity;

/// Apply a lambda to every element of each list, keeping each list's length and validity.
///
/// The lambda takes one parameter, the element. The only child is the list, of `List` dtype in any
/// encoding, and the result is a `List`. The lambda runs only on the elements of valid lists.
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
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &ListTransformOptions, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            _ => unreachable!("Invalid child index {child_idx} for list_transform()"),
        }
    }

    fn fmt_sql(
        &self,
        options: &ListTransformOptions,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}({}, {options})", self.id(), expr.display_child(0))
    }

    fn return_dtype(&self, options: &ListTransformOptions, args: &[DType]) -> VortexResult<DType> {
        let element = match options {
            ListTransformOptions::Lambda(lambda) => {
                lambda.return_dtype(&element_dtype(&args[0], lambda.params())?)?
            }
            ListTransformOptions::Bound(lambda) => {
                check_bound_to(lambda, &args[0])?;
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
        match options {
            ListTransformOptions::Lambda(lambda) => {
                let element = element_dtype(&args[0], lambda.params())?;
                Ok(Some(ListTransformOptions::Bound(lambda.bind(&element)?)))
            }
            ListTransformOptions::Bound(lambda) => {
                check_bound_to(lambda, &args[0])?;
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
        // Every element belongs to exactly one valid list, in order: nothing comes before the
        // first list or after the last, and a null list has no elements.
        let list = args.get(0)?.execute::<ListViewArray>(ctx)?;
        let list = list_from_list_view(empty_null_lists(list, ctx)?, ctx)?;
        let elements = list.elements();
        let values = lambda.evaluate(vec![elements.clone()], elements.len())?;
        vortex_ensure_eq!(values.len(), elements.len());
        // SAFETY: the offsets are the list's, over one value per element.
        Ok(unsafe {
            ListArray::new_unchecked(values, list.offsets().clone(), list.list_validity())
        }
        .into_array())
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

/// Check that a bound lambda's parameter has the element dtype of a list of this dtype.
fn check_bound_to(lambda: &BoundLambda, list: &DType) -> VortexResult<()> {
    let element = element_dtype(list, lambda.params().names().as_ref())?;
    vortex_ensure!(
        lambda.is_bound_to(&element),
        "list_transform()'s lambda {lambda} is bound to element dtype {}, not {}",
        lambda.params().fields().format(", "),
        element[0]
    );
    Ok(())
}

/// The dtype of the lambda's one parameter, the element, for a list of this dtype.
fn element_dtype(list: &DType, params: &[FieldName]) -> VortexResult<[DType; 1]> {
    let DType::List(element, _) = list else {
        vortex_bail!("list_transform() takes a List, got {list}");
    };
    vortex_ensure!(
        params.len() == 1,
        "list_transform() takes a lambda of one parameter, the element, not ({})",
        params.iter().format(", ")
    );
    Ok([element.as_ref().clone()])
}

/// The lists with every null list emptied, so that nothing under a null list reaches the lambda.
fn empty_null_lists(list: ListViewArray, ctx: &mut ExecutionCtx) -> VortexResult<ListViewArray> {
    let validity = list.listview_validity();
    if matches!(validity, Validity::NonNullable | Validity::AllValid) {
        return Ok(list);
    }
    let valid = validity.execute_mask(list.len(), ctx)?;
    let sizes = list.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = match_each_integer_ptype!(sizes.ptype(), |T| {
        let sizes = sizes.as_slice::<T>();
        let null = |index: usize| !valid.value(index);
        // Writers usually leave null lists empty already.
        if !(0..sizes.len()).any(|index| null(index) && sizes[index] != T::default()) {
            return Ok(list);
        }
        PrimitiveArray::from_iter((0..sizes.len()).map(|index| {
            if null(index) {
                T::default()
            } else {
                sizes[index]
            }
        }))
    });
    // SAFETY: only sizes changed, and only to 0, so every view stays within the elements. The new
    // array is not marked zero-copy to a list: an emptied list leaves its elements behind as a gap,
    // which the rebuild must drop.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            list.elements().clone(),
            list.offsets().clone(),
            sizes.into_array(),
            validity,
        )
    })
}

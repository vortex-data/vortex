// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
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
use crate::validity::Validity;

/// Apply a lambda to every element of each list, keeping each list's length and validity.
///
/// The lambda takes one parameter, the element. The only child is the list, of `List` dtype in any
/// encoding, and the result is a `List` of the same shape. A lambda that can fail runs only on the
/// elements of valid lists.
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
        let list = if lambda.is_infallible() {
            Lists::as_they_are(args.get(0)?, ctx)?
        } else {
            Lists::visible_elements_only(args.get(0)?, ctx)?
        };
        list.with_elements(|elements| lambda.evaluate(vec![elements.clone()], elements.len()))
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

/// The lists the lambda runs over: their elements buffer, and the offsets (and sizes) that put the
/// results back into lists of the same shape.
enum Lists {
    List(ListArray),
    View(ListViewArray),
}

impl Lists {
    /// The lists as they are. For a lambda that can't fail: it runs over the whole elements buffer,
    /// and results for elements no valid list shows are never read.
    fn as_they_are(list: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        Ok(match list.try_downcast::<List>() {
            Ok(list) => Self::List(list),
            Err(list) => Self::View(list.execute::<ListViewArray>(ctx)?),
        })
    }

    /// The lists with nothing in their elements buffer but the elements of valid lists. For a lambda
    /// that can fail, which must not fail on a value nobody sees.
    ///
    /// The lists keep their shape, trimmed of what comes before the first list and after the last,
    /// unless a null list holds elements or a list view's lists aren't laid out back to back. Only
    /// then are the elements copied.
    fn visible_elements_only(list: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let list = match list.try_downcast::<List>() {
            Ok(list) => match trim_list(&list, ctx)? {
                Some(trimmed) => return Ok(Self::List(trimmed)),
                None => list.into_array(),
            },
            Err(list) => list,
        };
        let view = list.execute::<ListViewArray>(ctx)?;
        match empty_null_lists(&view, ctx)? {
            None if view.is_zero_copy_to_list() => Ok(Self::View(
                view.rebuild(ListViewRebuildMode::TrimElements, ctx)?,
            )),
            None => Ok(Self::List(list_from_list_view(view, ctx)?)),
            Some(emptied) => Ok(Self::List(list_from_list_view(emptied, ctx)?)),
        }
    }

    /// The same lists, with `f` of their elements buffer as their elements.
    fn with_elements(
        self,
        f: impl FnOnce(&ArrayRef) -> VortexResult<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        match self {
            Self::List(list) => {
                let values = f(list.elements())?;
                vortex_ensure_eq!(values.len(), list.elements().len());
                // SAFETY: the offsets and validity are the list's, over one value per element.
                Ok(unsafe {
                    ListArray::new_unchecked(values, list.offsets().clone(), list.list_validity())
                }
                .into_array())
            }
            Self::View(view) => {
                let values = f(view.elements())?;
                vortex_ensure_eq!(values.len(), view.elements().len());
                // SAFETY: the offsets, sizes and validity are the view's, over one value per
                // element, so the views keep their bounds and their layout.
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
        }
    }
}

/// The list without what comes before its first list and after its last, or `None` when a null
/// list holds elements, which only a rebuild can drop.
fn trim_list(list: &ListArray, ctx: &mut ExecutionCtx) -> VortexResult<Option<ListArray>> {
    let offsets = list.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let nulls = null_runs(&list.list_validity(), list.len(), ctx)?;
    match_each_integer_ptype!(offsets.ptype(), |O| {
        let offsets = offsets.as_slice::<O>();
        // The offsets are sorted, so a run of null lists is empty when it starts and ends at the
        // same offset.
        if nulls
            .iter()
            .any(|&(start, end)| offsets[start] != offsets[end])
        {
            return Ok(None);
        }
        let first = offsets[0];
        let (start, end): (usize, usize) = (first.as_(), offsets[offsets.len() - 1].as_());
        if start == 0 && end == list.elements().len() {
            return Ok(Some(list.clone()));
        }
        let shifted: Buffer<O> = offsets.iter().map(|&offset| offset - first).collect();
        // SAFETY: the elements are cut to the lists' range and the offsets shifted by its start,
        // so every list keeps its elements.
        Ok(Some(unsafe {
            ListArray::new_unchecked(
                list.elements().slice(start..end)?,
                shifted.into_array(),
                list.list_validity(),
            )
        }))
    })
}

/// The runs of null rows, as `(start, end)` ranges.
fn null_runs(
    validity: &Validity,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<(usize, usize)>> {
    if matches!(validity, Validity::NonNullable | Validity::AllValid) {
        return Ok(Vec::new());
    }
    Ok(match (!&validity.execute_mask(len, ctx)?).slices() {
        AllOr::All => vec![(0, len)],
        AllOr::None => Vec::new(),
        AllOr::Some(runs) => runs.to_vec(),
    })
}

/// The lists with every null list emptied, so that nothing under a null list reaches the lambda, or
/// `None` when no null list holds elements.
fn empty_null_lists(
    list: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ListViewArray>> {
    let validity = list.listview_validity();
    let nulls = null_runs(&validity, list.len(), ctx)?;
    if nulls.is_empty() {
        return Ok(None);
    }
    let sizes = list.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = match_each_integer_ptype!(sizes.ptype(), |T| {
        let sizes = sizes.as_slice::<T>();
        // Writers usually leave null lists empty already.
        if nulls
            .iter()
            .all(|&(start, end)| sizes[start..end].iter().all(|size| *size == T::default()))
        {
            return Ok(None);
        }
        let mut sizes = BufferMut::copy_from(sizes);
        for &(start, end) in &nulls {
            sizes[start..end].fill(T::default());
        }
        PrimitiveArray::new(sizes.freeze(), Validity::NonNullable)
    });
    // SAFETY: only sizes changed, and only to 0, so every view stays within the elements. The new
    // array is not marked zero-copy to a list: an emptied list leaves its elements behind as a gap,
    // which the rebuild must drop.
    Ok(Some(unsafe {
        ListViewArray::new_unchecked(
            list.elements().clone(),
            list.offsets().clone(),
            sizes.into_array(),
            validity,
        )
    }))
}

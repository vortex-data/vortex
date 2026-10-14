// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod elements;
#[cfg(test)]
mod tests;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use elements::ListElements;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::Lambda;
use crate::expr::display::ExprDisplay;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeValidity;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::is_not_null_node;

/// Apply a lambda to every element of each list, keeping each list's length and validity.
///
/// The lambda takes the element and, if it takes two parameters, the element's zero-based `u64`
/// index within its list. The children are the list, then the lambda's captures. A lambda that can
/// fail never runs on the elements of a null list. `List`, `ListView` and `FixedSizeList` inputs
/// all give a `List`.
#[derive(Clone)]
pub struct ListTransform;

impl ScalarFnVTable for ListTransform {
    type Options = Lambda;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.transform");
        *ID
    }

    fn serialize(&self, _lambda: &Lambda) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("lambdas are not serializable yet")
    }

    fn arity(&self, lambda: &Lambda) -> Arity {
        Arity::Exact(1 + lambda.capture_names().len())
    }

    fn child_name(&self, lambda: &Lambda, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("list"),
            capture => ChildName::from(Arc::clone(lambda.capture_names()[capture - 1].inner())),
        }
    }

    fn fmt_sql(
        &self,
        lambda: &Lambda,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}(", self.id())?;
        for child in 0..expr.display_children_count() {
            Display::fmt(expr.display_child(child), f)?;
            f.write_str(", ")?;
        }
        write!(f, "{lambda})")
    }

    fn return_dtype(&self, lambda: &Lambda, args: &[DType]) -> VortexResult<DType> {
        let Some((list, captures)) = args.split_first() else {
            vortex_bail!("list_transform() takes a list");
        };
        let params = param_dtypes(list, lambda.params().len())?;
        let element = lambda.return_dtype(&params, captures)?;
        Ok(DType::List(Arc::new(element), list.nullability()))
    }

    fn execute(
        &self,
        lambda: &Lambda,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // A body that can fail must not see the elements of a null list. One that cannot may run
        // over them, which saves copying the other lists' elements, and its results stay hidden
        // behind the lists' validity.
        let elements = ListElements::new(args.get(0)?, lambda.is_infallible(), ctx)?;

        let mut params = vec![elements.elements().clone()];
        if lambda.params().len() == 2 {
            params.push(elements.indices(ctx)?);
        }
        let captures = (1..args.num_inputs())
            .map(|capture| elements.repeat(args.get(capture)?, ctx))
            .try_collect()?;

        let values = lambda.evaluate(params, captures, elements.len())?;
        elements.rebuild(values)
    }

    fn validity<T: ReduceNode>(
        &self,
        _lambda: &Lambda,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        Ok(ReduceNodeValidity::Reduced(is_not_null_node(
            &node.child(0),
        )?))
    }

    fn is_infallible(&self, lambda: &Lambda) -> bool {
        lambda.is_infallible()
    }
}

/// The lambda's parameter dtypes: the element, and its `u64` index if the lambda takes two.
fn param_dtypes(list: &DType, param_count: usize) -> VortexResult<Vec<DType>> {
    let element = match list {
        DType::List(element, _) | DType::FixedSizeList(element, ..) => element.as_ref().clone(),
        other => vortex_bail!("list_transform() takes a List or FixedSizeList, got {other}"),
    };
    let index = DType::Primitive(PType::U64, Nullability::NonNullable);
    match param_count {
        1 => Ok(vec![element]),
        2 => Ok(vec![element, index]),
        n => vortex_bail!(
            "list_transform()'s lambda takes the element and its index, not {n} parameters"
        ),
    }
}

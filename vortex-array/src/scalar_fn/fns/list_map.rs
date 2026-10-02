// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::display::ExprDisplay;
use crate::expr::proto::ExprSerializeProtoExt;
use crate::proto::expr as pb;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::list_length::AnyList;

/// Applies an expression to every element of a list, keeping the list structure.
///
/// The element expression is evaluated with [`Expression::Root`] bound to the list elements, so
/// `list_map(xs, is_not_null(root()))` turns a list of values into a list of booleans of the same
/// shape. Null lists stay null.
#[derive(Clone)]
pub struct ListMap;

/// The expression [`ListMap`] applies to each list element.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ListMapOptions(pub Expression);

impl Display for ListMapOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "$ => {}", self.0)
    }
}

impl ScalarFnVTable for ListMap {
    type Options = ListMapOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.map");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.0.serialize_proto()?.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let proto = pb::Expr::decode(metadata)?;
        Ok(ListMapOptions(Expression::from_proto(&proto, session)?))
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for list_map()"),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}({}, {options})", self.id(), expr.display_child(0))
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        match &arg_dtypes[0] {
            DType::List(elem, nullability) => Ok(DType::List(
                Arc::new(options.0.return_dtype(elem)?),
                *nullability,
            )),
            DType::FixedSizeList(elem, size, nullability) => Ok(DType::FixedSizeList(
                Arc::new(options.0.return_dtype(elem)?),
                *size,
                *nullability,
            )),
            other => vortex_bail!("list_map() requires List or FixedSizeList, got {other}"),
        }
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?.execute_until::<AnyList>(ctx)?;
        let element_expr = &options.0;

        if let Some(list) = input.as_opt::<ListView>() {
            let elements = list.elements().clone().apply(element_expr)?;
            // SAFETY: the offsets, sizes and validity are unchanged, and `apply` keeps one value
            // per element.
            let mapped = unsafe {
                ListViewArray::new_unchecked(
                    elements,
                    list.offsets().clone(),
                    list.sizes().clone(),
                    list.listview_validity(),
                )
                .with_zero_copy_to_list(list.is_zero_copy_to_list())
            };
            return Ok(mapped.into_array());
        }

        if let Some(list) = input.as_opt::<List>() {
            let elements = list.elements().clone().apply(element_expr)?;
            // SAFETY: the offsets and validity are unchanged, and `apply` keeps one value per
            // element.
            let mapped = unsafe {
                ListArray::new_unchecked(elements, list.offsets().clone(), list.list_validity())
            };
            return Ok(mapped.into_array());
        }

        if let Some(list) = input.as_opt::<FixedSizeList>() {
            let elements = list.elements().clone().apply(element_expr)?;
            let mapped = FixedSizeListArray::try_new(
                elements,
                list.list_size(),
                list.fixed_size_list_validity(),
                list.len(),
            )?;
            return Ok(mapped.into_array());
        }

        vortex_bail!(
            "list_map() requires List, ListView, or FixedSizeList but got {}",
            input.dtype()
        )
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        // The element expression may fail, e.g. on an overflowing cast.
        false
    }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::expr::Expression;
    use crate::expr::eq;
    use crate::expr::is_not_null;
    use crate::expr::list_map;
    use crate::expr::list_sum;
    use crate::expr::lit;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::proto::expr as pb;
    use crate::validity::Validity;

    fn elements() -> ArrayRef {
        PrimitiveArray::from_option_iter([Some(0u64), Some(1), None, Some(2), Some(1), None])
            .into_array()
    }

    #[test]
    fn test_list_map_is_not_null() -> VortexResult<()> {
        let list = ListArray::try_new(
            elements(),
            buffer![0u32, 3, 3, 6].into_array(),
            Validity::from_iter([true, false, true]),
        )?
        .into_array();

        let result = list.apply(&list_map(root(), is_not_null(root())))?;

        let mut ctx = array_session().create_execution_ctx();
        let expected = ListArray::try_new(
            BoolArray::from_iter([true, true, false, true, true, false]).into_array(),
            buffer![0u32, 3, 3, 6].into_array(),
            Validity::from_iter([true, false, true]),
        )?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_list_view_map_then_sum_counts_matches() -> VortexResult<()> {
        // The second list view skips element 2 and the third reuses elements of the first.
        let list = ListViewArray::try_new(
            elements(),
            buffer![0u32, 3, 0].into_array(),
            buffer![3u32, 3, 2].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let result = list.apply(&list_sum(list_map(root(), eq(root(), lit(1u64)))))?;

        let mut ctx = array_session().create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter([Some(1u64), Some(1), Some(1)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_fixed_size_list_map() -> VortexResult<()> {
        let list =
            FixedSizeListArray::try_new(elements(), 2, Validity::NonNullable, 3)?.into_array();

        let result = list.apply(&list_sum(list_map(root(), is_not_null(root()))))?;

        let mut ctx = array_session().create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter([Some(2u64), Some(1), Some(1)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_list_map_proto_roundtrip() -> VortexResult<()> {
        let expr = list_sum(list_map(root(), eq(root(), lit(2u64))));
        let bytes = expr.serialize_proto()?.encode_to_vec();
        let decoded =
            Expression::from_proto(&pb::Expr::decode(bytes.as_slice())?, &array_session())?;
        assert_eq!(decoded, expr);
        Ok(())
    }
}

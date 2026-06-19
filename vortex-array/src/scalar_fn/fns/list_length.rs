// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::List;
use crate::arrays::ListView;
use crate::arrays::list::ListArrayExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::Expression;
use crate::matcher::Matcher;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;

/// Number of elements in each list of a `List` or `FixedSizeList` typed array.
///
/// This is computed purely from the list's offsets (`ListArray`), sizes (`ListViewArray`), or
/// dtype (`FixedSizeListArray`) without reading the element *values*. Validity is carried over
/// from the original array.
#[derive(Clone)]
pub struct ListLength;

impl ScalarFnVTable for ListLength {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.length");
        *ID
    }

    fn serialize(&self, _instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for list_length()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        match &arg_dtypes[0] {
            DType::List(_, nullable) | DType::FixedSizeList(_, _, nullable) => {
                Ok(DType::Primitive(PType::U64, *nullable))
            }
            other => vortex_bail!("list_length() requires List or FixedSizeList, got {other}"),
        }
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let nullability = input.dtype().nullability();

        if let Some(scalar) = input.as_constant() {
            let len_scalar = scalar_list_length(&scalar, nullability)?;
            return Ok(ConstantArray::new(len_scalar, args.row_count()).into_array());
        }

        list_length(&input, nullability, ctx)
    }

    fn reduce(
        &self,
        _options: &Self::Options,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        // The length of nonnullable fixed-size list is constant
        if let DType::FixedSizeList(_, size, Nullability::NonNullable) =
            node.child(0).node_dtype()?
        {
            let length = Scalar::primitive(size as u64, Nullability::NonNullable);
            return Ok(Some(ctx.new_node(Literal.bind(length), &[])?));
        }
        Ok(None)
    }

    fn validity(
        &self,
        _: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

fn scalar_list_length(scalar: &Scalar, nullability: Nullability) -> VortexResult<Scalar> {
    if scalar.is_null() {
        let dtype = DType::Primitive(PType::U64, Nullability::Nullable);
        return Ok(Scalar::null(dtype));
    }
    let len: u64 = scalar.as_list().len().as_();
    Ok(Scalar::primitive(len, nullability))
}

pub(crate) fn list_length(
    array: &ArrayRef,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (lengths, validity) = match array.dtype() {
        // The length of fixed-size list is constant, so just need to carry over validity
        DType::FixedSizeList(_, size, _) => {
            let lengths = ConstantArray::new(
                Scalar::primitive(*size as u64, Nullability::NonNullable),
                array.len(),
            )
            .into_array();
            (lengths, array.validity()?)
        }
        DType::List(..) => {
            let list = array.clone().execute_until::<AnyList>(ctx)?;

            if let Some(list) = list.as_opt::<List>() {
                let lengths = list_length_from_offsets(list)?;
                (lengths, list.list_validity())
            } else if let Some(list_view) = list.as_opt::<ListView>() {
                // Length array is exactly the sizes child
                (list_view.sizes().clone(), list_view.listview_validity())
            } else {
                unreachable!("AnyList matcher guarantees List or ListView")
            }
        }
        other => vortex_bail!("list_length() requires List or FixedSizeList, got {other}"),
    };

    // Cast to `U64`
    let len = lengths.len();
    let lengths = lengths.cast(DType::Primitive(PType::U64, nullability))?;

    // Carry over validity mask for nullable arrays
    if matches!(nullability, Nullability::Nullable) {
        lengths.mask(validity.to_array(len))
    } else {
        Ok(lengths)
    }
}

/// Calculate the lengths of `ListArray` elements via the `offsets` child:
/// `length[i] = offsets[i + 1] - offsets[i]`.
fn list_length_from_offsets(list: ArrayView<'_, List>) -> VortexResult<ArrayRef> {
    let offsets = list.offsets();
    let n = offsets.len().saturating_sub(1);

    offsets
        .slice(1..offsets.len())?
        .binary(offsets.slice(0..n)?, Operator::Sub)
}

/// Matches an `Array<List>` or `Array<ListView>`.
struct AnyList;

impl Matcher for AnyList {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.as_opt::<List>().is_some() || array.as_opt::<ListView>().is_some()).then_some(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::list_length;
    use crate::expr::root;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::literal::Literal;
    use crate::validity::Validity;

    fn create_list_elements() -> ArrayRef {
        PrimitiveArray::from_option_iter::<i32, _>([
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            None,
        ])
        .into_array()
    }

    #[rstest]
    #[case(buffer![0u32, 2, 5, 5, 7].into_array())]
    #[case(buffer![0u64, 2, 5, 5, 7].into_array())]
    fn test_list_length(#[case] offsets: ArrayRef) -> VortexResult<()> {
        let elements = create_list_elements();
        let list = ListArray::try_new(elements, offsets, Validity::NonNullable)?.into_array();
        let result = list.apply(&list_length(root()))?;
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2u64, 3, 0, 2]));
        Ok(())
    }

    #[rstest]
    #[case(buffer![0u32, 2, 5, 5, 7].into_array())]
    #[case(buffer![0u64, 2, 5, 5, 7].into_array())]
    fn test_nullable_list_length(#[case] offsets: ArrayRef) -> VortexResult<()> {
        let elements = create_list_elements();
        let list = ListArray::try_new(
            elements,
            offsets,
            Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array()),
        )?
        .into_array();
        let result = list.apply(&list_length(root()))?;

        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(0), None]);

        assert_arrays_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_null_scalar_list_length() -> VortexResult<()> {
        let null_scalar = Scalar::null(DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        ));
        let array = ConstantArray::new(null_scalar, 2).into_array();
        let result = array.apply(&list_length(root()))?;

        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        assert!(!result.is_valid(0, &mut ctx)?);
        assert!(!result.is_valid(1, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn test_listview_length() -> VortexResult<()> {
        let elements = create_list_elements();
        let lv = ListViewArray::new(
            elements,
            buffer![5u32, 0, 4, 1].into_array(),
            buffer![2u32, 3, 0, 2].into_array(),
            Validity::NonNullable,
        )
        .into_array();
        let result = lv.apply(&list_length(root()))?;
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2u64, 3, 0, 2]));
        Ok(())
    }

    #[test]
    fn test_listview_length_nullable() -> VortexResult<()> {
        let elements = create_list_elements();
        let lv = ListViewArray::new(
            elements,
            buffer![5u32, 0, 4, 1].into_array(),
            buffer![2u32, 3, 0, 2].into_array(),
            Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array()),
        )
        .into_array();
        let result = lv.apply(&list_length(root()))?;

        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(0), None]);
        assert_arrays_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn test_list_length_take() -> VortexResult<()> {
        let elements = create_list_elements();
        let list = ListArray::try_new(
            elements,
            buffer![0u32, 2, 5, 5, 7].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let taken = list.take(buffer![3u64, 0, 2].into_array())?;

        let result = taken.apply(&list_length(root()))?;
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2u64, 2, 0]));
        Ok(())
    }

    fn create_fixed_size_list(validity: Validity) -> ArrayRef {
        // 4 lists of size 2 over 8 primitive elements.
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8]).into_array();
        FixedSizeListArray::new(elements, 2, validity, 4).into_array()
    }

    #[test]
    fn test_fixed_size_list_length() -> VortexResult<()> {
        let fsl = create_fixed_size_list(Validity::NonNullable);
        let result = fsl.apply(&list_length(root()))?;

        // A non-nullable fixed-size list reduces to a constant literal length, never touching the
        // `ListLength` execution path.
        assert!(
            result
                .as_opt::<ScalarFn>()
                .is_some_and(|f| f.scalar_fn().as_opt::<Literal>().is_some()),
            "list_length over a non-nullable FixedSizeList must reduce to a constant literal"
        );
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2u64, 2, 2, 2]));
        Ok(())
    }

    #[test]
    fn test_fixed_size_list_length_nullable() -> VortexResult<()> {
        let fsl = create_fixed_size_list(Validity::Array(
            BoolArray::from_iter([true, false, true, false]).into_array(),
        ));
        let result = fsl.apply(&list_length(root()))?;

        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(2), None]);
        assert_arrays_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = list_length(root());
        assert_eq!(expr.to_string(), "vortex.list.length($)");
    }
}

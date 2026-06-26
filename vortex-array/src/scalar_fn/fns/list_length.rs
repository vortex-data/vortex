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
use crate::arrays::FixedSizeList;
use crate::arrays::List;
use crate::arrays::ListView;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
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
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
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
    let any_list = array.clone().execute_until::<AnyList>(ctx)?;

    let (lengths, validity) = if let Some(fsl) = any_list.as_opt::<FixedSizeList>() {
        // The length of fixed-size list is constant, so just need to carry over validity
        let size = fsl.list_size() as u64;
        let lengths =
            ConstantArray::new(Scalar::primitive(size, Nullability::NonNullable), fsl.len())
                .into_array();
        (lengths, fsl.validity()?)
    } else if let Some(lv) = any_list.as_opt::<ListView>() {
        // Length array is exactly the sizes child
        (lv.sizes().clone(), lv.listview_validity())
    } else if let Some(l) = any_list.as_opt::<List>() {
        let lengths = list_length_from_offsets(l)?;
        (lengths, l.list_validity())
    } else {
        let dtype = any_list.dtype();
        vortex_bail!("list_length() requires List, ListView, or FixedSizeList but got {dtype}")
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

/// Matches an `Array<List>`, `Array<ListView>`, or `Array<FixedSizeList>`
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
    use std::sync::Arc;

    use insta::assert_snapshot;
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::cast;
    use crate::expr::list_length;
    use crate::expr::root;
    use crate::scalar::Scalar;
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
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2u64, 3, 0, 2]), &mut ctx);
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

        let mut ctx = array_session().create_execution_ctx();
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(0), None]);

        assert_arrays_eq!(result, expected, &mut ctx);

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

        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2u64, 3, 0, 2]), &mut ctx);
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

        let mut ctx = array_session().create_execution_ctx();
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(0), None]);
        assert_arrays_eq!(result, expected, &mut ctx);
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
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2u64, 2, 0]), &mut ctx);
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

        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2u64, 2, 2, 2]), &mut ctx);
        Ok(())
    }

    #[test]
    fn test_fixed_size_list_length_nullable() -> VortexResult<()> {
        let fsl = create_fixed_size_list(Validity::Array(
            BoolArray::from_iter([true, false, true, false]).into_array(),
        ));
        let result = fsl.apply(&list_length(root()))?;

        let mut ctx = array_session().create_execution_ctx();
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(2), None]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_fallible_child_expression_fails() -> VortexResult<()> {
        let fsl = create_fixed_size_list(Validity::Array(
            BoolArray::from_iter([true, false, true, false]).into_array(),
        ));
        let failing_cast_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::NonNullable,
        );

        let lengths = fsl.apply(&list_length(cast(root(), failing_cast_dtype)))?;

        let mut ctx = array_session().create_execution_ctx();
        let result = lengths.execute::<ArrayRef>(&mut ctx);

        assert_snapshot!(
            result.unwrap_err().to_string(),
            @"Invalid argument error: Cannot cast array with invalid values to non-nullable type."

        );
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = list_length(root());
        assert_eq!(expr.to_string(), "vortex.list.length($)");
    }
}

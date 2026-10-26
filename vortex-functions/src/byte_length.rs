// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::varbinview::VarBinViewArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

pub trait ByteLengthKernel: VTable {
    fn byte_length(
        array: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

#[derive(Default, Debug)]
pub struct ByteLengthExecuteAdaptor<V>(pub V);

impl<V: ByteLengthKernel> ExecuteParentKernel<V> for ByteLengthExecuteAdaptor<V> {
    type Parent = ExactScalarFn<ByteLength>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        _parent: ScalarFnArrayView<'_, ByteLength>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        vortex_ensure_eq!(child_idx, 0);
        V::byte_length(array, ctx)
    }
}

/// Byte length of each element in a Utf8 or Binary array.
#[derive(Clone)]
pub struct ByteLength;

impl ScalarFnVTable for ByteLength {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.byte_length");
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
            _ => unreachable!("Invalid child index {child_idx} for byte_length()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        match &arg_dtypes[0] {
            DType::Utf8(nullable) | DType::Binary(nullable) => {
                Ok(DType::Primitive(PType::U64, *nullable))
            }
            other => vortex_bail!("byte_length() requires Utf8 or Binary, got {other}"),
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
            let len_scalar = scalar_byte_length(&scalar, nullability)?;
            return Ok(ConstantArray::new(len_scalar, args.row_count()).into_array());
        }

        match input.dtype() {
            DType::Utf8(_) | DType::Binary(_) => byte_length(&input, nullability, ctx),
            other => vortex_bail!("byte_length() requires Utf8 or Binary, got {other}"),
        }
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

fn scalar_byte_length(scalar: &Scalar, nullability: Nullability) -> VortexResult<Scalar> {
    if scalar.is_null() {
        let dtype = DType::Primitive(PType::U64, Nullability::Nullable);
        return Ok(Scalar::null(dtype));
    }
    let len = match scalar.dtype() {
        DType::Utf8(_) => scalar
            .as_utf8()
            .value()
            .vortex_expect("null utf-8 scalar")
            .len(),
        DType::Binary(_) => scalar
            .as_binary()
            .value()
            .vortex_expect("null binary scalar")
            .len(),
        other => vortex_bail!("byte_length() requires Utf8 or Binary, got {other}"),
    };
    let len: u64 = len.as_();
    Ok(Scalar::primitive(len, nullability))
}

pub(crate) fn byte_length(
    array: &ArrayRef,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let array = array.clone().execute::<VarBinViewArray>(ctx)?;
    let validity = array.varbinview_validity();
    let lengths: Buffer<u64> = array.views().iter().map(|v| v.len() as u64).collect();
    Ok(PrimitiveArray::new(lengths, validity.union_nullability(nullability)).into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::expr::byte_length;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();

        crate::initialize(&session);

        session
    });

    #[rstest]
    #[case(VarBinArray::from_strs(vec!["hello", "world", ""]).into_array(), vec![5u64, 5, 0])]
    #[case(VarBinArray::from_bytes(vec![b"ab".as_ref(), b"cde"]).into_array(), vec![2u64, 3])]
    #[case(VarBinArray::from_strs(vec!["Пуховички"]).into_array(), vec![18u64])]
    #[case(VarBinArray::from_bytes(vec!["Пуховички".as_ref()]).into_array(), vec![18u64])]
    fn test_bytes_byte_length(
        #[case] array: ArrayRef,
        #[case] expected_lens: Vec<u64>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let result = array.apply(&byte_length(root()))?;
        let expected = PrimitiveArray::from_iter(expected_lens);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_varbinview_byte_length() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinViewArray::from_iter_str(["short", "a longer string here"]).into_array();
        let result = array.apply(&byte_length(root()))?;
        let expected = PrimitiveArray::from_iter(vec![5u64, 20]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nullable_string_byte_length() -> VortexResult<()> {
        let array = VarBinArray::from_nullable_strs(vec![Some("hello"), None, Some("Пуховички")])
            .into_array();
        let result = array.apply(&byte_length(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        assert!(result.is_valid(0, &mut ctx)?);
        assert!(!result.is_valid(1, &mut ctx)?);
        assert!(result.is_valid(2, &mut ctx)?);
        assert_eq!(
            result.execute_scalar(0, &mut SESSION.create_execution_ctx())?,
            Scalar::primitive(5u64, Nullability::Nullable),
        );
        assert_eq!(
            result.execute_scalar(2, &mut SESSION.create_execution_ctx())?,
            Scalar::primitive(18u64, Nullability::Nullable),
        );
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(5u64), None, Some(18)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_null_scalar_byte_length() -> VortexResult<()> {
        let null_scalar = Scalar::null(DType::Utf8(Nullability::Nullable));
        let array = ConstantArray::new(null_scalar, 2).into_array();
        let result = array.apply(&byte_length(root()))?;
        let mut ctx = SESSION.create_execution_ctx();
        assert!(!result.is_valid(0, &mut ctx)?);
        assert!(!result.is_valid(1, &mut ctx)?);
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([None::<u64>, None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = byte_length(root());
        assert_eq!(expr.to_string(), "vortex.byte_length($)");
    }
}

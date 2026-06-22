// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Extract the storage values from an extension array.
#[derive(Clone)]
pub struct ExtStorage;

impl ScalarFnVTable for ExtStorage {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.ext.storage");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
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

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for ext_storage()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let DType::Extension(ext_dtype) = &arg_dtypes[0] else {
            vortex_bail!("ext_storage() requires Extension, got {}", arg_dtypes[0]);
        };

        Ok(ext_dtype.storage_dtype().clone())
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;

        if !matches!(input.dtype(), DType::Extension(_)) {
            vortex_bail!("ext_storage() requires Extension, got {}", input.dtype());
        }

        if let Some(scalar) = input.as_constant() {
            let storage_scalar = scalar.as_extension().to_storage_scalar();
            return Ok(ConstantArray::new(storage_scalar, args.row_count()).into_array());
        }

        let input = input.execute::<ExtensionArray>(ctx)?;
        Ok(input.storage_array().clone())
    }

    fn validity(
        &self,
        _options: &Self::Options,
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

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDTypeRef;
    use crate::expr::ext_storage;
    use crate::expr::root;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;

    fn ext_dtype(nullability: Nullability) -> ExtDTypeRef {
        Timestamp::new(TimeUnit::Nanoseconds, nullability).erased()
    }

    #[test]
    fn extracts_extension_storage_array() -> VortexResult<()> {
        let storage = buffer![2i64, 4, 6].into_array();
        let array =
            ExtensionArray::new(ext_dtype(Nullability::NonNullable), storage.clone()).into_array();

        let result = array.apply(&ext_storage(root()))?;

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, storage, &mut crate::array_execution_ctx());
        Ok(())
    }

    #[test]
    fn extracts_nullable_extension_storage_array() -> VortexResult<()> {
        let storage = PrimitiveArray::from_option_iter([Some(2i64), None, Some(6)]).into_array();
        let array =
            ExtensionArray::new(ext_dtype(Nullability::Nullable), storage.clone()).into_array();

        let result = array.apply(&ext_storage(root()))?;

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        assert_arrays_eq!(result, storage, &mut crate::array_execution_ctx());
        Ok(())
    }

    #[test]
    fn extracts_constant_extension_storage_scalar() -> VortexResult<()> {
        let storage_scalar = Scalar::primitive(4i64, Nullability::NonNullable);
        let scalar =
            Scalar::extension_ref(ext_dtype(Nullability::NonNullable), storage_scalar.clone());
        let array = ConstantArray::new(scalar, 3).into_array();

        let result = array.apply(&ext_storage(root()))?;

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_arrays_eq!(
            result,
            ConstantArray::new(storage_scalar, 3),
            &mut crate::array_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn rejects_non_extension_input() {
        let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let err = ext_storage(root()).return_dtype(&dtype).unwrap_err();
        assert!(err.to_string().contains("requires Extension"));
    }

    #[test]
    fn test_display() {
        assert_eq!(ext_storage(root()).to_string(), "vortex.ext.storage($)");
    }
}

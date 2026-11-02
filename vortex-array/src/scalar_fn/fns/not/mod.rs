// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

pub use kernel::*;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::bool::BoolArrayExt;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnExecutionResult;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// Expression that logically inverts boolean values.
#[derive(Clone)]
pub struct Not;

impl Not {
    /// Creates a lazy boolean inversion of `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is not boolean data.
    pub fn try_new(input: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Not.bind(EmptyOptions), [input])
    }
}

impl ScalarFnVTable for Not {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.not");
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
            _ => unreachable!("Invalid child index {} for Not expression", child_idx),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let child_dtype = &arg_dtypes[0];
        if !matches!(child_dtype, DType::Bool(_)) {
            vortex_bail!(
                "Not expression expects a boolean child, got: {}",
                child_dtype
            );
        }
        Ok(child_dtype.clone())
    }

    fn execute(
        &self,
        _options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_panic!("Not implements execute_step, so execute is never called")
    }

    fn execute_step(
        &self,
        _data: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult> {
        match not_columnar(&args.get(0)?, args.row_count())? {
            Some(result) => Ok(ScalarFnExecutionResult::done(result)),
            None => Ok(ScalarFnExecutionResult::execute_input::<AnyColumnar>(0)),
        }
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Inverts a constant or canonical boolean `child`, returning `None` for any other encoding.
fn not_columnar(child: &ArrayRef, row_count: usize) -> VortexResult<Option<ArrayRef>> {
    if let Some(scalar) = child.as_constant() {
        let value = match scalar.as_bool().value() {
            Some(b) => Scalar::bool(!b, child.dtype().nullability()),
            None => Scalar::null(child.dtype().clone()),
        };
        return Ok(Some(ConstantArray::new(value, row_count).into_array()));
    }

    if let Some(bool) = child.as_opt::<Bool>() {
        return Ok(Some(
            BoolArray::new(!bool.to_bit_buffer(), bool.validity()?).into_array(),
        ));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::not;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::scalar_fn::fns::not::BoolArray;
    use crate::scalar_fn::fns::not::Not;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn is_strict() {
        assert!(
            not(root())
                .as_scalar()
                .is_some_and(|f| f.signature().is_strict())
        );
    }

    #[test]
    fn executes_non_columnar_input_iteratively() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let chunks = vec![
            BoolArray::from_iter([Some(true), None]).into_array(),
            BoolArray::from_iter([Some(false)]).into_array(),
        ];
        let dtype = chunks[0].dtype().clone();
        let chunked = ChunkedArray::try_new(chunks, dtype)?.into_array();
        let nested = Not::try_new(Not::try_new(Not::try_new(chunked)?.into_array())?.into_array())?;

        let result = nested.into_array().execute::<Canonical>(&mut ctx)?;

        assert_arrays_eq!(
            result.into_array(),
            BoolArray::from_iter([Some(false), None, Some(true)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn preserves_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = BoolArray::from_iter([Some(false), None, Some(true)]).into_array();

        let result = input.apply(&not(root()))?;

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([Some(true), None, Some(false)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn invert_booleans() {
        let mut ctx = SESSION.create_execution_ctx();
        let not_expr = not(root());
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        let result = bools.into_array().apply(&not_expr).unwrap();
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, false, true, true, false, false]),
            &mut ctx
        );
    }

    #[test]
    fn test_display_order_of_operations() {
        let a = not(get_item("a", root()));
        let b = get_item("a", not(root()));
        assert_ne!(a.to_string(), b.to_string());
        assert_eq!(a.to_string(), "vortex.not($.a)");
        assert_eq!(b.to_string(), "vortex.not($).a");
    }

    #[test]
    fn dtype() {
        let not_expr = not(root());
        let dtype = DType::Bool(Nullability::NonNullable);
        assert_eq!(
            not_expr.return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );

        let dtype = test_harness::struct_dtype();
        assert_eq!(
            not(col("bool1")).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }
}

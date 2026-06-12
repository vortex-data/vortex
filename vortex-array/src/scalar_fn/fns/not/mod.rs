// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

pub use kernel::*;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// BoundExpr that logically inverts boolean values.
#[derive(Clone)]
pub struct Not;

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
        _data: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = args.get(0)?;

        // For constant boolean
        if let Some(scalar) = child.as_constant() {
            let value = match scalar.as_bool().value() {
                Some(b) => Scalar::bool(!b, child.dtype().nullability()),
                None => Scalar::null(child.dtype().clone()),
            };
            return Ok(ConstantArray::new(value, args.row_count()).into_array());
        }

        // For boolean array
        if let Some(bool) = child.as_opt::<Bool>() {
            return Ok(BoolArray::new(!bool.to_bit_buffer(), bool.validity()?).into_array());
        }

        // Otherwise, execute and try again
        child.execute::<ArrayRef>(ctx)?.not()
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
    use crate::IntoArray;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::arrays::bool::BoolArrayExt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::not;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::scalar_fn::fns::not::BoolArray;

    #[test]
    fn invert_booleans() {
        let not_expr = not(root(DType::Bool(Nullability::NonNullable)));
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        #[expect(deprecated)]
        let result = bools.into_array().apply(&not_expr).unwrap().to_bool();
        assert_eq!(
            result.to_bit_buffer().iter().collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }

    #[test]
    fn test_display_order_of_operations() {
        let scope = DType::struct_(
            [("a", DType::Bool(Nullability::NonNullable))],
            Nullability::NonNullable,
        );
        let a = not(get_item("a", root(scope)));
        // GetItem renders as a `.field` suffix on its child, including when the child is itself
        // a call rather than the root.
        let b = not(get_item(
            "a",
            pack([("a", lit(true))], Nullability::NonNullable),
        ));
        assert_ne!(a.to_string(), b.to_string());
        assert_eq!(a.to_string(), "vortex.not($.a)");
        assert_eq!(b.to_string(), "vortex.not(pack(a: true).a)");
    }

    #[test]
    fn dtype() {
        let dtype = DType::Bool(Nullability::NonNullable);
        let not_expr = not(root(dtype));
        assert_eq!(not_expr.dtype(), &DType::Bool(Nullability::NonNullable));

        let dtype = test_harness::struct_dtype();
        assert_eq!(
            not(col("bool1", &dtype)).dtype(),
            &DType::Bool(Nullability::NonNullable)
        );
    }
}

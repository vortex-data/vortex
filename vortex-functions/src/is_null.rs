// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::slice;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ReduceNode;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::is_not_null::reduce_null_check;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// Expression that checks for null values.
#[derive(Clone)]
pub struct IsNull;

impl IsNull {
    /// Creates a lazy null check over `input`.
    #[expect(clippy::new_ret_no_self, reason = "constructs the lazy result array")]
    pub fn new(input: ArrayRef) -> ScalarFnArray {
        ScalarFnArray::try_new(IsNull.bind(EmptyOptions), [input])
            .vortex_expect("IsNull has one child and an infallible return dtype")
    }
}

impl ScalarFnVTable for IsNull {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.is_null");
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
            _ => unreachable!("Invalid child index {} for IsNull expression", child_idx),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(
        &self,
        _data: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let child = args.get(0)?;
        if let Some(scalar) = child.as_constant() {
            return Ok(ConstantArray::new(scalar.is_null(), args.row_count()).into_array());
        }

        match child.validity()? {
            Validity::NonNullable | Validity::AllValid => {
                Ok(ConstantArray::new(false, args.row_count()).into_array())
            }
            Validity::AllInvalid => Ok(ConstantArray::new(true, args.row_count()).into_array()),
            Validity::Array(a) => a.not(),
        }
    }

    fn reduce<T: ReduceNode>(&self, _options: &Self::Options, node: &T) -> VortexResult<Option<T>> {
        reduce_null_check(node, true, |value| {
            value.new_node(IsNull.bind(EmptyOptions), slice::from_ref(value))
        })
    }

    fn is_strict(&self, _instance: &Self::Options) -> bool {
        // Null input produces the non-null boolean value `true`.
        false
    }

    fn is_infallible(&self, _instance: &Self::Options) -> bool {
        true
    }
}

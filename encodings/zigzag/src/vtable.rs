// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::compress::zigzag_decode;

/// ZigZag encoding maps signed integers to unsigned integers so that small absolute values
/// have small encoded values.
#[derive(Clone)]
pub struct ZigZag;

impl ScalarFnVTable for ZigZag {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.zigzag")
    }

    fn serialize(&self, _options: &EmptyOptions) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<EmptyOptions> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &EmptyOptions) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &EmptyOptions, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("encoded"),
            _ => unreachable!("Invalid child index {child_idx} for ZigZag"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &EmptyOptions,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "zigzag_decode(")?;
        expr.children()[0].fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &EmptyOptions, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let encoded_dtype = &arg_dtypes[0];
        let ptype = PType::try_from(encoded_dtype)?;
        vortex_ensure!(
            ptype.is_unsigned_int(),
            "ZigZag encoded child must be unsigned integer, got {encoded_dtype}"
        );
        Ok(DType::from(ptype.to_signed()).with_nullability(encoded_dtype.nullability()))
    }

    fn execute(
        &self,
        _options: &EmptyOptions,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let encoded = args.get(0)?;
        let decoded = zigzag_decode(encoded.execute::<PrimitiveArray>(ctx)?);
        Ok(decoded.into_array())
    }

    fn validity(
        &self,
        _options: &EmptyOptions,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_null_sensitive(&self, _options: &EmptyOptions) -> bool {
        false
    }

    fn is_fallible(&self, _options: &EmptyOptions) -> bool {
        false
    }
}

impl ScalarFnArrayVTable for ZigZag {
    fn serialize(
        &self,
        _view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        vortex_ensure!(
            metadata.is_empty(),
            "ZigZag expects empty metadata, got {} bytes",
            metadata.len()
        );
        vortex_ensure!(
            children.len() == 1,
            "ZigZag expects 1 child, got {}",
            children.len()
        );

        let ptype = PType::try_from(dtype)?;
        let encoded_dtype = DType::Primitive(ptype.to_unsigned(), dtype.nullability());
        let encoded = children.get(0, &encoded_dtype, len)?;

        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: vec![encoded],
        })
    }
}

/// Construct a ZigZag-encoded array from an unsigned encoded child.
pub fn zigzag_try_new(encoded: ArrayRef) -> VortexResult<ArrayRef> {
    let len = encoded.len();
    ZigZag.try_new_array(len, EmptyOptions, [encoded])
}

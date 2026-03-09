// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Options for the `VariantGet` scalar function.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantGetOptions {
    /// The variant field path to extract.
    pub path: String,
    /// The expected return type.
    pub dtype: DType,
}

impl fmt::Display for VariantGetOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "variant_get({}, {:?})", self.path, self.dtype)
    }
}

/// Scalar function that extracts data by path and dtype from variant arrays.
#[derive(Clone)]
pub struct VariantGet;

impl ScalarFnVTable for VariantGet {
    type Options = VariantGetOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.variant_get")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::VariantGetOpts {
                path: instance.path.clone(),
                dtype: Some((&instance.dtype).try_into()?),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let opts = pb::VariantGetOpts::decode(metadata)?;
        let dtype = DType::from_proto(
            opts.dtype
                .as_ref()
                .ok_or_else(|| vortex_err!("VariantGetOpts missing dtype"))?,
            session,
        )?;
        Ok(VariantGetOptions {
            path: opts.path,
            dtype,
        })
    }

    fn arity(&self, _options: &VariantGetOptions) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!(
                "Invalid child index {} for VariantGet expression",
                child_idx
            ),
        }
    }

    fn fmt_sql(
        &self,
        options: &VariantGetOptions,
        expr: &crate::expr::Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        expr.children()[0].fmt_sql(f)?;
        write!(f, ".{}", options.path)
    }

    fn return_dtype(
        &self,
        options: &VariantGetOptions,
        _arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        // Always return nullable since Variant data is always nullable
        Ok(options.dtype.with_nullability(Nullability::Nullable))
    }

    fn execute(
        &self,
        _options: &VariantGetOptions,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "VariantGet should be pushed down via parent reduction rules, not executed directly"
        )
    }

    fn is_null_sensitive(&self, _options: &VariantGetOptions) -> bool {
        true
    }

    fn is_fallible(&self, _options: &VariantGetOptions) -> bool {
        false
    }
}

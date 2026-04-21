// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Variant;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::variant::VariantArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

mod path;

pub use path::VariantPath;
pub use path::VariantPathElement;

/// Extracts a nested path from a variant value, returning a new nullable variant.
///
/// Execution prefers encoding-specific parent kernels and otherwise falls back through the
/// canonical [`Variant`] boundary.
#[derive(Clone)]
pub struct VariantGet;

impl ScalarFnVTable for VariantGet {
    type Options = VariantPath;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.variant_get")
    }

    fn serialize(&self, path: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let path = path
            .iter()
            .cloned()
            .map(|element| match element {
                VariantPathElement::Field(name) => Ok(pb::VariantPathElement {
                    path_element: Some(pb::variant_path_element::PathElement::Field(
                        name.to_string(),
                    )),
                }),
                VariantPathElement::Index(index) => Ok(pb::VariantPathElement {
                    path_element: Some(pb::variant_path_element::PathElement::Index(
                        index.try_into().map_err(|_| {
                            vortex_err!("variant path index {index} does not fit in u32")
                        })?,
                    )),
                }),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Some(pb::VariantGetOpts { path }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::VariantGetOpts::decode(metadata)?;
        opts.path
            .into_iter()
            .map(|element| match element.path_element {
                Some(pb::variant_path_element::PathElement::Field(name)) => {
                    Ok(VariantPathElement::Field(name.into()))
                }
                Some(pb::variant_path_element::PathElement::Index(index)) => {
                    Ok(VariantPathElement::Index(index as usize))
                }
                None => vortex_bail!("variant_get path element must be set"),
            })
            .collect()
    }

    fn arity(&self, _path: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _path: &Self::Options, child_idx: usize) -> ChildName {
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
        path: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "variant_get(")?;
        expr.children()[0].fmt_sql(f)?;
        write!(f, ", '{path}')")
    }

    fn return_dtype(&self, _path: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        if !matches!(arg_dtypes[0], DType::Variant(_)) {
            vortex_bail!(
                "variant_get requires a Variant input, got {:?}",
                arg_dtypes[0]
            );
        }
        Ok(DType::Variant(Nullability::Nullable))
    }

    fn execute(
        &self,
        path: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;

        if let Some(variant) = input.as_opt::<Variant>() {
            // Canonical Variant is the stable boundary: keep delegating through core_storage until
            // an encoding-specific parent kernel or a lower-level execute path can handle it.
            return self
                .try_new_array(
                    args.row_count(),
                    path.clone(),
                    [variant.core_storage().clone()],
                )?
                .execute::<ArrayRef>(ctx);
        }

        let executed_input = input.clone().execute::<ArrayRef>(ctx)?;
        if ArrayRef::ptr_eq(&input, &executed_input) {
            vortex_bail!(
                "variant_get could not make progress on {} input {}",
                input.dtype(),
                input.encoding_id()
            );
        }

        self.try_new_array(args.row_count(), path.clone(), [executed_input])?
            .execute::<ArrayRef>(ctx)
    }

    fn is_null_sensitive(&self, _path: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _path: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use vortex_session::VortexSession;

    use super::VariantGet;
    use super::VariantPath;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar_fn::ScalarFnVTable;

    #[test]
    fn variant_get_path_proto_round_trip() {
        let path = VariantPath::from_name("outer").join(1usize).join("inner");

        let metadata = VariantGet.serialize(&path).unwrap().unwrap();
        let decoded = VariantGet
            .deserialize(&metadata, &VortexSession::empty())
            .unwrap();

        assert_eq!(decoded, path);
        assert_eq!(decoded.to_string(), "outer[1].inner");
        assert_eq!(
            VariantGet
                .return_dtype(&decoded, &[DType::Variant(Nullability::NonNullable)])
                .unwrap(),
            DType::Variant(Nullability::Nullable)
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::sync::Arc;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Variant;
use crate::arrays::variant::VariantArrayExt;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// Extracts a field from a variant object by name, returning a new variant.
///
/// This is analogous to [`GetItem`](super::get_item::GetItem) for structs, but operates on
/// semi-structured variant data. The result is always `DType::Variant(Nullable)` since the
/// requested field may not exist in every row.
///
/// Execution is handled by variant encodings (e.g. `ParquetVariantArray`) via `execute_parent`.
/// The canonical `VariantArray` does not support direct execution; a `reduce` rule unwraps
/// the `VariantArray` wrapper to expose the underlying encoding.
#[derive(Clone)]
pub struct VariantGet;

impl ScalarFnVTable for VariantGet {
    type Options = FieldName;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.variant_get")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::VariantGetOpts {
                path: instance.to_string(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::VariantGetOpts::decode(metadata)?;
        Ok(FieldName::from(opts.path))
    }

    fn arity(&self, _field_name: &FieldName) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
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
        field_name: &FieldName,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "variant_get(")?;
        expr.children()[0].fmt_sql(f)?;
        write!(f, ", '{}')", field_name)
    }

    fn return_dtype(&self, _field_name: &FieldName, arg_dtypes: &[DType]) -> VortexResult<DType> {
        if !matches!(arg_dtypes[0], DType::Variant(_)) {
            vortex_bail!(
                "variant_get requires a Variant input, got {:?}",
                arg_dtypes[0]
            );
        }
        // Always nullable: the field may not exist in every variant value.
        Ok(DType::Variant(Nullability::Nullable))
    }

    fn execute(
        &self,
        _field_name: &FieldName,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "variant_get cannot be executed directly; \
             it must be pushed down to a variant encoding via execute_parent"
        )
    }

    fn reduce(
        &self,
        field_name: &FieldName,
        node: &dyn ReduceNode,
        ctx: &dyn ReduceCtx,
    ) -> VortexResult<Option<ReduceNodeRef>> {
        // If the child is a canonical VariantArray wrapper, unwrap it to expose the
        // underlying encoding (e.g. ParquetVariantArray) so that execute_parent can
        // handle the operation.
        let child = node.child(0);
        if let Some(child_array) = child.as_any().downcast_ref::<ArrayRef>()
            && child_array.is::<Variant>()
        {
            let inner = child_array.as_::<Variant>().child().clone();
            return Ok(Some(ctx.new_node(
                VariantGet.bind(field_name.clone()),
                &[Arc::new(inner) as ReduceNodeRef],
            )?));
        }
        Ok(None)
    }

    fn is_null_sensitive(&self, _field_name: &FieldName) -> bool {
        true
    }

    fn is_fallible(&self, _field_name: &FieldName) -> bool {
        false
    }
}

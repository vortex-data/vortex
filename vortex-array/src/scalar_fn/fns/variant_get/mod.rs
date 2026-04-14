// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Variant;
use crate::arrays::variant::VariantArrayExt;
use crate::dtype::DType;
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

mod path;

pub use path::VariantPath;
pub use path::VariantPathElement;

/// Extracts a nested path from a variant value, returning a new variant.
///
/// This is analogous to [`GetItem`](super::get_item::GetItem) for structs, but operates on
/// semi-structured variant data. The result is always `DType::Variant(Nullable)` since the
/// requested path may not exist in every row.
///
/// Execution is handled by variant encodings (e.g. `ParquetVariantArray`) via `execute_parent`.
/// The canonical `VariantArray` does not support direct execution, so `VariantGet` keeps a small
/// `reduce` rule that unwraps a direct `VariantArray` child to expose the underlying encoding.
/// Wrapper arrays such as `Slice` and `Filter` forward `VariantGet` from their own
/// `execute_parent` hooks so the expression can still reach the underlying variant encoding
/// without teaching `VariantGet` about wrapper-specific array shapes.
#[derive(Clone)]
pub struct VariantGet;

/// Options for [`VariantGet`].
///
/// `path` selects the nested variant value to extract. `as_dtype`, when set, asks the encoding
/// to materialize the result directly as that logical type instead of another variant.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct VariantGetOptions {
    path: VariantPath,
    as_dtype: Option<DType>,
}

impl VariantGetOptions {
    /// Creates options for extracting the given path as another variant value.
    pub fn new(path: impl Into<VariantPath>) -> Self {
        Self {
            path: path.into(),
            as_dtype: None,
        }
    }

    /// Returns the requested path.
    pub fn path(&self) -> &VariantPath {
        &self.path
    }

    /// Returns the requested output type, if any.
    pub fn as_dtype(&self) -> Option<&DType> {
        self.as_dtype.as_ref()
    }

    /// Returns new options that request direct materialization as `as_dtype`.
    pub fn with_as_dtype(mut self, as_dtype: Option<DType>) -> Self {
        self.as_dtype = as_dtype;
        self
    }

    /// Returns the logical output dtype for this expression.
    pub fn return_dtype(&self) -> DType {
        match self.effective_as_dtype() {
            Some(dtype) => dtype.as_nullable(),
            None => DType::Variant(Nullability::Nullable),
        }
    }

    /// Returns the dtype to materialize directly, if it differs from the default variant output.
    pub fn effective_as_dtype(&self) -> Option<&DType> {
        self.as_dtype
            .as_ref()
            .filter(|dtype| !matches!(dtype, DType::Variant(_)))
    }
}

impl Display for VariantGetOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path)?;
        if let Some(as_dtype) = &self.as_dtype {
            write!(f, " as {as_dtype}")?;
        }
        Ok(())
    }
}

impl<T: Into<VariantPath>> From<T> for VariantGetOptions {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl ScalarFnVTable for VariantGet {
    type Options = VariantGetOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.variant_get")
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let path = instance
            .path()
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
        let as_dtype = instance.as_dtype().map(TryInto::try_into).transpose()?;
        Ok(Some(pb::VariantGetOpts { path, as_dtype }.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let opts = pb::VariantGetOpts::decode(metadata)?;
        let path = opts
            .path
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
            .collect::<VortexResult<VariantPath>>()?;
        let as_dtype = opts
            .as_dtype
            .as_ref()
            .map(|dtype| DType::from_proto(dtype, session))
            .transpose()?;

        Ok(VariantGetOptions { path, as_dtype })
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
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "variant_get(")?;
        expr.children()[0].fmt_sql(f)?;
        write!(f, ", '{}'", options.path())?;
        if let Some(as_dtype) = options.as_dtype() {
            write!(f, ", {as_dtype}")?;
        }
        write!(f, ")")
    }

    fn return_dtype(
        &self,
        options: &VariantGetOptions,
        arg_dtypes: &[DType],
    ) -> VortexResult<DType> {
        if !matches!(arg_dtypes[0], DType::Variant(_)) {
            vortex_bail!(
                "variant_get requires a Variant input, got {:?}",
                arg_dtypes[0]
            );
        }
        Ok(options.return_dtype())
    }

    fn execute(
        &self,
        options: &VariantGetOptions,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let _ = (options, args, ctx);
        vortex_bail!("variant_get cannot be executed directly")
    }

    fn reduce(
        &self,
        options: &VariantGetOptions,
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
                VariantGet.bind(options.clone()),
                &[Arc::new(inner) as ReduceNodeRef],
            )?));
        }
        Ok(None)
    }

    fn is_null_sensitive(&self, _options: &VariantGetOptions) -> bool {
        true
    }

    fn is_fallible(&self, _options: &VariantGetOptions) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use vortex_session::VortexSession;

    use super::VariantGet;
    use super::VariantGetOptions;
    use super::VariantPath;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar_fn::ScalarFnVTable;

    #[test]
    fn variant_get_path_proto_round_trip() {
        let options =
            VariantGetOptions::new(VariantPath::from_name("outer").join(1usize).join("inner"))
                .with_as_dtype(Some(DType::Primitive(PType::I64, Nullability::NonNullable)));

        let metadata = VariantGet.serialize(&options).unwrap().unwrap();
        let decoded = VariantGet
            .deserialize(&metadata, &VortexSession::empty())
            .unwrap();

        assert_eq!(decoded, options);
        assert_eq!(decoded.to_string(), "outer[1].inner as i64");
        assert_eq!(
            decoded.return_dtype(),
            DType::Primitive(PType::I64, Nullability::Nullable)
        );
    }
}

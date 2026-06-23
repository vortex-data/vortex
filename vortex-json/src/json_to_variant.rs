// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.json_to_variant` scalar function.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Json;

/// Parses JSON strings into Variant values, optionally shredding fields.
///
/// Accepts [`Json`] extension inputs and returns `Variant` values with the input's nullability.
/// Null rows stay null, the JSON literal `null` becomes a variant-null value, and any row that
/// fails to parse as JSON fails the whole expression.
///
/// A non-empty [`ShreddingSpec`] additionally shreds the selected paths into a typed shredded
/// child, following the [Parquet Variant shredding] rules: rows whose value does not match the
/// requested type stay readable through the residual variant value.
///
/// # Execution
///
/// Building a Variant requires a concrete Variant encoding, so this function does not perform the
/// conversion itself. The Variant encoding registered with the session supplies it as an
/// `execute_parent` kernel keyed on the extension encoding for a [`Json`] input. The fallback
/// [`execute`](ScalarFnVTable::execute) here only canonicalizes the input to that encoding and
/// re-dispatches so that kernel runs; it errors if no Variant encoding is registered with the
/// session.
///
/// # Normalization
///
/// `json_to_variant` is a lossy, normalizing conversion: the parsed Variant does not round-trip
/// back to the exact source JSON text.
/// - JSON whitespace is not preserved.
/// - Object keys are stored in Variant metadata in sorted order, not source order.
/// - Number representations are normalized (e.g. `1.0` and `1` may parse to the same value;
///   exponent forms and very large numbers are re-encoded).
/// - Duplicate object keys are collapsed to a single entry.
/// - Unicode escape sequences are normalized (e.g. `A` becomes `A`).
///
/// [Parquet Variant shredding]: https://github.com/apache/parquet-format/blob/master/VariantShredding.md
#[derive(Clone)]
pub struct JsonToVariant;

impl ScalarFnVTable for JsonToVariant {
    type Options = JsonToVariantOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.json_to_variant");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let shredding = options
            .shredding()
            .fields()
            .iter()
            .map(|(path, dtype)| {
                Ok(pb::ShreddingSpecField {
                    path: path
                        .elements()
                        .iter()
                        .map(VariantPathElement::to_proto)
                        .collect(),
                    dtype: Some(dtype.try_into()?),
                })
            })
            .collect::<VortexResult<_>>()?;

        Ok(Some(pb::JsonToVariantOpts { shredding }.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let opts = pb::JsonToVariantOpts::decode(metadata)?;
        let fields = opts
            .shredding
            .into_iter()
            .map(|field| {
                let path = field
                    .path
                    .into_iter()
                    .map(VariantPathElement::from_proto)
                    .collect::<VortexResult<VariantPath>>()?;
                let dtype = field
                    .dtype
                    .as_ref()
                    .ok_or_else(|| vortex_err!("ShreddingSpecField missing dtype"))
                    .and_then(|dtype| DType::from_proto(dtype, session))?;
                Ok((path, dtype))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(JsonToVariantOptions::new(ShreddingSpec::try_new(fields)?))
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for JsonToVariant expression"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input_dtype = &arg_dtypes[0];
        vortex_ensure!(
            input_dtype
                .as_extension_opt()
                .is_some_and(|ext_dtype| ext_dtype.is::<Json>()),
            "JsonToVariant input must be a Json extension, found {input_dtype}"
        );

        Ok(DType::Variant(input_dtype.nullability()))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;

        // This function does not build Variants itself: the Variant encoding registered with the
        // session supplies the conversion as an `execute_parent` kernel keyed on the extension
        // encoding for a `Json` input. Reaching this fallback means no such kernel fired, so
        // canonicalize the input to that encoding and re-dispatch. If the input is already
        // canonical here, no kernel is registered for it; bail with a clear error rather than
        // looping.
        let no_kernel = || {
            vortex_err!(
                "json_to_variant requires a registered Variant encoding to build Variant values \
                 from JSON, but none is registered with this session"
            )
        };

        let canonical = if input
            .dtype()
            .as_extension_opt()
            .is_some_and(|ext_dtype| ext_dtype.is::<Json>())
        {
            if input.is::<Extension>() {
                return Err(no_kernel());
            }
            input.execute::<ExtensionArray>(ctx)?.into_array()
        } else {
            vortex_bail!(
                "JsonToVariant input must be a Json extension, found {}",
                input.dtype()
            );
        };

        self.try_new_array(canonical.len(), options.clone(), [canonical])?
            .execute::<ArrayRef>(ctx)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        // `json_to_variant` maps null rows to null rows and the JSON literal `null` to a
        // variant-null value independently of which other rows are null, so it commutes with
        // validity masking. Marking it not null-sensitive also lets it push through dictionaries
        // into their JSON extension values, where the kernel fires directly.
        false
    }
}

/// Creates a [`JsonToVariant`] expression that parses `child`'s JSON strings into Variant
/// values, shredding the paths selected by `shredding`.
///
/// `child` must produce [`Json`] extension values; the result is `Variant` with the input's
/// nullability. Rows containing invalid JSON fail the expression.
///
/// Note that this is a lossy, normalizing conversion. See [`JsonToVariant`] for the full list of
/// caveats.
pub fn json_to_variant(child: Expression, shredding: ShreddingSpec) -> Expression {
    JsonToVariant.new_expr(JsonToVariantOptions::new(shredding), [child])
}

/// A list of `(path, dtype)` directives describing which Variant paths to shred and as what
/// type.
///
/// Paths must contain only object-field elements; list index elements are rejected because
/// Parquet Variant shredding schemas cannot express list element shredding yet. The root path
/// (`$`) shreds the top-level value itself. When entries overlap (e.g. `$.a` and `$.a.b`),
/// later entries overwrite earlier ones.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShreddingSpec(Vec<(VariantPath, DType)>);

impl ShreddingSpec {
    /// Creates an empty spec, meaning no shredding is performed.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Creates a spec from `(path, dtype)` directives.
    ///
    /// # Errors
    ///
    /// Returns an error if any path contains a list index element.
    pub fn try_new(fields: impl IntoIterator<Item = (VariantPath, DType)>) -> VortexResult<Self> {
        let fields = fields.into_iter().collect::<Vec<_>>();
        for (path, _) in &fields {
            vortex_ensure!(
                path.elements()
                    .iter()
                    .all(|element| matches!(element, VariantPathElement::Field(_))),
                "ShreddingSpec paths must only contain object fields, found list index in {path}"
            );
        }
        Ok(Self(fields))
    }

    /// Returns the `(path, dtype)` directives.
    pub fn fields(&self) -> &[(VariantPath, DType)] {
        &self.0
    }

    /// Returns whether this spec contains no directives.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Display for ShreddingSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (idx, (path, dtype)) in self.0.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{path} as {dtype}")?;
        }
        Ok(())
    }
}

/// Options for [`JsonToVariant`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JsonToVariantOptions {
    /// The paths to shred into typed storage, if any.
    shredding: ShreddingSpec,
}

impl JsonToVariantOptions {
    /// Creates options that shred the paths selected by `shredding`.
    pub fn new(shredding: ShreddingSpec) -> Self {
        Self { shredding }
    }

    /// Creates options that perform no shredding.
    pub fn unshredded() -> Self {
        Self::new(ShreddingSpec::empty())
    }

    /// Returns the shredding spec.
    pub fn shredding(&self) -> &ShreddingSpec {
        &self.shredding
    }
}

impl Display for JsonToVariantOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.shredding.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::missing_docs_in_private_items)]

    use vortex_array::ArrayRef;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::session::DTypeSession;
    use vortex_array::expr::proto::ExprSerializeProtoExt;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_array::scalar_fn::session::ScalarFnSessionExt;
    use vortex_array::session::ArraySession;

    use super::*;

    fn i64_dtype() -> DType {
        DType::Primitive(PType::I64, Nullability::Nullable)
    }

    /// A session that knows the `JsonToVariant` definition but has no Variant encoding registered,
    /// so executing the function exercises the fallback that errors when no kernel is present.
    fn session() -> VortexSession {
        let session = VortexSession::empty()
            .with::<ArraySession>()
            .with::<DTypeSession>()
            .with::<ScalarFnSession>();
        session.scalar_fns().register(JsonToVariant);
        session
    }

    #[test]
    fn shredding_spec_rejects_index_paths() {
        let err = ShreddingSpec::try_new([(
            VariantPath::new([VariantPathElement::field("a"), VariantPathElement::index(0)]),
            i64_dtype(),
        )])
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("ShreddingSpec paths must only contain object fields"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn expression_roundtrip_serialization() -> VortexResult<()> {
        let spec = ShreddingSpec::try_new([(VariantPath::field("a"), i64_dtype())])?;
        let expr: Expression = json_to_variant(root(), spec);
        let proto = expr.serialize_proto()?;
        let actual = Expression::from_proto(&proto, &session())?;

        assert_eq!(actual, expr);
        Ok(())
    }

    #[test]
    fn utf8_input_is_rejected() {
        let input = VarBinViewArray::from_iter_str(["1", "2"]).into_array();
        let err = JsonToVariant
            .try_new_array(input.len(), JsonToVariantOptions::unshredded(), [input])
            .unwrap_err();

        assert!(
            err.to_string().contains("Json extension"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn execute_without_variant_kernel_errors() -> VortexResult<()> {
        // With no Variant encoding registered, executing over an already-canonical input must
        // surface a clear error rather than looping.
        let input = ExtensionArray::try_new_from_vtable(
            Json,
            EmptyMetadata,
            VarBinViewArray::from_iter_str(["1", "2"]).into_array(),
        )?
        .into_array();

        let dtype = input.dtype().clone();
        let array = JsonToVariant.try_new_array(
            input.len(),
            JsonToVariantOptions::unshredded(),
            [input],
        )?;

        let err = array
            .execute::<ArrayRef>(&mut session().create_execution_ctx())
            .unwrap_err();

        assert!(
            err.to_string().contains("Variant encoding"),
            "unexpected error for {dtype}: {err}"
        );
        Ok(())
    }
}

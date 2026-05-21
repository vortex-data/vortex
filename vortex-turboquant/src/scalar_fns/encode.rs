// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant encode scalar function.

use std::fmt;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_tensor::vector::AnyVector;

use super::metadata::deserialize_config;
use super::metadata::serialize_config;
use crate::TurboQuantConfig;
use crate::config::MIN_DIMENSION;
use crate::vector::normalize::tq_normalize_as_l2_denorm;
use crate::vector::quantize::empty_quantization;
use crate::vector::quantize::turboquant_quantize_core;
use crate::vector::storage::build_codes_child;
use crate::vector::storage::build_storage;
use crate::vector::tq_padded_dim;
use crate::vtable::TurboQuant;
use crate::vtable::TurboQuantMetadata;
use crate::vtable::tq_storage_dtype;

/// TurboQuant vector encode scalar function.
///
/// `TQEncode` itself is a `ScalarFnVTable` and so its options round-trip through expression
/// serialization.
///
/// Unlike `TQDecode`, it deliberately does **not** implement `ScalarFnArrayVTable` since the
/// persisted artifact would be the original vector array, not the TurboQuant-quantized array.
#[derive(Clone)]
pub struct TQEncode;

impl TQEncode {
    /// Creates a new [`TypedScalarFnInstance`] wrapping TurboQuant encoding.
    pub fn new(config: &TurboQuantConfig) -> TypedScalarFnInstance<TQEncode> {
        TypedScalarFnInstance::new(TQEncode, config.clone())
    }

    /// Constructs a [`ScalarFnArray`] that lazily encodes a `Vector` child into `TurboQuant`.
    pub fn try_new_array(
        child: ArrayRef,
        config: &TurboQuantConfig,
    ) -> VortexResult<ScalarFnArray> {
        let len = child.len();
        ScalarFnArray::try_new(TQEncode::new(config).erased(), vec![child], len)
    }
}

impl ScalarFnVTable for TQEncode {
    type Options = TurboQuantConfig;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.turboquant.encode")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(serialize_config(options)))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        deserialize_config(metadata)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("vector"),
            _ => unreachable!("TQEncode must have exactly one child"),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "tq_encode(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", {options})")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input_dtype = &arg_dtypes[0];
        let vector_metadata = input_dtype
            .as_extension_opt()
            .and_then(|ext_dtype| ext_dtype.metadata_opt::<AnyVector>())
            .ok_or_else(|| {
                vortex_err!("TQEncode expects a Vector extension array, got {input_dtype}")
            })?;

        let dimensions = vector_metadata.dimensions();
        vortex_ensure!(
            dimensions >= MIN_DIMENSION,
            "TurboQuant requires dimension >= {MIN_DIMENSION}, got {dimensions}",
        );
        tq_padded_dim(dimensions)?;

        let metadata = TurboQuantMetadata {
            element_ptype: vector_metadata.element_ptype(),
            dimensions,
            bit_width: options.bit_width(),
            seed: options.seed(),
            num_rounds: options.num_rounds(),
        };
        let storage_dtype = tq_storage_dtype(&metadata, input_dtype.nullability())?;
        let ext_dtype = ExtDType::<TurboQuant>::try_new(metadata, storage_dtype)?.erased();

        Ok(DType::Extension(ext_dtype))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        encode_vector(args.get(0)?, options, ctx)
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

/// Lossily encode a `Vector` extension array into a `TurboQuant` extension array.
///
/// Valid rows are normalized internally before SORF transform and scalar quantization. The original
/// row norms are stored explicitly, and original vector nulls are preserved on the storage struct
/// and both row-aligned child arrays.
pub(crate) fn encode_vector(
    input: ArrayRef,
    config: &TurboQuantConfig,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let num_vectors = input.len();
    let vector_metadata = input
        .dtype()
        .as_extension_opt()
        .and_then(|ext_dtype| ext_dtype.metadata_opt::<AnyVector>())
        .ok_or_else(|| vortex_err!("TurboQuant encode expects a Vector extension array"))?;

    let element_ptype = vector_metadata.element_ptype();

    let dimensions = vector_metadata.dimensions();
    vortex_ensure!(
        dimensions >= MIN_DIMENSION,
        "TurboQuant requires dimension >= {MIN_DIMENSION}, got {dimensions}",
    );
    let padded_dim = tq_padded_dim(dimensions)?;

    let vector_validity = input.validity()?;

    let l2_denorm = tq_normalize_as_l2_denorm(input, ctx)?;
    let normalized = l2_denorm.child_at(0).clone();
    let norms = l2_denorm.child_at(1).clone();

    let normalized_ext = normalized
        .as_opt::<Extension>()
        .ok_or_else(|| vortex_err!("normalized TurboQuant input must be a Vector extension"))?;
    let normalized_fsl: FixedSizeListArray = normalized_ext.storage_array().clone().execute(ctx)?;

    let core = if normalized_fsl.is_empty() {
        empty_quantization(padded_dim)
    } else {
        // SAFETY: `tq_normalize_as_l2_denorm` returned this normalized Vector child.
        unsafe { turboquant_quantize_core(&normalized_fsl, config, ctx)? }
    };
    let codes = build_codes_child(num_vectors, core, vector_validity.clone())?;

    let metadata = TurboQuantMetadata {
        element_ptype,
        dimensions,
        bit_width: config.bit_width(),
        seed: config.seed(),
        num_rounds: config.num_rounds(),
    };
    let storage = build_storage(norms, codes, num_vectors, vector_validity)?;

    Ok(ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage)?.into_array())
}

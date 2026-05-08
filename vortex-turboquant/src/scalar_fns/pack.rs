// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant pack scalar function.

use std::fmt;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ScalarFnArray;
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
use crate::vector::pack::pack_vector;
use crate::vector::tq_padded_dim;
use crate::vtable::TurboQuant;
use crate::vtable::TurboQuantMetadata;
use crate::vtable::tq_storage_dtype;

/// TurboQuant vector pack scalar function.
///
/// `TQPack` itself is a `ScalarFnVTable` and so its options round-trip through expression
/// serialization.
///
/// Unlike `TQUnpack`, it deliberately does **not** implement `ScalarFnArrayVTable` since the
/// persisted artifact would be the original vector array, not the TurboQuant-quantized array.
#[derive(Clone)]
pub struct TQPack;

impl TQPack {
    /// Creates a new [`TypedScalarFnInstance`] wrapping TurboQuant packing.
    pub fn new(config: &TurboQuantConfig) -> TypedScalarFnInstance<TQPack> {
        TypedScalarFnInstance::new(TQPack, config.clone())
    }

    /// Constructs a [`ScalarFnArray`] that lazily packs a `Vector` child into `TurboQuant`.
    pub fn try_new_array(
        child: ArrayRef,
        config: &TurboQuantConfig,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(TQPack::new(config).erased(), vec![child], len)
    }
}

impl ScalarFnVTable for TQPack {
    type Options = TurboQuantConfig;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.turboquant.pack")
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
            _ => unreachable!("TQPack must have exactly one child"),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "tq_pack(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", {options})")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input_dtype = &arg_dtypes[0];
        let vector_metadata = input_dtype
            .as_extension_opt()
            .and_then(|ext_dtype| ext_dtype.metadata_opt::<AnyVector>())
            .ok_or_else(|| {
                vortex_err!("TQPack expects a Vector extension array, got {input_dtype}")
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
        pack_vector(args.get(0)?, options, ctx)
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

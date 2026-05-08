// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant decode scalar function.

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_tensor::vector::AnyVector;
use vortex_tensor::vector::Vector;

use super::metadata::deserialize_config;
use super::metadata::serialize_config;
use crate::TurboQuantConfig;
use crate::vector::decode::decode_vector;
use crate::vtable::TurboQuant;
use crate::vtable::TurboQuantMetadata;
use crate::vtable::tq_metadata;
use crate::vtable::tq_storage_dtype;

/// Lazy TurboQuant vector decode scalar function.
#[derive(Clone)]
pub struct TQDecode;

impl TQDecode {
    /// Creates a new [`TypedScalarFnInstance`] wrapping TurboQuant decoding.
    pub fn new(config: &TurboQuantConfig) -> TypedScalarFnInstance<TQDecode> {
        TypedScalarFnInstance::new(TQDecode, config.clone())
    }

    /// Constructs a [`ScalarFnArray`] that lazily decodes a `TurboQuant` child into a `Vector`.
    pub fn try_new_array(
        child: ArrayRef,
        config: &TurboQuantConfig,
        len: usize,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(TQDecode::new(config).erased(), vec![child], len)
    }
}

impl ScalarFnVTable for TQDecode {
    type Options = TurboQuantConfig;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.turboquant.decode")
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
            0 => ChildName::from("turboquant"),
            _ => unreachable!("TQDecode must have exactly one child"),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "tq_decode(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", {options})")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let child_dtype = &arg_dtypes[0];
        let metadata = tq_metadata(child_dtype)?;
        validate_config_matches_metadata(options, &metadata)?;

        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(
                metadata.element_ptype,
                Nullability::NonNullable,
            )),
            metadata.dimensions,
            child_dtype.nullability(),
        );
        let ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, storage_dtype)?.erased();

        Ok(DType::Extension(ext_dtype))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        decode_vector(args.get(0)?, ctx)
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

impl ScalarFnArrayVTable for TQDecode {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(serialize_config(view.options)))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let options = deserialize_config(metadata)?;
        let vector_metadata = dtype
            .as_extension_opt()
            .and_then(|ext_dtype| ext_dtype.metadata_opt::<AnyVector>())
            .ok_or_else(|| {
                vortex_err!("TQDecode parent dtype must be a Vector extension array, got {dtype}")
            })?;

        let metadata = TurboQuantMetadata {
            element_ptype: vector_metadata.element_ptype(),
            dimensions: vector_metadata.dimensions(),
            bit_width: options.bit_width(),
            seed: options.seed(),
            num_rounds: options.num_rounds(),
        };
        let storage_dtype = tq_storage_dtype(&metadata, dtype.nullability())?;
        let child_dtype =
            DType::Extension(ExtDType::<TurboQuant>::try_new(metadata, storage_dtype)?.erased());
        let child = children.get(0, &child_dtype, len)?;

        Ok(ScalarFnArrayParts {
            options,
            children: vec![child],
        })
    }
}

fn validate_config_matches_metadata(
    config: &TurboQuantConfig,
    metadata: &TurboQuantMetadata,
) -> VortexResult<()> {
    vortex_ensure_eq!(
        config.bit_width(),
        metadata.bit_width,
        "TQDecode config bit_width must match TurboQuant child metadata"
    );
    vortex_ensure_eq!(
        config.seed(),
        metadata.seed,
        "TQDecode config seed must match TurboQuant child metadata"
    );
    vortex_ensure_eq!(
        config.num_rounds(),
        metadata.num_rounds,
        "TQDecode config num_rounds must match TurboQuant child metadata"
    );
    Ok(())
}

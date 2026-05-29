// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant encode scalar function.

use std::fmt;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
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
use crate::config::validate_block_shape;
use crate::config::validate_block_sum;
use crate::vector::quantize::prepare_block_state;
use crate::vector::quantize::turboquant_encode_blocks;
use crate::vector::storage::build_storage;
use crate::vtable::TurboQuant;
use crate::vtable::TurboQuantMetadata;
use crate::vtable::tq_storage_dtype;

/// TurboQuant vector encode scalar function.
///
/// `TQEncode` itself is a `ScalarFnVTable` and so its options round-trip through expression
/// serialization.
///
/// Unlike `TQDecode`, it deliberately does NOT implement `ScalarFnArrayVTable` since the
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
        let block_sizes = resolve_block_sizes(options.block_sizes(), dimensions, false)?;

        let metadata = TurboQuantMetadata {
            element_ptype: vector_metadata.element_ptype(),
            dimensions,
            bit_width: options.bit_width(),
            seed: options.seed(),
            num_rounds: options.num_rounds(),
            block_sizes,
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

/// Encode a `Vector` extension array into a block-decomposed `TurboQuant` extension array.
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

    let dimensions = vector_metadata.dimensions();
    vortex_ensure!(
        dimensions >= MIN_DIMENSION,
        "TurboQuant requires dimension >= {MIN_DIMENSION}, got {dimensions}",
    );

    let block_sizes = resolve_block_sizes(config.block_sizes(), dimensions, true)?;
    let vector_validity = input.validity()?;

    let state = prepare_block_state(
        config.seed(),
        config.num_rounds(),
        config.bit_width(),
        &block_sizes,
    )?;

    // Encode all blocks independently with the TurboQuant quantization algorithm.
    let blocks =
        turboquant_encode_blocks(input, &block_sizes, &state, vector_validity.clone(), ctx)?;

    let storage = build_storage(blocks, &block_sizes, num_vectors, vector_validity)?;
    let metadata = TurboQuantMetadata {
        element_ptype: vector_metadata.element_ptype(),
        dimensions,
        bit_width: config.bit_width(),
        seed: config.seed(),
        num_rounds: config.num_rounds(),
        block_sizes,
    };

    Ok(ExtensionArray::try_new_from_vtable(TurboQuant, metadata, storage)?.into_array())
}

/// Resolve the block list, validate the dim-dependent rules, and emit soft warnings.
///
/// `warn = false` skips the `tracing::warn!` emission so `return_dtype` can be called from
/// places where logging would be noisy.
fn resolve_block_sizes(
    config_block_sizes: Option<&[u32]>,
    dimensions: u32,
    warn: bool,
) -> VortexResult<Vec<u32>> {
    let block_sizes = match config_block_sizes {
        Some(block_sizes) => block_sizes.to_vec(),
        None => vec![dimensions.checked_next_power_of_two().ok_or_else(|| {
            vortex_err!(
                "TurboQuant dimensions {dimensions} overflow u32 when rounded up to a power of two"
            )
        })?],
    };

    // Validate the resolved blocks. This covers the default single-block path, which is not
    // validated at config-construction time, and re-checks user blocks harmlessly. The
    // `sum >= dimensions` coverage rule is enforced by `validate_block_sum` (u64-accumulated).
    validate_block_shape(&block_sizes)?;
    validate_block_sum(&block_sizes, dimensions)?;

    // TODO(connor): We NEED to make sure that this is propagated to any users. Should we just do
    // this unconditionally?
    if warn {
        let sum: u64 = block_sizes.iter().map(|&block| block as u64).sum();
        let mut covered: u32 = 0;
        for (index, &block) in block_sizes.iter().enumerate() {
            if covered >= dimensions {
                tracing::warn!(
                    block_index = index,
                    block = block,
                    dimensions = dimensions,
                    "TurboQuant block lies entirely past dimensions; it will only store \
                     padding-derived codes"
                );
            }
            covered = covered.saturating_add(block);
        }

        if sum > (dimensions as u64).saturating_mul(2) {
            tracing::warn!(
                sum = sum,
                dimensions = dimensions,
                "TurboQuant block_sizes sum exceeds 2 * dimensions; significant padding overhead"
            );
        }
    }

    Ok(block_sizes)
}

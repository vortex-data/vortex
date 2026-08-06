// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! L2 norm expression for tensor-like types.

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ScalarFn as ScalarFnArrayEncoding;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::ElementSink;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::serde::ArrayChildren;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::encodings::normalized::Normalized;
use crate::scalar_fns::row::TensorRow;
use crate::scalar_fns::row::tensor_element_ptype;
use crate::utils::extract_normalized_children;
use crate::utils::l2_norm_row;
use crate::utils::validate_tensor_float_input;

/// L2 norm (Euclidean norm) of a tensor or vector column.
///
/// Computes `||v|| = sqrt(sum(v_i^2))` over the flat backing buffer of each tensor-like type.
///
/// The input must be a tensor-like extension array with a float element type. The output is a float
/// column of the same float type.
///
/// When the input is [`Normalized`]-encoded, this operator treats the stored norms as
/// authoritative. For lossy normalized children, that means `L2Norm` intentionally reads the
/// stored norms instead of re-deriving them from fully decoded coordinates. That behavior is part
/// of the storage contract, not a separate lossy-compute mode.
///
/// [`Normalized`]: crate::encodings::normalized::Normalized
#[derive(Clone, Debug, Default)]
pub struct L2Norm;

impl RowFn for L2Norm {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.l2_norm");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        match_each_float_ptype!(tensor_element_ptype(args)?, |T| {
            visitor.visit_prepared_into::<(TensorRow<T>,), ElementSink<T>, _, _>(
                |_| (),
                |&(), (row,), output| *output = l2_norm_row(row),
            )
        })
    }

    /// `L2Norm` over a [`Normalized`]-encoded column is defined to read back the authoritative
    /// stored norms. Callers of lossy encodings opt into that storage semantics instead of forcing
    /// a decode-and-recompute path here.
    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let input = &args[0];
        if !input.is::<Normalized>() {
            return Ok(None);
        }
        let element_ptype = validate_tensor_float_input(input.dtype())?.element_ptype();
        let (_, norms) = extract_normalized_children(input);
        vortex_ensure_eq!(norms.dtype().as_ptype(), element_ptype);
        Ok(Some(norms))
    }
}

/// Metadata for a serialized [`L2Norm`] array: the single `input` child's [`DType`], which carries
/// the extension type (`FixedShapeTensor` vs `Vector`), dimension, and nullability that are not
/// recoverable from the parent's primitive-float output.
#[derive(Clone, prost::Message)]
pub(super) struct L2NormMetadata {
    #[prost(message, optional, tag = "1")]
    input_dtype: Option<pb::DType>,
}

impl ScalarFnArrayVTable for L2Norm {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let scalar_fn_array = view.as_::<ScalarFnArrayEncoding>();
        let input_dtype = Some(scalar_fn_array.child_at(0).dtype().try_into()?);
        Ok(Some(L2NormMetadata { input_dtype }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let metadata = L2NormMetadata::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode L2NormMetadata: {e}"))?;
        let input_pb = metadata
            .input_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("L2NormMetadata missing input_dtype"))?;
        let input_dtype = DType::from_proto(input_pb, session)?;
        let child = children.get(0, &input_dtype, len)?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: vec![child],
        })
    }
}

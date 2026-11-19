// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.protobuf.encode` scalar function, which serializes messages in the wire format.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Protobuf;
use crate::encode::encode_messages;
use crate::encode::encode_value;

/// Serializes each message of a protobuf extension array in the protobuf wire format, as
/// [`encode_messages`] does.
///
/// The result is `binary` with the nullability of the input, and null where the message is null.
/// Encoding fails for a message larger than a binary value can hold, and for a JSON value in a
/// dedicated form that is not valid JSON of its message type.
///
/// [`protobuf_encode`] applies the function to a message expression.
#[derive(Clone, Debug)]
pub struct Encode;

impl ScalarFnVTable for Encode {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.protobuf.encode");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(Vec::new()))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("messages"),
            _ => unreachable!("Invalid child index {child_idx} for protobuf encode()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        match &args[0] {
            DType::Extension(ext_dtype) if ext_dtype.is::<Protobuf>() => {
                Ok(DType::Binary(ext_dtype.nullability()))
            }
            other => vortex_bail!(
                "Encoding protobuf messages requires a {} input, got {other}",
                Protobuf::ID
            ),
        }
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let messages = args.get(0)?;
        if let Some(message) = messages.as_constant() {
            return Ok(ConstantArray::new(encode_scalar(message)?, args.row_count()).into_array());
        }
        Ok(encode_messages(&messages, ctx)?.into_array())
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Returns an expression that serializes each message in `message` in the protobuf wire format,
/// with [`Encode`].
///
/// `message` must evaluate to a protobuf extension array. The expression reads whole messages,
/// so a scan reads every storage column of the message.
///
/// ```
/// use vortex_array::expr::root;
/// use vortex_protobuf::protobuf_encode;
///
/// let expr = protobuf_encode(root());
/// assert_eq!(expr.to_string(), "vortex.protobuf.encode($)");
/// ```
pub fn protobuf_encode(message: Expression) -> Expression {
    Encode.new_expr(EmptyOptions, [message])
}

/// Encodes one protobuf message scalar, which gives a null scalar for a null message.
fn encode_scalar(message: &Scalar) -> VortexResult<Scalar> {
    let Some(metadata) = message
        .dtype()
        .as_extension_opt()
        .and_then(|ext_dtype| ext_dtype.metadata_opt::<Protobuf>())
    else {
        vortex_bail!(
            "Encoding protobuf messages requires a {} input, got {}",
            Protobuf::ID,
            message.dtype()
        );
    };
    let nullability = message.dtype().nullability();
    Ok(match message.value() {
        Some(storage) => Scalar::binary(encode_value(metadata.plan(), storage)?, nullability),
        None => Scalar::null(DType::Binary(nullability)),
    })
}

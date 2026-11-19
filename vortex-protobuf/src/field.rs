// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expressions that read message fields: single fields, the names of enum values and the set
//! member of a oneof.

use vortex_array::expr::Expression;
use vortex_array::expr::ext_storage;
use vortex_array::expr::get_item;
use vortex_array::expr::is_not_null;
use vortex_array::expr::lit;
use vortex_array::expr::nested_case_when;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::EnumName;
use crate::EnumNameOptions;
use crate::ProtobufMetadata;
use crate::mapping::FieldPlan;
use crate::mapping::FieldShape;
use crate::mapping::MessagePlan;
use crate::mapping::ValuePlan;

/// Returns an expression that reads the field at the dotted `path` of each message in `message`.
///
/// `message` must evaluate to a protobuf extension array. Each segment of `path` names a field of
/// the message selected by the segments before it, so `"inner.name"` reads the `name` field of the
/// `inner` message field. The expression reads the extension storage once and then selects struct
/// fields, as in `get_item("name", get_item("inner", ext_storage(message)))`, so a scan only needs
/// the columns of the selected field.
///
/// The result has the storage dtype of the field, made nullable when the field or any message on
/// the path is nullable. Binding the expression fails if a segment does not name a field, or names
/// a field inside a repeated or map field.
///
/// ```
/// use vortex_array::expr::root;
/// use vortex_protobuf::protobuf_field;
///
/// let expr = protobuf_field("inner.name", root());
/// assert_eq!(expr.to_string(), "vortex.ext.storage($).inner.name");
/// ```
pub fn protobuf_field(path: &str, message: Expression) -> Expression {
    path.split('.')
        .fold(ext_storage(message), |expr, field| get_item(field, expr))
}

/// Returns an expression that reads the names of the enum values of the field at the dotted
/// `path`, as [`protobuf_field`] reads the field.
///
/// `metadata` is the metadata of the protobuf dtype of `message`, and the field must be a singular
/// enum field. The result is nullable `utf8`, which is null where the field is null or holds a
/// number that the enum does not define. [`EnumName`] describes the mapping.
///
/// # Errors
///
/// Returns an error if `path` does not lead through singular message fields stored as structs to
/// a singular enum field.
pub fn protobuf_enum_name(
    metadata: &ProtobufMetadata,
    path: &str,
    message: Expression,
) -> VortexResult<Expression> {
    let (parent, name) = split_path(path);
    let field = find_field(find_message(metadata.plan(), parent)?, name)?;
    let FieldShape::Single(ValuePlan::Scalar {
        enum_type: Some(enum_type),
        ..
    }) = &field.shape
    else {
        vortex_bail!("Protobuf field {path} is not a singular enum field");
    };
    Ok(EnumName.new_expr(
        EnumNameOptions::new(enum_type),
        [protobuf_field(path, message)],
    ))
}

/// Returns an expression that reads which member of the oneof at the dotted `path` is set: the
/// member's field name, or null where no member is set.
///
/// The last segment of `path` names a oneof of the message that the segments before it select,
/// as in [`protobuf_field`], so `"inner.choice"` reads the `choice` oneof of the `inner` message
/// field. `metadata` is the metadata of the protobuf dtype of `message`. The result is nullable
/// `utf8`, null also where a message on the path is null. The expression checks the members in
/// declaration order with `case_when` over `is_not_null`, so a scan reads the columns of all
/// members.
///
/// # Errors
///
/// Returns an error if `path` does not lead through singular message fields stored as structs to
/// a oneof.
pub fn protobuf_oneof_case(
    metadata: &ProtobufMetadata,
    path: &str,
    message: Expression,
) -> VortexResult<Expression> {
    let (parent, name) = split_path(path);
    let plan = find_message(metadata.plan(), parent)?;
    let oneof = plan
        .descriptor
        .oneofs()
        .find(|oneof| oneof.name() == name)
        .ok_or_else(|| {
            vortex_err!(
                "Protobuf message {} has no oneof {name}",
                plan.descriptor.full_name()
            )
        })?;
    let cases = oneof
        .fields()
        .map(|member| {
            let member_path = match parent {
                Some(parent) => format!("{parent}.{}", member.name()),
                None => member.name().to_owned(),
            };
            (
                is_not_null(protobuf_field(&member_path, message.clone())),
                lit(member.name()),
            )
        })
        .collect::<Vec<_>>();
    Ok(nested_case_when(cases, None))
}

/// Splits a dotted path into the path of the parent message, if any, and the last segment.
fn split_path(path: &str) -> (Option<&str>, &str) {
    match path.rsplit_once('.') {
        Some((parent, name)) => (Some(parent), name),
        None => (None, path),
    }
}

/// Returns the plan of the message at the dotted `path` of singular message fields, or `root`
/// without a path.
fn find_message<'a>(root: &'a MessagePlan, path: Option<&str>) -> VortexResult<&'a MessagePlan> {
    path.into_iter()
        .flat_map(|path| path.split('.'))
        .try_fold(root, |message, name| {
            match &find_field(message, name)?.shape {
                FieldShape::Single(ValuePlan::Message(message)) => Ok(message.as_ref()),
                _ => vortex_bail!(
                    "Protobuf field {name} of {} is not a singular message stored as a struct",
                    message.descriptor.full_name()
                ),
            }
        })
}

/// Returns the plan of the field `name` of `message`.
fn find_field<'a>(message: &'a MessagePlan, name: &str) -> VortexResult<&'a FieldPlan> {
    message
        .fields
        .iter()
        .find(|field| field.descriptor.name() == name)
        .ok_or_else(|| {
            vortex_err!(
                "Protobuf message {} has no field {name}",
                message.descriptor.full_name()
            )
        })
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expressions that read single message fields.

use vortex_array::expr::Expression;
use vortex_array::expr::ext_storage;
use vortex_array::expr::get_item;

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

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions converting between JSON strings and Variant values.

mod json_to_variant;
mod variant_to_json;

pub use json_to_variant::JsonToVariant;
pub use json_to_variant::JsonToVariantOptions;
pub use json_to_variant::ShreddingSpec;
pub use variant_to_json::VariantToJson;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;

/// Creates a [`JsonToVariant`] expression that parses `child`'s JSON strings into Variant
/// values, shredding the paths selected by `shredding`.
///
/// `child` must produce `Utf8` or [`Json`](vortex_json::Json) extension values; the result is
/// `Variant` with the input's nullability. Rows containing invalid JSON fail the expression.
///
/// Note that this is NOT an inverse of [`variant_to_json()`]: both conversions normalize their
/// input. See [`JsonToVariant`] for the full list of caveats.
pub fn json_to_variant(child: Expression, shredding: ShreddingSpec) -> Expression {
    JsonToVariant.new_expr(JsonToVariantOptions::new(shredding), [child])
}

/// Creates a [`VariantToJson`] expression that renders `child`'s Variant values as JSON
/// strings with the [`Json`](vortex_json::Json) extension dtype.
///
/// Shredded inputs are unshredded before rendering, and the result keeps the input's
/// nullability.
///
/// Note that this is NOT an inverse of [`json_to_variant()`]: both conversions normalize their
/// input, and Variant-only types (dates, timestamps, UUIDs, binary, decimals) are rendered as
/// plain JSON strings or numbers. See [`VariantToJson`] for the full list of caveats.
pub fn variant_to_json(child: Expression) -> Expression {
    VariantToJson.new_expr(EmptyOptions, [child])
}

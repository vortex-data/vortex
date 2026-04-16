// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extract a query vector from a cosine-similarity expression.
//!
//! The IVF reader uses this to detect queries of the shape
//! `Binary(Cmp, [CosineSimilarity(root, literal_query), threshold])` (or its symmetric variant).
//! Only the literal query vector is needed — the threshold comparison is handled by the data
//! child's filter evaluation.

use vortex_array::expr::Expression;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;

/// If `expr` contains a cosine-similarity expression against a constant query vector,
/// extract the query as a `Vec<f32>`. Returns `None` otherwise.
///
/// Matches shapes like:
/// - `CosineSimilarity(root, literal)` inside any binary comparison
/// - `literal` and `root` operands may appear on either side
pub fn extract_cosine_query(expr: &Expression) -> Option<Vec<f32>> {
    let cosine = find_cosine_similarity(expr)?;
    // cosine has two children; one should be `root()` and the other a literal.
    let lhs = cosine.child(0);
    let rhs = cosine.child(1);
    extract_literal_vector(lhs).or_else(|| extract_literal_vector(rhs))
}

fn find_cosine_similarity(expr: &Expression) -> Option<&Expression> {
    if expr.is::<CosineSimilarity>() {
        return Some(expr);
    }
    for child in expr.children().iter() {
        if let Some(found) = find_cosine_similarity(child) {
            return Some(found);
        }
    }
    None
}

/// Try to extract a constant query vector from an expression.
///
/// Returns `None` if the expression is not a `literal` carrying an extension value backed by a
/// `FixedSizeList<f32>` or `FixedSizeList<f64>` representing a single vector.
fn extract_literal_vector(expr: &Expression) -> Option<Vec<f32>> {
    let lit_ref = expr.downcast_ref::<Literal>()?;
    let scalar: &Scalar = lit_ref.options();
    extract_vector_from_scalar(scalar)
}

/// Extract a flat f32 vector from a `Scalar` that stores a `Vector<dim, float>` extension value.
///
/// The representation is: extension scalar whose storage scalar is a fixed-size list of floats.
/// We unwrap the extension to get the storage scalar, then iterate its list elements.
fn extract_vector_from_scalar(scalar: &Scalar) -> Option<Vec<f32>> {
    // The scalar might be an extension scalar — unwrap it to its storage representation.
    let inner: Scalar = if let Some(ext) = scalar.as_extension_opt() {
        ext.to_storage_scalar()
    } else {
        scalar.clone()
    };

    let list = inner.as_list_opt()?;
    let len = list.len();
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let elem = list.element(i)?;
        let prim = elem.as_primitive_opt()?;
        let value = prim.typed_value::<f32>().or_else(|| {
            // Try f64 and convert (e.g. OpenAI vectors are f64).
            #[expect(clippy::cast_possible_truncation, reason = "IVF operates in f32")]
            prim.typed_value::<f64>().map(|v| v as f32)
        })?;
        out.push(value);
    }
    Some(out)
}

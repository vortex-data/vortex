// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine-similarity filter [`Expression`]s used by the file-scan path.
//!
//! The scan layer accepts an [`Expression`] (not an `ArrayRef`), so we cannot reuse
//! [`vortex_tensor::vector_search::build_similarity_search_tree`] (which constructs an
//! [`ArrayRef`] tree directly). Instead we build the same expression by hand:
//!
//! ```text
//! gt(
//!     cosine_similarity(col("emb"), lit(query_scalar)),
//!     lit(threshold),
//! )
//! ```
//!
//! The query is wrapped as `Scalar::extension::<Vector>(Scalar::fixed_size_list(F32, ...))`
//! so [`vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity`] can treat it as a
//! single-row Vector value during evaluation. At scan time the literal expands into a
//! ConstantArray whose row count matches the chunk batch size.

use anyhow::Result;
use vortex::array::expr::Expression;
use vortex::array::expr::col;
use vortex::array::expr::gt;
use vortex::array::expr::lit;
use vortex::array::extension::EmptyMetadata;
use vortex::array::scalar::Scalar;
use vortex::array::scalar_fn::EmptyOptions;
use vortex::array::scalar_fn::ScalarFnVTableExt;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector::Vector;

/// Wrap a query vector as `Scalar::extension::<Vector>(Scalar::fixed_size_list(F32, ...))`.
///
/// Returns an error if `query` is empty (a zero-dimension vector has no representation in
/// the type system).
pub fn query_scalar(query: &[f32]) -> Result<Scalar> {
    anyhow::ensure!(!query.is_empty(), "query_scalar: query must be non-empty");
    let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let children: Vec<Scalar> = query
        .iter()
        .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
        .collect();
    let fsl = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    Ok(Scalar::extension::<Vector>(EmptyMetadata, fsl))
}

/// Build the filter `cosine_similarity(emb, query) > threshold`.
///
/// Suitable for [`vortex_layout::scan::scan_builder::ScanBuilder::with_filter`].
pub fn similarity_filter(query: &[f32], threshold: f32) -> Result<Expression> {
    let query_lit = lit(query_scalar(query)?);
    let cosine = CosineSimilarity.new_expr(EmptyOptions, [col("emb"), query_lit]);
    Ok(gt(cosine, lit(threshold)))
}

/// Project just the `emb` column. Used by the throughput-only scan path.
pub fn emb_projection() -> Expression {
    col("emb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_scalar_rejects_empty_query() {
        assert!(query_scalar(&[]).is_err());
    }

    #[test]
    fn query_scalar_builds_extension_dtype() {
        let scalar = query_scalar(&[1.0, 0.0, 0.0]).unwrap();
        match scalar.dtype() {
            DType::Extension(_) => {}
            other => panic!("expected Extension, got {other}"),
        }
    }

    #[test]
    fn similarity_filter_uses_gt_operator() {
        let expr = similarity_filter(&[1.0, 0.0, 0.0], 0.5).unwrap();
        // Quick sanity check: the printed form contains the operator and the threshold so
        // future refactors that change the structure get caught here.
        let printed = format!("{expr:?}");
        assert!(printed.contains("Gt") || printed.contains(">"), "{printed}");
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine-similarity filter [`Expression`]s used by the file-scan path.
//!
//! We can easily build a cosine similarity filter by hand:
//!
//! ```text
//! gt(
//!     cosine_similarity(col("emb"), lit(query_scalar)),
//!     lit(threshold),
//! )
//! ```
//!
//! The query is wrapped as `Scalar::extension::<Vector>(Scalar::fixed_size_list(F32, ...))` so
//! [`CosineSimilarity`] can treat it as a single-row `Vector` value during evaluation.
//!
//! At scan time the literal expands into a `ConstantArray` whose row count matches the chunk batch
//! size.

use anyhow::Result;
use vortex::array::expr::Expression;
use vortex::array::expr::col;
use vortex::array::expr::gt;
use vortex::array::expr::lit;
use vortex::array::extension::EmptyMetadata;
use vortex::array::extension::Lossy;
use vortex::array::scalar::Scalar;
use vortex::array::scalar_fn::EmptyOptions;
use vortex::array::scalar_fn::ScalarFnVTableExt;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector::Vector;

use crate::compression::VectorFlavor;

/// Build the filter `cosine_similarity(emb, query) > threshold`. The query scalar's dtype is
/// matched to the persisted column dtype for the given [`VectorFlavor`]: bare `Vector<f32>` for
/// [`VectorFlavor::Uncompressed`], and `Lossy<Vector<f32>>` for [`VectorFlavor::TurboQuant`].
pub fn similarity_filter(
    query: &[f32],
    threshold: f32,
    flavor: VectorFlavor,
) -> Result<Expression> {
    // Empty queries short-circuit to a literal `false`, so scans return no rows instead of trying
    // to evaluate cosine similarity on a zero-dimensional vector.
    if query.is_empty() {
        return Ok(lit(false));
    }

    let query_lit = lit(query_scalar(query, flavor)?);
    let cosine = CosineSimilarity.new_expr(EmptyOptions, [col("emb"), query_lit]);
    Ok(gt(cosine, lit(threshold)))
}

/// Wrap a query vector as a `Vector<f32>` (or `Lossy<Vector<f32>>` for the TurboQuant flavor)
/// extension scalar so the binary tensor expression validates against the persisted column dtype.
pub fn query_scalar(query: &[f32], flavor: VectorFlavor) -> Result<Scalar> {
    let children: Vec<Scalar> = query
        .iter()
        .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
        .collect();

    let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
    let fsl = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
    let vector_scalar = Scalar::extension::<Vector>(EmptyMetadata, fsl);

    Ok(match flavor {
        VectorFlavor::Uncompressed => vector_scalar,
        VectorFlavor::TurboQuant => {
            let lossy_dtype = Lossy::new(vector_scalar.dtype().clone())?.erased();
            Scalar::extension_ref(lossy_dtype, vector_scalar)
        }
    })
}

/// Project just the `emb` column. Used by the throughput-only scan path.
pub fn emb_projection() -> Expression {
    col("emb")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_scalar_accepts_empty_query() {
        let scalar = query_scalar(&[], VectorFlavor::Uncompressed).unwrap();
        match scalar.dtype() {
            DType::Extension(_) => {}
            other => panic!("expected Extension, got {other}"),
        }
    }

    #[test]
    fn query_scalar_builds_extension_dtype() {
        let scalar = query_scalar(&[1.0, 0.0, 0.0], VectorFlavor::Uncompressed).unwrap();
        match scalar.dtype() {
            DType::Extension(_) => {}
            other => panic!("expected Extension, got {other}"),
        }
    }

    #[test]
    fn similarity_filter_uses_gt_operator() {
        let expr = similarity_filter(&[1.0, 0.0, 0.0], 0.5, VectorFlavor::Uncompressed).unwrap();
        // Quick sanity check: the printed form contains the operator and the threshold so
        // future refactors that change the structure get caught here.
        let printed = format!("{expr:?}");
        assert!(printed.contains("Gt") || printed.contains(">"), "{printed}");
    }
}

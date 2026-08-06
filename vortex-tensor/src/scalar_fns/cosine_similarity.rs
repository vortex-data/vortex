// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity between two tensor columns.

use num_traits::Float;
use num_traits::Zero;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::ElementSink;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::encodings::normalized::NormalizedOrientation;
use crate::scalar_fns::inner_product::InnerProduct;
use crate::scalar_fns::l2_norm::L2Norm;
use crate::scalar_fns::row::TensorRow;
#[cfg(test)]
use crate::scalar_fns::row::probe;
use crate::scalar_fns::row::tensor_element_ptype;
use crate::utils::BinaryTensorOpMetadata;
use crate::utils::extract_normalized_children;
use crate::utils::l2_norm_row;

/// Cosine similarity between two columns.
///
/// Computes `dot(a, b) / (||a|| * ||b||)` over the flat backing buffer of each tensor or vector.
/// The shape and permutation do not affect the result because cosine similarity only depends on the
/// element values, not their logical arrangement. A zero norm on either side yields `0.0`.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// When either input is [`Normalized`]-encoded, this operator treats the stored norms and
/// normalized children as authoritative. For lossy normalized children, that means the optimized
/// read-through path may intentionally differ slightly from decoding both sides to dense
/// coordinates and recomputing cosine from scratch.
///
/// [`FixedShapeTensor`]: crate::fixed_shape_tensor::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
/// [`Normalized`]: crate::encodings::normalized::Normalized
#[derive(Clone, Debug, Default)]
pub struct CosineSimilarity;

impl RowFn for CosineSimilarity {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.tensor.cosine_similarity");
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
            visitor.visit_prepared_into::<(TensorRow<T>, TensorRow<T>), ElementSink<T>, _, _>(
                |(lhs, rhs)| {
                    #[cfg(test)]
                    probe::record(lhs.is_some(), rhs.is_some());
                    ConstNorms {
                        lhs: lhs.map(l2_norm_row),
                        rhs: rhs.map(l2_norm_row),
                    }
                },
                |norms, (lhs, rhs), output| {
                    *output = cosine_similarity_row_prepared(norms, lhs, rhs);
                },
            )
        })
    }

    /// [`Normalized`]-encoded operands make the *stored* norms and normalized children
    /// authoritative: `cos(D(x, s), D(y, t)) = dot(x, y)` and `cos(D(x, s), y) = dot(x, y) /
    /// ||y||`, in both cases forced to `0.0` on rows where any authoritative norm is `0.0` (even
    /// for lossy children whose decoded coordinates are nonzero).
    ///
    /// [`Normalized`]: crate::encodings::normalized::Normalized
    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let lhs = args[0].clone();
        let rhs = args[1].clone();

        match NormalizedOrientation::classify(&lhs, &rhs) {
            NormalizedOrientation::Both { lhs, rhs } => {
                cosine_both_normalized(lhs, rhs, ctx).map(Some)
            }
            NormalizedOrientation::One {
                normalized_array,
                plain,
            } => cosine_one_normalized(normalized_array, plain, ctx).map(Some),
            NormalizedOrientation::Neither => Ok(None),
        }
    }
}

impl ScalarFnArrayVTable for CosineSimilarity {
    fn serialize(
        &self,
        view: &ScalarFnArrayView<Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(BinaryTensorOpMetadata::encode_from_view(view)?))
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        len: usize,
        metadata: &[u8],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ScalarFnArrayParts<Self>> {
        let reconstructed =
            BinaryTensorOpMetadata::decode_children(metadata, len, children, session)?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: reconstructed,
        })
    }
}

/// Per-batch state for the cosine row kernel: the L2 norm of each operand that is constant for
/// the batch.
///
/// A broadcast query vector holds the same elements in every row, so its norm is the same in
/// every row too. Computing it in the prepare step hoists an `O(width)` pass and a `sqrt` per row
/// out of the row loop. `None` marks an operand that varies by row, whose norm the row closure
/// computes exactly as it did before the hoist.
struct ConstNorms<T> {
    /// The norm of the lhs when it is batch-constant.
    lhs: Option<T>,

    /// The norm of the rhs when it is batch-constant.
    rhs: Option<T>,
}

/// Computes the cosine similarity of one row, taking any hoisted norm from `norms` and computing
/// the rest exactly as [`cosine_similarity_row`] does.
///
/// Each arm accumulates the same values in the same order as [`cosine_similarity_row`], and the
/// denominator keeps its lhs-times-rhs order, so the result is bit-identical whether a norm was
/// hoisted or not. The match costs one predictable branch per row: the arm is the same for the
/// whole batch.
fn cosine_similarity_row_prepared<T: Float + NativePType>(
    norms: &ConstNorms<T>,
    a: &[T],
    b: &[T],
) -> T {
    match (norms.lhs, norms.rhs) {
        (None, None) => cosine_similarity_row(a, b),
        (Some(norm_a), None) => {
            let mut dot = T::zero();
            let mut norm_sq_b = T::zero();
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot = dot + x * y;
                norm_sq_b = norm_sq_b + y * y;
            }
            cosine_from_parts(dot, norm_a * norm_sq_b.sqrt())
        }
        (None, Some(norm_b)) => {
            let mut dot = T::zero();
            let mut norm_sq_a = T::zero();
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot = dot + x * y;
                norm_sq_a = norm_sq_a + x * x;
            }
            cosine_from_parts(dot, norm_sq_a.sqrt() * norm_b)
        }
        (Some(norm_a), Some(norm_b)) => {
            let mut dot = T::zero();
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot = dot + x * y;
            }
            cosine_from_parts(dot, norm_a * norm_b)
        }
    }
}

/// Computes the cosine similarity of two equal-length float slices.
///
/// Returns `dot(a, b) / (||a|| * ||b||)`, or `0.0` when either norm is zero.
fn cosine_similarity_row<T: Float + NativePType>(a: &[T], b: &[T]) -> T {
    let mut dot = T::zero();
    let mut norm_sq_a = T::zero();
    let mut norm_sq_b = T::zero();
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot = dot + x * y;
        norm_sq_a = norm_sq_a + x * x;
        norm_sq_b = norm_sq_b + y * y;
    }

    cosine_from_parts(dot, norm_sq_a.sqrt() * norm_sq_b.sqrt())
}

/// The shared tail of every cosine arm: `dot / denom`, guarded to `0.0` when the denominator is
/// zero.
fn cosine_from_parts<T: Float>(dot: T, denom: T) -> T {
    if denom == T::zero() {
        T::zero()
    } else {
        dot / denom
    }
}

/// Both sides are [`Normalized`]-encoded: the normalized children are authoritative, so their dot
/// product is the cosine similarity, except that a row with a zero *stored* norm is a zero vector.
///
/// Unlike [`InnerProduct::reduce_encoded`], which composes lazy `Mul` arrays over the norm columns,
/// this executes and materializes. The zero-norm guard is a conditional per row rather than an
/// arithmetic factor, so there is no lazy array that expresses it; the norm columns are one value
/// per row rather than one per coordinate, so materializing them is cheap next to the decode this
/// avoids.
///
/// [`InnerProduct::reduce_encoded`]: InnerProduct::reduce_encoded
/// [`Normalized`]: crate::encodings::normalized::Normalized
fn cosine_both_normalized(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let (normalized_l, norms_l) = extract_normalized_children(lhs);
    let (normalized_r, norms_r) = extract_normalized_children(rhs);

    let dot: PrimitiveArray = InnerProduct
        .try_new_array(len, EmptyOptions, [normalized_l, normalized_r])?
        .execute(ctx)?;
    let norms_l: PrimitiveArray = norms_l.execute(ctx)?;
    let norms_r: PrimitiveArray = norms_r.execute(ctx)?;

    match_each_float_ptype!(dot.ptype(), |T| {
        let dots = dot.as_slice::<T>();
        let norms_l = norms_l.as_slice::<T>();
        let norms_r = norms_r.as_slice::<T>();
        // Zipped rather than indexed by `0..len`: one bounds check per iterator instead of three
        // per row. A length disagreement between the children shortens the result, which the
        // lifting reports against the batch row count rather than panicking mid-loop.
        let buffer: Buffer<T> = dots
            .iter()
            .zip(norms_l)
            .zip(norms_r)
            .map(|((&dot, &norm_l), &norm_r)| {
                if norm_l.is_zero() || norm_r.is_zero() {
                    T::zero()
                } else {
                    dot
                }
            })
            .collect();

        Ok(PrimitiveArray::new(buffer, Validity::NonNullable).into_array())
    })
}

/// One side is [`Normalized`]-encoded: `cos = dot(normalized, plain) / ||plain||`, forced to `0.0`
/// on rows where the stored norm or the plain norm is `0.0`.
///
/// [`Normalized`]: crate::encodings::normalized::Normalized
fn cosine_one_normalized(
    normalized_array: &ArrayRef,
    plain: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = normalized_array.len();
    let (normalized, normalized_norms) = extract_normalized_children(normalized_array);

    let dot: PrimitiveArray = InnerProduct
        .try_new_array(len, EmptyOptions, [normalized, plain.clone()])?
        .execute(ctx)?;
    let normalized_norms: PrimitiveArray = normalized_norms.execute(ctx)?;
    let plain_norm: PrimitiveArray = L2Norm
        .try_new_array(len, EmptyOptions, [plain.clone()])?
        .execute(ctx)?;

    match_each_float_ptype!(dot.ptype(), |T| {
        let dots = dot.as_slice::<T>();
        let normalized_norms = normalized_norms.as_slice::<T>();
        let plain_norms = plain_norm.as_slice::<T>();
        // Zipped for the same reason as [`cosine_both_normalized`].
        let buffer: Buffer<T> = dots
            .iter()
            .zip(normalized_norms)
            .zip(plain_norms)
            .map(|((&dot, &stored_norm), &plain_norm)| {
                if stored_norm.is_zero() || plain_norm.is_zero() {
                    T::zero()
                } else {
                    dot / plain_norm
                }
            })
            .collect();

        Ok(PrimitiveArray::new(buffer, Validity::NonNullable).into_array())
    })
}

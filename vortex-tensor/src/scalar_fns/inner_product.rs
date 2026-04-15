// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inner product expression for tensor-like types.

use std::fmt::Formatter;
use std::sync::Arc;

use num_traits::Float;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::ScalarFnVTable as ScalarFnArrayEncoding;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayParts;
use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::extension::EmptyMetadata;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFn;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::serde::ArrayChildren;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::matcher::AnyTensor;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::sorf_transform::SorfMatrix;
use crate::scalar_fns::sorf_transform::SorfTransform;
use crate::utils::extract_flat_elements;
use crate::utils::extract_l2_denorm_children;
use crate::vector::Vector;

/// Inner product (dot product) between two columns.
///
/// Computes `sum(a_i * b_i)` over the flat backing buffer of each tensor or vector. For vectors
/// this is the standard dot product; for higher-rank ([`FixedShapeTensor`]) arrays this is the
/// Frobenius inner product.
///
/// Both inputs must be tensor-like extension arrays ([`FixedShapeTensor`] or [`Vector`]) with the
/// same dtype and a float element type. The output is a float column of the same float type.
///
/// [`FixedShapeTensor`]: crate::fixed_shape::FixedShapeTensor
/// [`Vector`]: crate::vector::Vector
#[derive(Clone)]
pub struct InnerProduct;

impl InnerProduct {
    /// Creates a new [`ScalarFn`] wrapping the inner product operation.
    pub fn new() -> ScalarFn<InnerProduct> {
        ScalarFn::new(InnerProduct, EmptyOptions)
    }

    /// Constructs a [`ScalarFnArray`] that lazily computes the inner product between `lhs` and
    /// `rhs`.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`ScalarFnArray`] cannot be constructed (e.g. due to dtype
    /// mismatches).
    pub fn try_new_array(lhs: ArrayRef, rhs: ArrayRef, len: usize) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(InnerProduct::new().erased(), vec![lhs, rhs], len)
    }
}

impl ScalarFnVTable for InnerProduct {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.tensor.inner_product")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("InnerProduct must have exactly two children"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "inner_product(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        // Both must have the same dtype (ignoring top-level nullability).
        vortex_ensure!(
            lhs.eq_ignore_nullability(rhs),
            "InnerProduct requires both inputs to have the same dtype, got {lhs} and {rhs}"
        );

        // Both inputs must be tensor-like extension types.
        let lhs_ext = lhs
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("InnerProduct lhs must be an extension type, got {lhs}"))?;

        vortex_ensure!(
            lhs_ext.is::<AnyTensor>(),
            "InnerProduct inputs must be an `AnyTensor`, got {lhs}"
        );

        let tensor_match = lhs_ext
            .metadata_opt::<AnyTensor>()
            .ok_or_else(|| vortex_err!("InnerProduct inputs must be an `AnyTensor`, got {lhs}"))?;
        let ptype = tensor_match.element_ptype();
        // TODO(connor): This should support integer tensors!
        vortex_ensure!(
            ptype.is_float(),
            "InnerProduct element dtype must be a float primitive, got {ptype}"
        );

        let nullability = Nullability::from(lhs.is_nullable() || rhs.is_nullable());
        Ok(DType::Primitive(ptype, nullability))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let mut lhs_ref = args.get(0)?;
        let mut rhs_ref = args.get(1)?;
        let len = args.row_count();

        // Check if any of our children have be already normalized.
        {
            let lhs_is_denorm = lhs_ref.is::<ExactScalarFn<L2Denorm>>();
            let rhs_is_denorm = rhs_ref.is::<ExactScalarFn<L2Denorm>>();

            if lhs_is_denorm && rhs_is_denorm {
                return self.execute_both_denorm(&lhs_ref, &rhs_ref, len, ctx);
            } else if lhs_is_denorm || rhs_is_denorm {
                if rhs_is_denorm {
                    (lhs_ref, rhs_ref) = (rhs_ref, lhs_ref);
                }
                return self.execute_one_denorm(&lhs_ref, &rhs_ref, len, ctx);
            }
        }

        // Reduction case 1: `InnerProduct(SorfTransform(x), const)` rewrites to
        // `InnerProduct(x, forward_rotate(zero_pad(const)))`. Re-executes recursively so
        // case 2 can fire on the rewritten tree.
        if let Some(rewritten) = self.try_execute_sorf_constant(&lhs_ref, &rhs_ref, len, ctx)? {
            return Ok(rewritten);
        }

        // Reduction case 2: `InnerProduct(Vector[FSL(Dict(u8, f32))], const)` is computed by
        // gather-summing `q[j] * values[codes[j] as usize]` per row, reading the codebook
        // directly instead of decoding the column into dense vectors.
        if let Some(result) = self.try_execute_dict_constant(&lhs_ref, &rhs_ref, len, ctx)? {
            return Ok(result);
        }

        // Compute combined validity.
        let validity = lhs_ref.validity()?.and(rhs_ref.validity()?)?;

        // Canonicalize so we can perform the math directly.
        let lhs: ExtensionArray = lhs_ref.execute(ctx)?;
        let rhs: ExtensionArray = rhs_ref.execute(ctx)?;

        // We validated that both inputs have the same type.
        let ext = lhs.dtype().as_extension();
        let tensor_match = ext
            .metadata_opt::<AnyTensor>()
            .vortex_expect("we already validated this in `return_dtype`");
        let dimensions = tensor_match.list_size();

        // Extract the storage array from each extension input. We pass the storage (FSL) rather
        // than the extension array to avoid canonicalizing the extension wrapper.
        let lhs_storage = lhs.storage_array();
        let rhs_storage = rhs.storage_array();

        let lhs_flat = extract_flat_elements(lhs_storage, dimensions, ctx)?;
        let rhs_flat = extract_flat_elements(rhs_storage, dimensions, ctx)?;

        match_each_float_ptype!(lhs_flat.ptype(), |T| {
            let buffer: Buffer<T> = (0..len)
                .map(|i| inner_product_row(lhs_flat.row::<T>(i), rhs_flat.row::<T>(i)))
                .collect();

            // SAFETY: The buffer length equals `row_count`, which matches the source validity
            // length.
            Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // The result is null if either input tensor is null.
        let lhs_validity = expression.child(0).validity()?;
        let rhs_validity = expression.child(1).validity()?;

        Ok(Some(and(lhs_validity, rhs_validity)))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Metadata for a serialized binary tensor-op array (shared by [`InnerProduct`] and
/// [`CosineSimilarity`]). Both operands share the same extension dtype up to nullability
/// (enforced by their `return_dtype` checks), but their individual nullabilities are lost in the
/// parent's unioned output, so both are persisted.
///
/// [`CosineSimilarity`]: crate::scalar_fns::cosine_similarity::CosineSimilarity
#[derive(Clone, prost::Message)]
pub(crate) struct BinaryTensorOpMetadata {
    #[prost(message, optional, tag = "1")]
    pub(crate) lhs_dtype: Option<pb::DType>,
    #[prost(message, optional, tag = "2")]
    pub(crate) rhs_dtype: Option<pb::DType>,
}

impl BinaryTensorOpMetadata {
    /// Encodes the two children of `view` into a [`BinaryTensorOpMetadata`] byte blob.
    pub(crate) fn encode_from_view<V: ScalarFnVTable>(
        view: &ScalarFnArrayView<V>,
    ) -> VortexResult<Vec<u8>> {
        let scalar_fn_array = view.as_::<ScalarFnArrayEncoding>();
        let lhs_dtype = Some(scalar_fn_array.child_at(0).dtype().try_into()?);
        let rhs_dtype = Some(scalar_fn_array.child_at(1).dtype().try_into()?);
        Ok(Self {
            lhs_dtype,
            rhs_dtype,
        }
        .encode_to_vec())
    }

    /// Decodes `metadata` and fetches both children from `children` using the decoded dtypes,
    /// validating that `lhs` and `rhs` agree modulo nullability.
    pub(crate) fn decode_children(
        metadata: &[u8],
        len: usize,
        children: &dyn ArrayChildren,
        session: &VortexSession,
        scalar_fn_name: &str,
    ) -> VortexResult<Vec<ArrayRef>> {
        let metadata = Self::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode BinaryTensorOpMetadata: {e}"))?;
        let lhs_pb = metadata
            .lhs_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("{scalar_fn_name} metadata missing lhs_dtype"))?;
        let rhs_pb = metadata
            .rhs_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("{scalar_fn_name} metadata missing rhs_dtype"))?;
        let lhs_dtype = DType::from_proto(lhs_pb, session)?;
        let rhs_dtype = DType::from_proto(rhs_pb, session)?;
        vortex_ensure!(
            lhs_dtype.eq_ignore_nullability(&rhs_dtype),
            "{scalar_fn_name} operand dtype mismatch: {lhs_dtype} vs {rhs_dtype}"
        );
        let lhs = children.get(0, &lhs_dtype, len)?;
        let rhs = children.get(1, &rhs_dtype, len)?;
        Ok(vec![lhs, rhs])
    }
}

impl ScalarFnArrayVTable for InnerProduct {
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
        let reconstructed = BinaryTensorOpMetadata::decode_children(
            metadata,
            len,
            children,
            session,
            "InnerProduct",
        )?;
        Ok(ScalarFnArrayParts {
            options: EmptyOptions,
            children: reconstructed,
        })
    }
}

impl InnerProduct {
    /// Both sides are `L2Denorm`: `inner_product = s_l * s_r * dot(n_l, n_r)`.
    fn execute_both_denorm(
        &self,
        lhs_ref: &ArrayRef,
        rhs_ref: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = lhs_ref.validity()?.and(rhs_ref.validity()?)?;

        let (normalized_l, norms_l) = extract_l2_denorm_children(lhs_ref);
        let (normalized_r, norms_r) = extract_l2_denorm_children(rhs_ref);

        let norms_l: PrimitiveArray = norms_l.execute(ctx)?;
        let norms_r: PrimitiveArray = norms_r.execute(ctx)?;

        let dot: PrimitiveArray = InnerProduct::try_new_array(normalized_l, normalized_r, len)?
            .into_array()
            .execute(ctx)?;

        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let nl = norms_l.as_slice::<T>();
            let nr = norms_r.as_slice::<T>();
            let buffer: Buffer<T> = (0..len).map(|i| nl[i] * nr[i] * dots[i]).collect();

            // SAFETY: The buffer length equals `len`, which matches the source validity length.
            Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
        })
    }

    /// One side is `L2Denorm`: `inner_product = s * dot(n, other)`.
    ///
    /// The caller must pass the denorm array as `denorm_ref` and the plain array as `plain_ref`.
    fn execute_one_denorm(
        &self,
        denorm_ref: &ArrayRef,
        plain_ref: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = denorm_ref.validity()?.and(plain_ref.validity()?)?;

        let (normalized, norms) = extract_l2_denorm_children(denorm_ref);
        let denorm_norms: PrimitiveArray = norms.execute(ctx)?;

        let dot: PrimitiveArray = InnerProduct::try_new_array(normalized, plain_ref.clone(), len)?
            .into_array()
            .execute(ctx)?;

        match_each_float_ptype!(dot.ptype(), |T| {
            let dots = dot.as_slice::<T>();
            let ns = denorm_norms.as_slice::<T>();
            let buffer: Buffer<T> = (0..len).map(|i| ns[i] * dots[i]).collect();

            // SAFETY: The buffer length equals `len`, which matches the source validity length.
            Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
        })
    }

    /// Fast path when one side is `ExactScalarFn<SorfTransform>` and the other side is a
    /// constant-backed tensor-like extension. Rewrites to
    /// `InnerProduct(sorf_child, forward_rotate(zero_pad(const_query)))` because SORF is
    /// orthogonal, so `<T(R^{-1} x), c> = <x, R · zero_pad(c)>` where `T` is the truncation
    /// from `padded_dim` to `dim` applied by `SorfTransform` and `R` is the SORF forward
    /// matrix. See the proof in the crate-level docs and in the plan file.
    ///
    /// Returns `Ok(None)` if neither side matches or when `element_ptype` is not `F32`. The
    /// caller is expected to fall through to the standard path in that case.
    ///
    /// # TODO(connor):
    ///
    /// This rewrite is only sound for `PType::F32` because `SorfTransform` applies an
    /// `f32 -> element_ptype` cast at the end of its execute (see `sorf_transform/vtable.rs`
    /// line ~218). For F16/F64 the cast changes the inner product's rounding and would
    /// change the semantics of the rewrite. Until we push the cast through `InnerProduct`,
    /// this path only fires for F32.
    fn try_execute_sorf_constant(
        &self,
        lhs_ref: &ArrayRef,
        rhs_ref: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Identify which side is the SorfTransform, if any.
        let (sorf_view, const_ref) =
            if let Some(view) = lhs_ref.as_opt::<ExactScalarFn<SorfTransform>>() {
                (view, rhs_ref)
            } else if let Some(view) = rhs_ref.as_opt::<ExactScalarFn<SorfTransform>>() {
                (view, lhs_ref)
            } else {
                return Ok(None);
            };

        // TODO(connor): pull-through is only sound for F32 because SorfTransform applies an
        // `f32 -> element_ptype` cast at the end of its execute. For F16/F64 the rewrite
        // would change the inner product's rounding semantics. Fall through so the standard
        // path (which does the cast before inner product) handles it.
        if sorf_view.options.element_ptype != PType::F32 {
            return Ok(None);
        }

        // The other side must be a constant-backed tensor-like extension whose scalar is
        // non-null.
        let Some(const_ext) = const_ref.as_opt::<Extension>() else {
            return Ok(None);
        };
        let const_storage = const_ext.storage_array();
        let Some(const_backing) = const_storage.as_opt::<Constant>() else {
            return Ok(None);
        };
        if const_backing.scalar().is_null() {
            return Ok(None);
        }

        let dim = sorf_view.options.dimension as usize;
        let num_rounds = sorf_view.options.num_rounds as usize;
        let seed = sorf_view.options.seed;
        let padded_dim = dim.next_power_of_two();

        // Extract the single stored row of the constant via the stride-0 short-circuit.
        let flat = extract_flat_elements(const_storage, dim, ctx)?;
        if flat.ptype() != PType::F32 {
            // TODO(connor): as above, f16/f64 are not supported by this rewrite yet. The
            // standard path handles them correctly.
            return Ok(None);
        }

        // Zero-pad the query from `dim` to `padded_dim` and forward-rotate.
        let mut padded_query = vec![0.0f32; padded_dim];
        padded_query[..dim].copy_from_slice(flat.row::<f32>(0));

        let rotation = SorfMatrix::try_new(seed, dim, num_rounds)?;
        let mut rotated_query = vec![0.0f32; padded_dim];
        rotation.rotate(&padded_query, &mut rotated_query);

        // Build the rewritten constant as a `Vector<padded_dim, f32>` extension wrapping a
        // `ConstantArray` of length `len`. We reuse the original storage FSL nullability so
        // the new extension dtype stays consistent with whatever the original tree expected.
        let storage_fsl_nullability = const_storage.dtype().nullability();
        let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
        let children: Vec<Scalar> = rotated_query
            .into_iter()
            .map(|v| Scalar::primitive(v, Nullability::NonNullable))
            .collect();
        let fsl_scalar =
            Scalar::fixed_size_list(element_dtype.clone(), children, storage_fsl_nullability);
        let new_storage = ConstantArray::new(fsl_scalar, len).into_array();

        // Build a fresh `Vector<padded_dim, f32>` extension dtype. We cannot reuse the
        // original extension dtype because that one has `dim`, not `padded_dim`.
        let padded_dim_u32 = u32::try_from(padded_dim).vortex_expect("padded_dim fits u32");
        let new_fsl_dtype = DType::FixedSizeList(
            Arc::new(element_dtype),
            padded_dim_u32,
            storage_fsl_nullability,
        );
        let new_ext_dtype = ExtDType::<Vector>::try_new(EmptyMetadata, new_fsl_dtype)?.erased();
        let new_constant = ExtensionArray::new(new_ext_dtype, new_storage).into_array();

        // Extract the SorfTransform child (the already-padded Vector<padded_dim, f32>).
        let sorf_child = sorf_view
            .nth_child(0)
            .vortex_expect("SorfTransform must have exactly one child");

        // Recursively execute the rewritten inner product. This allows case 2 to fire on
        // the rewritten tree if the sorf child is `Vector[FSL(Dict)]`. Termination is
        // guaranteed because the rewrite strictly removes a `SorfTransform` scalar-fn node
        // from the tree and SORFs cannot be nested.
        let rewritten = InnerProduct::try_new_array(sorf_child, new_constant, len)?
            .into_array()
            .execute(ctx)?;
        Ok(Some(rewritten))
    }

    /// Fast path when one side is an extension whose storage is `FSL(Dict(u8, f32))` and
    /// the other side is a constant-backed tensor extension with an F32 element ptype.
    ///
    /// Computes each row's inner product as
    ///   `out[i] = sum_{j in 0..padded_dim} q[j] * values[codes[i * padded_dim + j] as usize]`
    /// using a direct codebook lookup in the hot loop. An explicit product table
    /// `P[j, k] = q[j] * values[k]` (size `padded_dim * num_centroids * 4B`, ~1 MiB for the
    /// common 1024/256 case) was tried and measured ~10% *slower* on the
    /// `similarity_search` bench because the 1 KiB `values` table stays in L1 across all
    /// rows, while the 1 MiB product table does not.
    ///
    /// Returns `Ok(None)` when the pattern doesn't match; the caller should fall through to
    /// the standard path.
    fn try_execute_dict_constant(
        &self,
        lhs_ref: &ArrayRef,
        rhs_ref: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Try each orientation. The oriented helper navigates each side exactly once, so
        // the only redundant work here is the failed navigation of the first side when the
        // dict happens to be on the right.
        if let Some(result) = self.try_execute_dict_constant_oriented(lhs_ref, rhs_ref, len, ctx)? {
            return Ok(Some(result));
        }
        self.try_execute_dict_constant_oriented(rhs_ref, lhs_ref, len, ctx)
    }

    /// Orientation-specific helper for [`Self::try_execute_dict_constant`]. `dict_candidate`
    /// is tried as `Extension[FSL[Dict]]`; `const_candidate` is tried as a constant-backed
    /// tensor extension. Returns `Ok(None)` if either navigation fails or any gate rejects.
    fn try_execute_dict_constant_oriented(
        &self,
        dict_candidate: &ArrayRef,
        const_candidate: &ArrayRef,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Navigate the dict side.
        let Some(dict_ext) = dict_candidate.as_opt::<Extension>() else {
            return Ok(None);
        };
        let Some(fsl) = dict_ext.storage_array().as_opt::<FixedSizeList>() else {
            return Ok(None);
        };
        let Some(dict) = fsl.elements().as_opt::<Dict>() else {
            return Ok(None);
        };

        // Navigate the constant side and require its scalar be non-null.
        let Some(const_ext) = const_candidate.as_opt::<Extension>() else {
            return Ok(None);
        };
        let const_storage = const_ext.storage_array();
        let Some(const_backing) = const_storage.as_opt::<Constant>() else {
            return Ok(None);
        };
        if const_backing.scalar().is_null() {
            return Ok(None);
        }

        // Canonicalize codes and values. Codes may be e.g. BitPacked; executing is cheaper
        // than falling through to the standard path (which would also canonicalize).
        let codes_prim: PrimitiveArray = dict.codes().clone().execute(ctx)?;
        let values_prim: PrimitiveArray = dict.values().clone().execute(ctx)?;

        // Gate: u8 codes and f32 centroids.
        if codes_prim.ptype() != PType::U8 {
            // TODO(connor): support wider code widths (u16, u32). TurboQuant only emits u8
            // codes today, so this is the only path we need for now.
            return Ok(None);
        }
        if values_prim.ptype() != PType::F32 {
            // TODO(connor): direct-lookup path only supports f32 centroids. SorfTransform
            // forces f32 anyway, so this is the only shape we need for now.
            return Ok(None);
        }

        let padded_dim = usize::try_from(fsl.list_size()).vortex_expect("fsl list_size fits usize");

        let flat = extract_flat_elements(const_storage, padded_dim, ctx)?;
        if flat.ptype() != PType::F32 {
            // TODO(connor): case 2 is f32-only. For f16/f64 we fall through to the standard
            // path, which computes the inner product with the correct element type.
            return Ok(None);
        }

        // Combine the input validities up front; the per-row arithmetic may write garbage
        // into null rows but the validity mask hides it (matching the standard path).
        let validity = dict_candidate
            .validity()?
            .and(const_candidate.validity()?)?;

        // Fast path for the empty case: skip allocating and touching the codes buffer.
        if len == 0 {
            let empty = PrimitiveArray::empty::<f32>(validity.nullability());
            return Ok(Some(empty.into_array()));
        }

        let q: &[f32] = flat.row::<f32>(0);
        debug_assert_eq!(q.len(), padded_dim);
        let codes: &[u8] = codes_prim.as_slice::<u8>();
        let values: &[f32] = values_prim.as_slice::<f32>();
        debug_assert_eq!(codes.len(), len * padded_dim);

        // The hot loop is extracted into [`execute_dict_constant_inner_product`] with
        // unchecked indexing so the compiler can vectorize the inner gather-accumulate.
        let out = execute_dict_constant_inner_product(q, values, codes, len, padded_dim);

        // SAFETY: the buffer length equals `len`, which matches the validity length.
        let result = unsafe { PrimitiveArray::new_unchecked(out.freeze(), validity) }.into_array();
        Ok(Some(result))
    }
}

/// Computes the inner product (dot product) of two equal-length float slices.
///
/// Returns `sum(a_i * b_i)`.
fn inner_product_row<T: Float + NativePType>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, v| acc + v)
}

/// Compute inner products between a constant query vector and dictionary-encoded rows.
///
/// For each row, computes `sum(q[j] * values[codes[row * dim + j]])` using the codebook
/// `values` directly instead of decoding the dictionary into dense vectors.
///
/// The inner loop uses four independent accumulators so the CPU can pipeline FP additions
/// instead of waiting for each `fadd` to retire before starting the next.
fn execute_dict_constant_inner_product(
    q: &[f32],
    values: &[f32],
    codes: &[u8],
    num_rows: usize,
    dim: usize,
) -> BufferMut<f32> {
    let mut out = BufferMut::<f32>::with_capacity(num_rows);

    const PARTIAL_SUMS: usize = 8;

    for row_codes in codes.chunks_exact(dim) {
        let mut acc = [0.0f32; PARTIAL_SUMS];

        let code_chunks = row_codes.chunks_exact(PARTIAL_SUMS);
        let q_chunks = q.chunks_exact(PARTIAL_SUMS);
        let code_rem = code_chunks.remainder();
        let q_rem = q_chunks.remainder();

        for (cc, qd) in code_chunks.zip(q_chunks) {
            for i in 0..PARTIAL_SUMS {
                acc[i] = qd[i].mul_add(values[cc[i] as usize], acc[i]);
            }
        }

        for (&code, &q_val) in code_rem.iter().zip(q_rem.iter()) {
            acc[0] = q_val.mul_add(values[code as usize], acc[0]);
        }

        // SAFETY: we reserved `num_rows` slots and push exactly once per row.
        unsafe { out.push_unchecked(acc.iter().sum::<f32>()) };
    }

    out
}

#[cfg(test)]
mod tests {

    use rstest::rstest;
    use vortex_array::ArrayPlugin;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::ScalarFnArray;
    use vortex_array::arrays::scalar_fn::plugin::ScalarFnArrayPlugin;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use crate::scalar_fns::inner_product::InnerProduct;
    use crate::scalar_fns::l2_denorm::L2Denorm;
    use crate::tests::SESSION;
    use crate::utils::test_helpers::assert_close;
    use crate::utils::test_helpers::tensor_array;
    use crate::utils::test_helpers::vector_array;

    /// Evaluates inner product between two tensor arrays and returns the result as `Vec<f64>`.
    fn eval_inner_product(lhs: ArrayRef, rhs: ArrayRef, len: usize) -> VortexResult<Vec<f64>> {
        let scalar_fn = InnerProduct::new().erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], len)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;
        Ok(prim.as_slice::<f64>().to_vec())
    }

    /// Single-row inner product for various vector pairs.
    #[rstest]
    // Orthogonal: [1, 0] . [0, 1] = 0.
    #[case::orthogonal(&[2], &[1.0, 0.0], &[0.0, 1.0], &[0.0])]
    // Parallel: [3, 4] . [3, 4] = 9 + 16 = 25.
    #[case::parallel(&[2], &[3.0, 4.0], &[3.0, 4.0], &[25.0])]
    // Antiparallel: [1, 2] . [-1, -2] = -1 + -4 = -5.
    #[case::antiparallel(&[2], &[1.0, 2.0], &[-1.0, -2.0], &[-5.0])]
    // Scaled: [2, 0] . [3, 0] = 6.
    #[case::scaled(&[2], &[2.0, 0.0], &[3.0, 0.0], &[6.0])]
    fn single_row(
        #[case] shape: &[usize],
        #[case] lhs_elems: &[f64],
        #[case] rhs_elems: &[f64],
        #[case] expected: &[f64],
    ) -> VortexResult<()> {
        let lhs = tensor_array(shape, lhs_elems)?;
        let rhs = tensor_array(shape, rhs_elems)?;
        assert_close(&eval_inner_product(lhs, rhs, 1)?, expected);
        Ok(())
    }

    #[test]
    fn multiple_rows() -> VortexResult<()> {
        let lhs = tensor_array(
            &[3],
            &[
                1.0, 0.0, 0.0, // tensor 0
                3.0, 4.0, 0.0, // tensor 1
                1.0, 1.0, 1.0, // tensor 2
            ],
        )?;
        let rhs = tensor_array(
            &[3],
            &[
                0.0, 1.0, 0.0, // tensor 0: dot = 0
                3.0, 4.0, 0.0, // tensor 1: dot = 25
                2.0, 2.0, 2.0, // tensor 2: dot = 6
            ],
        )?;
        assert_close(&eval_inner_product(lhs, rhs, 3)?, &[0.0, 25.0, 6.0]);
        Ok(())
    }

    #[test]
    fn vector_inner_product() -> VortexResult<()> {
        let lhs = vector_array(
            2,
            &[
                3.0, 4.0, // vector 0
                1.0, 0.0, // vector 1
            ],
        )?;
        let rhs = vector_array(
            2,
            &[
                3.0, 4.0, // vector 0: dot = 25
                0.0, 1.0, // vector 1: dot = 0
            ],
        )?;
        assert_close(&eval_inner_product(lhs, rhs, 2)?, &[25.0, 0.0]);
        Ok(())
    }

    #[test]
    fn null_input_row() -> VortexResult<()> {
        // 3 rows of dim-2 vectors. Row 1 of lhs is masked as null.
        let lhs = tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let rhs = tensor_array(&[2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
        let lhs = MaskedArray::try_new(lhs, Validity::from_iter([true, false, true]))?.into_array();

        let scalar_fn = InnerProduct::new().erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 3)?;
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: 1*7 + 2*8 = 23, row 1: null, row 2: 5*11 + 6*12 = 127.
        assert!(prim.is_valid(0)?);
        assert!(!prim.is_valid(1)?);
        assert!(prim.is_valid(2)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[23.0]);
        assert_close(&[prim.as_slice::<f64>()[2]], &[127.0]);
        Ok(())
    }

    #[test]
    fn rejects_non_extension_dtype() {
        let lhs = PrimitiveArray::from_iter([1.0_f64, 2.0]).into_array();
        let rhs = PrimitiveArray::from_iter([3.0_f64, 4.0]).into_array();
        let result = InnerProduct::try_new_array(lhs, rhs, 2);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_mismatched_dtypes() -> VortexResult<()> {
        let lhs = tensor_array(&[2], &[1.0_f64, 2.0])?;
        let rhs = vector_array(2, &[3.0_f64, 4.0])?;
        let result = InnerProduct::try_new_array(lhs, rhs, 1);
        assert!(result.is_err());
        Ok(())
    }

    /// Creates an `L2Denorm` scalar function array from pre-normalized elements and norms.
    fn l2_denorm_array(
        shape: &[usize],
        normalized_elements: &[f64],
        norms: &[f64],
    ) -> VortexResult<ArrayRef> {
        use vortex_array::IntoArray;

        let len = norms.len();
        let normalized = tensor_array(shape, normalized_elements)?;
        let norms = PrimitiveArray::from_iter(norms.iter().copied()).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        Ok(L2Denorm::try_new_array(normalized, norms, len, &mut ctx)?.into_array())
    }

    #[test]
    fn both_denorm() -> VortexResult<()> {
        // LHS: [3.0, 4.0] = L2Denorm([0.6, 0.8], 5.0).
        // RHS: [1.0, 0.0] = L2Denorm([1.0, 0.0], 1.0).
        // dot([3.0, 4.0], [1.0, 0.0]) = 3.0.
        let lhs = l2_denorm_array(&[2], &[0.6, 0.8], &[5.0])?;
        let rhs = l2_denorm_array(&[2], &[1.0, 0.0], &[1.0])?;

        // Expected: 5.0 * 1.0 * dot([0.6, 0.8], [1.0, 0.0]) = 5.0 * 0.6 = 3.0.
        assert_close(&eval_inner_product(lhs, rhs, 1)?, &[3.0]);
        Ok(())
    }

    #[test]
    fn both_denorm_multiple_rows() -> VortexResult<()> {
        // Row 0: [3.0, 4.0] dot [3.0, 4.0] = 25.0.
        // Row 1: [1.0, 0.0] dot [0.0, 1.0] = 0.0.
        let lhs = l2_denorm_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0])?;
        let rhs = l2_denorm_array(&[2], &[0.6, 0.8, 0.0, 1.0], &[5.0, 1.0])?;

        assert_close(&eval_inner_product(lhs, rhs, 2)?, &[25.0, 0.0]);
        Ok(())
    }

    #[test]
    fn one_side_denorm_lhs() -> VortexResult<()> {
        // LHS: L2Denorm([0.6, 0.8], 5.0) representing [3.0, 4.0].
        // RHS: plain [1.0, 2.0].
        // dot([3.0, 4.0], [1.0, 2.0]) = 3.0 + 8.0 = 11.0.
        let lhs = l2_denorm_array(&[2], &[0.6, 0.8], &[5.0])?;
        let rhs = tensor_array(&[2], &[1.0, 2.0])?;

        assert_close(&eval_inner_product(lhs, rhs, 1)?, &[11.0]);
        Ok(())
    }

    #[test]
    fn one_side_denorm_rhs() -> VortexResult<()> {
        // LHS: plain [1.0, 2.0].
        // RHS: L2Denorm([0.6, 0.8], 5.0) representing [3.0, 4.0].
        // dot([1.0, 2.0], [3.0, 4.0]) = 3.0 + 8.0 = 11.0.
        let lhs = tensor_array(&[2], &[1.0, 2.0])?;
        let rhs = l2_denorm_array(&[2], &[0.6, 0.8], &[5.0])?;

        assert_close(&eval_inner_product(lhs, rhs, 1)?, &[11.0]);
        Ok(())
    }

    #[test]
    fn both_denorm_null_norms() -> VortexResult<()> {
        // Row 0: valid, row 1: null (via nullable norms on lhs).
        let normalized_l = tensor_array(&[2], &[0.6, 0.8, 1.0, 0.0])?;
        let norms_l = PrimitiveArray::from_option_iter([Some(5.0f64), None]).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let lhs = L2Denorm::try_new_array(normalized_l, norms_l, 2, &mut ctx)?.into_array();
        let rhs = l2_denorm_array(&[2], &[0.6, 0.8, 1.0, 0.0], &[5.0, 1.0])?;

        let scalar_fn = InnerProduct::new().erased();
        let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], 2)?;
        let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;

        // Row 0: 5.0 * 5.0 * dot([0.6, 0.8], [0.6, 0.8]) = 25.0, row 1: null.
        assert!(prim.is_valid(0)?);
        assert!(!prim.is_valid(1)?);
        assert_close(&[prim.as_slice::<f64>()[0]], &[25.0]);
        Ok(())
    }

    #[rstest]
    #[case::vector(
        vector_array(3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        vector_array(3, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap(),
        2,
    )]
    #[case::fixed_shape_tensor(
        tensor_array(&[2], &[1.0, 2.0, 3.0, 4.0]).unwrap(),
        tensor_array(&[2], &[5.0, 6.0, 7.0, 8.0]).unwrap(),
        2,
    )]
    fn serde_round_trip(
        #[case] lhs: ArrayRef,
        #[case] rhs: ArrayRef,
        #[case] len: usize,
    ) -> VortexResult<()> {
        let original = InnerProduct::try_new_array(lhs.clone(), rhs.clone(), len)?.into_array();

        let plugin = ScalarFnArrayPlugin::new(InnerProduct);
        let metadata = plugin
            .serialize(&original, &SESSION)?
            .expect("InnerProduct serialize must produce metadata");

        let children = vec![lhs, rhs];
        let recovered = plugin.deserialize(
            original.dtype(),
            original.len(),
            &metadata,
            &[],
            &children,
            &SESSION,
        )?;

        assert_eq!(recovered.dtype(), original.dtype());
        assert_eq!(recovered.len(), original.len());
        assert_eq!(recovered.encoding_id(), original.encoding_id());
        Ok(())
    }

    // ---- Tests for the `SorfTransform + constant` and `Dict + constant` fast paths ----

    #[allow(
        clippy::cast_possible_truncation,
        reason = "tests build small fixtures with deterministic in-range indices"
    )]
    mod constant_query_optimizations {
        use std::sync::LazyLock;

        use rstest::rstest;
        use vortex_array::ArrayRef;
        use vortex_array::IntoArray;
        use vortex_array::VortexSessionExecute;
        use vortex_array::arrays::ConstantArray;
        use vortex_array::arrays::ExtensionArray;
        use vortex_array::arrays::FixedSizeListArray;
        use vortex_array::arrays::PrimitiveArray;
        use vortex_array::arrays::ScalarFnArray;
        use vortex_array::arrays::dict::DictArray;
        use vortex_array::dtype::DType;
        use vortex_array::dtype::Nullability;
        use vortex_array::dtype::PType;
        use vortex_array::dtype::extension::ExtDType;
        use vortex_array::extension::EmptyMetadata;
        use vortex_array::scalar::Scalar;
        use vortex_array::session::ArraySession;
        use vortex_array::validity::Validity;
        use vortex_buffer::Buffer;
        use vortex_error::VortexResult;
        use vortex_session::VortexSession;

        use crate::scalar_fns::inner_product::InnerProduct;
        use crate::scalar_fns::sorf_transform::SorfMatrix;
        use crate::scalar_fns::sorf_transform::SorfOptions;
        use crate::scalar_fns::sorf_transform::SorfTransform;
        use crate::vector::Vector;

        static SESSION: LazyLock<VortexSession> =
            LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

        /// Compact f32 Vector<dim> extension over a column-major `elements` slice.
        fn vector_f32(dim: u32, elements: &[f32]) -> VortexResult<ArrayRef> {
            let row_count = elements.len() / dim as usize;
            let elems: ArrayRef = Buffer::copy_from(elements).into_array();
            let fsl = FixedSizeListArray::new(elems, dim, Validity::NonNullable, row_count);
            let ext_dtype =
                ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
            Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
        }

        /// Compact constant-backed f32 Vector<dim> extension with a single stored row.
        fn constant_vector_f32(elements: &[f32], len: usize) -> VortexResult<ArrayRef> {
            let element_dtype = DType::Primitive(PType::F32, Nullability::NonNullable);
            let children: Vec<Scalar> = elements
                .iter()
                .map(|&v| Scalar::primitive(v, Nullability::NonNullable))
                .collect();
            let storage_scalar =
                Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
            let storage = ConstantArray::new(storage_scalar, len).into_array();
            let ext_dtype =
                ExtDType::<Vector>::try_new(EmptyMetadata, storage.dtype().clone())?.erased();
            Ok(ExtensionArray::new(ext_dtype, storage).into_array())
        }

        /// Build an `ExtensionArray<Vector<list_size, f32>>` whose storage is
        /// `FSL(DictArray(codes: u8, values: f32))`. This mirrors the shape that
        /// TurboQuant produces as the SorfTransform child.
        fn dict_vector_f32(list_size: u32, codes: &[u8], values: &[f32]) -> VortexResult<ArrayRef> {
            let num_rows = codes.len() / list_size as usize;
            let codes_arr =
                PrimitiveArray::new::<u8>(Buffer::copy_from(codes), Validity::NonNullable)
                    .into_array();
            let values_arr =
                PrimitiveArray::new::<f32>(Buffer::copy_from(values), Validity::NonNullable)
                    .into_array();
            let dict = DictArray::try_new(codes_arr, values_arr)?;
            let fsl = FixedSizeListArray::try_new(
                dict.into_array(),
                list_size,
                Validity::NonNullable,
                num_rows,
            )?;
            let ext_dtype =
                ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
            Ok(ExtensionArray::new(ext_dtype, fsl.into_array()).into_array())
        }

        /// Execute an inner product and return the flat `f32` results.
        fn eval_ip_f32(lhs: ArrayRef, rhs: ArrayRef, len: usize) -> VortexResult<Vec<f32>> {
            let scalar_fn = InnerProduct::new().erased();
            let result = ScalarFnArray::try_new(scalar_fn, vec![lhs, rhs], len)?;
            let mut ctx = SESSION.create_execution_ctx();
            let prim: PrimitiveArray = result.into_array().execute(&mut ctx)?;
            Ok(prim.as_slice::<f32>().to_vec())
        }

        fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32) {
            assert_eq!(actual.len(), expected.len(), "length mismatch");
            for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() < tol,
                    "row {i}: got {a}, expected {e} (diff = {})",
                    (a - e).abs()
                );
            }
        }

        /// Build a SorfTransform ScalarFnArray whose child is a `Vector<padded_dim, f32>`
        /// wrapping `FSL(Dict(codes, values))`. Returns `(sorf_array, codes, values,
        /// padded_dim)`.
        fn build_sorf_with_dict_child(
            dim: u32,
            num_rows: usize,
            seed: u64,
            num_rounds: u8,
        ) -> VortexResult<(ArrayRef, Vec<u8>, Vec<f32>, usize)> {
            let padded_dim = (dim as usize).next_power_of_two();
            // Small hand-picked codebook of 8 f32 centroids.
            let values: Vec<f32> = vec![-1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5];
            // Deterministic codes in 0..values.len() covering every position.
            let codes: Vec<u8> = (0..num_rows * padded_dim)
                .map(|i| (i as u8) % (values.len() as u8))
                .collect();

            let padded_vector = dict_vector_f32(padded_dim as u32, &codes, &values)?;
            let sorf_options = SorfOptions {
                seed,
                num_rounds,
                dimension: dim,
                element_ptype: PType::F32,
            };
            let sorf =
                SorfTransform::try_new_array(&sorf_options, padded_vector, num_rows)?.into_array();
            Ok((sorf, codes, values, padded_dim))
        }

        /// Decode a SorfTransform-wrapped dict-vector to a flat `Vec<f32>` of `num_rows *
        /// dim` post-rotation, post-truncation values. This is the ground truth against
        /// which we compare the fast-path result.
        fn decode_sorf_dict(
            codes: &[u8],
            values: &[f32],
            padded_dim: usize,
            dim: usize,
            num_rows: usize,
            seed: u64,
            num_rounds: u8,
        ) -> VortexResult<Vec<f32>> {
            let rotation = SorfMatrix::try_new(seed, dim, num_rounds as usize)?;
            let mut padded = vec![0.0f32; padded_dim];
            let mut rotated = vec![0.0f32; padded_dim];
            let mut out = Vec::with_capacity(num_rows * dim);
            for row in 0..num_rows {
                for j in 0..padded_dim {
                    padded[j] = values[codes[row * padded_dim + j] as usize];
                }
                rotation.inverse_rotate(&padded, &mut rotated);
                out.extend_from_slice(&rotated[..dim]);
            }
            Ok(out)
        }

        fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
        }

        // ---- Case 1: SorfTransform + Constant pull-through ----

        /// Case 1: SorfTransform on LHS, constant query on RHS, with `dim < padded_dim`
        /// so the zero-padding branch is exercised.
        #[test]
        fn case1_sorf_lhs_constant_rhs_padded_gt_dim() -> VortexResult<()> {
            let dim: u32 = 100;
            let num_rows = 7usize;
            let seed = 42u64;
            let num_rounds = 3u8;
            let padded_dim = (dim as usize).next_power_of_two();
            assert!(padded_dim > dim as usize, "test must exercise padding");

            let (sorf_lhs, codes, values, padded_dim_computed) =
                build_sorf_with_dict_child(dim, num_rows, seed, num_rounds)?;
            assert_eq!(padded_dim_computed, padded_dim);

            // Query has `dim` elements.
            let query_elems: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let const_rhs = constant_vector_f32(&query_elems, num_rows)?;

            // Ground truth: decode LHS to plain f32 vectors, dot each with the query.
            let decoded = decode_sorf_dict(
                &codes,
                &values,
                padded_dim,
                dim as usize,
                num_rows,
                seed,
                num_rounds,
            )?;
            let expected: Vec<f32> = (0..num_rows)
                .map(|i| {
                    naive_dot(
                        &decoded[i * dim as usize..(i + 1) * dim as usize],
                        &query_elems,
                    )
                })
                .collect();

            let actual = eval_ip_f32(sorf_lhs, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-3);
            Ok(())
        }

        /// Case 1: SorfTransform on RHS, constant query on LHS (mirrored).
        #[test]
        fn case1_constant_lhs_sorf_rhs_mirrored() -> VortexResult<()> {
            let dim: u32 = 100;
            let num_rows = 5usize;
            let seed = 7u64;
            let num_rounds = 3u8;

            let (sorf, codes, values, padded_dim) =
                build_sorf_with_dict_child(dim, num_rows, seed, num_rounds)?;

            let query_elems: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
            let const_lhs = constant_vector_f32(&query_elems, num_rows)?;

            let decoded = decode_sorf_dict(
                &codes,
                &values,
                padded_dim,
                dim as usize,
                num_rows,
                seed,
                num_rounds,
            )?;
            let expected: Vec<f32> = (0..num_rows)
                .map(|i| {
                    naive_dot(
                        &decoded[i * dim as usize..(i + 1) * dim as usize],
                        &query_elems,
                    )
                })
                .collect();

            let actual = eval_ip_f32(const_lhs, sorf, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-3);
            Ok(())
        }

        /// Case 1: `dim == padded_dim` (power-of-two, no zero padding).
        #[test]
        fn case1_padded_equals_dim() -> VortexResult<()> {
            let dim: u32 = 128;
            let num_rows = 4usize;
            let seed = 11u64;
            let num_rounds = 3u8;

            let (sorf, codes, values, padded_dim) =
                build_sorf_with_dict_child(dim, num_rows, seed, num_rounds)?;
            assert_eq!(padded_dim, dim as usize);

            let query_elems: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01 - 0.5).collect();
            let const_rhs = constant_vector_f32(&query_elems, num_rows)?;

            let decoded = decode_sorf_dict(
                &codes,
                &values,
                padded_dim,
                dim as usize,
                num_rows,
                seed,
                num_rounds,
            )?;
            let expected: Vec<f32> = (0..num_rows)
                .map(|i| {
                    naive_dot(
                        &decoded[i * dim as usize..(i + 1) * dim as usize],
                        &query_elems,
                    )
                })
                .collect();

            let actual = eval_ip_f32(sorf, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-3);
            Ok(())
        }

        /// Case 1: empty `len == 0`. The fast path should handle this without exploding.
        #[test]
        fn case1_empty_len_zero() -> VortexResult<()> {
            let dim: u32 = 100;
            let num_rows = 0usize;
            let seed = 42u64;
            let num_rounds = 3u8;

            let (sorf, _codes, _values, _padded_dim) =
                build_sorf_with_dict_child(dim, num_rows, seed, num_rounds)?;

            let query_elems: Vec<f32> = vec![0.0; dim as usize];
            let const_rhs = constant_vector_f32(&query_elems, num_rows)?;

            let actual = eval_ip_f32(sorf, const_rhs, num_rows)?;
            assert_eq!(actual.len(), 0);
            Ok(())
        }

        // ---- Case 2: Dict + Constant direct-lookup path ----

        /// Case 2: Vector[FSL[Dict(u8, f32)]] on LHS, constant query on RHS.
        #[test]
        fn case2_dict_lhs_constant_rhs_matches_naive() -> VortexResult<()> {
            let list_size: u32 = 8;
            let num_rows = 10usize;
            // 8 centroids, tiny table.
            let values: Vec<f32> = vec![-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0];
            // Deterministic codes.
            let codes: Vec<u8> = (0..num_rows * list_size as usize)
                .map(|i| (i as u8) % (values.len() as u8))
                .collect();
            let dict_lhs = dict_vector_f32(list_size, &codes, &values)?;

            let query: Vec<f32> = (0..list_size).map(|i| (i as f32 + 1.0) * 0.3).collect();
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            let expected: Vec<f32> = (0..num_rows)
                .map(|row| {
                    let mut acc = 0.0f32;
                    for j in 0..list_size as usize {
                        let k = codes[row * list_size as usize + j] as usize;
                        acc += query[j] * values[k];
                    }
                    acc
                })
                .collect();

            let actual = eval_ip_f32(dict_lhs, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-5);
            Ok(())
        }

        /// Case 2: constant query on LHS, dict column on RHS (mirrored).
        #[test]
        fn case2_constant_lhs_dict_rhs_mirrored() -> VortexResult<()> {
            let list_size: u32 = 4;
            let num_rows = 6usize;
            let values: Vec<f32> = vec![0.1, 0.4, 0.7, 1.0];
            let codes: Vec<u8> = (0..num_rows * list_size as usize)
                .map(|i| ((i * 3) as u8) % (values.len() as u8))
                .collect();
            let dict_rhs = dict_vector_f32(list_size, &codes, &values)?;

            let query: Vec<f32> = vec![0.5, -1.0, 2.5, -0.25];
            let const_lhs = constant_vector_f32(&query, num_rows)?;

            let expected: Vec<f32> = (0..num_rows)
                .map(|row| {
                    let mut acc = 0.0f32;
                    for j in 0..list_size as usize {
                        let k = codes[row * list_size as usize + j] as usize;
                        acc += query[j] * values[k];
                    }
                    acc
                })
                .collect();

            let actual = eval_ip_f32(const_lhs, dict_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-5);
            Ok(())
        }

        /// Case 2: dict with `u16` codes (and hence more than 256 values) falls through to
        /// the standard path but still produces the correct result. The direct-lookup path
        /// only handles `u8` codes today.
        #[test]
        fn case2_u16_codes_falls_through() -> VortexResult<()> {
            let list_size: u32 = 4;
            let num_rows = 3usize;
            let num_values = 300usize;
            let values: Vec<f32> = (0..num_values).map(|i| i as f32 * 0.01).collect();
            // Codes must be u16 because 300 > 255. dict_vector_f32 only supports u8 so we
            // build the dict by hand here.
            let codes_u16: Vec<u16> = (0..(num_rows * 4))
                .map(|i| (i % num_values) as u16)
                .collect();
            let codes_arr =
                PrimitiveArray::new::<u16>(Buffer::copy_from(codes_u16), Validity::NonNullable)
                    .into_array();
            let values_arr =
                PrimitiveArray::new::<f32>(Buffer::copy_from(&values), Validity::NonNullable)
                    .into_array();
            let dict = DictArray::try_new(codes_arr, values_arr)?;
            let fsl = FixedSizeListArray::try_new(
                dict.into_array(),
                list_size,
                Validity::NonNullable,
                num_rows,
            )?;
            let ext_dtype =
                ExtDType::<Vector>::try_new(EmptyMetadata, fsl.dtype().clone())?.erased();
            let dict_lhs = ExtensionArray::new(ext_dtype, fsl.into_array()).into_array();

            let query: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            // Build expected by decoding by hand.
            let expected: Vec<f32> = (0..num_rows)
                .map(|row| {
                    let mut acc = 0.0f32;
                    for j in 0..4 {
                        let code = (row * 4 + j) % num_values;
                        acc += query[j] * values[code];
                    }
                    acc
                })
                .collect();

            let actual = eval_ip_f32(dict_lhs, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-5);
            Ok(())
        }

        /// Case 2: plain (non-dict) FSL with a constant RHS falls through to the standard
        /// path and produces the correct result.
        #[test]
        fn case2_plain_fsl_falls_through() -> VortexResult<()> {
            let dim: u32 = 4;
            let num_rows = 3usize;
            let lhs_elems: Vec<f32> = (0..num_rows * dim as usize)
                .map(|i| i as f32 * 0.25)
                .collect();
            let plain_lhs = vector_f32(dim, &lhs_elems)?;

            let query: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            let expected: Vec<f32> = (0..num_rows)
                .map(|row| {
                    naive_dot(
                        &lhs_elems[row * dim as usize..(row + 1) * dim as usize],
                        &query,
                    )
                })
                .collect();

            let actual = eval_ip_f32(plain_lhs, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-5);
            Ok(())
        }

        /// Case 2: empty `len == 0` fast path returns an empty primitive array without
        /// touching the codes buffer.
        #[test]
        fn case2_empty_len_zero() -> VortexResult<()> {
            let list_size: u32 = 4;
            let num_rows = 0usize;
            let values: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
            let codes: Vec<u8> = Vec::new();
            let dict_lhs = dict_vector_f32(list_size, &codes, &values)?;

            let query: Vec<f32> = vec![0.0; 4];
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            let actual = eval_ip_f32(dict_lhs, const_rhs, num_rows)?;
            assert_eq!(actual.len(), 0);
            Ok(())
        }

        /// Case 1 + Case 2 end-to-end: the SorfTransform-wrapped dict column hits Case 1
        /// then Case 2 via recursive execution.
        #[test]
        fn end_to_end_sorf_plus_dict_cosine_path() -> VortexResult<()> {
            let dim: u32 = 100;
            let num_rows = 9usize;
            let seed = 99u64;
            let num_rounds = 3u8;

            let (sorf, codes, values, padded_dim) =
                build_sorf_with_dict_child(dim, num_rows, seed, num_rounds)?;

            let query_elems: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.15).sin() * 0.4).collect();
            let const_rhs = constant_vector_f32(&query_elems, num_rows)?;

            // Ground truth via full decode + naive dot.
            let decoded = decode_sorf_dict(
                &codes,
                &values,
                padded_dim,
                dim as usize,
                num_rows,
                seed,
                num_rounds,
            )?;
            let expected: Vec<f32> = (0..num_rows)
                .map(|i| {
                    naive_dot(
                        &decoded[i * dim as usize..(i + 1) * dim as usize],
                        &query_elems,
                    )
                })
                .collect();

            let actual = eval_ip_f32(sorf, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-3);
            Ok(())
        }

        // ---- Additional correctness / stress tests (all with loose tolerances) ----

        /// A tiny in-place xorshift64 PRNG so these tests don't depend on `rand`. Producing
        /// deterministic pseudo-random f32 values lets the correctness checks exercise
        /// realistic data instead of smooth sin/cos patterns.
        struct XorShift64(u64);

        impl XorShift64 {
            fn new(seed: u64) -> Self {
                // Any nonzero seed is fine; xorshift fixed-points at 0.
                Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
            }

            fn next_u64(&mut self) -> u64 {
                let mut x = self.0;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.0 = x;
                x
            }

            /// Uniform f32 in `[-1.0, 1.0)`.
            fn next_f32(&mut self) -> f32 {
                // Top 24 bits -> mantissa in [0, 1), then shift to [-1, 1).
                let bits = (self.next_u64() >> 40) as u32; // 24 bits
                (bits as f32) / (1u32 << 24) as f32 * 2.0 - 1.0
            }
        }

        /// Case 2 stress: u8-coded dict with 200 centroids (formerly blocked by the
        /// `values.len() <= 256` gate). The direct-lookup path must now handle it.
        #[test]
        fn case2_large_u8_codebook_direct_lookup() -> VortexResult<()> {
            let list_size: u32 = 16;
            let num_rows = 20usize;
            let num_centroids = 200usize;
            assert!(num_centroids > 8 && num_centroids <= 256);

            let mut rng = XorShift64::new(0xDEAD_BEEF);
            let values: Vec<f32> = (0..num_centroids).map(|_| rng.next_f32()).collect();
            let codes: Vec<u8> = (0..num_rows * list_size as usize)
                .map(|_| (rng.next_u64() % num_centroids as u64) as u8)
                .collect();

            let dict_lhs = dict_vector_f32(list_size, &codes, &values)?;
            let query: Vec<f32> = (0..list_size).map(|_| rng.next_f32()).collect();
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            let expected: Vec<f32> = (0..num_rows)
                .map(|row| {
                    let mut acc = 0.0f32;
                    for j in 0..list_size as usize {
                        let k = codes[row * list_size as usize + j] as usize;
                        acc += query[j] * values[k];
                    }
                    acc
                })
                .collect();

            let actual = eval_ip_f32(dict_lhs, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-4);
            Ok(())
        }

        /// Parameterized sweep over the full `InnerProduct(SorfTransform(Vector[FSL(Dict)]),
        /// ConstantArray)` tree, exercising the case 1 + case 2 chain for a realistic mix
        /// of dimensions, row counts, seeds, and number of SORF rounds. Tolerance is
        /// deliberately loose because the rewrite introduces an f32-domain rotation that
        /// accumulates a small numerical drift versus a naive decode.
        #[rstest]
        #[case::small_no_pad(128, 11, 1, 1)]
        #[case::small_no_pad_rounds3(128, 23, 1_234, 3)]
        #[case::small_padded(100, 17, 42, 3)]
        #[case::mid_padded(200, 13, 2024, 3)]
        #[case::mid_power_of_two(256, 31, 7, 3)]
        #[case::larger_padded(300, 9, 99, 3)]
        #[case::max_rounds(128, 5, 31_415, 5)]
        fn case1_sorf_random_sweep(
            #[case] dim: u32,
            #[case] num_rows: usize,
            #[case] seed: u64,
            #[case] num_rounds: u8,
        ) -> VortexResult<()> {
            let (sorf, codes, values, padded_dim) =
                build_sorf_with_dict_child(dim, num_rows, seed, num_rounds)?;

            // Use a pseudo-random query with both positive and negative entries so the sum
            // has cancellation.
            let mut rng = XorShift64::new(seed ^ 0xABCD_1234);
            let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            let decoded = decode_sorf_dict(
                &codes,
                &values,
                padded_dim,
                dim as usize,
                num_rows,
                seed,
                num_rounds,
            )?;
            let expected: Vec<f32> = (0..num_rows)
                .map(|i| naive_dot(&decoded[i * dim as usize..(i + 1) * dim as usize], &query))
                .collect();

            // Loose tolerance: the sorf transform works in f32 with a k-round butterfly, so
            // the rewrite path and the decoded path accumulate slightly different rounding
            // even though the math is equivalent in exact arithmetic.
            let actual = eval_ip_f32(sorf, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-2);
            Ok(())
        }

        /// Parameterized sweep over plain `Vector[FSL(Dict(u8, f32))]` + constant query,
        /// without SorfTransform in the mix. This directly exercises case 2 across a
        /// variety of list sizes, num_rows, and codebook sizes including large ones that
        /// the old `<= 256` gate would have rejected.
        #[rstest]
        #[case::small(4, 7, 8)]
        #[case::medium(16, 50, 64)]
        #[case::larger(32, 100, 150)]
        #[case::very_large_codebook(8, 25, 250)]
        fn case2_random_sweep(
            #[case] list_size: u32,
            #[case] num_rows: usize,
            #[case] num_centroids: usize,
        ) -> VortexResult<()> {
            let mut rng = XorShift64::new((list_size as u64) * 31 + num_rows as u64);
            let values: Vec<f32> = (0..num_centroids).map(|_| rng.next_f32()).collect();
            assert!(num_centroids <= 256, "u8 codes cap at 256 centroids");
            let codes: Vec<u8> = (0..num_rows * list_size as usize)
                .map(|_| (rng.next_u64() % num_centroids as u64) as u8)
                .collect();

            let dict_lhs = dict_vector_f32(list_size, &codes, &values)?;
            let query: Vec<f32> = (0..list_size).map(|_| rng.next_f32()).collect();
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            let expected: Vec<f32> = (0..num_rows)
                .map(|row| {
                    let mut acc = 0.0f32;
                    for j in 0..list_size as usize {
                        let k = codes[row * list_size as usize + j] as usize;
                        acc += query[j] * values[k];
                    }
                    acc
                })
                .collect();

            // Tight tolerance here because no SorfTransform rotation is involved — the
            // arithmetic should agree bit-for-bit up to float reassociation.
            let actual = eval_ip_f32(dict_lhs, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-4);
            Ok(())
        }

        /// End-to-end regression: for a plausible vector-search configuration (SORF rounds
        /// = 3, dim = 128, num_rows = 64, u8 codes, 64 centroids), the fast-path result
        /// must track a fully naive computation within 1e-2.
        #[test]
        fn end_to_end_dim128_rows64_bit6_regression() -> VortexResult<()> {
            let dim: u32 = 128;
            let num_rows = 64usize;
            let seed = 0xFACE_F00D;
            let num_rounds = 3u8;

            // Use 64 centroids (6 bits), a typical TurboQuant configuration.
            let num_centroids = 64usize;
            let padded_dim = (dim as usize).next_power_of_two();
            let mut rng = XorShift64::new(seed);
            let values: Vec<f32> = (0..num_centroids).map(|_| rng.next_f32()).collect();
            let codes: Vec<u8> = (0..num_rows * padded_dim)
                .map(|_| (rng.next_u64() % num_centroids as u64) as u8)
                .collect();

            let padded_vector = dict_vector_f32(padded_dim as u32, &codes, &values)?;
            let sorf_options = SorfOptions {
                seed,
                num_rounds,
                dimension: dim,
                element_ptype: PType::F32,
            };
            let sorf =
                SorfTransform::try_new_array(&sorf_options, padded_vector, num_rows)?.into_array();

            let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            let decoded = decode_sorf_dict(
                &codes,
                &values,
                padded_dim,
                dim as usize,
                num_rows,
                seed,
                num_rounds,
            )?;
            let expected: Vec<f32> = (0..num_rows)
                .map(|i| naive_dot(&decoded[i * dim as usize..(i + 1) * dim as usize], &query))
                .collect();

            let actual = eval_ip_f32(sorf, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-2);

            // Also verify the max relative error is small. The SORF rotation does not
            // amplify error, so both measures should be bounded.
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                let denom = e.abs().max(1.0);
                let rel = (a - e).abs() / denom;
                assert!(
                    rel < 1e-3,
                    "row {i}: rel err {rel} too large (a={a}, e={e})"
                );
            }
            Ok(())
        }

        /// Case 1 + Case 2 end-to-end with varying `num_rounds`. The rotation becomes
        /// progressively more chaotic as rounds increase, so this catches any off-by-one
        /// bug in the round-indexing that would not show up in the 3-round default.
        #[rstest]
        #[case(1)]
        #[case(2)]
        #[case(3)]
        #[case(4)]
        #[case(5)]
        fn case1_various_num_rounds(#[case] num_rounds: u8) -> VortexResult<()> {
            let dim: u32 = 128;
            let num_rows = 8usize;
            let seed = 0x1234_5678;

            let (sorf, codes, values, padded_dim) =
                build_sorf_with_dict_child(dim, num_rows, seed, num_rounds)?;

            let mut rng = XorShift64::new(seed ^ (num_rounds as u64));
            let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            let const_rhs = constant_vector_f32(&query, num_rows)?;

            let decoded = decode_sorf_dict(
                &codes,
                &values,
                padded_dim,
                dim as usize,
                num_rows,
                seed,
                num_rounds,
            )?;
            let expected: Vec<f32> = (0..num_rows)
                .map(|i| naive_dot(&decoded[i * dim as usize..(i + 1) * dim as usize], &query))
                .collect();

            let actual = eval_ip_f32(sorf, const_rhs, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-2);
            Ok(())
        }

        /// Swap LHS and RHS on the full tree to prove the side-detection and the scalar
        /// argument-order handling are symmetric for both cases simultaneously.
        #[test]
        fn end_to_end_constant_lhs_sorf_rhs_mirrored() -> VortexResult<()> {
            let dim: u32 = 256;
            let num_rows = 12usize;
            let seed = 0xBEEF_CAFE;
            let num_rounds = 3u8;

            let (sorf, codes, values, padded_dim) =
                build_sorf_with_dict_child(dim, num_rows, seed, num_rounds)?;

            let mut rng = XorShift64::new(seed);
            let query: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            let const_lhs = constant_vector_f32(&query, num_rows)?;

            let decoded = decode_sorf_dict(
                &codes,
                &values,
                padded_dim,
                dim as usize,
                num_rows,
                seed,
                num_rounds,
            )?;
            let expected: Vec<f32> = (0..num_rows)
                .map(|i| naive_dot(&decoded[i * dim as usize..(i + 1) * dim as usize], &query))
                .collect();

            let actual = eval_ip_f32(const_lhs, sorf, num_rows)?;
            assert_close_f32(&actual, &expected, 1e-2);
            Ok(())
        }
    }
}

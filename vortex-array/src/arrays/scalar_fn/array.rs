// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::ScalarFn;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::scalar_fn::ScalarFnRef;

// ScalarFnArray has a variable number of slots (one per child)

#[derive(Clone, Debug)]
pub struct ScalarFnData {
    pub(super) scalar_fn: ScalarFnRef,
}

impl Display for ScalarFnData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "scalar_fn: {}", self.scalar_fn)
    }
}

impl ScalarFnData {
    /// Create a new ScalarFnArray from a scalar function and its children.
    pub fn build(
        scalar_fn: ScalarFnRef,
        children: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            children.iter().all(|c| c.len() == len),
            "ScalarFnArray must have children equal to the array length"
        );
        Ok(Self { scalar_fn })
    }

    /// Get the scalar function bound to this array.
    #[inline(always)]
    pub fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }
}

pub trait ScalarFnArrayExt: TypedArrayRef<ScalarFn> {
    fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }

    fn child_at(&self, idx: usize) -> &ArrayRef {
        self.as_ref().slots()[idx]
            .as_ref()
            .vortex_expect("ScalarFnArray child slot")
    }

    fn child_count(&self) -> usize {
        self.as_ref().slots().len()
    }

    fn nchildren(&self) -> usize {
        self.child_count()
    }

    fn get_child(&self, idx: usize) -> &ArrayRef {
        self.child_at(idx)
    }

    fn iter_children(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        (0..self.child_count()).map(|idx| self.child_at(idx))
    }

    fn children(&self) -> Vec<ArrayRef> {
        self.iter_children().cloned().collect()
    }
}
impl<T: TypedArrayRef<ScalarFn>> ScalarFnArrayExt for T {}

impl Array<ScalarFn> {
    /// Create a new ScalarFnArray from a scalar function and its children.
    ///
    /// When a child has a refinement extension dtype
    /// (`ExtVTable::is_refinement() == true`) and the scalar function rejects the original
    /// input shape, refinement children are transparently peeled one level at a time until
    /// the fn accepts the shape or no refinement children remain to peel.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        mut children: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        let dtype = peel_refinements_and_resolve_dtype(&scalar_fn, &mut children)?;

        let data = ScalarFnData::build(scalar_fn.clone(), children.clone(), len)?;
        let vtable = ScalarFn { id: scalar_fn.id() };
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(vtable, dtype, len, data)
                    .with_slots(children.into_iter().map(Some).collect()),
            )
        })
    }
}

// TODO(connor): Refinement-preserving (e.g. `add(PositiveInt, PositiveInt) -> PositiveInt`) and
// refinement-changing (e.g. `negate(PositiveInt) -> NegativeInt`) semantics require an
// inverted-control hook like `ExtVTable::closure_result(&self, scalar_fn, arg_dtypes) ->
// Option<DType>` so the refinement can inspect the fn and preserve/rewrite itself. Until then,
// refinements are lost through any scalar fn whose `return_dtype` doesn't explicitly acceptthem
// (the peel loop strips the refinement and operates on storage instead).
/// Resolves the scalar function's return dtype against `children`, transparently peeling
/// refinement-typed children when the fn doesn't accept the original shape.
///
/// Why this exists:
///
/// Refinement extensions are logically stricter views of a storage dtype. For example, a
/// `DivisibleInt` is still represented as `u64`, and many scalar fns are semantically valid on
/// the storage values even though their `return_dtype` implementation only understands `u64`.
/// That means `add(DivisibleInt, DivisibleInt)` should be allowed to fall back to
/// `add(u64, u64)` when the fn does not explicitly accept the refinement.
///
/// Why this happens during construction rather than as a normal optimizer rewrite:
///
/// Public array construction goes through `ScalarFnFactoryExt::try_new_array`, which must call
/// `return_dtype` before a `ScalarFnArray` exists. A post-construction reduce rule would fire too
/// late. So refinement peeling has to be a construction-time fallback for dtype resolution.
///
/// Why peeling is intentionally narrow:
///
/// Only arrays that are *representation wrappers* can be peeled safely. Today that means:
/// - `ExtensionArray`, which can peel to `storage_array()`
/// - `ConstantArray` holding an extension scalar, which can peel to a constant storage scalar
///
/// Other encodings may preserve a refinement dtype without exposing storage as child slot 0. For
/// example, `mask(refined, mask)` still has a refinement dtype, but child 0 is the masked input
/// expression, not "the refinement storage". Peeling such arrays structurally via `nth_child(0)`
/// silently drops semantics and can produce wrong results. In those cases we must reject
/// construction until we have a semantics-preserving rewrite.
///
/// The algorithm is a fixpoint:
///
/// 1. Ask the scalar fn for its return dtype on the current children's dtypes.
/// 2. If it succeeds, return the result. This covers both the all-non-refinement case and the case
///    where the fn explicitly accepts a refinement type as a child.
/// 3. If it errors, try peeling one level from every child whose dtype is an extension dtype with
///    [`is_refinement`] set. Extension arrays peel to their storage arrays; constant extension
///    literals peel to constant storage scalars.
/// 4. If no children were peeled, return the original error.
/// 5. Otherwise, loop back to step 1 with the peeled children.
///
/// [`is_refinement`]: crate::dtype::extension::ExtVTable::is_refinement
pub(crate) fn peel_refinements_and_resolve_dtype(
    scalar_fn: &ScalarFnRef,
    children: &mut [ArrayRef],
) -> VortexResult<DType> {
    loop {
        let children_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();

        match scalar_fn.return_dtype(&children_dtypes) {
            Ok(dtype) => return Ok(dtype),
            Err(err) => {
                let any_peeled = peel_refinement_layers(children);
                if !any_peeled {
                    return Err(err);
                }
            }
        }
    }
}

// TODO(connor): Is it correct/ok to peel away refinement on all children at once? Do we need
// small-step semantics here?
/// Peels one layer of refinement extensions from all `children`.
///
/// Returns a flag indicating whether any child was actually peeled.
fn peel_refinement_layers(children: &mut [ArrayRef]) -> bool {
    let mut any_peeled = false;

    for child in children.iter_mut() {
        if let Some(peeled) = peel_refinement_child(child) {
            *child = peeled;
            any_peeled = true;
        }
    }

    any_peeled
}

/// Peel exactly one refinement wrapper when the array is a real representation wrapper.
///
/// This must not inspect arbitrary structural children. Only `ExtensionArray` and refinement
/// literals in `ConstantArray` are safe to unwrap here.
fn peel_refinement_child(child: &ArrayRef) -> Option<ArrayRef> {
    let DType::Extension(ext_dtype) = child.dtype() else {
        return None;
    };
    if !ext_dtype.is_refinement() {
        return None;
    }

    if let Some(ext_array) = child.as_opt::<Extension>() {
        return Some(ext_array.storage_array().clone());
    }

    if let Some(const_array) = child.as_opt::<Constant>() {
        let constant = const_array.scalar();
        let ext_scalar = constant.as_extension_opt()?;
        return Some(ConstantArray::new(ext_scalar.to_storage_scalar(), child.len()).into_array());
    }

    None
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::extension::EmptyMetadata;
    use crate::extension::tests::divisible_int::DivisibleInt;
    use crate::extension::tests::divisible_int::Divisor;
    use crate::extension::tests::even_divisible_int::EvenDivisibleInt;
    use crate::extension::uuid::Uuid;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::mask::Mask;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    fn divisible_int_dtype(divisor: u64) -> VortexResult<crate::dtype::extension::ExtDTypeRef> {
        ExtDType::<DivisibleInt>::try_new(
            Divisor(divisor),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .map(|dtype| dtype.erased())
    }

    fn divisible_int_array(divisor: u64, values: Vec<u64>) -> VortexResult<ArrayRef> {
        let ext_dtype = divisible_int_dtype(divisor)?;
        let storage =
            PrimitiveArray::new::<u64>(Buffer::<u64>::copy_from(&values), Validity::NonNullable)
                .into_array();
        Ok(ExtensionArray::try_new(ext_dtype, storage)?.into_array())
    }

    fn even_divisible_int_array(divisor: u64, values: Vec<u64>) -> VortexResult<ArrayRef> {
        let inner_dtype = ExtDType::<DivisibleInt>::try_new(
            Divisor(divisor),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .erased();
        let outer_dtype = ExtDType::<EvenDivisibleInt>::try_new(
            EmptyMetadata,
            DType::Extension(inner_dtype.clone()),
        )?
        .erased();
        let primitive =
            PrimitiveArray::new::<u64>(Buffer::<u64>::copy_from(&values), Validity::NonNullable)
                .into_array();
        let inner_ext = ExtensionArray::try_new(inner_dtype, primitive)?.into_array();
        Ok(ExtensionArray::try_new(outer_dtype, inner_ext)?.into_array())
    }

    fn uuid_array(row_count: usize) -> VortexResult<ArrayRef> {
        let fsl_element = DType::Primitive(PType::U8, Nullability::NonNullable);
        let fsl = DType::FixedSizeList(
            std::sync::Arc::new(fsl_element),
            16,
            Nullability::NonNullable,
        );
        let uuid_dtype = ExtDType::<Uuid>::try_new(Default::default(), fsl)?.erased();
        let bytes: Buffer<u8> = Buffer::copy_from(vec![0u8; row_count * 16]);
        let primitive = PrimitiveArray::new::<u8>(bytes, Validity::NonNullable).into_array();
        let storage = FixedSizeListArray::try_new(primitive, 16, Validity::NonNullable, row_count)?
            .into_array();
        Ok(ExtensionArray::try_new(uuid_dtype, storage)?.into_array())
    }

    /// `Binary(Add)` applied to two `DivisibleInt` children must peel one level, producing
    /// a ScalarFnArray whose dtype is `Primitive(U64)`. The refinement is lost by design
    /// (category A).
    #[test]
    fn peels_single_level_refinement_through_strict_add() -> VortexResult<()> {
        let lhs = divisible_int_array(3, vec![0u64, 3, 6])?;
        let rhs = divisible_int_array(3, vec![0u64, 3, 6])?;

        let sfn = Binary.bind(Operator::Add);
        let arr = Array::<ScalarFn>::try_new(sfn, vec![lhs, rhs], 3)?;

        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
        );
        Ok(())
    }

    /// `Binary(Add)` applied to two `EvenDivisibleInt` children must peel through both
    /// refinement layers (EvenDivisibleInt → DivisibleInt → U64) via the fixpoint loop.
    #[test]
    fn peels_two_level_refinement_chain_through_strict_add() -> VortexResult<()> {
        let lhs = even_divisible_int_array(3, vec![0u64, 6, 12])?;
        let rhs = even_divisible_int_array(3, vec![0u64, 6, 12])?;

        let sfn = Binary.bind(Operator::Add);
        let arr = Array::<ScalarFn>::try_new(sfn, vec![lhs, rhs], 3)?;

        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
        );
        Ok(())
    }

    #[test]
    fn peels_refinement_constant_literal_through_strict_add() -> VortexResult<()> {
        let ext_dtype = divisible_int_dtype(3)?;
        let lhs = divisible_int_array(3, vec![0u64, 3, 6])?;
        let rhs = ConstantArray::new(
            Scalar::extension_ref(ext_dtype, Scalar::from(3u64)),
            lhs.len(),
        )
        .into_array();

        let arr = Array::<ScalarFn>::try_new(Binary.bind(Operator::Add), vec![lhs, rhs], 3)?;

        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
        );
        Ok(())
    }

    #[test]
    fn array_builtins_binary_reuses_refinement_peeling() -> VortexResult<()> {
        let lhs = divisible_int_array(3, vec![0u64, 3, 6])?;
        let rhs = divisible_int_array(3, vec![0u64, 3, 6])?;

        let arr = lhs.binary(rhs, Operator::Add)?;

        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
        );
        Ok(())
    }

    /// Regression for an easy-to-make `nth_child(0)` peel bug.
    ///
    /// `mask(refined, false)` preserves the refinement dtype, but it is not an `ExtensionArray`.
    /// Peeling it via child 0 would discard the `mask(...)` node entirely, after which the retry
    /// loop would eventually type-check `add(lhs_storage, rhs_storage)` and silently change the
    /// program. The correct behavior today is to reject the construction.
    #[test]
    fn does_not_drop_masked_refinement_via_child_zero_peel() -> VortexResult<()> {
        let lhs = divisible_int_array(3, vec![0u64, 3, 6])?;
        let rhs = divisible_int_array(3, vec![0u64, 3, 6])?;
        let mask = ConstantArray::new(false, lhs.len()).into_array();
        let masked_lhs =
            Array::<ScalarFn>::try_new(Mask.bind(EmptyOptions), vec![lhs, mask], 3)?.into_array();

        let result =
            Array::<ScalarFn>::try_new(Binary.bind(Operator::Add), vec![masked_lhs, rhs], 3);

        assert!(
            result.is_err(),
            "non-Extension refinement producers must not be peeled via structural slot access"
        );
        Ok(())
    }

    /// `Uuid` is a non-refinement extension (`is_refinement() == false`). When fed to a
    /// strict primitive-typed scalar fn, peeling must NOT happen — the fn's original error
    /// is surfaced as-is.
    #[test]
    fn does_not_peel_non_refinement_extension() -> VortexResult<()> {
        let lhs = uuid_array(2)?;
        let rhs = uuid_array(2)?;

        let sfn = Binary.bind(Operator::Add);
        let result = Array::<ScalarFn>::try_new(sfn, vec![lhs, rhs], 2);

        assert!(
            result.is_err(),
            "Uuid is not a refinement; peel must not fire"
        );
        Ok(())
    }

    /// When the scalar fn's `return_dtype` already accepts the refinement (category B / C /
    /// D — specialization path), the peel loop must short-circuit without touching
    /// children.
    #[test]
    fn does_not_peel_when_scalar_fn_accepts_refinement() -> VortexResult<()> {
        // `Binary(Eq)` is a comparison, and its return_dtype accepts any pair of matching
        // extension dtypes (see `binary::mod::return_dtype` — comparisons allow extensions
        // as long as the two sides share the same dtype). So `Eq(DivisibleInt,
        // DivisibleInt)` succeeds at return_dtype time with no peel.
        let lhs = divisible_int_array(3, vec![0u64, 3, 6])?;
        let rhs = divisible_int_array(3, vec![0u64, 3, 6])?;

        let sfn = Binary.bind(Operator::Eq);
        let arr = Array::<ScalarFn>::try_new(sfn, vec![lhs, rhs], 3)?;

        // Comparison returns Bool; children retain their refinement dtypes.
        assert_eq!(arr.dtype(), &DType::Bool(Nullability::NonNullable));
        let child0 = arr.child_at(0);
        assert!(
            matches!(child0.dtype(), DType::Extension(ext) if ext.is::<DivisibleInt>()),
            "child 0 retained its DivisibleInt refinement (got {})",
            child0.dtype(),
        );
        Ok(())
    }
}

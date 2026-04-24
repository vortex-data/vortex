// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::ScalarFn;
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
    /// the fn accepts the shape or no refinement children remain to peel. See
    /// [`peel_refinements_and_resolve_dtype`] for details.
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: Vec<ArrayRef>,
        len: usize,
    ) -> VortexResult<Self> {
        let (dtype, children) = peel_refinements_and_resolve_dtype(&scalar_fn, children)?;
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

/// Resolves the scalar function's return dtype against `children`, transparently peeling
/// refinement-typed children when the fn doesn't accept the original shape.
///
/// The algorithm is a fixpoint:
///
/// 1. Ask the scalar fn for its return dtype on the current children's dtypes.
/// 2. If it succeeds, return the result. This covers both the all-non-refinement case
///    and the case where the fn explicitly accepts a refinement (category B / C / D from
///    the plan — the fn has authored a specialization).
/// 3. If it errors, try peeling one level from every child whose dtype is an extension
///    dtype with [`is_refinement`] set. Replace each with its storage array.
/// 4. If no children were peeled, return the original error.
/// 5. Otherwise, loop back to step 1 with the peeled children.
///
/// This is the blanket implementation of category A (refinement-transparent scalar fns):
/// when a generic fn doesn't know about a refinement, the refinement is lost and the fn
/// operates on the source storage instead. Refinement-preserving semantics (categories C
/// and D) require the scalar fn to author `return_dtype` to accept the refinement, at
/// which point step 2 short-circuits.
///
// TODO(connor): Categories C (refinement-preserving, e.g. `add(PositiveInt, PositiveInt)
// -> PositiveInt`) and D (refinement-changing, e.g. `negate(PositiveInt) ->
// NegativeInt`) cannot be handled by per-fn specialization because `vortex-array`
// cannot depend on downstream crates that define refinements. The future direction is
// an inverted-control hook on the refinement vtable itself — something like
// `ExtVTable::closure_result(&self, scalar_fn, arg_dtypes) -> Option<DType>` — so the
// refinement gets to inspect the scalar fn it's being fed into and optionally preserve
// or rewrite itself in the output. Until that exists, refinements lose their
// refinement-ness through any scalar fn that doesn't explicitly know about them.
///
/// [`is_refinement`]: crate::dtype::extension::ExtVTable::is_refinement
fn peel_refinements_and_resolve_dtype(
    scalar_fn: &ScalarFnRef,
    mut children: Vec<ArrayRef>,
) -> VortexResult<(DType, Vec<ArrayRef>)> {
    loop {
        let arg_dtypes: Vec<_> = children.iter().map(|c| c.dtype().clone()).collect();
        match scalar_fn.return_dtype(&arg_dtypes) {
            Ok(dtype) => return Ok((dtype, children)),
            Err(err) => {
                let mut any_peeled = false;
                let mut new_children = Vec::with_capacity(children.len());
                for child in children.into_iter() {
                    let is_refinement = matches!(
                        child.dtype(),
                        DType::Extension(ext) if ext.is_refinement()
                    );
                    if is_refinement {
                        let storage = child.nth_child(0).ok_or_else(|| {
                            vortex_err!("refinement extension array is missing its storage slot")
                        })?;
                        new_children.push(storage);
                        any_peeled = true;
                    } else {
                        new_children.push(child);
                    }
                }
                if !any_peeled {
                    return Err(err);
                }
                children = new_children;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::extension::EmptyMetadata;
    use crate::extension::tests::divisible_int::DivisibleInt;
    use crate::extension::tests::divisible_int::Divisor;
    use crate::extension::tests::even_divisible_int::EvenDivisibleInt;
    use crate::extension::uuid::Uuid;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::operators::Operator;
    use crate::validity::Validity;

    fn divisible_int_array(divisor: u64, values: Vec<u64>) -> VortexResult<ArrayRef> {
        let ext_dtype = ExtDType::<DivisibleInt>::try_new(
            Divisor(divisor),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )?
        .erased();
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

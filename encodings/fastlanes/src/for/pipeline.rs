// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::WrappingAdd;
use vortex_array::Array;
use vortex_array::compute::Operator as BinaryOperator;
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::operators::{
    BindContext, Operator, OperatorRef, ScalarCompareOperator,
};
use vortex_array::pipeline::vec::VectorId;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{Element, Kernel, KernelContext, PipelineVTable, VType};
use vortex_dtype::{NativePType, PType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::{FoRArray, FoRVTable};

impl PipelineVTable<FoRVTable> for FoRVTable {
    fn to_operator(array: &FoRArray) -> VortexResult<Option<OperatorRef>> {
        let Some(op) = array.encoded.to_operator()? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(FoROperator {
            child: [op],
            reference: array.reference.clone(),
            ptype: array.ptype(),
            encoded_ptype: array.encoded.dtype().as_ptype(),
        })))
    }
}

#[derive(Debug, Hash)]
pub struct FoROperator {
    child: [OperatorRef; 1],
    reference: Scalar,
    ptype: PType,
    encoded_ptype: PType,
}

impl Operator for FoROperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        VType::Primitive(self.ptype)
    }

    fn children(&self) -> &[OperatorRef] {
        &self.child
    }

    fn with_children(&self, mut children: Vec<OperatorRef>) -> OperatorRef {
        assert_eq!(children.len(), 1);
        Arc::new(FoROperator {
            child: [children.remove(0)],
            reference: self.reference.clone(),
            ptype: self.ptype,
            encoded_ptype: self.encoded_ptype,
        })
    }

    // TODO(joe): support in-place, FoR is in-place, but this is not implemented.
    fn in_place(&self) -> bool {
        false
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let VType::Primitive(ptype) = self.vtype() else {
            vortex_bail!("FoROperator only supports primitive types");
        };

        match_each_integer_ptype!(ptype, |T| {
            match_each_integer_ptype!(self.encoded_ptype, |E| {
                Ok(Box::new(FoRKernel::<T, E> {
                    child: ctx.children()[0],
                    reference: self
                        .reference
                        .as_primitive()
                        .typed_value::<T>()
                        .vortex_expect("reference value not of type T"),
                    _marker: PhantomData,
                }))
            })
        })
    }

    fn reduce_parent(&self, parent: OperatorRef) -> Option<OperatorRef> {
        let compare = parent.as_any().downcast_ref::<ScalarCompareOperator>()?;
        if compare.op != BinaryOperator::Eq && compare.op != BinaryOperator::NotEq {
            return None;
        }

        let new_ref = match_each_integer_ptype!(self.reference.as_primitive().ptype(), |P| {
            let compare = compare
                .scalar
                .as_primitive()
                .typed_value::<P>()
                .vortex_expect("must have ptype");
            let reference = self
                .reference
                .as_primitive()
                .typed_value::<P>()
                .vortex_expect("must have ptype");
            // TODO: handle overflow
            Scalar::from(compare.wrapping_sub(reference))
        });

        Some(Arc::new(ScalarCompareOperator::new(
            self.children()[0].clone(),
            compare.op,
            new_ref,
        )))
    }
}

// We could replace this with a binaryOp kernel
pub(crate) struct FoRKernel<T: NativePType, E: NativePType> {
    child: VectorId,
    reference: T,
    _marker: PhantomData<E>,
}

impl<T, E> Kernel for FoRKernel<T, E>
where
    T: NativePType + Element + WrappingAdd,
    E: NativePType + Element,
{
    fn seek(&mut self, _chunk_idx: usize) -> VortexResult<()> {
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &KernelContext,
        _selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let vec = ctx.vector(self.child);
        let values = unsafe { std::mem::transmute::<&[E], &[T]>(vec.as_slice::<E>()) };
        let out = out.as_slice_mut::<T>();

        values.iter().zip(out).for_each(|(value, out)| {
            *out = value.wrapping_add(&self.reference);
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::filter;
    use vortex_array::pipeline::export_canonical_pipeline_expr;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::BufferMut;
    use vortex_mask::Mask;

    use super::*;
    use crate::bitpack_to_best_bit_width;

    fn create_for_bitpacked_array<T: NativePType>(values: BufferMut<T>) -> VortexResult<FoRArray> {
        let primitive_array = values.into_array().to_primitive();

        // First apply FoR encoding
        let for_array = FoRArray::encode(primitive_array)?;

        // Then bitpack the residuals
        let residuals = for_array.encoded().to_primitive();
        let bitpacked = bitpack_to_best_bit_width(&residuals)?;

        // Create a new FoR array with bitpacked residuals
        FoRArray::try_new(bitpacked.into_array(), for_array.reference_scalar().clone())
    }

    #[test]
    fn test_for_pipeline() {
        let len = 8093usize;
        let mut rng = StdRng::seed_from_u64(0);
        let prim = (0i32..i32::try_from(len).unwrap())
            .map(|_| rng.random_range(0..120000))
            .collect::<PrimitiveArray>();
        let mask = Mask::AllTrue(len);
        let bitpack = bitpack_to_best_bit_width(&prim).unwrap();
        let array = FoRArray::try_new(bitpack.to_array(), Scalar::from(100i32)).unwrap();

        let res = export_canonical_pipeline_expr(
            array.dtype(),
            array.len(),
            array.to_operator().unwrap().unwrap().as_ref(),
            &mask,
        )
        .unwrap()
        .into_array();

        let expect = filter(array.as_ref(), &mask).unwrap();

        for i in 0..mask.true_count() {
            assert_eq!(res.scalar_at(i), expect.scalar_at(i), "{i}",);
        }
    }

    #[test]
    fn test_for_pipeline2() {
        let frac = 0.99;
        let len = 10;
        let mut rng = StdRng::seed_from_u64(0);
        let values = (0i16..len)
            .map(|_| rng.random_range(50..150))
            .collect::<BufferMut<_>>();
        let array = create_for_bitpacked_array(values).unwrap();

        let mask = (0..len)
            .map(|_| rng.random_bool(frac))
            .collect::<BooleanBuffer>();
        let mask = Mask::from_buffer(mask);

        let result = export_canonical_pipeline_expr(
            array.dtype(),
            array.len(),
            array.to_operator().unwrap().unwrap().as_ref(),
            &mask,
        )
        .unwrap()
        .into_array();

        let expect = filter(array.to_canonical().as_ref(), &mask).unwrap();

        for i in 0..mask.true_count() {
            assert_eq!(result.scalar_at(i), expect.scalar_at(i), "{}, {}", i, frac);
        }
    }
}

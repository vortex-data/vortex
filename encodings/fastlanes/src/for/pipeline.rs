// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::marker::PhantomData;
use std::task::Poll;

use num_traits::{Unsigned, WrappingAdd};
use vortex_array::Array;
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::operators::{BindContext, Operator};
use vortex_array::pipeline::types::{Element, VType};
use vortex_array::pipeline::vector::VectorId;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{Kernel, KernelContext};
use vortex_array::vtable::PipelineVTable;
use vortex_dtype::{
    NativePType, PType, match_each_integer_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::{FoRArray, FoRVTable};

impl PipelineVTable<FoRVTable> for FoRVTable {
    fn to_operator(array: &FoRArray) -> VortexResult<Box<dyn Operator>> {
        Ok(Box::new(FoROperator {
            child: [array.encoded.to_pipeline_plan()?],
            reference: array.reference.clone(),
            ptype: array.ptype(),
            encoded_ptype: array.encoded.dtype().as_ptype(),
        }))
    }

    fn to_pipeline(_array: &FoRArray) -> VortexResult<Box<dyn Kernel>> {
        todo!()
    }
}

#[derive(Debug, Hash)]
pub struct FoROperator {
    child: [Box<dyn Operator>; 1],
    reference: Scalar,
    ptype: PType,
    encoded_ptype: PType,
}

impl Operator for FoROperator {
    fn vtype(&self) -> VType {
        VType::Primitive(self.ptype)
    }

    fn children(&self) -> &[Box<dyn Operator>] {
        &self.child
    }

    fn in_place(&self) -> bool {
        true
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let VType::Primitive(ptype) = self.vtype() else {
            vortex_bail!("FoROperator only supports primitive types");
        };

        match_each_integer_ptype!(ptype, |T| {
            match_each_unsigned_integer_ptype!(self.encoded_ptype, |E| {
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
}

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
pub(crate) struct FoRKernel<T: NativePType, E: NativePType + Unsigned> {
    child: VectorId,
    reference: T,
    _marker: PhantomData<E>,
}

impl<T, E> Kernel for FoRKernel<T, E>
where
    T: NativePType + Element + WrappingAdd,
    E: NativePType + Element + Unsigned,
{
    fn seek(&mut self, _chunk_idx: usize) -> VortexResult<()> {
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let vec = ctx.vector(self.child);
        let values = unsafe { std::mem::transmute::<&[E], &[T]>(vec.as_slice::<E>()) };
        let out = out.as_slice_mut::<T>();
        for i in 0..selected.true_count() {
            out[i] = values[i].wrapping_add(&self.reference);
        }
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use itertools::Itertools;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::filter;
    use vortex_array::display::{DisplayArrayAs, DisplayOptions};
    use vortex_array::pipeline::canonical::export_canonical_pipeline_expr;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::BufferMut;
    use vortex_mask::Mask;

    use super::*;
    use crate::{FoRArray, bitpack_to_best_bit_width};

    fn create_for_bitpacked_array<T: NativePType>(values: BufferMut<T>) -> VortexResult<FoRArray> {
        let primitive_array = values.into_array().to_primitive().unwrap();

        println!("values: {}", primitive_array.display_tree());

        // First apply FoR encoding
        let for_array = FoRArray::encode(primitive_array.clone())?;
        println!("for_array: {}", for_array.display_tree());

        // Then bitpack the residuals
        let residuals = for_array.encoded().to_primitive()?;
        let bitpacked = bitpack_to_best_bit_width(&residuals)?;
        println!("bitpacked: {}", primitive_array.clone().display_tree());

        // Create a new FoR array with bitpacked residuals
        Ok(FoRArray::try_new(
            bitpacked.into_array(),
            for_array.reference_scalar().clone(),
        )?)
    }

    #[test]
    fn test_for_pipeline() {
        let len = 1024;
        let prim = (0i32..len).map(|x| x % 32).collect::<PrimitiveArray>();
        let mask = (0..len).map(|i| i % 32 != 0).collect::<Mask>();
        let bitpack = bitpack_to_best_bit_width(&prim).unwrap();
        let array = FoRArray::try_new(bitpack.to_array(), Scalar::from(100i32)).unwrap();

        let res = export_canonical_pipeline_expr(
            array.dtype(),
            array.len(),
            array.to_pipeline_plan().unwrap().as_ref(),
            &mask,
        )
        .unwrap()
        .into_array();

        println!("mask: {:?}", mask.to_boolean_buffer().iter().collect_vec());

        println!(
            "result: {}",
            DisplayArrayAs(
                res.as_ref(),
                DisplayOptions::CommaSeparatedScalars {
                    omit_comma_after_space: false
                }
            )
        );
        let expect = filter(array.as_ref(), &mask).unwrap();
        println!(
            "expect: {}",
            DisplayArrayAs(
                expect.as_ref(),
                DisplayOptions::CommaSeparatedScalars {
                    omit_comma_after_space: false
                }
            )
        );

        for i in 0..mask.true_count() {
            assert_eq!(
                res.scalar_at(i as usize).unwrap(),
                expect.scalar_at(i as usize).unwrap(),
                "{i}",
            );
        }
    }

    #[test]
    fn test_for_pipeline2() {
        // for frac in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999] {
        for frac in [0.99] {
            let len = 1024;
            let mut rng = StdRng::seed_from_u64(0);
            let values = (0i16..len)
                .map(|_| rng.random_range(50..150))
                .collect::<BufferMut<_>>();
            let array = create_for_bitpacked_array(values.clone()).unwrap();

            let mask = (0..len)
                .map(|_| rng.random_bool(frac))
                .collect::<BooleanBuffer>();
            let mask = Mask::from_buffer(mask);

            let result = export_canonical_pipeline_expr(
                array.dtype(),
                array.len(),
                array.to_pipeline_plan().unwrap().as_ref(),
                &mask,
            )
            .unwrap()
            .into_array();

            let expect = filter(array.to_canonical().unwrap().as_ref(), &mask).unwrap();

            println!(
                "mask: {:?}, tc {}",
                mask.to_boolean_buffer().set_indices().collect::<Vec<_>>(),
                mask.true_count()
            );

            println!(
                "\nresult: {}",
                DisplayArrayAs(
                    result.as_ref(),
                    DisplayOptions::CommaSeparatedScalars {
                        omit_comma_after_space: false
                    }
                )
            );
            println!(
                "\nexpect: {}",
                DisplayArrayAs(
                    &expect,
                    DisplayOptions::CommaSeparatedScalars {
                        omit_comma_after_space: false
                    }
                )
            );

            for i in 0..mask.true_count() {
                assert_eq!(
                    result.scalar_at(i).unwrap(),
                    expect.scalar_at(i).unwrap(),
                    "{}, {}",
                    i,
                    frac
                );
            }
        }
    }
}

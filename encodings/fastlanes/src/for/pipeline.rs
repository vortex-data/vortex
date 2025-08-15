// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use std::task::Poll;

use num_traits::WrappingAdd;
use vortex_array::Array;
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::operators::scalar_compare::ScalarCompareOperator;
use vortex_array::pipeline::operators::{BindContext, Operator};
use vortex_array::pipeline::types::{Element, VType};
use vortex_array::pipeline::vector::VectorId;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{Kernel, KernelContext};
use vortex_array::vtable::PipelineVTable;
use vortex_dtype::{NativePType, PType, match_each_integer_ptype, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::{FoRArray, FoRVTable};

impl PipelineVTable<FoRVTable> for FoRVTable {
    fn to_operator(array: &FoRArray) -> VortexResult<Arc<dyn Operator>> {
        Ok(Arc::new(FoROperator {
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
    child: [Arc<dyn Operator>; 1],
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

    fn children(&self) -> &[Arc<dyn Operator>] {
        &self.child
    }

    fn with_children(&self, mut children: Vec<Arc<dyn Operator>>) -> Arc<dyn Operator> {
        assert_eq!(children.len(), 1);
        Arc::new(FoROperator {
            child: [children.remove(0)],
            reference: self.reference.clone(),
            ptype: self.ptype,
            encoded_ptype: self.encoded_ptype,
        })
    }

    fn in_place(&self) -> bool {
        true
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

    fn reduce_parent(&self, parent: Arc<dyn Operator>) -> Option<Arc<dyn Operator>> {
        let compare = parent.as_any().downcast_ref::<ScalarCompareOperator>()?;

        let new_ref = match_each_native_ptype!(self.reference.as_primitive().ptype(), |P| {
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
            Scalar::from(compare - reference)
        });

        Some(Arc::new(ScalarCompareOperator::new(
            self.children()[0].clone(),
            compare.op,
            new_ref,
        )))
    }
}

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
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
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_expr::traversal::NodeExt;
    use vortex_expr::{ExprOperatorConverter, Scope, gt, lit, reduce_operator, root};
    use vortex_mask::Mask;

    use super::*;
    use crate::bitpack_to_best_bit_width;

    fn create_for_bitpacked_array<T: NativePType>(values: BufferMut<T>) -> VortexResult<FoRArray> {
        let primitive_array = values.into_array().to_primitive().unwrap();

        // First apply FoR encoding
        let for_array = FoRArray::encode(primitive_array)?;

        // Then bitpack the residuals
        let residuals = for_array.encoded().to_primitive()?;
        let bitpacked = bitpack_to_best_bit_width(&residuals)?;

        println!(
            "bitpacked: {}",
            DisplayArrayAs(bitpacked.as_ref(), DisplayOptions::default())
        );

        // Create a new FoR array with bitpacked residuals
        FoRArray::try_new(bitpacked.into_array(), for_array.reference_scalar().clone())
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
                res.scalar_at(i).unwrap(),
                expect.scalar_at(i).unwrap(),
                "{i}",
            );
        }
    }

    #[test]
    fn test_for_pipeline2() {
        for frac in [0.99] {
            let len = 10;
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

    #[test]
    fn test_expr_operator_converter() {
        const N: usize = 100;
        let array =
            create_for_bitpacked_array((0..N as i32).map(|x| x % 32).collect::<BufferMut<_>>())
                .unwrap()
                .to_array();
        let expr = gt(root(), lit(2));

        let mut m = vec![true; N];
        m[2] = false;
        let mask = Mask::from_iter(m.into_iter());
        println!("mask: {}", mask.true_count());

        let expect = expr
            .evaluate(&Scope::new(filter(&array, &mask).unwrap()))
            .unwrap();

        let mut converter = ExprOperatorConverter::new(array.clone());
        let operator = expr.fold(&mut converter).unwrap().value();
        let operator = reduce_operator(operator).unwrap();

        let result = export_canonical_pipeline_expr(
            &DType::Bool(NonNullable),
            array.len(),
            operator.as_ref(),
            &mask,
        )
        .unwrap()
        .into_array();

        println!(
            "result[..{}]: {}",
            result.len(),
            DisplayArrayAs(result.as_ref(), DisplayOptions::default()),
        );
        println!(
            "expect[..{}]: {}",
            expect.len(),
            DisplayArrayAs(expect.as_ref(), DisplayOptions::default())
        );

        for i in 0..mask.true_count() {
            assert_eq!(
                result.scalar_at(i).unwrap(),
                expect.scalar_at(i).unwrap(),
                "i: {i}"
            );
        }
    }
}

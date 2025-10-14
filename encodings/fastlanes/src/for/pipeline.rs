// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::WrappingAdd;
use vortex_array::Array;
use vortex_array::operator::{
    LengthBounds, Operator, OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{
    BindContext, Element, Kernel, KernelContext, PipelinedOperator, RowSelection, VectorId,
};
use vortex_array::vtable::PipelineVTable;
use vortex_dtype::{DType, NativePType, PType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::{FoRArray, FoRVTable};

impl PipelineVTable<FoRVTable> for FoRVTable {
    fn to_operator(array: &FoRArray) -> VortexResult<Option<OperatorRef>> {
        let Some(op) = array.encoded.to_operator()? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(FoROperator {
            child: op,
            dtype: array.dtype().clone(),
            reference: array.reference.clone(),
            ptype: array.ptype(),
            encoded_ptype: array.encoded.dtype().as_ptype(),
        })))
    }
}

#[derive(Debug)]
pub struct FoROperator {
    child: OperatorRef,
    reference: Scalar,
    dtype: DType,
    ptype: PType,
    encoded_ptype: PType,
}

impl OperatorHash for FoROperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.child.operator_hash(state);
        self.reference.hash(state);
        self.dtype.hash(state);
        self.ptype.hash(state);
        self.encoded_ptype.hash(state);
    }
}

impl OperatorEq for FoROperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.child.operator_eq(&other.child)
            && self.reference == other.reference
            && self.dtype == other.dtype
            && self.ptype == other.ptype
            && self.encoded_ptype == other.encoded_ptype
    }
}

impl Operator for FoROperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("fastlanes.for")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn bounds(&self) -> LengthBounds {
        self.child.bounds()
    }

    fn children(&self) -> &[OperatorRef] {
        std::slice::from_ref(&self.child)
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(FoROperator {
            child: children.into_iter().next().vortex_expect("missing child"),
            reference: self.reference.clone(),
            dtype: self.dtype.clone(),
            ptype: self.ptype,
            encoded_ptype: self.encoded_ptype,
        }))
    }

    fn reduce_parent(
        &self,
        _parent: OperatorRef,
        _child_idx: usize,
    ) -> VortexResult<Option<OperatorRef>> {
        Ok(None)
        // let Some(compare) = parent.as_any().downcast_ref::<CompareOperator>() else {
        //     return Ok(None);
        // };
        // if compare.op() != BinaryOperator::Eq && compare.op() != BinaryOperator::NotEq {
        //     return Ok(None);
        // }
        //
        // let new_ref = match_each_integer_ptype!(self.reference.as_primitive().ptype(), |P| {
        //     let compare = compare
        //         .scalar
        //         .as_primitive()
        //         .typed_value::<P>()
        //         .vortex_expect("must have ptype");
        //     let reference = self
        //         .reference
        //         .as_primitive()
        //         .typed_value::<P>()
        //         .vortex_expect("must have ptype");
        //     // TODO: handle overflow
        //     Scalar::from(compare.wrapping_sub(reference))
        // });
        //
        // Some(Arc::new(CompareOperator::new(
        //     self.children()[0].clone(),
        //     compare.op,
        //     new_ref,
        // )))
    }
}

impl PipelinedOperator for FoROperator {
    fn row_selection(&self) -> RowSelection {
        self.child
            .as_pipelined()
            .map(|p| p.row_selection())
            .unwrap_or(RowSelection::All)
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let DType::Primitive(ptype, _) = self.dtype() else {
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
    //
    // // TODO(joe): support in-place, FoR is in-place, but this is not implemented.
    // fn in_place(&self) -> bool {
    //     false
    // }

    fn vector_children(&self) -> Vec<usize> {
        vec![0]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
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
    fn step(
        &self,
        ctx: &KernelContext,
        _chunk_idx: usize,
        _selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let vec = ctx.vector(self.child);

        let values = unsafe { std::mem::transmute::<&[E], &[T]>(vec.as_array::<E>()) };
        let out_values = out.as_array_mut::<T>();

        // TODO(ngates): decide whether to iter ones of the selection mask
        values.iter().zip(out_values).for_each(|(value, out)| {
            *out = value.wrapping_add(&self.reference);
        });
        out.set_selection(vec.selection());

        Ok(())
    }
}
//
// #[cfg(test)]
// mod tests {
//     use arrow_buffer::BooleanBuffer;
//     use rand::prelude::StdRng;
//     use rand::{Rng, SeedableRng};
//     use vortex_array::arrays::PrimitiveArray;
//     use vortex_array::compute::filter;
//     use vortex_array::{IntoArray, ToCanonical};
//     use vortex_buffer::BufferMut;
//     use vortex_mask::Mask;
//
//     use super::*;
//     use crate::bitpack_to_best_bit_width;
//
//     fn create_for_bitpacked_array<T: NativePType>(values: BufferMut<T>) -> VortexResult<FoRArray> {
//         let primitive_array = values.into_array().to_primitive();
//
//         // First apply FoR encoding
//         let for_array = FoRArray::encode(primitive_array)?;
//
//         // Then bitpack the residuals
//         let residuals = for_array.encoded().to_primitive();
//         let bitpacked = bitpack_to_best_bit_width(&residuals)?;
//
//         // Create a new FoR array with bitpacked residuals
//         FoRArray::try_new(bitpacked.into_array(), for_array.reference_scalar().clone())
//     }
//
//     #[test]
//     fn test_for_pipeline() {
//         let len = 8093usize;
//         let mut rng = StdRng::seed_from_u64(0);
//         let prim = (0i32..i32::try_from(len).unwrap())
//             .map(|_| rng.random_range(0..120000))
//             .collect::<PrimitiveArray>();
//         let mask = Mask::AllTrue(len);
//         let bitpack = bitpack_to_best_bit_width(&prim).unwrap();
//         let array = FoRArray::try_new(bitpack.to_array(), Scalar::from(100i32)).unwrap();
//
//         let res = export_canonical_pipeline_expr(
//             array.dtype(),
//             array.len(),
//             array.to_operator().unwrap().unwrap().as_ref(),
//             &mask,
//         )
//         .unwrap()
//         .into_array();
//
//         let expect = filter(array.as_ref(), &mask).unwrap();
//
//         for i in 0..mask.true_count() {
//             assert_eq!(res.scalar_at(i), expect.scalar_at(i), "{i}",);
//         }
//     }
//
//     #[test]
//     fn test_for_pipeline2() {
//         let frac = 0.99;
//         let len = 10;
//         let mut rng = StdRng::seed_from_u64(0);
//         let values = (0i16..len)
//             .map(|_| rng.random_range(50..150))
//             .collect::<BufferMut<_>>();
//         let array = create_for_bitpacked_array(values).unwrap();
//
//         let mask = (0..len)
//             .map(|_| rng.random_bool(frac))
//             .collect::<BooleanBuffer>();
//         let mask = Mask::from_buffer(mask);
//
//         let result = export_canonical_pipeline_expr(
//             array.dtype(),
//             array.len(),
//             array.to_operator().unwrap().unwrap().as_ref(),
//             &mask,
//         )
//         .unwrap()
//         .into_array();
//
//         let expect = filter(array.to_canonical().as_ref(), &mask).unwrap();
//
//         for i in 0..mask.true_count() {
//             assert_eq!(result.scalar_at(i), expect.scalar_at(i), "{}, {}", i, frac);
//         }
//     }
// }

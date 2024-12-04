use vortex_array::compute::SumFn;
use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::{DictArray, DictEncoding};

impl SumFn<DictArray> for DictEncoding {
    fn sum(&self, _array: &DictArray, _ends: &[u64]) -> VortexResult<ArrayData> {
        // let mut hist = vec![0; array.values().len()];
        //
        // let values = array.values();
        //
        // array
        //     .codes()
        //     .into_primitive()
        //     .unwrap()
        //     .with_iterator(|iter| {
        //         iter.filter_map(|x| x).for_each(|scalar: &u32| {
        //             let idx = *scalar as usize;
        //             hist[idx] += 1
        //         })
        //     })?;
        //
        // if values.dtype().is_float() {
        //     Ok(Scalar::from(hist.into_iter().enumerate().fold(
        //         0 as f32,
        //         |acc, (idx, count)| {
        //             acc + (f32::try_from(scalar_at(&values, idx).expect("value")).expect("cast")
        //                 * count as f32)
        //         },
        //     )))
        // } else if values.dtype().is_int() {
        //     Ok(Scalar::from(hist.into_iter().enumerate().fold(
        //         0 as i32,
        //         |acc, (idx, count)| {
        //             acc + (i32::try_from(scalar_at(&values, idx).expect("value")).expect("cast")
        //                 * count)
        //         },
        //     )))
        // } else {
        //     todo!()
        // }
        todo!()
    }
}

use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::FoRArray;

impl ArrayOperationsImpl for FoRArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        FoRArray::try_new(
            self.encoded().slice(start, stop)?,
            self.reference_scalar().clone(),
        )
        .map(|a| a.into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let encoded_pvalue = self
            .encoded()
            .scalar_at(index)?
            .reinterpret_cast(self.ptype());
        let encoded_pvalue = encoded_pvalue.as_primitive();
        let reference = self.reference_scalar();
        let reference = reference.as_primitive();

        Ok(match_each_integer_ptype!(self.ptype(), |$P| {
            encoded_pvalue
                .typed_value::<$P>()
                .map(|v|
                     v.wrapping_add(
                         reference
                             .typed_value::<$P>()
                             .vortex_expect("FoRArray Reference value cannot be null")))
                .map(|v| Scalar::primitive::<$P>(v, self.dtype().nullability()))
                .unwrap_or_else(|| Scalar::null(self.dtype().clone()))
        }))
    }
}

#[cfg(test)]
mod test {
    use vortex_array::Array;
    use vortex_array::arrays::PrimitiveArray;

    use crate::FoRArray;

    #[test]
    fn for_scalar_at() {
        let for_arr =
            FoRArray::encode(PrimitiveArray::from_iter([-100, 1100, 1500, 1900])).unwrap();
        assert_eq!(for_arr.scalar_at(0).unwrap(), (-100).into());
        assert_eq!(for_arr.scalar_at(1).unwrap(), 1100.into());
        assert_eq!(for_arr.scalar_at(2).unwrap(), 1500.into());
        assert_eq!(for_arr.scalar_at(3).unwrap(), 1900.into());
    }
}

mod compare;

use vortex_array::compute::{
    filter, scalar_at, slice, take, CompareFn, FilterFn, ScalarAtFn, SliceFn, TakeFn,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::{match_each_alp_float_ptype, ALPArray, ALPEncoding, ALPFloat};

impl ComputeVTable for ALPEncoding {
    fn compare_fn(&self) -> Option<&dyn CompareFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }
}

impl ScalarAtFn<ALPArray> for ALPEncoding {
    fn scalar_at(&self, array: &ALPArray, index: usize) -> VortexResult<Scalar> {
        if !array.encoded().is_valid(index)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        if let Some(patches) = array.patches() {
            if let Some(patch) = patches.get_patched(index)? {
                return patch.cast(array.dtype());
            }
        }

        let encoded_val = scalar_at(array.encoded(), index)?;

        Ok(match_each_alp_float_ptype!(array.ptype(), |$T| {
            let encoded_val: <$T as ALPFloat>::ALPInt = encoded_val.as_ref().try_into().unwrap();
            Scalar::primitive(<$T as ALPFloat>::decode_single(
                encoded_val,
                array.exponents(),
            ), array.dtype().nullability())
        }))
    }
}

impl TakeFn<ALPArray> for ALPEncoding {
    fn take(&self, array: &ALPArray, indices: &Array) -> VortexResult<Array> {
        let taken_encoded = take(array.encoded(), indices)?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices))
            .transpose()?
            .flatten()
            .map(|p| {
                p.cast_values(
                    &array
                        .dtype()
                        .with_nullability(taken_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(ALPArray::try_new(taken_encoded, array.exponents(), taken_patches)?.into_array())
    }
}

impl SliceFn<ALPArray> for ALPEncoding {
    fn slice(&self, array: &ALPArray, start: usize, end: usize) -> VortexResult<Array> {
        Ok(ALPArray::try_new(
            slice(array.encoded(), start, end)?,
            array.exponents(),
            array
                .patches()
                .map(|p| p.slice(start, end))
                .transpose()?
                .flatten(),
        )?
        .into_array())
    }
}

impl FilterFn<ALPArray> for ALPEncoding {
    fn filter(&self, array: &ALPArray, mask: &Mask) -> VortexResult<Array> {
        let patches = array
            .patches()
            .map(|p| p.filter(mask))
            .transpose()?
            .flatten();

        Ok(
            ALPArray::try_new(filter(&array.encoded(), mask)?, array.exponents(), patches)?
                .into_array(),
        )
    }
}

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::array::primitive::PrimitiveArray;
use crate::array::varbinview::VarBinViewArray;
use crate::array::BinaryView;
use crate::validity::ArrayValidity;
use crate::IntoCanonical;

impl ArrayAccessor<[u8]> for VarBinViewArray {
    fn with_iterator<F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a [u8]>>) -> R, R>(
        &self,
        f: F,
    ) -> VortexResult<R> {
        let bytes: Vec<PrimitiveArray> = (0..self.metadata().buffer_lens.len())
            .map(|i| self.buffer(i).into_canonical()?.into_primitive())
            .try_collect()?;
        let bytes_slices: Vec<&[u8]> = bytes.iter().map(|b| b.maybe_null_slice::<u8>()).collect();
        let views: Vec<BinaryView> = self.binary_views()?.collect();
        let validity = self.logical_validity().to_null_buffer()?;

        match validity {
            None => {
                let mut iter = views.iter().map(|view| {
                    if view.is_inlined() {
                        Some(view.as_inlined().value())
                    } else {
                        Some(
                            &bytes_slices[view.as_view().buffer_index() as usize]
                                [view.as_view().to_range()],
                        )
                    }
                });
                Ok(f(&mut iter))
            }
            Some(validity) => {
                let mut iter = views.iter().zip(validity.iter()).map(|(view, valid)| {
                    if valid {
                        if view.is_inlined() {
                            Some(view.as_inlined().value())
                        } else {
                            Some(
                                &bytes_slices[view.as_view().buffer_index() as usize]
                                    [view.as_view().to_range()],
                            )
                        }
                    } else {
                        None
                    }
                });
                Ok(f(&mut iter))
            }
        }
    }
}

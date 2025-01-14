use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::array::varbinview::VarBinViewArray;
// use crate::array::BinaryView;
use crate::validity::ArrayValidity;

impl ArrayAccessor<[u8]> for VarBinViewArray {
    fn with_iterator<F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a [u8]>>) -> R, R>(
        &self,
        f: F,
    ) -> VortexResult<R> {
        let bytes = (0..self.metadata().buffer_lens.len())
            .map(|i| self.buffer(i))
            .collect::<Vec<_>>();

        let views = self.views();
        let validity = self.logical_validity().to_null_buffer()?;

        match validity {
            None => {
                let mut iter = views.iter().map(|view| {
                    if view.is_inlined() {
                        Some(view.as_inlined().value())
                    } else {
                        Some(
                            &bytes[view.as_view().buffer_index() as usize]
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
                                &bytes[view.as_view().buffer_index() as usize]
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

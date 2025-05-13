use std::iter;

use vortex_error::VortexResult;

use crate::ToCanonical;
use crate::accessor::ArrayAccessor;
use crate::arrays::varbinview::VarBinViewArray;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl ArrayAccessor<[u8]> for VarBinViewArray {
    fn with_iterator<F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a [u8]>>) -> R, R>(
        &self,
        f: F,
    ) -> VortexResult<R> {
        let bytes = (0..self.nbuffers())
            .map(|i| self.buffer(i))
            .collect::<Vec<_>>();

        let views = self.views();

        match self.validity() {
            Validity::NonNullable | Validity::AllValid => {
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
            Validity::AllInvalid => Ok(f(&mut iter::repeat_n(None, views.len()))),
            Validity::Array(v) => {
                let validity = v.to_bool()?;
                let mut iter = views
                    .iter()
                    .zip(validity.boolean_buffer())
                    .map(|(view, valid)| {
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

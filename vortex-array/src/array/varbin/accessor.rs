use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::array::varbin::VarBinArray;
use crate::validity::ArrayValidity;
use crate::variants::PrimitiveArrayTrait;
use crate::IntoArrayVariant;

impl ArrayAccessor<[u8]> for VarBinArray {
    fn with_iterator<F, R>(&self, f: F) -> VortexResult<R>
    where
        F: for<'a> FnOnce(&mut (dyn Iterator<Item = Option<&'a [u8]>>)) -> R,
    {
        let offsets = self.offsets().into_primitive()?;
        let validity = self.logical_validity().to_null_buffer()?;

        // TODO(ngates): what happens if bytes is much larger than sliced_bytes?
        let bytes = self.bytes();
        let bytes = bytes.as_slice();

        match_each_integer_ptype!(offsets.ptype(), |$T| {
            let offsets = offsets.as_slice::<$T>();

            match validity {
                None => {
                    let mut iter = offsets
                        .iter()
                        .zip(offsets.iter().skip(1))
                        .map(|(start, end)| Some(&bytes[*start as usize..*end as usize]));
                    Ok(f(&mut iter))
                }
                Some(validity) => {
                    let mut iter = offsets
                        .iter()
                        .zip(offsets.iter().skip(1))
                        .zip(validity.iter())
                        .map(|((start, end), valid)| {
                            if valid {
                                Some(&bytes[*start as usize..*end as usize])
                            } else {
                                None
                            }
                        });
                    Ok(f(&mut iter))
                }
            }
        })
    }
}

use std::iter;

use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::arrays::varbin::VarBinArray;
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::IntoArrayVariant;

impl ArrayAccessor<[u8]> for VarBinArray {
    fn with_iterator<F, R>(&self, f: F) -> VortexResult<R>
    where
        F: for<'a> FnOnce(&mut (dyn Iterator<Item = Option<&'a [u8]>>)) -> R,
    {
        let offsets = self.offsets().into_primitive()?;
        let validity = self.validity();

        let bytes = self.bytes();
        let bytes = bytes.as_slice();

        match_each_integer_ptype!(offsets.ptype(), |$T| {
            let offsets = offsets.as_slice::<$T>();

            match validity {
                Validity::NonNullable | Validity::AllValid => {
                    let mut iter = offsets
                        .windows(2)
                        .map(|w| Some(&bytes[w[0] as usize..w[1] as usize]));
                    Ok(f(&mut iter))
                }
                Validity::AllInvalid => Ok(f(&mut iter::repeat_n(None, self.len()))),
                Validity::Array(v) => {
                    let validity_buf = v.into_bool()?.boolean_buffer();
                    let mut iter = offsets
                        .windows(2)
                        .zip(validity_buf.iter())
                        .map(|(w, valid)| {
                            if valid {
                                Some(&bytes[w[0] as usize..w[1] as usize])
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

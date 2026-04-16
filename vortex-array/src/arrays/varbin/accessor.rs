// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use vortex_error::VortexExpect;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::accessor::ArrayAccessor;
use crate::arrays::VarBinArray;
use crate::arrays::varbin::VarBinArrayExt;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

impl ArrayAccessor<[u8]> for VarBinArray {
    fn with_iterator<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a [u8]>>) -> R,
    {
        #[expect(deprecated)]
        let offsets = self.offsets().to_primitive();
        let validity = self
            .validity()
            .vortex_expect("varbin validity should be derivable");

        let bytes = self.bytes();
        let bytes = bytes.as_slice();

        match_each_integer_ptype!(offsets.ptype(), |T| {
            let offsets = offsets.as_slice::<T>();

            #[allow(clippy::cast_possible_truncation)]
            match validity {
                Validity::NonNullable | Validity::AllValid => {
                    let mut iter = offsets
                        .windows(2)
                        .map(|w| Some(&bytes[w[0] as usize..w[1] as usize]));
                    f(&mut iter)
                }
                Validity::AllInvalid => f(&mut iter::repeat_n(None, self.len())),
                Validity::Array(v) => {
                    #[expect(deprecated)]
                    let validity = v.to_bool().into_bit_buffer();
                    let mut iter = offsets
                        .windows(2)
                        .zip(validity.iter())
                        .map(|(w, valid)| valid.then(|| &bytes[w[0] as usize..w[1] as usize]));
                    f(&mut iter)
                }
            }
        })
    }
}

impl ArrayAccessor<[u8]> for &VarBinArray {
    fn with_iterator<F, R>(&self, f: F) -> R
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a [u8]>>) -> R,
    {
        <VarBinArray as ArrayAccessor<[u8]>>::with_iterator(*self, f)
    }
}

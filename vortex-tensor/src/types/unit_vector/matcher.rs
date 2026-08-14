// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;

use crate::types::unit_vector::UnitVector;
use crate::types::vector::VectorMatcherMetadata;
use crate::types::vector::match_vector_storage;

/// Matches exactly the [`UnitVector`] extension type.
pub struct AnyUnitVector;

impl Matcher for AnyUnitVector {
    type Match<'a> = VectorMatcherMetadata;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        ext_dtype
            .is::<UnitVector>()
            .then(|| match_vector_storage(ext_dtype))
    }
}

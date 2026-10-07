// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::BetweenReduce;

impl BetweenReduce for Extension {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // Matching metadata is required before comparing storage (e.g. timestamp units).
        if !array.dtype().eq_ignore_nullability(lower.dtype())
            || !array.dtype().eq_ignore_nullability(upper.dtype())
        {
            return Ok(None);
        }
        let (Some(lower), Some(upper)) = (storage_bound(lower), storage_bound(upper)) else {
            return Ok(None);
        };

        array
            .storage_array()
            .clone()
            .between(lower, upper, options.clone())
            .map(Some)
    }
}

fn storage_bound(bound: &ArrayRef) -> Option<ArrayRef> {
    if let Some(scalar) = bound.as_constant() {
        return Some(
            ConstantArray::new(scalar.as_extension().to_storage_scalar(), bound.len()).into_array(),
        );
    }
    bound
        .as_opt::<Extension>()
        .map(|extension| extension.storage_array().clone())
}

#[cfg(test)]
mod tests;

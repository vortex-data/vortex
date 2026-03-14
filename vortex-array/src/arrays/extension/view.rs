// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use crate::ArrayRef;
use crate::DynArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtVTable;
use crate::matcher::Matcher;

/// A typed view of an extension array.
pub struct ExtArray<'a, V: ExtVTable> {
    ext_dtype: &'a ExtDType<V>,
    array: &'a ExtensionArray,
}

impl<'a, V: ExtVTable> ExtArray<'a, V> {
    pub fn try_new(array: &'a ExtensionArray) -> Option<Self> {
        let ext_dtype = array.ext_dtype().downcast_ref::<V>()?;
        Some(Self { ext_dtype, array })
    }

    pub fn ext_dtype(&self) -> &ExtDType<V> {
        self.ext_dtype
    }

    pub fn storage_array(&self) -> &ArrayRef {
        self.array.storage_array()
    }
}

/// A matcher that matches an [`ExtensionArray`] with a specific [`ExtVTable`] type.
///
/// Similar to [`ExactScalarFn`](crate::arrays::scalar_fn::ExactScalarFn) for scalar functions,
/// this provides typed access to the extension array view ([`ExtArray<V>`]).
#[derive(Debug, Default)]
pub struct ExactExtArray<V: ExtVTable>(PhantomData<V>);

impl<V: ExtVTable> Matcher for ExactExtArray<V> {
    type Match<'a> = ExtArray<'a, V>;

    fn matches(array: &dyn DynArray) -> bool {
        if let Some(ext_array) = array.as_opt::<Extension>() {
            ext_array.downcast_ref::<V>().is_some()
        } else {
            false
        }
    }

    fn try_match(array: &dyn DynArray) -> Option<Self::Match<'_>> {
        let ext_array = array.as_opt::<Extension>()?;
        ext_array.downcast_ref::<V>()
    }
}

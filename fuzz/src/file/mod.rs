// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod corruption;
#[cfg(feature = "native")]
mod io;
mod session;
mod settings;

use std::panic;
use std::sync::Once;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
pub use corruption::*;
#[cfg(feature = "native")]
pub use io::run_file_io;
pub use session::*;
pub use settings::*;
use vortex::expr::Expression;
use vortex_array::ArrayRef;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::dtype::DType;
use vortex_array::dtype::StructFields;
use vortex_expr::arbitrary::filter_expr;
use vortex_expr::arbitrary::projection_expr;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_set::HashSet;

use crate::FUZZ_FILE_ARRAY_MAX_LEN;
use crate::FuzzSource;

#[derive(Debug)]
pub struct FuzzFileAction {
    pub array: ArrayRef,
    pub projection_expr: Option<Expression>,
    pub filter_expr: Option<Expression>,
    pub write_settings: WriteSettings,
}

impl<'a> Arbitrary<'a> for FuzzFileAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Self::generate(u)
    }
}

impl FuzzFileAction {
    /// Generate an array, write settings, and the projection and filter to scan it back with.
    pub fn generate(src: &mut impl FuzzSource) -> arbitrary::Result<Self> {
        let array = src
            .bytes(|u| {
                ArbitraryArray::arbitrary_with_config(
                    u,
                    &ArbitraryArrayConfig {
                        dtype: None,
                        len: 0..=FUZZ_FILE_ARRAY_MAX_LEN,
                    },
                )
            })?
            .0;
        let dtype = array.dtype().clone();
        let mut write_settings = src.bytes(|u| u.arbitrary::<WriteSettings>())?;
        write_settings.disable_editions = false;
        write_settings.unrestricted = false;
        Ok(FuzzFileAction {
            array,
            projection_expr: src.bytes(|u| projection_expr(u, &dtype))?,
            filter_expr: src.bytes(|u| filter_expr(u, &dtype))?,
            write_settings,
        })
    }
}

pub(crate) fn report_on_panic(report: fn()) {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            report();
            previous(info);
        }));
    });
}

pub fn has_nullable_struct(dtype: &DType) -> bool {
    dtype.is_struct() && dtype.is_nullable()
        || dtype
            .as_struct_fields_opt()
            .map(|sdt| sdt.fields().any(|dtype| has_nullable_struct(&dtype)))
            .unwrap_or(false)
        || dtype
            .as_list_element_opt()
            .map(|e| has_nullable_struct(e.as_ref()))
            .unwrap_or(false)
}

pub fn has_duplicate_field_names(dtype: &DType) -> bool {
    if let Some(struct_dtype) = dtype.as_struct_fields_opt() {
        struct_has_duplicate_names(struct_dtype)
    } else if let Some(list_elem) = dtype.as_list_element_opt() {
        has_duplicate_field_names(list_elem)
    } else {
        false
    }
}

fn struct_has_duplicate_names(struct_dtype: &StructFields) -> bool {
    HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
        != struct_dtype.names().len()
        || struct_dtype
            .fields()
            .any(|dtype| has_duplicate_field_names(&dtype))
}

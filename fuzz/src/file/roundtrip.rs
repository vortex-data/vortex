// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use itertools::Itertools;
use vortex::expr::lit;
use vortex::expr::root;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_error::VortexExpect;
use vortex_expr::ArrayExprExt;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;

use super::FuzzFileAction;
use super::has_duplicate_field_names;
use super::has_nullable_struct;
use crate::RUNTIME;
use crate::SESSION;

/// Inputs to the `file_io` target.
#[derive(Debug)]
pub enum FileRoundTrip {
    /// An arbitrary file with generated write settings and scan expressions.
    File(FuzzFileAction),
    /// A Variant column and an optional input batch boundary.
    Variant {
        array: ArrayRef,
        split_at: Option<usize>,
    },
}

impl<'a> Arbitrary<'a> for FileRoundTrip {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Preserve existing corpus decoding outside the Variant prefix.
        if u.peek_bytes(1) == Some(&[0xff]) {
            u.bytes(1)?;
            return super::variant::arbitrary_file(u);
        }
        Ok(Self::File(u.arbitrary()?))
    }
}

impl FileRoundTrip {
    /// Writes, reopens, and scans the file. Returns false for unsupported schemas.
    pub fn check(self) -> bool {
        let (array_data, split_at, projection_expr, filter_expr, write_options) = match self {
            Self::File(action) => (
                action.array,
                None,
                action.projection_expr,
                action.filter_expr,
                action.write_settings.write_options(),
            ),
            Self::Variant { array, split_at } => {
                (array, split_at, None, None, SESSION.write_options())
            }
        };
        if has_nullable_struct(array_data.dtype()) || has_duplicate_field_names(array_data.dtype())
        {
            return false;
        }

        let mut ctx = SESSION.create_execution_ctx();
        let expected_array = if filter_expr.is_none() && projection_expr.is_none() {
            array_data.clone()
        } else {
            let bool_mask = array_data
                .clone()
                .apply(&filter_expr.clone().unwrap_or_else(|| lit(true)))
                .vortex_expect("filter expression evaluation should succeed in fuzz test");
            let bool_mask_bool = bool_mask
                .execute::<BoolArray>(&mut ctx)
                .vortex_expect("execute bool");
            let mask = bool_mask_bool.to_mask_fill_null_false(&mut ctx);
            let filtered = array_data
                .filter(mask)
                .vortex_expect("filter operation should succeed in fuzz test");
            filtered
                .apply(&projection_expr.clone().unwrap_or_else(root))
                .vortex_expect("projection expression evaluation should succeed in fuzz test")
        };

        let input = if let Some(split) = split_at {
            let first = array_data.slice(0..split).vortex_expect("valid split");
            let second = array_data
                .slice(split..array_data.len())
                .vortex_expect("valid split");
            ChunkedArray::from_iter([first, second]).into_array()
        } else {
            array_data
        };

        let mut full_buff = Vec::new();
        let _footer = write_options
            .blocking(&*RUNTIME)
            .write(&mut full_buff, input.to_array_iterator())
            .vortex_expect("file write should succeed in fuzz test");

        let file = SESSION
            .open_options()
            .open_buffer(full_buff)
            .vortex_expect("open_buffer should succeed in fuzz test");
        let projection = projection_expr
            .unwrap_or_else(root)
            .bind(file.dtype())
            .and_then(|expr| expr.optimize_recursive())
            .vortex_expect("projection should bind in fuzz test");
        let filter = filter_expr
            .map(|filter| {
                filter
                    .bind(file.dtype())
                    .and_then(|expr| expr.optimize_recursive())
            })
            .transpose()
            .vortex_expect("filter should bind in fuzz test");
        let mut output = file
            .scan()
            .vortex_expect("scan should succeed in fuzz test")
            .with_projection(projection)
            .with_some_filter(filter)
            .into_array_iter(&*RUNTIME)
            .vortex_expect("into_array_iter should succeed in fuzz test")
            .try_collect::<_, Vec<_>, _>()
            .vortex_expect("collect should succeed in fuzz test");

        let output_array = match output.len() {
            0 => Canonical::empty(expected_array.dtype()).into_array(),
            1 => output.pop().vortex_expect("one output"),
            _ => ChunkedArray::from_iter(output).into_array(),
        };

        assert_arrays_eq!(output_array, expected_array, &mut ctx);

        true
    }
}

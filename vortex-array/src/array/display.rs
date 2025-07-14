// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use itertools::Itertools as _;
use vortex_error::VortexExpect as _;

use crate::Array;

pub struct DisplayArray<'a>(pub &'a dyn Array);

impl Display for DisplayArray<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_as(f, &DisplayOptions::default())
    }
}

pub enum DisplayOptions {
    /// `EncodingId(dtype, len=123)`
    MetadataOnly,
    /// `[1i32, 2i32, 3i32]`
    CommaSeparatedScalars { space_after_comma: bool },
}

impl Default for DisplayOptions {
    fn default() -> Self {
        Self::CommaSeparatedScalars {
            space_after_comma: true,
        }
    }
}

pub struct DisplayArrayAs<'a>(pub &'a dyn Array, pub &'a DisplayOptions);

impl Display for DisplayArrayAs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_as(f, self.1)
    }
}

impl dyn Array + '_ {
    pub fn fmt_as(
        &self,
        f: &mut std::fmt::Formatter,
        options: &DisplayOptions,
    ) -> std::fmt::Result {
        match options {
            DisplayOptions::MetadataOnly => {
                write!(
                    f,
                    "{}({}, len={})",
                    self.encoding_id(),
                    self.dtype(),
                    self.len()
                )
            }
            DisplayOptions::CommaSeparatedScalars { space_after_comma } => {
                write!(f, "[")?;
                let sep = if *space_after_comma { ", " } else { "," };
                write!(
                    f,
                    "{}",
                    (0..self.len())
                        .map(|i| self.scalar_at(i).vortex_expect("index is in bounds"))
                        .format(sep)
                )?;
                write!(f, "]")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::{Buffer, buffer};
    use vortex_dtype::FieldNames;

    use crate::IntoArray as _;
    use crate::array::Array;
    use crate::arrays::{BoolArray, ListArray, StructArray};
    use crate::validity::Validity;

    #[test]
    fn test_primitive() {
        let x = Buffer::<u32>::empty().into_array();
        assert_eq!(x.display().to_string(), "[]");

        let x = buffer![1].into_array();
        assert_eq!(x.display().to_string(), "[1i32]");

        let x = buffer![1, 2, 3, 4].into_array();
        assert_eq!(x.display().to_string(), "[1i32, 2i32, 3i32, 4i32]");
    }

    #[test]
    fn test_empty_struct() {
        let s = StructArray::try_new(
            FieldNames::from(vec![]),
            vec![],
            3,
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        )
        .unwrap()
        .into_array();
        assert_eq!(s.display().to_string(), "[{}, null, {}]");
    }

    #[test]
    fn test_simple_struct() {
        let s = StructArray::from_fields(&[
            ("x", buffer![1, 2, 3, 4].into_array()),
            ("y", buffer![-1, -2, -3, -4].into_array()),
        ])
        .unwrap()
        .into_array();
        assert_eq!(
            s.display().to_string(),
            "[{x: 1i32, y: -1i32}, {x: 2i32, y: -2i32}, {x: 3i32, y: -3i32}, {x: 4i32, y: -4i32}]"
        );
    }

    #[test]
    fn test_list() {
        let x = ListArray::try_new(
            buffer![1, 2, 3, 4].into_array(),
            buffer![0, 0, 1, 1, 2, 4].into_array(),
            Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array()),
        )
        .unwrap()
        .into_array();
        assert_eq!(
            x.display().to_string(),
            "[[], [1i32], null, [2i32], [3i32, 4i32]]"
        );
    }
}

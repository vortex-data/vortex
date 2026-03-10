// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod tree;

use std::fmt::Display;

use itertools::Itertools as _;
use tree::TreeDisplayWrapper;

use crate::DynArray;

/// Describe how to convert an array to a string.
///
/// See also:
/// [Array::display_as](../trait.Array.html#method.display_as)
/// and [DisplayArrayAs].
pub enum DisplayOptions {
    /// Only the top-level encoding id and limited metadata: `vortex.primitive(i16, len=5)`.
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// assert_eq!(
    ///     format!("{}", array.display_as(DisplayOptions::MetadataOnly)),
    ///     "vortex.primitive(i16, len=5)",
    /// );
    /// ```
    MetadataOnly,
    /// Only the logical values of the array: `[0i16, 1i16, 2i16, 3i16, 4i16]`.
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// assert_eq!(
    ///     format!("{}", array.display_as(DisplayOptions::default())),
    ///     "[0i16, 1i16, 2i16, 3i16, 4i16]",
    /// );
    /// assert_eq!(
    ///     format!("{}", array.display_as(DisplayOptions::default())),
    ///     format!("{}", array.display_values()),
    /// );
    /// ```
    CommaSeparatedScalars { omit_comma_after_space: bool },
    /// The tree of encodings without any concrete values.
    ///
    /// With buffers, metadata, and stats:
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5) nbytes=10 B (100.00%)
    ///   metadata: EmptyMetadata
    ///   buffer: values host 10 B (align=2) (100.00%)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: true, metadata: true, stats: true })), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2) nbytes=16 B (100.00%)
    ///   metadata: EmptyMetadata
    ///   x: vortex.primitive(i32, len=2) nbytes=8 B (50.00%)
    ///     metadata: EmptyMetadata
    ///     buffer: values host 8 B (align=4) (100.00%)
    ///   y: vortex.primitive(i32, len=2) nbytes=8 B (50.00%)
    ///     metadata: EmptyMetadata
    ///     buffer: values host 8 B (align=4) (100.00%)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: true, metadata: true, stats: true })), expected);
    /// ```
    ///
    /// With metadata and stats but no buffers:
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5) nbytes=10 B (100.00%)
    ///   metadata: EmptyMetadata
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: false, metadata: true, stats: true })), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2) nbytes=16 B (100.00%)
    ///   metadata: EmptyMetadata
    ///   x: vortex.primitive(i32, len=2) nbytes=8 B (50.00%)
    ///     metadata: EmptyMetadata
    ///   y: vortex.primitive(i32, len=2) nbytes=8 B (50.00%)
    ///     metadata: EmptyMetadata
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: false, metadata: true, stats: true })), expected);
    /// ```
    ///
    /// With metadata and buffers but no stats:
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5)
    ///   metadata: EmptyMetadata
    ///   buffer: values host 10 B (align=2)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: true, metadata: true, stats: false })), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2)
    ///   metadata: EmptyMetadata
    ///   x: vortex.primitive(i32, len=2)
    ///     metadata: EmptyMetadata
    ///     buffer: values host 8 B (align=4)
    ///   y: vortex.primitive(i32, len=2)
    ///     metadata: EmptyMetadata
    ///     buffer: values host 8 B (align=4)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: true, metadata: true, stats: false })), expected);
    /// ```
    ///
    /// With buffers and stats but no metadata:
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5) nbytes=10 B (100.00%)
    ///   buffer: values host 10 B (align=2) (100.00%)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: true, metadata: false, stats: true })), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2) nbytes=16 B (100.00%)
    ///   x: vortex.primitive(i32, len=2) nbytes=8 B (50.00%)
    ///     buffer: values host 8 B (align=4) (100.00%)
    ///   y: vortex.primitive(i32, len=2) nbytes=8 B (50.00%)
    ///     buffer: values host 8 B (align=4) (100.00%)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: true, metadata: false, stats: true })), expected);
    /// ```
    ///
    /// With just buffers:
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5)
    ///   buffer: values host 10 B (align=2)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: true, metadata: false, stats: false })), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2)
    ///   x: vortex.primitive(i32, len=2)
    ///     buffer: values host 8 B (align=4)
    ///   y: vortex.primitive(i32, len=2)
    ///     buffer: values host 8 B (align=4)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: true, metadata: false, stats: false })), expected);
    /// ```
    ///
    /// With just metadata:
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5)
    ///   metadata: EmptyMetadata
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: false, metadata: true, stats: false })), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2)
    ///   metadata: EmptyMetadata
    ///   x: vortex.primitive(i32, len=2)
    ///     metadata: EmptyMetadata
    ///   y: vortex.primitive(i32, len=2)
    ///     metadata: EmptyMetadata
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: false, metadata: true, stats: false })), expected);
    /// ```
    ///
    /// With just stats:
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5) nbytes=10 B (100.00%)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: false, metadata: false, stats: true })), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2) nbytes=16 B (100.00%)
    ///   x: vortex.primitive(i32, len=2) nbytes=8 B (50.00%)
    ///   y: vortex.primitive(i32, len=2) nbytes=8 B (50.00%)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: false, metadata: false, stats: true })), expected);
    /// ```
    ///
    /// With neither buffers, metadata, stats, nor values:
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5)\n";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: false, metadata: false, stats: false })), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2)
    ///   x: vortex.primitive(i32, len=2)
    ///   y: vortex.primitive(i32, len=2)
    /// ";
    /// assert_eq!(format!("{}", array.display_as(DisplayOptions::TreeDisplay { buffers: false, metadata: false, stats: false })), expected);
    /// ```
    TreeDisplay {
        buffers: bool,
        metadata: bool,
        stats: bool,
    },
    /// Display values in a formatted table with columns.
    ///
    /// For struct arrays, displays a column for each field in the struct.
    /// For regular arrays, displays a single column with values.
    ///
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::arrays::StructArray;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let s = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "
    /// ┌──────┬──────┐
    /// │  x   │  y   │
    /// ├──────┼──────┤
    /// │ 1i32 │ 3i32 │
    /// ├──────┼──────┤
    /// │ 2i32 │ 4i32 │
    /// └──────┴──────┘".trim();
    /// assert_eq!(format!("{}", s.display_as(DisplayOptions::TableDisplay)), expected);
    /// ```
    #[cfg(feature = "table-display")]
    TableDisplay,
}

impl Default for DisplayOptions {
    fn default() -> Self {
        Self::CommaSeparatedScalars {
            omit_comma_after_space: false,
        }
    }
}

/// A shim used to display an array as specified in the options.
///
/// See also:
/// [Array::display_as](../trait.Array.html#method.display_as)
/// and [DisplayOptions].
pub struct DisplayArrayAs<'a>(pub &'a dyn DynArray, pub DisplayOptions);

impl Display for DisplayArrayAs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_as(f, &self.1)
    }
}

/// Display the encoding and limited metadata of this array.
///
/// # Examples
/// ```
/// # use vortex_array::IntoArray;
/// # use vortex_buffer::buffer;
/// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
/// assert_eq!(
///     format!("{}", array),
///     "vortex.primitive(i16, len=5)",
/// );
/// ```
impl Display for dyn DynArray + '_ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_as(f, &DisplayOptions::MetadataOnly)
    }
}

impl dyn DynArray + '_ {
    /// Display logical values of the array
    ///
    /// For example, an `i16` typed array containing the first five non-negative integers is displayed
    /// as: `[0i16, 1i16, 2i16, 3i16, 4i16]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// assert_eq!(
    ///     format!("{}", array.display_values()),
    ///     "[0i16, 1i16, 2i16, 3i16, 4i16]",
    /// )
    /// ```
    ///
    /// See also:
    /// [Array::display_as](..//trait.Array.html#method.display_as),
    /// [DisplayArrayAs], and [DisplayOptions].
    pub fn display_values(&self) -> impl Display {
        DisplayArrayAs(
            self,
            DisplayOptions::CommaSeparatedScalars {
                omit_comma_after_space: false,
            },
        )
    }

    /// Display the array as specified by the options.
    ///
    /// See [DisplayOptions] for examples.
    pub fn display_as(&self, options: DisplayOptions) -> impl Display {
        DisplayArrayAs(self, options)
    }

    /// Display the tree of array encodings and lengths without metadata, buffers, or stats.
    ///
    /// # Examples
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5)\n";
    /// assert_eq!(format!("{}", array.display_tree_encodings_only()), expected);
    ///
    /// # use vortex_array::arrays::StructArray;
    /// let array = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "root: vortex.struct({x=i32, y=i32}, len=2)
    ///   x: vortex.primitive(i32, len=2)
    ///   y: vortex.primitive(i32, len=2)
    /// ";
    /// assert_eq!(format!("{}", array.display_tree_encodings_only()), expected);
    /// ```
    pub fn display_tree_encodings_only(&self) -> impl Display {
        DisplayArrayAs(
            self,
            DisplayOptions::TreeDisplay {
                buffers: false,
                metadata: false,
                stats: false,
            },
        )
    }

    /// Display the tree of encodings of this array as an indented lists.
    ///
    /// While some metadata (such as length, bytes and validity-rate) are included, the logical
    /// values of the array are not displayed. To view the logical values see
    /// [Array::display_as](../trait.Array.html#method.display_as)
    /// and [DisplayOptions].
    ///
    /// # Examples
    /// ```
    /// # use vortex_array::display::DisplayOptions;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let array = buffer![0_i16, 1, 2, 3, 4].into_array();
    /// let expected = "root: vortex.primitive(i16, len=5) nbytes=10 B (100.00%)
    ///   metadata: EmptyMetadata
    ///   buffer: values host 10 B (align=2) (100.00%)
    /// ";
    /// assert_eq!(format!("{}", array.display_tree()), expected);
    /// ```
    pub fn display_tree(&self) -> impl Display {
        DisplayArrayAs(
            self,
            DisplayOptions::TreeDisplay {
                buffers: true,
                metadata: true,
                stats: true,
            },
        )
    }

    /// Display the array as a formatted table.
    ///
    /// For struct arrays, displays a column for each field in the struct.
    /// For regular arrays, displays a single column with values.
    ///
    /// # Examples
    /// ```
    /// # #[cfg(feature = "table-display")]
    /// # {
    /// # use vortex_array::arrays::StructArray;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// let s = StructArray::from_fields(&[
    ///     ("x", buffer![1, 2].into_array()),
    ///     ("y", buffer![3, 4].into_array()),
    /// ]).unwrap().into_array();
    /// let expected = "
    /// ┌──────┬──────┐
    /// │  x   │  y   │
    /// ├──────┼──────┤
    /// │ 1i32 │ 3i32 │
    /// ├──────┼──────┤
    /// │ 2i32 │ 4i32 │
    /// └──────┴──────┘".trim();
    /// assert_eq!(format!("{}", s.display_table()), expected);
    /// # }
    /// ```
    #[cfg(feature = "table-display")]
    pub fn display_table(&self) -> impl Display {
        DisplayArrayAs(self, DisplayOptions::TableDisplay)
    }

    fn fmt_as(&self, f: &mut std::fmt::Formatter, options: &DisplayOptions) -> std::fmt::Result {
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
            DisplayOptions::CommaSeparatedScalars {
                omit_comma_after_space,
            } => {
                write!(f, "{}", if f.alternate() { "[\n" } else { "[" })?;
                let sep = if *omit_comma_after_space { "," } else { ", " };
                let sep = if f.alternate() { ",\n" } else { sep };
                write!(
                    f,
                    "{}",
                    (0..self.len())
                        .map(|i| self
                            .scalar_at(i)
                            .map_or_else(|e| format!("<error: {e}>"), |s| s.to_string()))
                        .format(sep)
                )?;
                write!(f, "{}", if f.alternate() { "\n]" } else { "]" })
            }
            DisplayOptions::TreeDisplay {
                buffers,
                metadata,
                stats,
            } => {
                write!(
                    f,
                    "{}",
                    TreeDisplayWrapper {
                        array: self.to_array(),
                        buffers: *buffers,
                        metadata: *metadata,
                        stats: *stats
                    }
                )
            }
            #[cfg(feature = "table-display")]
            DisplayOptions::TableDisplay => {
                use crate::canonical::ToCanonical;
                use crate::dtype::DType;

                let mut builder = tabled::builder::Builder::default();

                // Special logic for struct arrays.
                let DType::Struct(sf, _) = self.dtype() else {
                    // For non-struct arrays, simply display a single column table without header.
                    for row_idx in 0..self.len() {
                        let value = self
                            .scalar_at(row_idx)
                            .map_or_else(|e| format!("<error: {e}>"), |s| s.to_string());
                        builder.push_record([value]);
                    }

                    let mut table = builder.build();
                    table.with(tabled::settings::Style::modern());

                    return write!(f, "{table}");
                };

                let struct_ = self.to_struct();
                builder.push_record(sf.names().iter().map(|name| name.to_string()));

                for row_idx in 0..self.len() {
                    if !self.is_valid(row_idx).unwrap_or(false) {
                        let null_row = vec!["null".to_string(); sf.names().len()];
                        builder.push_record(null_row);
                    } else {
                        let mut row = Vec::new();
                        for field_array in struct_.unmasked_fields().iter() {
                            let value = field_array
                                .scalar_at(row_idx)
                                .map_or_else(|e| format!("<error: {e}>"), |s| s.to_string());
                            row.push(value);
                        }
                        builder.push_record(row);
                    }
                }

                let mut table = builder.build();
                table.with(tabled::settings::Style::modern());

                // Center headers
                for col_idx in 0..sf.names().len() {
                    table.modify((0, col_idx), tabled::settings::Alignment::center());
                }

                for row_idx in 0..self.len() {
                    if !self.is_valid(row_idx).unwrap_or(false) {
                        table.modify(
                            (1 + row_idx, 0),
                            tabled::settings::Span::column(sf.names().len() as isize),
                        );
                        table.modify((1 + row_idx, 0), tabled::settings::Alignment::center());
                    }
                }

                write!(f, "{table}")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use crate::IntoArray as _;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::StructArray;
    use crate::dtype::FieldNames;
    use crate::validity::Validity;

    #[test]
    fn test_primitive() {
        let x = Buffer::<u32>::empty().into_array();
        assert_eq!(x.display_values().to_string(), "[]");

        let x = buffer![1].into_array();
        assert_eq!(x.display_values().to_string(), "[1i32]");

        let x = buffer![1, 2, 3, 4].into_array();
        assert_eq!(x.display_values().to_string(), "[1i32, 2i32, 3i32, 4i32]");
    }

    #[test]
    fn test_empty_struct() {
        let s = StructArray::try_new(
            FieldNames::empty(),
            vec![],
            3,
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        )
        .unwrap()
        .into_array();
        assert_eq!(s.display_values().to_string(), "[{}, null, {}]");
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
            s.display_values().to_string(),
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
            x.display_values().to_string(),
            "[[], [1i32], null, [2i32], [3i32, 4i32]]"
        );
    }

    #[test]
    fn test_table_display_primitive() {
        use crate::display::DisplayOptions;

        let array = buffer![1, 2, 3, 4].into_array();
        let table_display = array.display_as(DisplayOptions::TableDisplay);
        assert_eq!(
            table_display.to_string(),
            r"
┌──────┐
│ 1i32 │
├──────┤
│ 2i32 │
├──────┤
│ 3i32 │
├──────┤
│ 4i32 │
└──────┘"
                .trim()
        );
    }

    #[test]
    fn test_table_display() {
        use crate::display::DisplayOptions;

        let array = crate::arrays::PrimitiveArray::from_option_iter(vec![
            Some(-1),
            Some(-2),
            Some(-3),
            None,
        ])
        .into_array();

        let struct_ = StructArray::try_from_iter_with_validity(
            [("x", buffer![1, 2, 3, 4].into_array()), ("y", array)],
            Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array()),
        )
        .unwrap()
        .into_array();

        let table_display = struct_.display_as(DisplayOptions::TableDisplay);
        assert_eq!(
            table_display.to_string(),
            r"
┌──────┬───────┐
│  x   │   y   │
├──────┼───────┤
│ 1i32 │ -1i32 │
├──────┼───────┤
│     null     │
├──────┼───────┤
│ 3i32 │ -3i32 │
├──────┼───────┤
│ 4i32 │ null  │
└──────┴───────┘"
                .trim()
        );
    }
}

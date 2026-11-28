// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_cast::cast::cast_num_to_bool;
use arrow_cast::cast::cast_single_string_to_boolean_default;
use parquet_variant::Variant;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::BoolBuilder;

use super::ARRAY;
use super::Append;
use super::BASIC_TYPE_MASK;
use super::Names;
use super::OBJECT;
use super::PRIMITIVE;
use super::decode;

const BOOL_TRUE: u8 = 1;
const BOOL_FALSE: u8 = 2;

impl Append for BoolBuilder {
    #[inline]
    fn append(&mut self, _names: &Names<'_, '_>, value: &[u8]) -> Option<()> {
        let header = *value.first()?;
        let boolean = match (header & BASIC_TYPE_MASK, header >> 2) {
            (PRIMITIVE, BOOL_TRUE) => Some(true),
            (PRIMITIVE, BOOL_FALSE) => Some(false),
            (OBJECT | ARRAY, _) => None,
            _ => match decode(value)? {
                Variant::Int8(i) => Some(cast_num_to_bool(i)),
                Variant::Int16(i) => Some(cast_num_to_bool(i)),
                Variant::Int32(i) => Some(cast_num_to_bool(i)),
                Variant::Int64(i) => Some(cast_num_to_bool(i)),
                Variant::Float(f) => Some(cast_num_to_bool(f)),
                Variant::Double(f) => Some(cast_num_to_bool(f)),
                Variant::ShortString(s) => cast_single_string_to_boolean_default(s.as_str()),
                Variant::String(s) => cast_single_string_to_boolean_default(s),
                _ => None,
            },
        };
        match boolean {
            Some(value) => self.append_value(value),
            None => ArrayBuilder::append_null(self),
        }
        Some(())
    }

    fn append_null(&mut self) {
        ArrayBuilder::append_null(self);
    }
}

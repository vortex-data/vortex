// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Observability label maps: `Map<Utf8, Union<str, int, float, bool>>`.

use std::fmt::Write as _;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnionVariants;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::flat::Strings;
use crate::flat::build_map;

/// A typed label value.
#[derive(Clone, Debug, PartialEq)]
pub enum LabelValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl LabelValue {
    /// Infers the narrowest type that formats back to exactly `s`.
    pub fn infer(s: &str) -> Self {
        match s {
            "true" => return Self::Bool(true),
            "false" => return Self::Bool(false),
            _ => {}
        }
        if let Ok(i) = s.parse::<i64>()
            && i.to_string() == s
        {
            return Self::Int(i);
        }
        if let Ok(f) = s.parse::<f64>()
            && f.to_string() == s
        {
            return Self::Float(f);
        }
        Self::Str(s.to_string())
    }

    /// Formats the value as the label string it was inferred from.
    pub fn to_label_string(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => f.to_string(),
            Self::Bool(b) => b.to_string(),
        }
    }
}

/// The variants of a label value union, in tag order.
pub fn label_variants() -> UnionVariants {
    UnionVariants::new(
        ["str", "int", "float", "bool"].into(),
        vec![
            DType::Utf8(Nullability::NonNullable),
            DType::Primitive(PType::I64, Nullability::NonNullable),
            DType::Primitive(PType::F64, Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
        ],
    )
    .vortex_expect("valid label variants")
}

/// `Map<Utf8, Union<str, int, float, bool>>` with sorted keys.
pub fn label_map_dtype(value_nullability: Nullability) -> MapDType {
    MapDType::try_new(
        DType::Utf8(Nullability::NonNullable),
        DType::Union(label_variants(), value_nullability),
        true,
    )
    .vortex_expect("valid label map dtype")
}

/// Builds a label map column row by row into columnar buffers.
pub struct LabelMapBuilder {
    value_nullability: Nullability,
    map_nullability: Nullability,
    keys: VarBinViewBuilder,
    type_ids: Vec<u8>,
    value_valid: BitBufferMut,
    strs: VarBinViewBuilder,
    ints: Vec<i64>,
    floats: Vec<f64>,
    bools: BitBufferMut,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    row_valid: BitBufferMut,
}

impl LabelMapBuilder {
    pub fn new(value_nullability: Nullability, map_nullability: Nullability) -> Self {
        Self {
            value_nullability,
            map_nullability,
            keys: VarBinViewBuilder::with_capacity_in(DType::Utf8(Nullability::NonNullable), 0, BufferAllocatorRef::static_ref().clone()),
            type_ids: Vec::new(),
            value_valid: BitBufferMut::with_capacity(0),
            strs: VarBinViewBuilder::with_capacity_in(DType::Utf8(Nullability::NonNullable), 0, BufferAllocatorRef::static_ref().clone()),
            ints: Vec::new(),
            floats: Vec::new(),
            bools: BitBufferMut::with_capacity(0),
            offsets: Vec::new(),
            sizes: Vec::new(),
            row_valid: BitBufferMut::with_capacity(0),
        }
    }

    fn push_value(&mut self, value: Option<&LabelValue>) {
        let tag = match value {
            None | Some(LabelValue::Str(_)) => 0,
            Some(LabelValue::Int(_)) => 1,
            Some(LabelValue::Float(_)) => 2,
            Some(LabelValue::Bool(_)) => 3,
        };
        self.type_ids.push(tag);
        self.value_valid.append(value.is_some());
        match value {
            Some(LabelValue::Str(s)) => self.strs.append_value(s),
            _ => self.strs.append_value(""),
        }
        self.ints.push(match value {
            Some(LabelValue::Int(i)) => *i,
            _ => 0,
        });
        self.floats.push(match value {
            Some(LabelValue::Float(f)) => *f,
            _ => 0.0,
        });
        self.bools.append(matches!(value, Some(LabelValue::Bool(true))));
    }

    /// Appends a row. Its keys must be sorted.
    pub fn push_row<'a, K: AsRef<str> + 'a>(
        &mut self,
        entries: impl IntoIterator<Item = (K, Option<&'a LabelValue>)>,
    ) {
        let start = self.type_ids.len();
        for (key, value) in entries {
            self.keys.append_value(key.as_ref());
            self.push_value(value);
        }
        self.offsets.push(start as u64);
        self.sizes.push((self.type_ids.len() - start) as u64);
        self.row_valid.append(true);
    }

    /// Appends a row equal to the previous one, sharing its entries instead of copying them.
    ///
    /// The result is a list-view whose rows may overlap, so repeated label sets cost one
    /// `(offset, size)` pair per row.
    pub fn repeat_last_row(&mut self) {
        let (offset, size, valid) = match (self.offsets.last(), self.sizes.last()) {
            (Some(&o), Some(&s)) => (o, s, self.row_valid.value(self.row_valid.len() - 1)),
            _ => (0, 0, true),
        };
        self.offsets.push(offset);
        self.sizes.push(size);
        self.row_valid.append(valid);
    }

    /// Appends a null row.
    pub fn push_null(&mut self) {
        self.offsets.push(self.type_ids.len() as u64);
        self.sizes.push(0);
        self.row_valid.append(false);
    }

    pub fn finish(mut self) -> VortexResult<MapArray> {
        let n = self.type_ids.len();
        let type_ids_validity =
            Validity::from_bit_buffer(self.value_valid.freeze(), self.value_nullability);
        let type_ids =
            PrimitiveArray::new(Buffer::from(self.type_ids), type_ids_validity).into_array();
        let children = vec![
            self.strs.finish(),
            PrimitiveArray::new(Buffer::from(self.ints), Validity::NonNullable).into_array(),
            PrimitiveArray::new(Buffer::from(self.floats), Validity::NonNullable).into_array(),
            BoolArray::new(self.bools.freeze(), Validity::NonNullable).into_array(),
        ];
        let values = UnionArray::try_new(type_ids, label_variants(), children)?.into_array();
        let keys = self.keys.finish();
        debug_assert_eq!(keys.len(), n);
        build_map(
            &label_map_dtype(self.value_nullability),
            keys,
            values,
            self.offsets,
            self.sizes,
            Validity::from_bit_buffer(self.row_valid.freeze(), self.map_nullability),
        )
    }
}

/// `Map<Utf8, Utf8>` with sorted keys.
pub fn utf8_map_dtype(value_nullability: Nullability) -> MapDType {
    MapDType::try_new(
        DType::Utf8(Nullability::NonNullable),
        DType::Utf8(value_nullability),
        true,
    )
    .vortex_expect("valid utf8 map dtype")
}

/// Builds a `Map<Utf8, Utf8>` column row by row, optionally sharing repeated rows.
pub struct Utf8MapBuilder {
    value_nullability: Nullability,
    map_nullability: Nullability,
    keys: VarBinViewBuilder,
    values: VarBinViewBuilder,
    entries: usize,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    row_valid: BitBufferMut,
}

impl Utf8MapBuilder {
    pub fn new(value_nullability: Nullability, map_nullability: Nullability) -> Self {
        let allocator = BufferAllocatorRef::static_ref().clone();
        Self {
            value_nullability,
            map_nullability,
            keys: VarBinViewBuilder::with_capacity_in(
                DType::Utf8(Nullability::NonNullable),
                0,
                allocator.clone(),
            ),
            values: VarBinViewBuilder::with_capacity_in(DType::Utf8(value_nullability), 0, allocator),
            entries: 0,
            offsets: Vec::new(),
            sizes: Vec::new(),
            row_valid: BitBufferMut::with_capacity(0),
        }
    }

    /// Appends a row. Its keys must be sorted.
    pub fn push_row<K: AsRef<str>, V: AsRef<str>>(
        &mut self,
        entries: impl IntoIterator<Item = (K, Option<V>)>,
    ) {
        let start = self.entries;
        for (key, value) in entries {
            self.keys.append_value(key.as_ref());
            match value {
                Some(v) => self.values.append_value(v.as_ref()),
                None => self.values.append_null(),
            }
            self.entries += 1;
        }
        self.offsets.push(start as u64);
        self.sizes.push((self.entries - start) as u64);
        self.row_valid.append(true);
    }

    /// Appends a row equal to the previous one, sharing its entries.
    pub fn repeat_last_row(&mut self) {
        let (offset, size, valid) = match (self.offsets.last(), self.sizes.last()) {
            (Some(&o), Some(&s)) => (o, s, self.row_valid.value(self.row_valid.len() - 1)),
            _ => (0, 0, true),
        };
        self.offsets.push(offset);
        self.sizes.push(size);
        self.row_valid.append(valid);
    }

    /// Appends a null row.
    pub fn push_null(&mut self) {
        self.offsets.push(self.entries as u64);
        self.sizes.push(0);
        self.row_valid.append(false);
    }

    pub fn finish(mut self) -> VortexResult<MapArray> {
        build_map(
            &utf8_map_dtype(self.value_nullability),
            self.keys.finish(),
            self.values.finish(),
            self.offsets,
            self.sizes,
            Validity::from_bit_buffer(self.row_valid.freeze(), self.map_nullability),
        )
    }
}

/// A canonical child of a union, ready for per-row formatting.
enum Formatter {
    Utf8(VarBinViewArray),
    Primitive(PrimitiveArray),
    Bool(BitBuffer),
}

impl Formatter {
    fn new(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<(Self, Mask)> {
        let mask = array.validity()?.execute_mask(array.len(), ctx)?;
        let formatter = match array.execute::<Canonical>(ctx)? {
            Canonical::VarBinView(a) => Self::Utf8(a),
            Canonical::Primitive(a) => Self::Primitive(a),
            Canonical::Bool(a) => Self::Bool(a.to_bit_buffer()),
            other => vortex_bail!("cannot format {} values as strings", other.dtype()),
        };
        Ok((formatter, mask))
    }

    fn append(&self, strings: Option<&Strings<'_>>, row: usize, buf: &mut String) {
        match self {
            Self::Utf8(_) => {
                let bytes = strings.unwrap_or_else(|| unreachable!()).get(row);
                // SAFETY: the child has a UTF-8 dtype.
                buf.push_str(unsafe { std::str::from_utf8_unchecked(bytes) });
            }
            Self::Primitive(p) => match_each_native_ptype!(p.ptype(), |P| {
                let _ = write!(buf, "{}", p.as_slice::<P>()[row]);
            }),
            Self::Bool(b) => buf.push_str(if b.value(row) { "true" } else { "false" }),
        }
    }
}

/// Formats each value of a union or scalar array as a nullable UTF-8 string.
pub fn values_to_utf8(values: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if let Some(dict) = values.as_opt::<Dict>() {
        // Format each distinct value once.
        let strings = values_to_utf8(dict.values(), ctx)?;
        return Ok(DictArray::try_new(dict.codes().clone(), strings)?.into_array());
    }
    let len = values.len();
    if let DType::Utf8(_) = values.dtype() {
        return Ok(values.clone());
    }
    let mut builder = VarBinViewBuilder::with_capacity_in(DType::Utf8(Nullability::Nullable), len, BufferAllocatorRef::static_ref().clone());
    let mut buf = String::new();

    let DType::Union(..) = values.dtype() else {
        let (formatter, mask) = Formatter::new(values.clone(), ctx)?;
        for row in 0..len {
            if mask.value(row) {
                buf.clear();
                formatter.append(None, row, &mut buf);
                builder.append_value(&buf);
            } else {
                builder.append_null();
            }
        }
        return Ok(builder.finish());
    };

    let union = values.clone().execute::<UnionArray>(ctx)?;
    let type_ids = union.type_ids().clone().execute::<PrimitiveArray>(ctx)?;
    let ids_valid = type_ids.validity()?.execute_mask(len, ctx)?;
    let tags = type_ids.as_slice::<u8>();
    let mut child_of_tag = [usize::MAX; 256];
    for (child, &tag) in union.variants().type_ids().iter().enumerate() {
        child_of_tag[tag as usize] = child;
    }
    let children = union
        .iter_children()
        .map(|c| Formatter::new(c.clone(), ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let strings: Vec<Option<Strings<'_>>> = children
        .iter()
        .map(|(f, _)| match f {
            Formatter::Utf8(a) => Some(Strings::new(a)),
            _ => None,
        })
        .collect();

    for row in 0..len {
        if !ids_valid.value(row) {
            builder.append_null();
            continue;
        }
        let child = child_of_tag[tags[row] as usize];
        let (formatter, mask) = &children[child];
        if !mask.value(row) {
            builder.append_null();
            continue;
        }
        if let (Formatter::Utf8(_), Some(s)) = (formatter, &strings[child]) {
            builder.append_value(s.get(row));
        } else {
            buf.clear();
            formatter.append(strings[child].as_ref(), row, &mut buf);
            builder.append_value(&buf);
        }
    }
    Ok(builder.finish())
}

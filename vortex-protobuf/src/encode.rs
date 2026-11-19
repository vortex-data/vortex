// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding Vortex arrays of protobuf messages back to the protobuf wire format.

use std::ops::Range;

use prost::Message;
use prost::bytes::Bytes;
use prost_reflect::DynamicMessage;
use prost_reflect::FieldDescriptor;
use prost_reflect::Kind;
use prost_reflect::MessageDescriptor;
use prost_reflect::Value;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::memory::BufferAllocatorRef;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Protobuf;
use crate::mapping::FieldShape;
use crate::value::map_key;

/// Serializes each message of a protobuf extension array in the protobuf wire format.
///
/// Returns a binary array of the same length and nullability, with null where the input holds a
/// null message.
///
/// # Errors
///
/// Returns an error if `array` is not a protobuf extension array, or if executing it fails.
pub fn encode_messages(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<VarBinViewArray> {
    let Some(metadata) = array
        .dtype()
        .as_extension_opt()
        .and_then(|ext_dtype| ext_dtype.metadata_opt::<Protobuf>())
    else {
        vortex_bail!(
            "Encoding protobuf messages requires a {} array, got {}",
            Protobuf::ID,
            array.dtype()
        );
    };
    let storage = array
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();
    let validity = storage.validity()?.execute_mask(storage.len(), ctx)?;
    let messages = MessageReader::new(metadata.message_descriptor(), storage, ctx)?;

    let mut builder = VarBinViewBuilder::with_capacity_in(
        DType::Binary(array.dtype().nullability()),
        array.len(),
        BufferAllocatorRef::statically_allocated(),
    );
    for row in 0..array.len() {
        if validity.value(row) {
            builder.append_value(messages.read(row)?.encode_to_vec());
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish_into_varbinview())
}

/// Reads messages of one type from the canonical columns of their storage struct.
struct MessageReader {
    /// The message type.
    descriptor: MessageDescriptor,
    /// The fields of the message, in declaration order, with their columns.
    fields: Vec<(FieldDescriptor, ColumnReader)>,
}

impl MessageReader {
    /// Prepares to read messages of type `descriptor` from the storage struct `array`.
    ///
    /// The rows of `array` must be valid where messages are read. Its validity is not checked.
    fn new(
        descriptor: &MessageDescriptor,
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let array = array.execute::<StructArray>(ctx)?;
        let fields = descriptor
            .fields()
            .enumerate()
            .map(|(index, field)| {
                let column = ColumnReader::for_field(&field, array.unmasked_field(index), ctx)?;
                Ok((field, column))
            })
            .collect::<VortexResult<_>>()?;
        Ok(Self {
            descriptor: descriptor.clone(),
            fields,
        })
    }

    /// Reads the message at `row`, setting the fields whose values are not null.
    fn read(&self, row: usize) -> VortexResult<DynamicMessage> {
        let mut message = DynamicMessage::new(self.descriptor.clone());
        for (field, column) in &self.fields {
            if let Some(value) = column.read(row)? {
                message.set_field(field, value);
            }
        }
        Ok(message)
    }
}

/// Reads protobuf values from one canonical storage column.
struct ColumnReader {
    /// The rows that hold a value.
    validity: Mask,
    /// The values, which are undefined at null rows.
    values: Values,
}

/// The values of a [`ColumnReader`].
enum Values {
    /// `double` values.
    F64(Buffer<f64>),
    /// `float` values.
    F32(Buffer<f32>),
    /// `int32`, `sint32` or `sfixed32` values.
    I32(Buffer<i32>),
    /// Enum numbers.
    Enum(Buffer<i32>),
    /// `int64`, `sint64` or `sfixed64` values.
    I64(Buffer<i64>),
    /// `uint32` or `fixed32` values.
    U32(Buffer<u32>),
    /// `uint64` or `fixed64` values.
    U64(Buffer<u64>),
    /// `bool` values.
    Bool(BitBuffer),
    /// `string` values.
    String(VarBinViewArray),
    /// `bytes` values.
    Bytes(VarBinViewArray),
    /// Messages.
    Message(MessageReader),
    /// Repeated fields.
    List {
        /// The element range of each list.
        lists: Lists,
        /// The elements of all lists.
        elements: Box<ColumnReader>,
    },
    /// Map fields.
    Map {
        /// The entry range of each map.
        lists: Lists,
        /// The keys of all maps.
        keys: Box<ColumnReader>,
        /// The values of all maps.
        values: Box<ColumnReader>,
    },
}

impl ColumnReader {
    /// Prepares to read the values of `field` from its storage column `array`.
    fn for_field(
        field: &FieldDescriptor,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let kind = match FieldShape::of(field) {
            FieldShape::Single(kind) => return Self::for_kind(&kind, array.clone(), ctx),
            FieldShape::List(kind) => kind,
            FieldShape::Map { key, value } => {
                let (validity, lists, entries) = Self::list_parts(array, ctx)?;
                let entries = entries.execute::<StructArray>(ctx)?;
                let keys = Self::for_kind(&key, entries.unmasked_field(0).clone(), ctx)?;
                let values = Self::for_kind(&value, entries.unmasked_field(1).clone(), ctx)?;
                return Ok(Self {
                    validity,
                    values: Values::Map {
                        lists,
                        keys: Box::new(keys),
                        values: Box::new(values),
                    },
                });
            }
        };
        let (validity, lists, elements) = Self::list_parts(array, ctx)?;
        Ok(Self {
            validity,
            values: Values::List {
                lists,
                elements: Box::new(Self::for_kind(&kind, elements, ctx)?),
            },
        })
    }

    /// Splits a list column into its validity, its list ranges and its elements.
    fn list_parts(
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(Mask, Lists, ArrayRef)> {
        let array = array.clone().execute::<ListViewArray>(ctx)?;
        let validity = array.validity()?.execute_mask(array.len(), ctx)?;
        let parts = array.into_data_parts();
        let lists = Lists {
            offsets: integers_as_usize(parts.offsets, ctx)?,
            sizes: integers_as_usize(parts.sizes, ctx)?,
        };
        Ok((validity, lists, parts.elements))
    }

    /// Prepares to read single values of `kind` from the column `array`.
    fn for_kind(kind: &Kind, array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let validity = array.validity()?.execute_mask(array.len(), ctx)?;
        let values = match kind {
            Kind::Message(message) => Values::Message(MessageReader::new(message, array, ctx)?),
            Kind::Bool => Values::Bool(array.execute::<BoolArray>(ctx)?.into_bit_buffer()),
            Kind::String => Values::String(array.execute::<VarBinViewArray>(ctx)?),
            Kind::Bytes => Values::Bytes(array.execute::<VarBinViewArray>(ctx)?),
            Kind::Double => Values::F64(primitive_values(array, ctx)?),
            Kind::Float => Values::F32(primitive_values(array, ctx)?),
            Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => {
                Values::I32(primitive_values(array, ctx)?)
            }
            Kind::Enum(_) => Values::Enum(primitive_values(array, ctx)?),
            Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => {
                Values::I64(primitive_values(array, ctx)?)
            }
            Kind::Uint32 | Kind::Fixed32 => Values::U32(primitive_values(array, ctx)?),
            Kind::Uint64 | Kind::Fixed64 => Values::U64(primitive_values(array, ctx)?),
        };
        Ok(Self { validity, values })
    }

    /// Reads the value at `row`, or `None` if it is null.
    fn read(&self, row: usize) -> VortexResult<Option<Value>> {
        if !self.validity.value(row) {
            return Ok(None);
        }
        Ok(Some(match &self.values {
            Values::F64(values) => Value::F64(values[row]),
            Values::F32(values) => Value::F32(values[row]),
            Values::I32(values) => Value::I32(values[row]),
            Values::Enum(values) => Value::EnumNumber(values[row]),
            Values::I64(values) => Value::I64(values[row]),
            Values::U32(values) => Value::U32(values[row]),
            Values::U64(values) => Value::U64(values[row]),
            Values::Bool(values) => Value::Bool(values.value(row)),
            Values::String(values) => Value::String(
                str::from_utf8(values.bytes_at(row).as_slice())
                    .map_err(|err| vortex_err!("Invalid UTF-8 in a protobuf string: {err}"))?
                    .to_owned(),
            ),
            Values::Bytes(values) => {
                Value::Bytes(Bytes::copy_from_slice(values.bytes_at(row).as_slice()))
            }
            Values::Message(messages) => Value::Message(messages.read(row)?),
            Values::List { lists, elements } => Value::List(
                lists
                    .range(row)
                    .map(|element| elements.read_present(element))
                    .collect::<VortexResult<_>>()?,
            ),
            Values::Map {
                lists,
                keys,
                values,
            } => Value::Map(
                lists
                    .range(row)
                    .map(|entry| {
                        Ok((
                            map_key(keys.read_present(entry)?)?,
                            values.read_present(entry)?,
                        ))
                    })
                    .collect::<VortexResult<_>>()?,
            ),
        }))
    }

    /// Reads the value at `row` of a column that cannot hold nulls.
    fn read_present(&self, row: usize) -> VortexResult<Value> {
        self.read(row)?
            .ok_or_else(|| vortex_err!("Protobuf list elements and map entries cannot be null"))
    }
}

/// The element ranges of the lists in a list column.
struct Lists {
    /// The offset of each list into the elements.
    offsets: Vec<usize>,
    /// The number of elements of each list.
    sizes: Vec<usize>,
}

impl Lists {
    /// The element range of the list at `row`.
    fn range(&self, row: usize) -> Range<usize> {
        let start = self.offsets[row];
        start..start + self.sizes[row]
    }
}

/// Executes a primitive array of `T` values and returns its values.
fn primitive_values<T: NativePType>(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    Ok(array.execute::<PrimitiveArray>(ctx)?.into_buffer())
}

/// Executes an integer array and converts its values to `usize`.
fn integers_as_usize(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<usize>> {
    let primitive = array.execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_integer_ptype!(primitive.ptype(), |T| {
        widen(primitive.as_slice::<T>())
    }))
}

/// Converts integers that are known to be non-negative offsets or sizes to `usize`.
fn widen<T: IntegerPType>(values: &[T]) -> Vec<usize> {
    values.iter().map(|value| value.as_()).collect()
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.protobuf.enum_name` scalar function, which maps enum numbers to value names.

use std::fmt;
use std::sync::Arc;

use prost::Message;
use prost_reflect::EnumDescriptor;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// Maps the `i32` numbers of a protobuf enum to the names of its values.
///
/// The result is nullable `utf8`. A null number, and a number that the enum does not define, have
/// a null name. Only an open enum stores such numbers: the decoder keeps unknown numbers of a
/// proto3 enum, and drops those of a closed proto2 enum. A number that several values share,
/// through `allow_alias`, has the name of the value that the enum declares first.
///
/// [`protobuf_enum_name`](crate::protobuf_enum_name) applies the function to an enum field.
#[derive(Clone, Debug)]
pub struct EnumName;

/// The values of the enum that an [`EnumName`] call maps to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnumNameOptions {
    /// The full name of the enum type.
    enum_name: Arc<str>,
    /// The numbers of the enum in increasing order, each with its name.
    values: Arc<[(i32, Arc<str>)]>,
}

impl EnumNameOptions {
    /// Creates the options that map to the values of `enum_type`.
    pub fn new(enum_type: &EnumDescriptor) -> Self {
        let mut values = enum_type
            .enum_descriptor_proto()
            .value
            .iter()
            .map(|value| (value.number(), Arc::from(value.name())))
            .collect::<Vec<_>>();
        // The values are in declaration order, so the stable sort keeps the value that is declared
        // first in front of the others with its number.
        values.sort_by_key(|&(number, _)| number);
        values.dedup_by_key(|&mut (number, _)| number);
        Self {
            enum_name: Arc::from(enum_type.full_name()),
            values: values.into(),
        }
    }

    /// The full name of the enum type, such as `my.package.Color`.
    pub fn enum_name(&self) -> &str {
        &self.enum_name
    }

    /// Returns the position of `number` in the values, or `None` if the enum does not define it.
    fn position(&self, number: i32) -> Option<usize> {
        self.values
            .binary_search_by_key(&number, |&(number, _)| number)
            .ok()
    }

    /// Returns the name of `number`, or `None` if the enum does not define it.
    fn name(&self, number: i32) -> Option<&str> {
        self.position(number)
            .map(|position| self.values[position].1.as_ref())
    }
}

impl fmt::Display for EnumNameOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.enum_name)
    }
}

/// The serialized form of [`EnumNameOptions`].
#[derive(Clone, PartialEq, Message)]
struct SerializedEnumName {
    /// See [`EnumNameOptions::enum_name`].
    #[prost(string, tag = "1")]
    enum_name: String,
    /// The values in increasing order of their numbers.
    #[prost(message, repeated, tag = "2")]
    values: Vec<SerializedEnumValue>,
}

/// One value of a [`SerializedEnumName`].
#[derive(Clone, PartialEq, Message)]
struct SerializedEnumValue {
    /// The number of the value.
    #[prost(int32, tag = "1")]
    number: i32,
    /// The name of the value.
    #[prost(string, tag = "2")]
    name: String,
}

impl ScalarFnVTable for EnumName {
    type Options = EnumNameOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.protobuf.enum_name");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            SerializedEnumName {
                enum_name: options.enum_name.to_string(),
                values: options
                    .values
                    .iter()
                    .map(|(number, name)| SerializedEnumValue {
                        number: *number,
                        name: name.to_string(),
                    })
                    .collect(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let serialized = SerializedEnumName::decode(metadata)?;
        vortex_ensure!(
            !serialized.values.is_empty()
                && serialized
                    .values
                    .windows(2)
                    .all(|pair| pair[0].number < pair[1].number),
            "The values of protobuf enum {} must be in increasing order, and there must be one",
            serialized.enum_name
        );
        Ok(EnumNameOptions {
            enum_name: serialized.enum_name.into(),
            values: serialized
                .values
                .into_iter()
                .map(|value| (value.number, Arc::from(value.name)))
                .collect(),
        })
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("numbers"),
            _ => unreachable!("Invalid child index {child_idx} for enum_name()"),
        }
    }

    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        match &args[0] {
            DType::Primitive(PType::I32, _) => Ok(DType::Utf8(Nullability::Nullable)),
            other => vortex_bail!(
                "The names of protobuf enum {} need i32 numbers, got {other}",
                options.enum_name
            ),
        }
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        if let Some(scalar) = input.as_constant() {
            let name = scalar
                .as_primitive()
                .typed_value::<i32>()
                .and_then(|number| options.name(number));
            let name = match name {
                Some(name) => Scalar::utf8(name, Nullability::Nullable),
                None => Scalar::null(DType::Utf8(Nullability::Nullable)),
            };
            return Ok(ConstantArray::new(name, args.row_count()).into_array());
        }

        let numbers = input.execute::<PrimitiveArray>(ctx)?;
        let present = numbers.validity()?.execute_mask(numbers.len(), ctx)?;
        let mut codes = BufferMut::<u32>::with_capacity(numbers.len());
        let mut named = BitBufferMut::with_capacity(numbers.len());
        for (row, &number) in numbers.as_slice::<i32>().iter().enumerate() {
            let position = options.position(number).filter(|_| present.value(row));
            codes.push(position.map_or(0, |position| {
                u32::try_from(position).vortex_expect("enum values fit in u32")
            }));
            named.append(position.is_some());
        }
        let codes = PrimitiveArray::new(
            codes.freeze(),
            Validity::from_bit_buffer(named.freeze(), Nullability::Nullable),
        );
        let names =
            VarBinViewArray::from_iter_str(options.values.iter().map(|(_, name)| name.as_ref()));
        Ok(DictArray::try_new(codes.into_array(), names.into_array())?.into_array())
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

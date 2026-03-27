pub use root::*;

const _: () = ::planus::check_version_compatibility("planus-1.3.0");


/// The root namespace
/// 
/// Generated from these locations:
/// * File `flatbuffers/vortex-serde/message.fbs`
/// * File `flatbuffers/vortex-array/array.fbs`
/// * File `flatbuffers/vortex-dtype/dtype.fbs`
#[no_implicit_prelude]
#[allow(clippy::needless_lifetimes)]
mod root {
/// The enum `MessageVersion`
/// 
/// Generated from these locations:
/// * Enum `MessageVersion` in the file `flatbuffers/vortex-serde/message.fbs:7`
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, ::serde::Serialize, ::serde::Deserialize)]#[repr(u8)]pub enum MessageVersion {
    
        /// The variant `V0` in the enum `MessageVersion`
        V0 = 0,
    
}

impl MessageVersion {
    /// Array containing all valid variants of MessageVersion
    pub const ENUM_VALUES: [Self; 1] = [Self::V0,];
}

impl ::core::convert::TryFrom<u8> for MessageVersion {
    type Error = ::planus::errors::UnknownEnumTagKind;
    #[inline]
    fn try_from(value: u8) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind> {
        #[allow(clippy::match_single_binding)]
        match value {
            0 => ::core::result::Result::Ok(MessageVersion::V0),
            

            _ => ::core::result::Result::Err(::planus::errors::UnknownEnumTagKind { tag: value as i128 }),
        }
    }
}

impl ::core::convert::From<MessageVersion> for u8 {
    #[inline]
    fn from(value: MessageVersion) -> Self {
        value as u8
    }
}

/// # Safety
/// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
unsafe impl ::planus::Primitive for MessageVersion {
    const ALIGNMENT: usize = 1;
    const SIZE: usize = 1;
}

impl ::planus::WriteAsPrimitive<MessageVersion> for MessageVersion {
    #[inline]
    fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
        (*self as u8).write(cursor, buffer_position);
    }
}

impl ::planus::WriteAs<MessageVersion> for MessageVersion {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> MessageVersion {
        *self
    }
}

impl ::planus::WriteAsDefault<MessageVersion, MessageVersion> for MessageVersion {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder, default: &MessageVersion) -> ::core::option::Option<MessageVersion> {
        if self == default {
            ::core::option::Option::None
        } else {
            ::core::option::Option::Some(*self)
        }
    }
}

impl ::planus::WriteAsOptional<MessageVersion> for MessageVersion {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<MessageVersion> {
        ::core::option::Option::Some(*self)
    }
}

impl<'buf> ::planus::TableRead<'buf> for MessageVersion {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'buf>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        let n: u8 = ::planus::TableRead::from_buffer(buffer, offset)?;
        ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
    }
}

impl<'buf> ::planus::VectorReadInner<'buf> for MessageVersion {
    type Error = ::planus::errors::UnknownEnumTag;
    const STRIDE: usize = 1;
    #[inline]
    unsafe fn from_buffer(
        buffer: ::planus::SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag> {let value = unsafe { *buffer.buffer.get_unchecked(offset) };let value: ::core::result::Result<Self, _> = ::core::convert::TryInto::try_into(value);
        value.map_err(|error_kind| error_kind.with_error_location(
            "MessageVersion",
            "VectorRead::from_buffer",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<MessageVersion> for MessageVersion {
    const STRIDE: usize = 1;

    type Value = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
        *self
    }

    #[inline]
    unsafe fn write_values(
        values: &[Self],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 1];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                
                buffer_position - i as u32,
                
            );
        }
    }
}

    
///  Indicates the message body contains a flatbuffer Array message, followed by array buffers.
/// 
/// Generated from these locations:
/// * Table `ArrayMessage` in the file `flatbuffers/vortex-serde/message.fbs:12`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct ArrayMessage{
    
        ///  The row count of the array.
        pub row_count: u32,
        ///  The encodings referenced by the array.
        pub encodings: ::core::option::Option<::planus::alloc::vec::Vec<::planus::alloc::string::String>>,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for ArrayMessage {
    fn default() -> Self {
        Self {
        row_count: 0,
        encodings: ::core::default::Default::default(),
        
        }
    }
}


impl ArrayMessage {
    /// Creates a [ArrayMessageBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> ArrayMessageBuilder<()> {
        ArrayMessageBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_row_count: impl ::planus::WriteAsDefault<u32, u32>,
        field_encodings: impl ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
        
    ) -> ::planus::Offset<Self> {let prepared_row_count = field_row_count.prepare(builder, &0);let prepared_encodings = field_encodings.prepare(builder);

        let mut table_writer: ::planus::table_writer::TableWriter::<8> = ::core::default::Default::default();
        if prepared_row_count.is_some() {table_writer.write_entry::<u32>(0);}if prepared_encodings.is_some() {table_writer.write_entry::<::planus::Offset<[::planus::Offset<str>]>>(1);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_row_count) = prepared_row_count {object_writer.write::<_, _, 4>(&prepared_row_count);}if let ::core::option::Option::Some(prepared_encodings) = prepared_encodings {object_writer.write::<_, _, 4>(&prepared_encodings);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<ArrayMessage>> for ArrayMessage {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayMessage> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<ArrayMessage>> for ArrayMessage {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<ArrayMessage>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<ArrayMessage> for ArrayMessage {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayMessage> {
        ArrayMessage::create(
            builder,
        
            
            self.row_count,
            
        
            
            &self.encodings,
            
        
        )
    }
}

/// Builder for serializing an instance of the [ArrayMessage] type.
///
/// Can be created using the [ArrayMessage::builder] method.
#[derive(Debug)]
#[must_use]
pub struct ArrayMessageBuilder<State>(State);

impl<
    
> ArrayMessageBuilder
<
    (
    
    )
>
{
    /// Setter for the [`row_count` field](ArrayMessage#structfield.row_count).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn row_count<T0>(self, value: T0) -> ArrayMessageBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<u32, u32>
    {
        ArrayMessageBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`row_count` field](ArrayMessage#structfield.row_count) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn row_count_as_default(self) -> ArrayMessageBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.row_count(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
> ArrayMessageBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`encodings` field](ArrayMessage#structfield.encodings).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn encodings<T1>(self, value: T1) -> ArrayMessageBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>
    {
        
        let (
            
                v0,
            
        ) = self.0;ArrayMessageBuilder((
            
                v0,
            
            value,
        ))
    }

    

    
    /// Sets the [`encodings` field](ArrayMessage#structfield.encodings) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn encodings_as_null(self) -> ArrayMessageBuilder<(
        
            T0,
        
        (),
    )>
    {
        self.encodings(())
    }
    
}



impl<
    
        T0,
    
        T1,
    
> ArrayMessageBuilder<(
    
        T0,
    
        T1,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [ArrayMessage].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayMessage>
        where Self: ::planus::WriteAsOffset<ArrayMessage>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u32, u32>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
    
> ::planus::WriteAs<::planus::Offset<ArrayMessage>> for ArrayMessageBuilder<(
    
        T0,
    
        T1,
    
)> {
    type Prepared = ::planus::Offset<ArrayMessage>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayMessage> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u32, u32>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
    
> ::planus::WriteAsOptional<::planus::Offset<ArrayMessage>> for ArrayMessageBuilder<(
    
        T0,
    
        T1,
    
)> {
    type Prepared = ::planus::Offset<ArrayMessage>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<ArrayMessage>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u32, u32>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
    
> ::planus::WriteAsOffset<ArrayMessage> for ArrayMessageBuilder<(
    
        T0,
    
        T1,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayMessage> {
        
        let (
            
                v0,
            
                v1,
            
        ) = &self.0;ArrayMessage::create(
            builder,
            
                v0,
            
                v1,
            
        )
    }
}

/// Reference to a deserialized [ArrayMessage].
#[derive(Copy, Clone)]
pub struct ArrayMessageRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> ArrayMessageRef<'a> {
    
        /// Getter for the [`row_count` field](ArrayMessage#structfield.row_count).
        #[inline]
        pub fn row_count(&self) -> ::planus::Result<u32> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "ArrayMessage", "row_count")
              
            
            ?.unwrap_or(0))
            
        }
    
        /// Getter for the [`encodings` field](ArrayMessage#structfield.encodings).
        #[inline]
        pub fn encodings(&self) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, ::planus::Result<&'a ::core::primitive::str>>>> {
            
            
              
              self.0.access(1, "ArrayMessage", "encodings")
              
            
            
            
        }
    
}

impl<'a> ::core::fmt::Debug for ArrayMessageRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("ArrayMessageRef");
        f.field("row_count", &self.row_count());if let ::core::option::Option::Some(field_encodings) = self.encodings().transpose() {
                f.field("encodings", &field_encodings);
            }
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<ArrayMessageRef<'a>> for ArrayMessage {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: ArrayMessageRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            row_count: ::core::convert::TryInto::try_into(value.row_count()?)?,encodings: 
                            if let ::core::option::Option::Some(encodings) = value.encodings()? {
                                ::core::option::Option::Some(encodings.to_vec_result()?)
                            } else {
                                ::core::option::Option::None
                            }
                        ,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for ArrayMessageRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for ArrayMessageRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[ArrayMessageRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<ArrayMessage>> for ArrayMessage {
    type Value = ::planus::Offset<ArrayMessage>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<ArrayMessage>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for ArrayMessageRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[ArrayMessageRef]",
            "read_as_root",
            0,
        ))
    }
}

    
///  Indicates the body contains a regular byte buffer.
/// 
/// Generated from these locations:
/// * Table `BufferMessage` in the file `flatbuffers/vortex-serde/message.fbs:20`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct BufferMessage{
    
        /// The field `alignment_exponent` in the table `BufferMessage`
        pub alignment_exponent: u8,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for BufferMessage {
    fn default() -> Self {
        Self {
        alignment_exponent: 0,
        
        }
    }
}


impl BufferMessage {
    /// Creates a [BufferMessageBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> BufferMessageBuilder<()> {
        BufferMessageBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_alignment_exponent: impl ::planus::WriteAsDefault<u8, u8>,
        
    ) -> ::planus::Offset<Self> {let prepared_alignment_exponent = field_alignment_exponent.prepare(builder, &0);

        let mut table_writer: ::planus::table_writer::TableWriter::<6> = ::core::default::Default::default();
        if prepared_alignment_exponent.is_some() {table_writer.write_entry::<u8>(0);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_alignment_exponent) = prepared_alignment_exponent {object_writer.write::<_, _, 1>(&prepared_alignment_exponent);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<BufferMessage>> for BufferMessage {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<BufferMessage> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<BufferMessage>> for BufferMessage {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<BufferMessage>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<BufferMessage> for BufferMessage {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<BufferMessage> {
        BufferMessage::create(
            builder,
        
            
            self.alignment_exponent,
            
        
        )
    }
}

/// Builder for serializing an instance of the [BufferMessage] type.
///
/// Can be created using the [BufferMessage::builder] method.
#[derive(Debug)]
#[must_use]
pub struct BufferMessageBuilder<State>(State);

impl<
    
> BufferMessageBuilder
<
    (
    
    )
>
{
    /// Setter for the [`alignment_exponent` field](BufferMessage#structfield.alignment_exponent).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn alignment_exponent<T0>(self, value: T0) -> BufferMessageBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<u8, u8>
    {
        BufferMessageBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`alignment_exponent` field](BufferMessage#structfield.alignment_exponent) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn alignment_exponent_as_default(self) -> BufferMessageBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.alignment_exponent(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
> BufferMessageBuilder<(
    
        T0,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [BufferMessage].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<BufferMessage>
        where Self: ::planus::WriteAsOffset<BufferMessage>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u8, u8>,
    
> ::planus::WriteAs<::planus::Offset<BufferMessage>> for BufferMessageBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<BufferMessage>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<BufferMessage> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u8, u8>,
    
> ::planus::WriteAsOptional<::planus::Offset<BufferMessage>> for BufferMessageBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<BufferMessage>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<BufferMessage>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u8, u8>,
    
> ::planus::WriteAsOffset<BufferMessage> for BufferMessageBuilder<(
    
        T0,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<BufferMessage> {
        
        let (
            
                v0,
            
        ) = &self.0;BufferMessage::create(
            builder,
            
                v0,
            
        )
    }
}

/// Reference to a deserialized [BufferMessage].
#[derive(Copy, Clone)]
pub struct BufferMessageRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> BufferMessageRef<'a> {
    
        /// Getter for the [`alignment_exponent` field](BufferMessage#structfield.alignment_exponent).
        #[inline]
        pub fn alignment_exponent(&self) -> ::planus::Result<u8> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "BufferMessage", "alignment_exponent")
              
            
            ?.unwrap_or(0))
            
        }
    
}

impl<'a> ::core::fmt::Debug for BufferMessageRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("BufferMessageRef");
        f.field("alignment_exponent", &self.alignment_exponent());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<BufferMessageRef<'a>> for BufferMessage {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: BufferMessageRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            alignment_exponent: ::core::convert::TryInto::try_into(value.alignment_exponent()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for BufferMessageRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for BufferMessageRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[BufferMessageRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<BufferMessage>> for BufferMessage {
    type Value = ::planus::Offset<BufferMessage>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<BufferMessage>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for BufferMessageRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[BufferMessageRef]",
            "read_as_root",
            0,
        ))
    }
}

    
///  Indicates the body contains a flatbuffer DType message.
/// 
/// Generated from these locations:
/// * Table `DTypeMessage` in the file `flatbuffers/vortex-serde/message.fbs:25`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct DTypeMessage{
    }


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for DTypeMessage {
    fn default() -> Self {
        Self {
        
        }
    }
}


impl DTypeMessage {
    /// Creates a [DTypeMessageBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> DTypeMessageBuilder<()> {
        DTypeMessageBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        
    ) -> ::planus::Offset<Self> {

        let table_writer: ::planus::table_writer::TableWriter::<4> = ::core::default::Default::default();
        unsafe {
            table_writer.finish(builder, |_table_writer| {});
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<DTypeMessage>> for DTypeMessage {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DTypeMessage> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<DTypeMessage>> for DTypeMessage {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<DTypeMessage>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<DTypeMessage> for DTypeMessage {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DTypeMessage> {
        DTypeMessage::create(
            builder,
        
        )
    }
}

/// Builder for serializing an instance of the [DTypeMessage] type.
///
/// Can be created using the [DTypeMessage::builder] method.
#[derive(Debug)]
#[must_use]
pub struct DTypeMessageBuilder<State>(State);



impl<
    
> DTypeMessageBuilder<(
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [DTypeMessage].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<DTypeMessage>
        where Self: ::planus::WriteAsOffset<DTypeMessage>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
> ::planus::WriteAs<::planus::Offset<DTypeMessage>> for DTypeMessageBuilder<(
    
)> {
    type Prepared = ::planus::Offset<DTypeMessage>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DTypeMessage> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
> ::planus::WriteAsOptional<::planus::Offset<DTypeMessage>> for DTypeMessageBuilder<(
    
)> {
    type Prepared = ::planus::Offset<DTypeMessage>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<DTypeMessage>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
> ::planus::WriteAsOffset<DTypeMessage> for DTypeMessageBuilder<(
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DTypeMessage> {
        DTypeMessage::create(
            builder,
            
        )
    }
}

/// Reference to a deserialized [DTypeMessage].
#[derive(Copy, Clone)]
pub struct DTypeMessageRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> DTypeMessageRef<'a> {
    
}

impl<'a> ::core::fmt::Debug for DTypeMessageRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("DTypeMessageRef");
        
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<DTypeMessageRef<'a>> for DTypeMessage {
    type Error = ::planus::Error;


    fn try_from(_value: DTypeMessageRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            
        })
    }
}

impl<'a> ::planus::TableRead<'a> for DTypeMessageRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for DTypeMessageRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[DTypeMessageRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<DTypeMessage>> for DTypeMessage {
    type Value = ::planus::Offset<DTypeMessage>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<DTypeMessage>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for DTypeMessageRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[DTypeMessageRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The union `MessageHeader`
/// 
/// Generated from these locations:
/// * Union `MessageHeader` in the file `flatbuffers/vortex-serde/message.fbs:27`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize)]
pub enum MessageHeader{
    
        /// The variant of type `ArrayMessage` in the union `MessageHeader`
        ArrayMessage(::planus::alloc::boxed::Box<self::ArrayMessage>),
    
        /// The variant of type `BufferMessage` in the union `MessageHeader`
        BufferMessage(::planus::alloc::boxed::Box<self::BufferMessage>),
    
        /// The variant of type `DTypeMessage` in the union `MessageHeader`
        DTypeMessage(::planus::alloc::boxed::Box<self::DTypeMessage>),
    
}


impl MessageHeader {
    /// Creates a [MessageHeaderBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> MessageHeaderBuilder<::planus::Uninitialized> {
        MessageHeaderBuilder(::planus::Uninitialized)
    }

    #[inline]
    pub fn create_array_message(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::ArrayMessage>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(1, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_buffer_message(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::BufferMessage>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(2, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_d_type_message(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::DTypeMessage>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(3, value.prepare(builder).downcast())
    }

    
}



impl ::planus::WriteAsUnion<MessageHeader> for MessageHeader {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Self> {
        match self {
            Self::ArrayMessage(value) => Self::create_array_message(builder, value),
            Self::BufferMessage(value) => Self::create_buffer_message(builder, value),
            Self::DTypeMessage(value) => Self::create_d_type_message(builder, value),
            
        }
    }
}


impl ::planus::WriteAsOptionalUnion<MessageHeader> for MessageHeader {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Self>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}

/// Builder for serializing an instance of the [MessageHeader] type.
///
/// Can be created using the [MessageHeader::builder] method.
#[derive(Debug)]
#[must_use]
pub struct MessageHeaderBuilder<T>(T);

impl MessageHeaderBuilder<::planus::Uninitialized> {
    /// Creates an instance of the [`ArrayMessage` variant](MessageHeader#variant.ArrayMessage).
    #[inline]
    pub fn array_message<T>(
        self,
        value: T,
    ) -> MessageHeaderBuilder<::planus::Initialized<1, T>>
        where T: ::planus::WriteAsOffset<self::ArrayMessage> {
        MessageHeaderBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`BufferMessage` variant](MessageHeader#variant.BufferMessage).
    #[inline]
    pub fn buffer_message<T>(
        self,
        value: T,
    ) -> MessageHeaderBuilder<::planus::Initialized<2, T>>
        where T: ::planus::WriteAsOffset<self::BufferMessage> {
        MessageHeaderBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`DTypeMessage` variant](MessageHeader#variant.DTypeMessage).
    #[inline]
    pub fn d_type_message<T>(
        self,
        value: T,
    ) -> MessageHeaderBuilder<::planus::Initialized<3, T>>
        where T: ::planus::WriteAsOffset<self::DTypeMessage> {
        MessageHeaderBuilder(::planus::Initialized(value))
    }

    
}

impl<const N: u8, T> MessageHeaderBuilder<::planus::Initialized<N, T>> {
    /// Finish writing the builder to get an [UnionOffset](::planus::UnionOffset) to a serialized [MessageHeader].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<MessageHeader>
        where Self: ::planus::WriteAsUnion<MessageHeader>
    {
        ::planus::WriteAsUnion::prepare(&self, builder)
    }
}

impl<T> ::planus::WriteAsUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<1, T>>
    where T: ::planus::WriteAsOffset<self::ArrayMessage>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<MessageHeader> {
        ::planus::UnionOffset::new(1, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<1, T>>
    where T: ::planus::WriteAsOffset<self::ArrayMessage>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<MessageHeader>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<2, T>>
    where T: ::planus::WriteAsOffset<self::BufferMessage>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<MessageHeader> {
        ::planus::UnionOffset::new(2, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<2, T>>
    where T: ::planus::WriteAsOffset<self::BufferMessage>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<MessageHeader>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<3, T>>
    where T: ::planus::WriteAsOffset<self::DTypeMessage>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<MessageHeader> {
        ::planus::UnionOffset::new(3, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<MessageHeader> for MessageHeaderBuilder<::planus::Initialized<3, T>>
    where T: ::planus::WriteAsOffset<self::DTypeMessage>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<MessageHeader>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}


/// Reference to a deserialized [MessageHeader].
#[derive(Copy, Clone, Debug)]
pub enum MessageHeaderRef<'a>{
    ArrayMessage(self::ArrayMessageRef<'a>),
    BufferMessage(self::BufferMessageRef<'a>),
    DTypeMessage(self::DTypeMessageRef<'a>),
    
}


impl<'a> ::core::convert::TryFrom<MessageHeaderRef<'a>> for MessageHeader {
    type Error = ::planus::Error;

    fn try_from(value: MessageHeaderRef<'a>) -> ::planus::Result<Self> {
        ::core::result::Result::Ok(match value {
            
                MessageHeaderRef::ArrayMessage(value) => Self::ArrayMessage(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                MessageHeaderRef::BufferMessage(value) => Self::BufferMessage(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                MessageHeaderRef::DTypeMessage(value) => Self::DTypeMessage(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
        })
    }
}



impl<'a> ::planus::TableReadUnion<'a> for MessageHeaderRef<'a> {
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, tag: u8, field_offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        match tag {
            1 => ::core::result::Result::Ok(Self::ArrayMessage(::planus::TableRead::from_buffer(buffer, field_offset)?)),2 => ::core::result::Result::Ok(Self::BufferMessage(::planus::TableRead::from_buffer(buffer, field_offset)?)),3 => ::core::result::Result::Ok(Self::DTypeMessage(::planus::TableRead::from_buffer(buffer, field_offset)?)),_ => ::core::result::Result::Err(::planus::errors::ErrorKind::UnknownUnionTag { tag }),
        }
    }
}



impl<'a> ::planus::VectorReadUnion<'a> for MessageHeaderRef<'a> {
    const VECTOR_NAME: &'static str = "[MessageHeaderRef]";
}


    
/// The table `Message`
/// 
/// Generated from these locations:
/// * Table `Message` in the file `flatbuffers/vortex-serde/message.fbs:33`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Message{
    
        /// The field `version` in the table `Message`
        pub version: self::MessageVersion,
        /// The field `header` in the table `Message`
        pub header: ::core::option::Option<self::MessageHeader>,
        /// The field `body_size` in the table `Message`
        pub body_size: u64,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Message {
    fn default() -> Self {
        Self {
        version: self::MessageVersion::V0,
        header: ::core::default::Default::default(),
        body_size: 0,
        
        }
    }
}


impl Message {
    /// Creates a [MessageBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> MessageBuilder<()> {
        MessageBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_version: impl ::planus::WriteAsDefault<self::MessageVersion, self::MessageVersion>,
        field_header: impl ::planus::WriteAsOptionalUnion<self::MessageHeader>,
        field_body_size: impl ::planus::WriteAsDefault<u64, u64>,
        
    ) -> ::planus::Offset<Self> {let prepared_version = field_version.prepare(builder, &self::MessageVersion::V0);let prepared_header = field_header.prepare(builder);let prepared_body_size = field_body_size.prepare(builder, &0);

        let mut table_writer: ::planus::table_writer::TableWriter::<12> = ::core::default::Default::default();
        if prepared_body_size.is_some() {table_writer.write_entry::<u64>(3);}if prepared_header.is_some() {table_writer.write_entry::<::planus::Offset<self::MessageHeader>>(2);}if prepared_version.is_some() {table_writer.write_entry::<self::MessageVersion>(0);}if prepared_header.is_some() {table_writer.write_entry::<u8>(1);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_body_size) = prepared_body_size {object_writer.write::<_, _, 8>(&prepared_body_size);}if let ::core::option::Option::Some(prepared_header) = prepared_header {object_writer.write::<_, _, 4>(&prepared_header.offset());}if let ::core::option::Option::Some(prepared_version) = prepared_version {object_writer.write::<_, _, 1>(&prepared_version);}if let ::core::option::Option::Some(prepared_header) = prepared_header {object_writer.write::<_, _, 1>(&prepared_header.tag());}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Message>> for Message {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Message> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Message>> for Message {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Message>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Message> for Message {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Message> {
        Message::create(
            builder,
        
            
            self.version,
            
        
            
            &self.header,
            
        
            
            self.body_size,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Message] type.
///
/// Can be created using the [Message::builder] method.
#[derive(Debug)]
#[must_use]
pub struct MessageBuilder<State>(State);

impl<
    
> MessageBuilder
<
    (
    
    )
>
{
    /// Setter for the [`version` field](Message#structfield.version).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn version<T0>(self, value: T0) -> MessageBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<self::MessageVersion, self::MessageVersion>
    {
        MessageBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`version` field](Message#structfield.version) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn version_as_default(self) -> MessageBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.version(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
> MessageBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`header` field](Message#structfield.header).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn header<T1>(self, value: T1) -> MessageBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsOptionalUnion<self::MessageHeader>
    {
        
        let (
            
                v0,
            
        ) = self.0;MessageBuilder((
            
                v0,
            
            value,
        ))
    }

    

    
    /// Sets the [`header` field](Message#structfield.header) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn header_as_null(self) -> MessageBuilder<(
        
            T0,
        
        (),
    )>
    {
        self.header(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
> MessageBuilder
<
    (
    
        T0,
    
        T1,
    
    )
>
{
    /// Setter for the [`body_size` field](Message#structfield.body_size).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn body_size<T2>(self, value: T2) -> MessageBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
    )>
    where T2: ::planus::WriteAsDefault<u64, u64>
    {
        
        let (
            
                v0,
            
                v1,
            
        ) = self.0;MessageBuilder((
            
                v0,
            
                v1,
            
            value,
        ))
    }

    
    /// Sets the [`body_size` field](Message#structfield.body_size) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn body_size_as_default(self) -> MessageBuilder<(
        
            T0,
        
            T1,
        
        ::planus::DefaultValue,
    )>
    {
        self.body_size(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
        T1,
    
        T2,
    
> MessageBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Message].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Message>
        where Self: ::planus::WriteAsOffset<Message>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<self::MessageVersion, self::MessageVersion>,
    
        T1: ::planus::WriteAsOptionalUnion<self::MessageHeader>,
    
        T2: ::planus::WriteAsDefault<u64, u64>,
    
> ::planus::WriteAs<::planus::Offset<Message>> for MessageBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<Message>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Message> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<self::MessageVersion, self::MessageVersion>,
    
        T1: ::planus::WriteAsOptionalUnion<self::MessageHeader>,
    
        T2: ::planus::WriteAsDefault<u64, u64>,
    
> ::planus::WriteAsOptional<::planus::Offset<Message>> for MessageBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<Message>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Message>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<self::MessageVersion, self::MessageVersion>,
    
        T1: ::planus::WriteAsOptionalUnion<self::MessageHeader>,
    
        T2: ::planus::WriteAsDefault<u64, u64>,
    
> ::planus::WriteAsOffset<Message> for MessageBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Message> {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
        ) = &self.0;Message::create(
            builder,
            
                v0,
            
                v1,
            
                v2,
            
        )
    }
}

/// Reference to a deserialized [Message].
#[derive(Copy, Clone)]
pub struct MessageRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> MessageRef<'a> {
    
        /// Getter for the [`version` field](Message#structfield.version).
        #[inline]
        pub fn version(&self) -> ::planus::Result<self::MessageVersion> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "Message", "version")
              
            
            ?.unwrap_or(self::MessageVersion::V0))
            
        }
    
        /// Getter for the [`header` field](Message#structfield.header).
        #[inline]
        pub fn header(&self) -> ::planus::Result<::core::option::Option<self::MessageHeaderRef<'a>>> {
            
            
              
              self.0.access_union(1, "Message", "header")
              
            
            
            
        }
    
        /// Getter for the [`body_size` field](Message#structfield.body_size).
        #[inline]
        pub fn body_size(&self) -> ::planus::Result<u64> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(3, "Message", "body_size")
              
            
            ?.unwrap_or(0))
            
        }
    
}

impl<'a> ::core::fmt::Debug for MessageRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("MessageRef");
        f.field("version", &self.version());if let ::core::option::Option::Some(field_header) = self.header().transpose() {
                f.field("header", &field_header);
            }f.field("body_size", &self.body_size());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<MessageRef<'a>> for Message {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: MessageRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            version: ::core::convert::TryInto::try_into(value.version()?)?,header: 
                    if let ::core::option::Option::Some(header) = value.header()? {
                        ::core::option::Option::Some(::core::convert::TryInto::try_into(header)?)
                    } else {
                        ::core::option::Option::None
                    }
                ,body_size: ::core::convert::TryInto::try_into(value.body_size()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for MessageRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for MessageRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[MessageRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Message>> for Message {
    type Value = ::planus::Offset<Message>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Message>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for MessageRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[MessageRef]",
            "read_as_root",
            0,
        ))
    }
}

    
///  An Array describes the hierarchy of an array as well as the locations of the data buffers that appear
///  immediately after the message in the byte stream.
/// 
/// Generated from these locations:
/// * Table `Array` in the file `flatbuffers/vortex-array/array.fbs:6`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Array{
    
        ///  The array's hierarchical definition.
        pub root: ::core::option::Option<::planus::alloc::boxed::Box<self::ArrayNode>>,
        ///  The locations of the data buffers of the array
        pub buffers: ::core::option::Option<::planus::alloc::vec::Vec<self::Buffer>>,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Array {
    fn default() -> Self {
        Self {
        root: ::core::default::Default::default(),
        buffers: ::core::default::Default::default(),
        
        }
    }
}


impl Array {
    /// Creates a [ArrayBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> ArrayBuilder<()> {
        ArrayBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_root: impl ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>,
        field_buffers: impl ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
        
    ) -> ::planus::Offset<Self> {let prepared_root = field_root.prepare(builder);let prepared_buffers = field_buffers.prepare(builder);

        let mut table_writer: ::planus::table_writer::TableWriter::<8> = ::core::default::Default::default();
        if prepared_root.is_some() {table_writer.write_entry::<::planus::Offset<self::ArrayNode>>(0);}if prepared_buffers.is_some() {table_writer.write_entry::<::planus::Offset<[self::Buffer]>>(1);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_root) = prepared_root {object_writer.write::<_, _, 4>(&prepared_root);}if let ::core::option::Option::Some(prepared_buffers) = prepared_buffers {object_writer.write::<_, _, 4>(&prepared_buffers);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Array>> for Array {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Array>> for Array {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Array>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Array> for Array {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array> {
        Array::create(
            builder,
        
            
            &self.root,
            
        
            
            &self.buffers,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Array] type.
///
/// Can be created using the [Array::builder] method.
#[derive(Debug)]
#[must_use]
pub struct ArrayBuilder<State>(State);

impl<
    
> ArrayBuilder
<
    (
    
    )
>
{
    /// Setter for the [`root` field](Array#structfield.root).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn root<T0>(self, value: T0) -> ArrayBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>
    {
        ArrayBuilder((
            
            value,
        ))
    }

    

    
    /// Sets the [`root` field](Array#structfield.root) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn root_as_null(self) -> ArrayBuilder<(
        
        (),
    )>
    {
        self.root(())
    }
    
}

impl<
    
        T0,
    
> ArrayBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`buffers` field](Array#structfield.buffers).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn buffers<T1>(self, value: T1) -> ArrayBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>
    {
        
        let (
            
                v0,
            
        ) = self.0;ArrayBuilder((
            
                v0,
            
            value,
        ))
    }

    

    
    /// Sets the [`buffers` field](Array#structfield.buffers) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn buffers_as_null(self) -> ArrayBuilder<(
        
            T0,
        
        (),
    )>
    {
        self.buffers(())
    }
    
}



impl<
    
        T0,
    
        T1,
    
> ArrayBuilder<(
    
        T0,
    
        T1,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Array].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array>
        where Self: ::planus::WriteAsOffset<Array>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
    
> ::planus::WriteAs<::planus::Offset<Array>> for ArrayBuilder<(
    
        T0,
    
        T1,
    
)> {
    type Prepared = ::planus::Offset<Array>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
    
> ::planus::WriteAsOptional<::planus::Offset<Array>> for ArrayBuilder<(
    
        T0,
    
        T1,
    
)> {
    type Prepared = ::planus::Offset<Array>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Array>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
    
> ::planus::WriteAsOffset<Array> for ArrayBuilder<(
    
        T0,
    
        T1,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array> {
        
        let (
            
                v0,
            
                v1,
            
        ) = &self.0;Array::create(
            builder,
            
                v0,
            
                v1,
            
        )
    }
}

/// Reference to a deserialized [Array].
#[derive(Copy, Clone)]
pub struct ArrayRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> ArrayRef<'a> {
    
        /// Getter for the [`root` field](Array#structfield.root).
        #[inline]
        pub fn root(&self) -> ::planus::Result<::core::option::Option<self::ArrayNodeRef<'a>>> {
            
            
              
              self.0.access(0, "Array", "root")
              
            
            
            
        }
    
        /// Getter for the [`buffers` field](Array#structfield.buffers).
        #[inline]
        pub fn buffers(&self) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, self::BufferRef<'a>>>> {
            
            
              
              self.0.access(1, "Array", "buffers")
              
            
            
            
        }
    
}

impl<'a> ::core::fmt::Debug for ArrayRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("ArrayRef");
        if let ::core::option::Option::Some(field_root) = self.root().transpose() {
                f.field("root", &field_root);
            }if let ::core::option::Option::Some(field_buffers) = self.buffers().transpose() {
                f.field("buffers", &field_buffers);
            }
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<ArrayRef<'a>> for Array {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: ArrayRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            root: 
                                if let ::core::option::Option::Some(root) = value.root()? {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(root)?))
                                } else {
                                    ::core::option::Option::None
                                }
                            ,buffers: 
                            if let ::core::option::Option::Some(buffers) = value.buffers()? {
                                ::core::option::Option::Some(buffers.to_vec()?)
                            } else {
                                ::core::option::Option::None
                            }
                        ,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for ArrayRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for ArrayRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[ArrayRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Array>> for Array {
    type Value = ::planus::Offset<Array>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Array>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for ArrayRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[ArrayRef]",
            "read_as_root",
            0,
        ))
    }
}

    
///  The compression mechanism used to compress the buffer.
/// 
/// Generated from these locations:
/// * Enum `Compression` in the file `flatbuffers/vortex-array/array.fbs:14`
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, ::serde::Serialize, ::serde::Deserialize)]#[repr(u8)]pub enum Compression {
    
        /// The variant `None` in the enum `Compression`
        None = 0,
    
        /// The variant `LZ4` in the enum `Compression`
        Lz4 = 1,
    
}

impl Compression {
    /// Array containing all valid variants of Compression
    pub const ENUM_VALUES: [Self; 2] = [Self::None,Self::Lz4,];
}

impl ::core::convert::TryFrom<u8> for Compression {
    type Error = ::planus::errors::UnknownEnumTagKind;
    #[inline]
    fn try_from(value: u8) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind> {
        #[allow(clippy::match_single_binding)]
        match value {
            0 => ::core::result::Result::Ok(Compression::None),
            1 => ::core::result::Result::Ok(Compression::Lz4),
            

            _ => ::core::result::Result::Err(::planus::errors::UnknownEnumTagKind { tag: value as i128 }),
        }
    }
}

impl ::core::convert::From<Compression> for u8 {
    #[inline]
    fn from(value: Compression) -> Self {
        value as u8
    }
}

/// # Safety
/// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
unsafe impl ::planus::Primitive for Compression {
    const ALIGNMENT: usize = 1;
    const SIZE: usize = 1;
}

impl ::planus::WriteAsPrimitive<Compression> for Compression {
    #[inline]
    fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
        (*self as u8).write(cursor, buffer_position);
    }
}

impl ::planus::WriteAs<Compression> for Compression {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> Compression {
        *self
    }
}

impl ::planus::WriteAsDefault<Compression, Compression> for Compression {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder, default: &Compression) -> ::core::option::Option<Compression> {
        if self == default {
            ::core::option::Option::None
        } else {
            ::core::option::Option::Some(*self)
        }
    }
}

impl ::planus::WriteAsOptional<Compression> for Compression {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<Compression> {
        ::core::option::Option::Some(*self)
    }
}

impl<'buf> ::planus::TableRead<'buf> for Compression {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'buf>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        let n: u8 = ::planus::TableRead::from_buffer(buffer, offset)?;
        ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
    }
}

impl<'buf> ::planus::VectorReadInner<'buf> for Compression {
    type Error = ::planus::errors::UnknownEnumTag;
    const STRIDE: usize = 1;
    #[inline]
    unsafe fn from_buffer(
        buffer: ::planus::SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag> {let value = unsafe { *buffer.buffer.get_unchecked(offset) };let value: ::core::result::Result<Self, _> = ::core::convert::TryInto::try_into(value);
        value.map_err(|error_kind| error_kind.with_error_location(
            "Compression",
            "VectorRead::from_buffer",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<Compression> for Compression {
    const STRIDE: usize = 1;

    type Value = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
        *self
    }

    #[inline]
    unsafe fn write_values(
        values: &[Self],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 1];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                
                buffer_position - i as u32,
                
            );
        }
    }
}

    
///  A Buffer describes the location of a data buffer in the byte stream as a packed 64-bit struct.
/// 
/// Generated from these locations:
/// * Struct `Buffer` in the file `flatbuffers/vortex-array/array.fbs:20`
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,

::serde::Serialize, ::serde::Deserialize
)]
pub struct Buffer {
    
        ///  The length of any padding bytes written immediately before the buffer.
        pub padding: u16,
    
        ///  The minimum alignment of the buffer, stored as an exponent of 2.
        pub alignment_exponent: u8,
    
        ///  The compression algorithm used to compress the buffer.
        pub compression: self::Compression,
    
        ///  The length of the buffer in bytes.
        pub length: u32,
    
}

/// # Safety
/// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
unsafe impl ::planus::Primitive for Buffer {
    const ALIGNMENT: usize = 4;
    const SIZE: usize = 8;
}

#[allow(clippy::identity_op)]
impl ::planus::WriteAsPrimitive<Buffer> for Buffer {
    #[inline]
    fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
        let (cur, cursor) = cursor.split::<2 , 6>();
            self.padding.write(cur, buffer_position - 0);let (cur, cursor) = cursor.split::<1 , 5>();
            self.alignment_exponent.write(cur, buffer_position - 2);let (cur, cursor) = cursor.split::<1 , 4>();
            self.compression.write(cur, buffer_position - 3);let (cur, cursor) = cursor.split::<4 , 0>();
            self.length.write(cur, buffer_position - 4);
        cursor.finish([]);
    }
}

impl ::planus::WriteAsOffset<Buffer> for Buffer {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Buffer> {
        unsafe {
            builder.write_with(
                8,
                3,
                |buffer_position, bytes| {
                    let bytes = bytes.as_mut_ptr();

                    ::planus::WriteAsPrimitive::write(
                        self,
                        ::planus::Cursor::new(&mut *(bytes as *mut [::core::mem::MaybeUninit<u8>; 8])),
                        buffer_position,
                    );
                }
            );
        }
        builder.current_offset()
    }
}

impl ::planus::WriteAs<Buffer> for Buffer {
    type Prepared = Self;
    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
        *self
    }
}

impl ::planus::WriteAsOptional<Buffer> for Buffer {
    type Prepared = Self;
    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<Self> {
        ::core::option::Option::Some(*self)
    }
}

/// Reference to a deserialized [Buffer].
#[derive(Copy, Clone)]
pub struct BufferRef<'a>(::planus::ArrayWithStartOffset<'a, 8>);

impl<'a> BufferRef<'a> {
    
        /// Getter for the [`padding` field](Buffer#structfield.padding).
        pub fn padding(&self) -> u16 {
            let buffer = self.0.advance_as_array::<2>(0).unwrap();
            
            u16::from_le_bytes(*buffer.as_array())
            
        }
    
        /// Getter for the [`alignment_exponent` field](Buffer#structfield.alignment_exponent).
        pub fn alignment_exponent(&self) -> u8 {
            let buffer = self.0.advance_as_array::<1>(2).unwrap();
            
            u8::from_le_bytes(*buffer.as_array())
            
        }
    
        /// Getter for the [`compression` field](Buffer#structfield.compression).
        pub fn compression(&self) -> ::core::result::Result<self::Compression, ::planus::errors::UnknownEnumTag> {
            let buffer = self.0.advance_as_array::<1>(3).unwrap();
            
            let value: ::core::result::Result<self::Compression, _> = ::core::convert::TryInto::try_into(u8::from_le_bytes(*buffer.as_array()));
                    value.map_err(|e| e.with_error_location(
                        "BufferRef",
                        "compression",
                        buffer.offset_from_start,
                    ))
            
        }
    
        /// Getter for the [`length` field](Buffer#structfield.length).
        pub fn length(&self) -> u32 {
            let buffer = self.0.advance_as_array::<4>(4).unwrap();
            
            u32::from_le_bytes(*buffer.as_array())
            
        }
    
}

impl<'a> ::core::fmt::Debug for BufferRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("BufferRef");
        f.field("padding", &self.padding());f.field("alignment_exponent", &self.alignment_exponent());f.field("compression", &self.compression());f.field("length", &self.length());
        f.finish()
    }
}

impl<'a> ::core::convert::From<::planus::ArrayWithStartOffset<'a, 8>> for BufferRef<'a> {
    fn from(array: ::planus::ArrayWithStartOffset<'a, 8>) -> Self {
        Self(array)
    }
}

impl<'a> ::core::convert::TryFrom<BufferRef<'a>> for Buffer {
    type Error = ::planus::Error;

    #[allow(unreachable_code)]
    fn try_from(value: BufferRef<'a>) -> ::planus::Result<Self> {
        ::core::result::Result::Ok(Self {
        padding:
            value.padding(),
            alignment_exponent:
            value.alignment_exponent(),
            compression:
            value.compression()?,length:
            value.length(),
            
        })
    }
}

impl<'a> ::planus::TableRead<'a> for BufferRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        let buffer = buffer.advance_as_array::<8>(offset)?;
        ::core::result::Result::Ok(Self(buffer))
    }
}

impl<'a> ::planus::VectorRead<'a> for BufferRef<'a> {
    const STRIDE: usize = 8;

    #[inline]
    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> Self {
        Self(unsafe { buffer.unchecked_advance_as_array(offset) })
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<Buffer> for Buffer {
    const STRIDE: usize = 8;

    type Value = Buffer;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
        *self
    }

    #[inline]
    unsafe fn write_values(
        values: &[Buffer],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 8];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                
                buffer_position - (8 * i) as u32,
                
            );
        }
    }
}

    
/// The table `ArrayNode`
/// 
/// Generated from these locations:
/// * Table `ArrayNode` in the file `flatbuffers/vortex-array/array.fbs:32`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct ArrayNode{
    
        /// The field `encoding` in the table `ArrayNode`
        pub encoding: u16,
        /// The field `metadata` in the table `ArrayNode`
        pub metadata: ::core::option::Option<::planus::alloc::vec::Vec<u8>>,
        /// The field `children` in the table `ArrayNode`
        pub children: ::core::option::Option<::planus::alloc::vec::Vec<self::ArrayNode>>,
        /// The field `buffers` in the table `ArrayNode`
        pub buffers: ::core::option::Option<::planus::alloc::vec::Vec<u16>>,
        /// The field `stats` in the table `ArrayNode`
        pub stats: ::core::option::Option<::planus::alloc::boxed::Box<self::ArrayStats>>,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for ArrayNode {
    fn default() -> Self {
        Self {
        encoding: 0,
        metadata: ::core::default::Default::default(),
        children: ::core::default::Default::default(),
        buffers: ::core::default::Default::default(),
        stats: ::core::default::Default::default(),
        
        }
    }
}


impl ArrayNode {
    /// Creates a [ArrayNodeBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> ArrayNodeBuilder<()> {
        ArrayNodeBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_encoding: impl ::planus::WriteAsDefault<u16, u16>,
        field_metadata: impl ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        field_children: impl ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>,
        field_buffers: impl ::planus::WriteAsOptional<::planus::Offset<[u16]>>,
        field_stats: impl ::planus::WriteAsOptional<::planus::Offset<self::ArrayStats>>,
        
    ) -> ::planus::Offset<Self> {let prepared_encoding = field_encoding.prepare(builder, &0);let prepared_metadata = field_metadata.prepare(builder);let prepared_children = field_children.prepare(builder);let prepared_buffers = field_buffers.prepare(builder);let prepared_stats = field_stats.prepare(builder);

        let mut table_writer: ::planus::table_writer::TableWriter::<14> = ::core::default::Default::default();
        if prepared_metadata.is_some() {table_writer.write_entry::<::planus::Offset<[u8]>>(1);}if prepared_children.is_some() {table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>(2);}if prepared_buffers.is_some() {table_writer.write_entry::<::planus::Offset<[u16]>>(3);}if prepared_stats.is_some() {table_writer.write_entry::<::planus::Offset<self::ArrayStats>>(4);}if prepared_encoding.is_some() {table_writer.write_entry::<u16>(0);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_metadata) = prepared_metadata {object_writer.write::<_, _, 4>(&prepared_metadata);}if let ::core::option::Option::Some(prepared_children) = prepared_children {object_writer.write::<_, _, 4>(&prepared_children);}if let ::core::option::Option::Some(prepared_buffers) = prepared_buffers {object_writer.write::<_, _, 4>(&prepared_buffers);}if let ::core::option::Option::Some(prepared_stats) = prepared_stats {object_writer.write::<_, _, 4>(&prepared_stats);}if let ::core::option::Option::Some(prepared_encoding) = prepared_encoding {object_writer.write::<_, _, 2>(&prepared_encoding);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<ArrayNode>> for ArrayNode {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayNode> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<ArrayNode>> for ArrayNode {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<ArrayNode>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<ArrayNode> for ArrayNode {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayNode> {
        ArrayNode::create(
            builder,
        
            
            self.encoding,
            
        
            
            &self.metadata,
            
        
            
            &self.children,
            
        
            
            &self.buffers,
            
        
            
            &self.stats,
            
        
        )
    }
}

/// Builder for serializing an instance of the [ArrayNode] type.
///
/// Can be created using the [ArrayNode::builder] method.
#[derive(Debug)]
#[must_use]
pub struct ArrayNodeBuilder<State>(State);

impl<
    
> ArrayNodeBuilder
<
    (
    
    )
>
{
    /// Setter for the [`encoding` field](ArrayNode#structfield.encoding).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn encoding<T0>(self, value: T0) -> ArrayNodeBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<u16, u16>
    {
        ArrayNodeBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`encoding` field](ArrayNode#structfield.encoding) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn encoding_as_default(self) -> ArrayNodeBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.encoding(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
> ArrayNodeBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`metadata` field](ArrayNode#structfield.metadata).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn metadata<T1>(self, value: T1) -> ArrayNodeBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsOptional<::planus::Offset<[u8]>>
    {
        
        let (
            
                v0,
            
        ) = self.0;ArrayNodeBuilder((
            
                v0,
            
            value,
        ))
    }

    

    
    /// Sets the [`metadata` field](ArrayNode#structfield.metadata) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn metadata_as_null(self) -> ArrayNodeBuilder<(
        
            T0,
        
        (),
    )>
    {
        self.metadata(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
> ArrayNodeBuilder
<
    (
    
        T0,
    
        T1,
    
    )
>
{
    /// Setter for the [`children` field](ArrayNode#structfield.children).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn children<T2>(self, value: T2) -> ArrayNodeBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
    )>
    where T2: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>
    {
        
        let (
            
                v0,
            
                v1,
            
        ) = self.0;ArrayNodeBuilder((
            
                v0,
            
                v1,
            
            value,
        ))
    }

    

    
    /// Sets the [`children` field](ArrayNode#structfield.children) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn children_as_null(self) -> ArrayNodeBuilder<(
        
            T0,
        
            T1,
        
        (),
    )>
    {
        self.children(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
> ArrayNodeBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
    )
>
{
    /// Setter for the [`buffers` field](ArrayNode#structfield.buffers).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn buffers<T3>(self, value: T3) -> ArrayNodeBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
    )>
    where T3: ::planus::WriteAsOptional<::planus::Offset<[u16]>>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
        ) = self.0;ArrayNodeBuilder((
            
                v0,
            
                v1,
            
                v2,
            
            value,
        ))
    }

    

    
    /// Sets the [`buffers` field](ArrayNode#structfield.buffers) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn buffers_as_null(self) -> ArrayNodeBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
        (),
    )>
    {
        self.buffers(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
> ArrayNodeBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
    )
>
{
    /// Setter for the [`stats` field](ArrayNode#structfield.stats).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn stats<T4>(self, value: T4) -> ArrayNodeBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
    )>
    where T4: ::planus::WriteAsOptional<::planus::Offset<self::ArrayStats>>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
        ) = self.0;ArrayNodeBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
            value,
        ))
    }

    

    
    /// Sets the [`stats` field](ArrayNode#structfield.stats) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn stats_as_null(self) -> ArrayNodeBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
        (),
    )>
    {
        self.stats(())
    }
    
}



impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
> ArrayNodeBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [ArrayNode].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayNode>
        where Self: ::planus::WriteAsOffset<ArrayNode>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u16, u16>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>,
    
        T3: ::planus::WriteAsOptional<::planus::Offset<[u16]>>,
    
        T4: ::planus::WriteAsOptional<::planus::Offset<self::ArrayStats>>,
    
> ::planus::WriteAs<::planus::Offset<ArrayNode>> for ArrayNodeBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
)> {
    type Prepared = ::planus::Offset<ArrayNode>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayNode> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u16, u16>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>,
    
        T3: ::planus::WriteAsOptional<::planus::Offset<[u16]>>,
    
        T4: ::planus::WriteAsOptional<::planus::Offset<self::ArrayStats>>,
    
> ::planus::WriteAsOptional<::planus::Offset<ArrayNode>> for ArrayNodeBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
)> {
    type Prepared = ::planus::Offset<ArrayNode>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<ArrayNode>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u16, u16>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>,
    
        T3: ::planus::WriteAsOptional<::planus::Offset<[u16]>>,
    
        T4: ::planus::WriteAsOptional<::planus::Offset<self::ArrayStats>>,
    
> ::planus::WriteAsOffset<ArrayNode> for ArrayNodeBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayNode> {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
        ) = &self.0;ArrayNode::create(
            builder,
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
        )
    }
}

/// Reference to a deserialized [ArrayNode].
#[derive(Copy, Clone)]
pub struct ArrayNodeRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> ArrayNodeRef<'a> {
    
        /// Getter for the [`encoding` field](ArrayNode#structfield.encoding).
        #[inline]
        pub fn encoding(&self) -> ::planus::Result<u16> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "ArrayNode", "encoding")
              
            
            ?.unwrap_or(0))
            
        }
    
        /// Getter for the [`metadata` field](ArrayNode#structfield.metadata).
        #[inline]
        pub fn metadata(&self) -> ::planus::Result<::core::option::Option<&'a [u8]>> {
            
            
              
              self.0.access(1, "ArrayNode", "metadata")
              
            
            
            
        }
    
        /// Getter for the [`children` field](ArrayNode#structfield.children).
        #[inline]
        pub fn children(&self) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, ::planus::Result<self::ArrayNodeRef<'a>>>>> {
            
            
              
              self.0.access(2, "ArrayNode", "children")
              
            
            
            
        }
    
        /// Getter for the [`buffers` field](ArrayNode#structfield.buffers).
        #[inline]
        pub fn buffers(&self) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, u16>>> {
            
            
              
              self.0.access(3, "ArrayNode", "buffers")
              
            
            
            
        }
    
        /// Getter for the [`stats` field](ArrayNode#structfield.stats).
        #[inline]
        pub fn stats(&self) -> ::planus::Result<::core::option::Option<self::ArrayStatsRef<'a>>> {
            
            
              
              self.0.access(4, "ArrayNode", "stats")
              
            
            
            
        }
    
}

impl<'a> ::core::fmt::Debug for ArrayNodeRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("ArrayNodeRef");
        f.field("encoding", &self.encoding());if let ::core::option::Option::Some(field_metadata) = self.metadata().transpose() {
                f.field("metadata", &field_metadata);
            }if let ::core::option::Option::Some(field_children) = self.children().transpose() {
                f.field("children", &field_children);
            }if let ::core::option::Option::Some(field_buffers) = self.buffers().transpose() {
                f.field("buffers", &field_buffers);
            }if let ::core::option::Option::Some(field_stats) = self.stats().transpose() {
                f.field("stats", &field_stats);
            }
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<ArrayNodeRef<'a>> for ArrayNode {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: ArrayNodeRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            encoding: ::core::convert::TryInto::try_into(value.encoding()?)?,metadata: value.metadata()?.map(|v| v.to_vec()),children: 
                            if let ::core::option::Option::Some(children) = value.children()? {
                                ::core::option::Option::Some(children.to_vec_result()?)
                            } else {
                                ::core::option::Option::None
                            }
                        ,buffers: 
                            if let ::core::option::Option::Some(buffers) = value.buffers()? {
                                ::core::option::Option::Some(buffers.to_vec()?)
                            } else {
                                ::core::option::Option::None
                            }
                        ,stats: 
                                if let ::core::option::Option::Some(stats) = value.stats()? {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(stats)?))
                                } else {
                                    ::core::option::Option::None
                                }
                            ,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for ArrayNodeRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for ArrayNodeRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[ArrayNodeRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<ArrayNode>> for ArrayNode {
    type Value = ::planus::Offset<ArrayNode>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<ArrayNode>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for ArrayNodeRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[ArrayNodeRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The enum `Precision`
/// 
/// Generated from these locations:
/// * Enum `Precision` in the file `flatbuffers/vortex-array/array.fbs:40`
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, ::serde::Serialize, ::serde::Deserialize)]#[repr(u8)]pub enum Precision {
    
        /// The variant `Inexact` in the enum `Precision`
        Inexact = 0,
    
        /// The variant `Exact` in the enum `Precision`
        Exact = 1,
    
}

impl Precision {
    /// Array containing all valid variants of Precision
    pub const ENUM_VALUES: [Self; 2] = [Self::Inexact,Self::Exact,];
}

impl ::core::convert::TryFrom<u8> for Precision {
    type Error = ::planus::errors::UnknownEnumTagKind;
    #[inline]
    fn try_from(value: u8) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind> {
        #[allow(clippy::match_single_binding)]
        match value {
            0 => ::core::result::Result::Ok(Precision::Inexact),
            1 => ::core::result::Result::Ok(Precision::Exact),
            

            _ => ::core::result::Result::Err(::planus::errors::UnknownEnumTagKind { tag: value as i128 }),
        }
    }
}

impl ::core::convert::From<Precision> for u8 {
    #[inline]
    fn from(value: Precision) -> Self {
        value as u8
    }
}

/// # Safety
/// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
unsafe impl ::planus::Primitive for Precision {
    const ALIGNMENT: usize = 1;
    const SIZE: usize = 1;
}

impl ::planus::WriteAsPrimitive<Precision> for Precision {
    #[inline]
    fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
        (*self as u8).write(cursor, buffer_position);
    }
}

impl ::planus::WriteAs<Precision> for Precision {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> Precision {
        *self
    }
}

impl ::planus::WriteAsDefault<Precision, Precision> for Precision {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder, default: &Precision) -> ::core::option::Option<Precision> {
        if self == default {
            ::core::option::Option::None
        } else {
            ::core::option::Option::Some(*self)
        }
    }
}

impl ::planus::WriteAsOptional<Precision> for Precision {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<Precision> {
        ::core::option::Option::Some(*self)
    }
}

impl<'buf> ::planus::TableRead<'buf> for Precision {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'buf>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        let n: u8 = ::planus::TableRead::from_buffer(buffer, offset)?;
        ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
    }
}

impl<'buf> ::planus::VectorReadInner<'buf> for Precision {
    type Error = ::planus::errors::UnknownEnumTag;
    const STRIDE: usize = 1;
    #[inline]
    unsafe fn from_buffer(
        buffer: ::planus::SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag> {let value = unsafe { *buffer.buffer.get_unchecked(offset) };let value: ::core::result::Result<Self, _> = ::core::convert::TryInto::try_into(value);
        value.map_err(|error_kind| error_kind.with_error_location(
            "Precision",
            "VectorRead::from_buffer",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<Precision> for Precision {
    const STRIDE: usize = 1;

    type Value = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
        *self
    }

    #[inline]
    unsafe fn write_values(
        values: &[Self],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 1];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                
                buffer_position - i as u32,
                
            );
        }
    }
}

    
/// The table `ArrayStats`
/// 
/// Generated from these locations:
/// * Table `ArrayStats` in the file `flatbuffers/vortex-array/array.fbs:45`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct ArrayStats{
    
        ///  Protobuf serialized ScalarValue
        pub min: ::core::option::Option<::planus::alloc::vec::Vec<u8>>,
        /// The field `min_precision` in the table `ArrayStats`
        pub min_precision: self::Precision,
        /// The field `max` in the table `ArrayStats`
        pub max: ::core::option::Option<::planus::alloc::vec::Vec<u8>>,
        /// The field `max_precision` in the table `ArrayStats`
        pub max_precision: self::Precision,
        /// The field `sum` in the table `ArrayStats`
        pub sum: ::core::option::Option<::planus::alloc::vec::Vec<u8>>,
        /// The field `is_sorted` in the table `ArrayStats`
        pub is_sorted: ::core::option::Option<bool>,
        /// The field `is_strict_sorted` in the table `ArrayStats`
        pub is_strict_sorted: ::core::option::Option<bool>,
        /// The field `is_constant` in the table `ArrayStats`
        pub is_constant: ::core::option::Option<bool>,
        /// The field `null_count` in the table `ArrayStats`
        pub null_count: ::core::option::Option<u64>,
        /// The field `uncompressed_size_in_bytes` in the table `ArrayStats`
        pub uncompressed_size_in_bytes: ::core::option::Option<u64>,
        /// The field `nan_count` in the table `ArrayStats`
        pub nan_count: ::core::option::Option<u64>,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for ArrayStats {
    fn default() -> Self {
        Self {
        min: ::core::default::Default::default(),
        min_precision: self::Precision::Inexact,
        max: ::core::default::Default::default(),
        max_precision: self::Precision::Inexact,
        sum: ::core::default::Default::default(),
        is_sorted: ::core::default::Default::default(),
        is_strict_sorted: ::core::default::Default::default(),
        is_constant: ::core::default::Default::default(),
        null_count: ::core::default::Default::default(),
        uncompressed_size_in_bytes: ::core::default::Default::default(),
        nan_count: ::core::default::Default::default(),
        
        }
    }
}


impl ArrayStats {
    /// Creates a [ArrayStatsBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> ArrayStatsBuilder<()> {
        ArrayStatsBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_min: impl ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        field_min_precision: impl ::planus::WriteAsDefault<self::Precision, self::Precision>,
        field_max: impl ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        field_max_precision: impl ::planus::WriteAsDefault<self::Precision, self::Precision>,
        field_sum: impl ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        field_is_sorted: impl ::planus::WriteAsOptional<bool>,
        field_is_strict_sorted: impl ::planus::WriteAsOptional<bool>,
        field_is_constant: impl ::planus::WriteAsOptional<bool>,
        field_null_count: impl ::planus::WriteAsOptional<u64>,
        field_uncompressed_size_in_bytes: impl ::planus::WriteAsOptional<u64>,
        field_nan_count: impl ::planus::WriteAsOptional<u64>,
        
    ) -> ::planus::Offset<Self> {let prepared_min = field_min.prepare(builder);let prepared_min_precision = field_min_precision.prepare(builder, &self::Precision::Inexact);let prepared_max = field_max.prepare(builder);let prepared_max_precision = field_max_precision.prepare(builder, &self::Precision::Inexact);let prepared_sum = field_sum.prepare(builder);let prepared_is_sorted = field_is_sorted.prepare(builder);let prepared_is_strict_sorted = field_is_strict_sorted.prepare(builder);let prepared_is_constant = field_is_constant.prepare(builder);let prepared_null_count = field_null_count.prepare(builder);let prepared_uncompressed_size_in_bytes = field_uncompressed_size_in_bytes.prepare(builder);let prepared_nan_count = field_nan_count.prepare(builder);

        let mut table_writer: ::planus::table_writer::TableWriter::<26> = ::core::default::Default::default();
        if prepared_null_count.is_some() {table_writer.write_entry::<u64>(8);}if prepared_uncompressed_size_in_bytes.is_some() {table_writer.write_entry::<u64>(9);}if prepared_nan_count.is_some() {table_writer.write_entry::<u64>(10);}if prepared_min.is_some() {table_writer.write_entry::<::planus::Offset<[u8]>>(0);}if prepared_max.is_some() {table_writer.write_entry::<::planus::Offset<[u8]>>(2);}if prepared_sum.is_some() {table_writer.write_entry::<::planus::Offset<[u8]>>(4);}if prepared_min_precision.is_some() {table_writer.write_entry::<self::Precision>(1);}if prepared_max_precision.is_some() {table_writer.write_entry::<self::Precision>(3);}if prepared_is_sorted.is_some() {table_writer.write_entry::<bool>(5);}if prepared_is_strict_sorted.is_some() {table_writer.write_entry::<bool>(6);}if prepared_is_constant.is_some() {table_writer.write_entry::<bool>(7);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_null_count) = prepared_null_count {object_writer.write::<_, _, 8>(&prepared_null_count);}if let ::core::option::Option::Some(prepared_uncompressed_size_in_bytes) = prepared_uncompressed_size_in_bytes {object_writer.write::<_, _, 8>(&prepared_uncompressed_size_in_bytes);}if let ::core::option::Option::Some(prepared_nan_count) = prepared_nan_count {object_writer.write::<_, _, 8>(&prepared_nan_count);}if let ::core::option::Option::Some(prepared_min) = prepared_min {object_writer.write::<_, _, 4>(&prepared_min);}if let ::core::option::Option::Some(prepared_max) = prepared_max {object_writer.write::<_, _, 4>(&prepared_max);}if let ::core::option::Option::Some(prepared_sum) = prepared_sum {object_writer.write::<_, _, 4>(&prepared_sum);}if let ::core::option::Option::Some(prepared_min_precision) = prepared_min_precision {object_writer.write::<_, _, 1>(&prepared_min_precision);}if let ::core::option::Option::Some(prepared_max_precision) = prepared_max_precision {object_writer.write::<_, _, 1>(&prepared_max_precision);}if let ::core::option::Option::Some(prepared_is_sorted) = prepared_is_sorted {object_writer.write::<_, _, 1>(&prepared_is_sorted);}if let ::core::option::Option::Some(prepared_is_strict_sorted) = prepared_is_strict_sorted {object_writer.write::<_, _, 1>(&prepared_is_strict_sorted);}if let ::core::option::Option::Some(prepared_is_constant) = prepared_is_constant {object_writer.write::<_, _, 1>(&prepared_is_constant);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<ArrayStats>> for ArrayStats {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayStats> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<ArrayStats>> for ArrayStats {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<ArrayStats>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<ArrayStats> for ArrayStats {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayStats> {
        ArrayStats::create(
            builder,
        
            
            &self.min,
            
        
            
            self.min_precision,
            
        
            
            &self.max,
            
        
            
            self.max_precision,
            
        
            
            &self.sum,
            
        
            
            self.is_sorted,
            
        
            
            self.is_strict_sorted,
            
        
            
            self.is_constant,
            
        
            
            self.null_count,
            
        
            
            self.uncompressed_size_in_bytes,
            
        
            
            self.nan_count,
            
        
        )
    }
}

/// Builder for serializing an instance of the [ArrayStats] type.
///
/// Can be created using the [ArrayStats::builder] method.
#[derive(Debug)]
#[must_use]
pub struct ArrayStatsBuilder<State>(State);

impl<
    
> ArrayStatsBuilder
<
    (
    
    )
>
{
    /// Setter for the [`min` field](ArrayStats#structfield.min).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn min<T0>(self, value: T0) -> ArrayStatsBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsOptional<::planus::Offset<[u8]>>
    {
        ArrayStatsBuilder((
            
            value,
        ))
    }

    

    
    /// Sets the [`min` field](ArrayStats#structfield.min) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn min_as_null(self) -> ArrayStatsBuilder<(
        
        (),
    )>
    {
        self.min(())
    }
    
}

impl<
    
        T0,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`min_precision` field](ArrayStats#structfield.min_precision).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn min_precision<T1>(self, value: T1) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsDefault<self::Precision, self::Precision>
    {
        
        let (
            
                v0,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
            value,
        ))
    }

    
    /// Sets the [`min_precision` field](ArrayStats#structfield.min_precision) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn min_precision_as_default(self) -> ArrayStatsBuilder<(
        
            T0,
        
        ::planus::DefaultValue,
    )>
    {
        self.min_precision(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
        T1,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
    )
>
{
    /// Setter for the [`max` field](ArrayStats#structfield.max).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn max<T2>(self, value: T2) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
    )>
    where T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>
    {
        
        let (
            
                v0,
            
                v1,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
            value,
        ))
    }

    

    
    /// Sets the [`max` field](ArrayStats#structfield.max) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn max_as_null(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
        (),
    )>
    {
        self.max(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
    )
>
{
    /// Setter for the [`max_precision` field](ArrayStats#structfield.max_precision).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn max_precision<T3>(self, value: T3) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
    )>
    where T3: ::planus::WriteAsDefault<self::Precision, self::Precision>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
                v2,
            
            value,
        ))
    }

    
    /// Sets the [`max_precision` field](ArrayStats#structfield.max_precision) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn max_precision_as_default(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
        ::planus::DefaultValue,
    )>
    {
        self.max_precision(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
    )
>
{
    /// Setter for the [`sum` field](ArrayStats#structfield.sum).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn sum<T4>(self, value: T4) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
    )>
    where T4: ::planus::WriteAsOptional<::planus::Offset<[u8]>>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
            value,
        ))
    }

    

    
    /// Sets the [`sum` field](ArrayStats#structfield.sum) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn sum_as_null(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
        (),
    )>
    {
        self.sum(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
    )
>
{
    /// Setter for the [`is_sorted` field](ArrayStats#structfield.is_sorted).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn is_sorted<T5>(self, value: T5) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
    )>
    where T5: ::planus::WriteAsOptional<bool>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
            value,
        ))
    }

    

    
    /// Sets the [`is_sorted` field](ArrayStats#structfield.is_sorted) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn is_sorted_as_null(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
            T4,
        
        (),
    )>
    {
        self.is_sorted(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
    )
>
{
    /// Setter for the [`is_strict_sorted` field](ArrayStats#structfield.is_strict_sorted).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn is_strict_sorted<T6>(self, value: T6) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
    )>
    where T6: ::planus::WriteAsOptional<bool>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
            value,
        ))
    }

    

    
    /// Sets the [`is_strict_sorted` field](ArrayStats#structfield.is_strict_sorted) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn is_strict_sorted_as_null(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
            T4,
        
            T5,
        
        (),
    )>
    {
        self.is_strict_sorted(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
    )
>
{
    /// Setter for the [`is_constant` field](ArrayStats#structfield.is_constant).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn is_constant<T7>(self, value: T7) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
    )>
    where T7: ::planus::WriteAsOptional<bool>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
            value,
        ))
    }

    

    
    /// Sets the [`is_constant` field](ArrayStats#structfield.is_constant) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn is_constant_as_null(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
            T4,
        
            T5,
        
            T6,
        
        (),
    )>
    {
        self.is_constant(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
    )
>
{
    /// Setter for the [`null_count` field](ArrayStats#structfield.null_count).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn null_count<T8>(self, value: T8) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
    )>
    where T8: ::planus::WriteAsOptional<u64>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
                v7,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
                v7,
            
            value,
        ))
    }

    

    
    /// Sets the [`null_count` field](ArrayStats#structfield.null_count) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn null_count_as_null(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
            T4,
        
            T5,
        
            T6,
        
            T7,
        
        (),
    )>
    {
        self.null_count(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
    )
>
{
    /// Setter for the [`uncompressed_size_in_bytes` field](ArrayStats#structfield.uncompressed_size_in_bytes).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn uncompressed_size_in_bytes<T9>(self, value: T9) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
    )>
    where T9: ::planus::WriteAsOptional<u64>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
                v7,
            
                v8,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
                v7,
            
                v8,
            
            value,
        ))
    }

    

    
    /// Sets the [`uncompressed_size_in_bytes` field](ArrayStats#structfield.uncompressed_size_in_bytes) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn uncompressed_size_in_bytes_as_null(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
            T4,
        
            T5,
        
            T6,
        
            T7,
        
            T8,
        
        (),
    )>
    {
        self.uncompressed_size_in_bytes(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
> ArrayStatsBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
    )
>
{
    /// Setter for the [`nan_count` field](ArrayStats#structfield.nan_count).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nan_count<T10>(self, value: T10) -> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
        T10,
    
    )>
    where T10: ::planus::WriteAsOptional<u64>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
                v7,
            
                v8,
            
                v9,
            
        ) = self.0;ArrayStatsBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
                v7,
            
                v8,
            
                v9,
            
            value,
        ))
    }

    

    
    /// Sets the [`nan_count` field](ArrayStats#structfield.nan_count) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nan_count_as_null(self) -> ArrayStatsBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
            T4,
        
            T5,
        
            T6,
        
            T7,
        
            T8,
        
            T9,
        
        (),
    )>
    {
        self.nan_count(())
    }
    
}



impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
        T10,
    
> ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
        T10,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [ArrayStats].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayStats>
        where Self: ::planus::WriteAsOffset<ArrayStats>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T1: ::planus::WriteAsDefault<self::Precision, self::Precision>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T3: ::planus::WriteAsDefault<self::Precision, self::Precision>,
    
        T4: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T5: ::planus::WriteAsOptional<bool>,
    
        T6: ::planus::WriteAsOptional<bool>,
    
        T7: ::planus::WriteAsOptional<bool>,
    
        T8: ::planus::WriteAsOptional<u64>,
    
        T9: ::planus::WriteAsOptional<u64>,
    
        T10: ::planus::WriteAsOptional<u64>,
    
> ::planus::WriteAs<::planus::Offset<ArrayStats>> for ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
        T10,
    
)> {
    type Prepared = ::planus::Offset<ArrayStats>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayStats> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T1: ::planus::WriteAsDefault<self::Precision, self::Precision>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T3: ::planus::WriteAsDefault<self::Precision, self::Precision>,
    
        T4: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T5: ::planus::WriteAsOptional<bool>,
    
        T6: ::planus::WriteAsOptional<bool>,
    
        T7: ::planus::WriteAsOptional<bool>,
    
        T8: ::planus::WriteAsOptional<u64>,
    
        T9: ::planus::WriteAsOptional<u64>,
    
        T10: ::planus::WriteAsOptional<u64>,
    
> ::planus::WriteAsOptional<::planus::Offset<ArrayStats>> for ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
        T10,
    
)> {
    type Prepared = ::planus::Offset<ArrayStats>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<ArrayStats>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T1: ::planus::WriteAsDefault<self::Precision, self::Precision>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T3: ::planus::WriteAsDefault<self::Precision, self::Precision>,
    
        T4: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T5: ::planus::WriteAsOptional<bool>,
    
        T6: ::planus::WriteAsOptional<bool>,
    
        T7: ::planus::WriteAsOptional<bool>,
    
        T8: ::planus::WriteAsOptional<u64>,
    
        T9: ::planus::WriteAsOptional<u64>,
    
        T10: ::planus::WriteAsOptional<u64>,
    
> ::planus::WriteAsOffset<ArrayStats> for ArrayStatsBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
        T5,
    
        T6,
    
        T7,
    
        T8,
    
        T9,
    
        T10,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayStats> {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
                v7,
            
                v8,
            
                v9,
            
                v10,
            
        ) = &self.0;ArrayStats::create(
            builder,
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
                v5,
            
                v6,
            
                v7,
            
                v8,
            
                v9,
            
                v10,
            
        )
    }
}

/// Reference to a deserialized [ArrayStats].
#[derive(Copy, Clone)]
pub struct ArrayStatsRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> ArrayStatsRef<'a> {
    
        /// Getter for the [`min` field](ArrayStats#structfield.min).
        #[inline]
        pub fn min(&self) -> ::planus::Result<::core::option::Option<&'a [u8]>> {
            
            
              
              self.0.access(0, "ArrayStats", "min")
              
            
            
            
        }
    
        /// Getter for the [`min_precision` field](ArrayStats#structfield.min_precision).
        #[inline]
        pub fn min_precision(&self) -> ::planus::Result<self::Precision> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(1, "ArrayStats", "min_precision")
              
            
            ?.unwrap_or(self::Precision::Inexact))
            
        }
    
        /// Getter for the [`max` field](ArrayStats#structfield.max).
        #[inline]
        pub fn max(&self) -> ::planus::Result<::core::option::Option<&'a [u8]>> {
            
            
              
              self.0.access(2, "ArrayStats", "max")
              
            
            
            
        }
    
        /// Getter for the [`max_precision` field](ArrayStats#structfield.max_precision).
        #[inline]
        pub fn max_precision(&self) -> ::planus::Result<self::Precision> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(3, "ArrayStats", "max_precision")
              
            
            ?.unwrap_or(self::Precision::Inexact))
            
        }
    
        /// Getter for the [`sum` field](ArrayStats#structfield.sum).
        #[inline]
        pub fn sum(&self) -> ::planus::Result<::core::option::Option<&'a [u8]>> {
            
            
              
              self.0.access(4, "ArrayStats", "sum")
              
            
            
            
        }
    
        /// Getter for the [`is_sorted` field](ArrayStats#structfield.is_sorted).
        #[inline]
        pub fn is_sorted(&self) -> ::planus::Result<::core::option::Option<bool>> {
            
            
              
              self.0.access(5, "ArrayStats", "is_sorted")
              
            
            
            
        }
    
        /// Getter for the [`is_strict_sorted` field](ArrayStats#structfield.is_strict_sorted).
        #[inline]
        pub fn is_strict_sorted(&self) -> ::planus::Result<::core::option::Option<bool>> {
            
            
              
              self.0.access(6, "ArrayStats", "is_strict_sorted")
              
            
            
            
        }
    
        /// Getter for the [`is_constant` field](ArrayStats#structfield.is_constant).
        #[inline]
        pub fn is_constant(&self) -> ::planus::Result<::core::option::Option<bool>> {
            
            
              
              self.0.access(7, "ArrayStats", "is_constant")
              
            
            
            
        }
    
        /// Getter for the [`null_count` field](ArrayStats#structfield.null_count).
        #[inline]
        pub fn null_count(&self) -> ::planus::Result<::core::option::Option<u64>> {
            
            
              
              self.0.access(8, "ArrayStats", "null_count")
              
            
            
            
        }
    
        /// Getter for the [`uncompressed_size_in_bytes` field](ArrayStats#structfield.uncompressed_size_in_bytes).
        #[inline]
        pub fn uncompressed_size_in_bytes(&self) -> ::planus::Result<::core::option::Option<u64>> {
            
            
              
              self.0.access(9, "ArrayStats", "uncompressed_size_in_bytes")
              
            
            
            
        }
    
        /// Getter for the [`nan_count` field](ArrayStats#structfield.nan_count).
        #[inline]
        pub fn nan_count(&self) -> ::planus::Result<::core::option::Option<u64>> {
            
            
              
              self.0.access(10, "ArrayStats", "nan_count")
              
            
            
            
        }
    
}

impl<'a> ::core::fmt::Debug for ArrayStatsRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("ArrayStatsRef");
        if let ::core::option::Option::Some(field_min) = self.min().transpose() {
                f.field("min", &field_min);
            }f.field("min_precision", &self.min_precision());if let ::core::option::Option::Some(field_max) = self.max().transpose() {
                f.field("max", &field_max);
            }f.field("max_precision", &self.max_precision());if let ::core::option::Option::Some(field_sum) = self.sum().transpose() {
                f.field("sum", &field_sum);
            }if let ::core::option::Option::Some(field_is_sorted) = self.is_sorted().transpose() {
                f.field("is_sorted", &field_is_sorted);
            }if let ::core::option::Option::Some(field_is_strict_sorted) = self.is_strict_sorted().transpose() {
                f.field("is_strict_sorted", &field_is_strict_sorted);
            }if let ::core::option::Option::Some(field_is_constant) = self.is_constant().transpose() {
                f.field("is_constant", &field_is_constant);
            }if let ::core::option::Option::Some(field_null_count) = self.null_count().transpose() {
                f.field("null_count", &field_null_count);
            }if let ::core::option::Option::Some(field_uncompressed_size_in_bytes) = self.uncompressed_size_in_bytes().transpose() {
                f.field("uncompressed_size_in_bytes", &field_uncompressed_size_in_bytes);
            }if let ::core::option::Option::Some(field_nan_count) = self.nan_count().transpose() {
                f.field("nan_count", &field_nan_count);
            }
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<ArrayStatsRef<'a>> for ArrayStats {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: ArrayStatsRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            min: value.min()?.map(|v| v.to_vec()),min_precision: ::core::convert::TryInto::try_into(value.min_precision()?)?,max: value.max()?.map(|v| v.to_vec()),max_precision: ::core::convert::TryInto::try_into(value.max_precision()?)?,sum: value.sum()?.map(|v| v.to_vec()),is_sorted: 
                    if let ::core::option::Option::Some(is_sorted) = value.is_sorted()? {
                        ::core::option::Option::Some(::core::convert::TryInto::try_into(is_sorted)?)
                    } else {
                        ::core::option::Option::None
                    }
                ,is_strict_sorted: 
                    if let ::core::option::Option::Some(is_strict_sorted) = value.is_strict_sorted()? {
                        ::core::option::Option::Some(::core::convert::TryInto::try_into(is_strict_sorted)?)
                    } else {
                        ::core::option::Option::None
                    }
                ,is_constant: 
                    if let ::core::option::Option::Some(is_constant) = value.is_constant()? {
                        ::core::option::Option::Some(::core::convert::TryInto::try_into(is_constant)?)
                    } else {
                        ::core::option::Option::None
                    }
                ,null_count: 
                    if let ::core::option::Option::Some(null_count) = value.null_count()? {
                        ::core::option::Option::Some(::core::convert::TryInto::try_into(null_count)?)
                    } else {
                        ::core::option::Option::None
                    }
                ,uncompressed_size_in_bytes: 
                    if let ::core::option::Option::Some(uncompressed_size_in_bytes) = value.uncompressed_size_in_bytes()? {
                        ::core::option::Option::Some(::core::convert::TryInto::try_into(uncompressed_size_in_bytes)?)
                    } else {
                        ::core::option::Option::None
                    }
                ,nan_count: 
                    if let ::core::option::Option::Some(nan_count) = value.nan_count()? {
                        ::core::option::Option::Some(::core::convert::TryInto::try_into(nan_count)?)
                    } else {
                        ::core::option::Option::None
                    }
                ,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for ArrayStatsRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for ArrayStatsRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[ArrayStatsRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<ArrayStats>> for ArrayStats {
    type Value = ::planus::Offset<ArrayStats>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<ArrayStats>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for ArrayStatsRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[ArrayStatsRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The enum `PType`
/// 
/// Generated from these locations:
/// * Enum `PType` in the file `flatbuffers/vortex-dtype/dtype.fbs:4`
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, ::serde::Serialize, ::serde::Deserialize)]#[repr(u8)]pub enum PType {
    
        /// The variant `U8` in the enum `PType`
        U8 = 0,
    
        /// The variant `U16` in the enum `PType`
        U16 = 1,
    
        /// The variant `U32` in the enum `PType`
        U32 = 2,
    
        /// The variant `U64` in the enum `PType`
        U64 = 3,
    
        /// The variant `I8` in the enum `PType`
        I8 = 4,
    
        /// The variant `I16` in the enum `PType`
        I16 = 5,
    
        /// The variant `I32` in the enum `PType`
        I32 = 6,
    
        /// The variant `I64` in the enum `PType`
        I64 = 7,
    
        /// The variant `F16` in the enum `PType`
        F16 = 8,
    
        /// The variant `F32` in the enum `PType`
        F32 = 9,
    
        /// The variant `F64` in the enum `PType`
        F64 = 10,
    
}

impl PType {
    /// Array containing all valid variants of PType
    pub const ENUM_VALUES: [Self; 11] = [Self::U8,Self::U16,Self::U32,Self::U64,Self::I8,Self::I16,Self::I32,Self::I64,Self::F16,Self::F32,Self::F64,];
}

impl ::core::convert::TryFrom<u8> for PType {
    type Error = ::planus::errors::UnknownEnumTagKind;
    #[inline]
    fn try_from(value: u8) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind> {
        #[allow(clippy::match_single_binding)]
        match value {
            0 => ::core::result::Result::Ok(PType::U8),
            1 => ::core::result::Result::Ok(PType::U16),
            2 => ::core::result::Result::Ok(PType::U32),
            3 => ::core::result::Result::Ok(PType::U64),
            4 => ::core::result::Result::Ok(PType::I8),
            5 => ::core::result::Result::Ok(PType::I16),
            6 => ::core::result::Result::Ok(PType::I32),
            7 => ::core::result::Result::Ok(PType::I64),
            8 => ::core::result::Result::Ok(PType::F16),
            9 => ::core::result::Result::Ok(PType::F32),
            10 => ::core::result::Result::Ok(PType::F64),
            

            _ => ::core::result::Result::Err(::planus::errors::UnknownEnumTagKind { tag: value as i128 }),
        }
    }
}

impl ::core::convert::From<PType> for u8 {
    #[inline]
    fn from(value: PType) -> Self {
        value as u8
    }
}

/// # Safety
/// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
unsafe impl ::planus::Primitive for PType {
    const ALIGNMENT: usize = 1;
    const SIZE: usize = 1;
}

impl ::planus::WriteAsPrimitive<PType> for PType {
    #[inline]
    fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
        (*self as u8).write(cursor, buffer_position);
    }
}

impl ::planus::WriteAs<PType> for PType {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> PType {
        *self
    }
}

impl ::planus::WriteAsDefault<PType, PType> for PType {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder, default: &PType) -> ::core::option::Option<PType> {
        if self == default {
            ::core::option::Option::None
        } else {
            ::core::option::Option::Some(*self)
        }
    }
}

impl ::planus::WriteAsOptional<PType> for PType {
    type Prepared = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<PType> {
        ::core::option::Option::Some(*self)
    }
}

impl<'buf> ::planus::TableRead<'buf> for PType {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'buf>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        let n: u8 = ::planus::TableRead::from_buffer(buffer, offset)?;
        ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
    }
}

impl<'buf> ::planus::VectorReadInner<'buf> for PType {
    type Error = ::planus::errors::UnknownEnumTag;
    const STRIDE: usize = 1;
    #[inline]
    unsafe fn from_buffer(
        buffer: ::planus::SliceWithStartOffset<'buf>,
        offset: usize,
    ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag> {let value = unsafe { *buffer.buffer.get_unchecked(offset) };let value: ::core::result::Result<Self, _> = ::core::convert::TryInto::try_into(value);
        value.map_err(|error_kind| error_kind.with_error_location(
            "PType",
            "VectorRead::from_buffer",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<PType> for PType {
    const STRIDE: usize = 1;

    type Value = Self;

    #[inline]
    fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
        *self
    }

    #[inline]
    unsafe fn write_values(
        values: &[Self],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 1];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                
                buffer_position - i as u32,
                
            );
        }
    }
}

    
/// The table `Null`
/// 
/// Generated from these locations:
/// * Table `Null` in the file `flatbuffers/vortex-dtype/dtype.fbs:18`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Null{
    }


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Null {
    fn default() -> Self {
        Self {
        
        }
    }
}


impl Null {
    /// Creates a [NullBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> NullBuilder<()> {
        NullBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        
    ) -> ::planus::Offset<Self> {

        let table_writer: ::planus::table_writer::TableWriter::<4> = ::core::default::Default::default();
        unsafe {
            table_writer.finish(builder, |_table_writer| {});
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Null>> for Null {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Null>> for Null {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Null>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Null> for Null {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null> {
        Null::create(
            builder,
        
        )
    }
}

/// Builder for serializing an instance of the [Null] type.
///
/// Can be created using the [Null::builder] method.
#[derive(Debug)]
#[must_use]
pub struct NullBuilder<State>(State);



impl<
    
> NullBuilder<(
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Null].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null>
        where Self: ::planus::WriteAsOffset<Null>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
> ::planus::WriteAs<::planus::Offset<Null>> for NullBuilder<(
    
)> {
    type Prepared = ::planus::Offset<Null>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
> ::planus::WriteAsOptional<::planus::Offset<Null>> for NullBuilder<(
    
)> {
    type Prepared = ::planus::Offset<Null>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Null>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
> ::planus::WriteAsOffset<Null> for NullBuilder<(
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null> {
        Null::create(
            builder,
            
        )
    }
}

/// Reference to a deserialized [Null].
#[derive(Copy, Clone)]
pub struct NullRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> NullRef<'a> {
    
}

impl<'a> ::core::fmt::Debug for NullRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("NullRef");
        
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<NullRef<'a>> for Null {
    type Error = ::planus::Error;


    fn try_from(_value: NullRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            
        })
    }
}

impl<'a> ::planus::TableRead<'a> for NullRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for NullRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[NullRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Null>> for Null {
    type Value = ::planus::Offset<Null>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Null>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for NullRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[NullRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `Bool`
/// 
/// Generated from these locations:
/// * Table `Bool` in the file `flatbuffers/vortex-dtype/dtype.fbs:20`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Bool{
    
        /// The field `nullable` in the table `Bool`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Bool {
    fn default() -> Self {
        Self {
        nullable: false,
        
        }
    }
}


impl Bool {
    /// Creates a [BoolBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> BoolBuilder<()> {
        BoolBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<6> = ::core::default::Default::default();
        if prepared_nullable.is_some() {table_writer.write_entry::<bool>(0);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Bool>> for Bool {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Bool>> for Bool {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Bool>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Bool> for Bool {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool> {
        Bool::create(
            builder,
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Bool] type.
///
/// Can be created using the [Bool::builder] method.
#[derive(Debug)]
#[must_use]
pub struct BoolBuilder<State>(State);

impl<
    
> BoolBuilder
<
    (
    
    )
>
{
    /// Setter for the [`nullable` field](Bool#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T0>(self, value: T0) -> BoolBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<bool, bool>
    {
        BoolBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](Bool#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> BoolBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
> BoolBuilder<(
    
        T0,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Bool].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool>
        where Self: ::planus::WriteAsOffset<Bool>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<Bool>> for BoolBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<Bool>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<Bool>> for BoolBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<Bool>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Bool>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<Bool> for BoolBuilder<(
    
        T0,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool> {
        
        let (
            
                v0,
            
        ) = &self.0;Bool::create(
            builder,
            
                v0,
            
        )
    }
}

/// Reference to a deserialized [Bool].
#[derive(Copy, Clone)]
pub struct BoolRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> BoolRef<'a> {
    
        /// Getter for the [`nullable` field](Bool#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "Bool", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for BoolRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("BoolRef");
        f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<BoolRef<'a>> for Bool {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: BoolRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for BoolRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for BoolRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[BoolRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Bool>> for Bool {
    type Value = ::planus::Offset<Bool>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Bool>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for BoolRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[BoolRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `Primitive`
/// 
/// Generated from these locations:
/// * Table `Primitive` in the file `flatbuffers/vortex-dtype/dtype.fbs:24`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Primitive{
    
        /// The field `ptype` in the table `Primitive`
        pub ptype: self::PType,
        /// The field `nullable` in the table `Primitive`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Primitive {
    fn default() -> Self {
        Self {
        ptype: self::PType::U8,
        nullable: false,
        
        }
    }
}


impl Primitive {
    /// Creates a [PrimitiveBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> PrimitiveBuilder<()> {
        PrimitiveBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_ptype: impl ::planus::WriteAsDefault<self::PType, self::PType>,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_ptype = field_ptype.prepare(builder, &self::PType::U8);let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<8> = ::core::default::Default::default();
        if prepared_ptype.is_some() {table_writer.write_entry::<self::PType>(0);}if prepared_nullable.is_some() {table_writer.write_entry::<bool>(1);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_ptype) = prepared_ptype {object_writer.write::<_, _, 1>(&prepared_ptype);}if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Primitive>> for Primitive {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Primitive>> for Primitive {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Primitive>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Primitive> for Primitive {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive> {
        Primitive::create(
            builder,
        
            
            self.ptype,
            
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Primitive] type.
///
/// Can be created using the [Primitive::builder] method.
#[derive(Debug)]
#[must_use]
pub struct PrimitiveBuilder<State>(State);

impl<
    
> PrimitiveBuilder
<
    (
    
    )
>
{
    /// Setter for the [`ptype` field](Primitive#structfield.ptype).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn ptype<T0>(self, value: T0) -> PrimitiveBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<self::PType, self::PType>
    {
        PrimitiveBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`ptype` field](Primitive#structfield.ptype) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn ptype_as_default(self) -> PrimitiveBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.ptype(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
> PrimitiveBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`nullable` field](Primitive#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T1>(self, value: T1) -> PrimitiveBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsDefault<bool, bool>
    {
        
        let (
            
                v0,
            
        ) = self.0;PrimitiveBuilder((
            
                v0,
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](Primitive#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> PrimitiveBuilder<(
        
            T0,
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
        T1,
    
> PrimitiveBuilder<(
    
        T0,
    
        T1,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Primitive].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive>
        where Self: ::planus::WriteAsOffset<Primitive>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<self::PType, self::PType>,
    
        T1: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<Primitive>> for PrimitiveBuilder<(
    
        T0,
    
        T1,
    
)> {
    type Prepared = ::planus::Offset<Primitive>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<self::PType, self::PType>,
    
        T1: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<Primitive>> for PrimitiveBuilder<(
    
        T0,
    
        T1,
    
)> {
    type Prepared = ::planus::Offset<Primitive>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Primitive>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<self::PType, self::PType>,
    
        T1: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<Primitive> for PrimitiveBuilder<(
    
        T0,
    
        T1,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive> {
        
        let (
            
                v0,
            
                v1,
            
        ) = &self.0;Primitive::create(
            builder,
            
                v0,
            
                v1,
            
        )
    }
}

/// Reference to a deserialized [Primitive].
#[derive(Copy, Clone)]
pub struct PrimitiveRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> PrimitiveRef<'a> {
    
        /// Getter for the [`ptype` field](Primitive#structfield.ptype).
        #[inline]
        pub fn ptype(&self) -> ::planus::Result<self::PType> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "Primitive", "ptype")
              
            
            ?.unwrap_or(self::PType::U8))
            
        }
    
        /// Getter for the [`nullable` field](Primitive#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(1, "Primitive", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for PrimitiveRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("PrimitiveRef");
        f.field("ptype", &self.ptype());f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<PrimitiveRef<'a>> for Primitive {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: PrimitiveRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            ptype: ::core::convert::TryInto::try_into(value.ptype()?)?,nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for PrimitiveRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for PrimitiveRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[PrimitiveRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Primitive>> for Primitive {
    type Value = ::planus::Offset<Primitive>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Primitive>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for PrimitiveRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[PrimitiveRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `Decimal`
/// 
/// Generated from these locations:
/// * Table `Decimal` in the file `flatbuffers/vortex-dtype/dtype.fbs:29`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Decimal{
    
        /// The field `precision` in the table `Decimal`
        pub precision: u8,
        /// The field `scale` in the table `Decimal`
        pub scale: i8,
        /// The field `nullable` in the table `Decimal`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Decimal {
    fn default() -> Self {
        Self {
        precision: 0,
        scale: 0,
        nullable: false,
        
        }
    }
}


impl Decimal {
    /// Creates a [DecimalBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> DecimalBuilder<()> {
        DecimalBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_precision: impl ::planus::WriteAsDefault<u8, u8>,
        field_scale: impl ::planus::WriteAsDefault<i8, i8>,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_precision = field_precision.prepare(builder, &0);let prepared_scale = field_scale.prepare(builder, &0);let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<10> = ::core::default::Default::default();
        if prepared_precision.is_some() {table_writer.write_entry::<u8>(0);}if prepared_scale.is_some() {table_writer.write_entry::<i8>(1);}if prepared_nullable.is_some() {table_writer.write_entry::<bool>(2);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_precision) = prepared_precision {object_writer.write::<_, _, 1>(&prepared_precision);}if let ::core::option::Option::Some(prepared_scale) = prepared_scale {object_writer.write::<_, _, 1>(&prepared_scale);}if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Decimal>> for Decimal {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Decimal> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Decimal>> for Decimal {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Decimal>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Decimal> for Decimal {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Decimal> {
        Decimal::create(
            builder,
        
            
            self.precision,
            
        
            
            self.scale,
            
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Decimal] type.
///
/// Can be created using the [Decimal::builder] method.
#[derive(Debug)]
#[must_use]
pub struct DecimalBuilder<State>(State);

impl<
    
> DecimalBuilder
<
    (
    
    )
>
{
    /// Setter for the [`precision` field](Decimal#structfield.precision).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn precision<T0>(self, value: T0) -> DecimalBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<u8, u8>
    {
        DecimalBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`precision` field](Decimal#structfield.precision) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn precision_as_default(self) -> DecimalBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.precision(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
> DecimalBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`scale` field](Decimal#structfield.scale).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn scale<T1>(self, value: T1) -> DecimalBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsDefault<i8, i8>
    {
        
        let (
            
                v0,
            
        ) = self.0;DecimalBuilder((
            
                v0,
            
            value,
        ))
    }

    
    /// Sets the [`scale` field](Decimal#structfield.scale) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn scale_as_default(self) -> DecimalBuilder<(
        
            T0,
        
        ::planus::DefaultValue,
    )>
    {
        self.scale(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
        T1,
    
> DecimalBuilder
<
    (
    
        T0,
    
        T1,
    
    )
>
{
    /// Setter for the [`nullable` field](Decimal#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T2>(self, value: T2) -> DecimalBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
    )>
    where T2: ::planus::WriteAsDefault<bool, bool>
    {
        
        let (
            
                v0,
            
                v1,
            
        ) = self.0;DecimalBuilder((
            
                v0,
            
                v1,
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](Decimal#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> DecimalBuilder<(
        
            T0,
        
            T1,
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
        T1,
    
        T2,
    
> DecimalBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Decimal].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Decimal>
        where Self: ::planus::WriteAsOffset<Decimal>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u8, u8>,
    
        T1: ::planus::WriteAsDefault<i8, i8>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<Decimal>> for DecimalBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<Decimal>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Decimal> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u8, u8>,
    
        T1: ::planus::WriteAsDefault<i8, i8>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<Decimal>> for DecimalBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<Decimal>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Decimal>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u8, u8>,
    
        T1: ::planus::WriteAsDefault<i8, i8>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<Decimal> for DecimalBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Decimal> {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
        ) = &self.0;Decimal::create(
            builder,
            
                v0,
            
                v1,
            
                v2,
            
        )
    }
}

/// Reference to a deserialized [Decimal].
#[derive(Copy, Clone)]
pub struct DecimalRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> DecimalRef<'a> {
    
        /// Getter for the [`precision` field](Decimal#structfield.precision).
        #[inline]
        pub fn precision(&self) -> ::planus::Result<u8> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "Decimal", "precision")
              
            
            ?.unwrap_or(0))
            
        }
    
        /// Getter for the [`scale` field](Decimal#structfield.scale).
        #[inline]
        pub fn scale(&self) -> ::planus::Result<i8> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(1, "Decimal", "scale")
              
            
            ?.unwrap_or(0))
            
        }
    
        /// Getter for the [`nullable` field](Decimal#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(2, "Decimal", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for DecimalRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("DecimalRef");
        f.field("precision", &self.precision());f.field("scale", &self.scale());f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<DecimalRef<'a>> for Decimal {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: DecimalRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            precision: ::core::convert::TryInto::try_into(value.precision()?)?,scale: ::core::convert::TryInto::try_into(value.scale()?)?,nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for DecimalRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for DecimalRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[DecimalRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Decimal>> for Decimal {
    type Value = ::planus::Offset<Decimal>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Decimal>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for DecimalRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[DecimalRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `Utf8`
/// 
/// Generated from these locations:
/// * Table `Utf8` in the file `flatbuffers/vortex-dtype/dtype.fbs:35`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Utf8{
    
        /// The field `nullable` in the table `Utf8`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Utf8 {
    fn default() -> Self {
        Self {
        nullable: false,
        
        }
    }
}


impl Utf8 {
    /// Creates a [Utf8Builder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> Utf8Builder<()> {
        Utf8Builder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<6> = ::core::default::Default::default();
        if prepared_nullable.is_some() {table_writer.write_entry::<bool>(0);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Utf8>> for Utf8 {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Utf8>> for Utf8 {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Utf8>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Utf8> for Utf8 {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8> {
        Utf8::create(
            builder,
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Utf8] type.
///
/// Can be created using the [Utf8::builder] method.
#[derive(Debug)]
#[must_use]
pub struct Utf8Builder<State>(State);

impl<
    
> Utf8Builder
<
    (
    
    )
>
{
    /// Setter for the [`nullable` field](Utf8#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T0>(self, value: T0) -> Utf8Builder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<bool, bool>
    {
        Utf8Builder((
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](Utf8#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> Utf8Builder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
> Utf8Builder<(
    
        T0,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Utf8].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8>
        where Self: ::planus::WriteAsOffset<Utf8>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<Utf8>> for Utf8Builder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<Utf8>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<Utf8>> for Utf8Builder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<Utf8>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Utf8>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<Utf8> for Utf8Builder<(
    
        T0,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8> {
        
        let (
            
                v0,
            
        ) = &self.0;Utf8::create(
            builder,
            
                v0,
            
        )
    }
}

/// Reference to a deserialized [Utf8].
#[derive(Copy, Clone)]
pub struct Utf8Ref<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> Utf8Ref<'a> {
    
        /// Getter for the [`nullable` field](Utf8#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "Utf8", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for Utf8Ref<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("Utf8Ref");
        f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<Utf8Ref<'a>> for Utf8 {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: Utf8Ref<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for Utf8Ref<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for Utf8Ref<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[Utf8Ref]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Utf8>> for Utf8 {
    type Value = ::planus::Offset<Utf8>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Utf8>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for Utf8Ref<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[Utf8Ref]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `Binary`
/// 
/// Generated from these locations:
/// * Table `Binary` in the file `flatbuffers/vortex-dtype/dtype.fbs:39`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Binary{
    
        /// The field `nullable` in the table `Binary`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Binary {
    fn default() -> Self {
        Self {
        nullable: false,
        
        }
    }
}


impl Binary {
    /// Creates a [BinaryBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> BinaryBuilder<()> {
        BinaryBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<6> = ::core::default::Default::default();
        if prepared_nullable.is_some() {table_writer.write_entry::<bool>(0);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Binary>> for Binary {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Binary>> for Binary {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Binary>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Binary> for Binary {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary> {
        Binary::create(
            builder,
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Binary] type.
///
/// Can be created using the [Binary::builder] method.
#[derive(Debug)]
#[must_use]
pub struct BinaryBuilder<State>(State);

impl<
    
> BinaryBuilder
<
    (
    
    )
>
{
    /// Setter for the [`nullable` field](Binary#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T0>(self, value: T0) -> BinaryBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<bool, bool>
    {
        BinaryBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](Binary#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> BinaryBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
> BinaryBuilder<(
    
        T0,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Binary].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary>
        where Self: ::planus::WriteAsOffset<Binary>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<Binary>> for BinaryBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<Binary>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<Binary>> for BinaryBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<Binary>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Binary>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<Binary> for BinaryBuilder<(
    
        T0,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary> {
        
        let (
            
                v0,
            
        ) = &self.0;Binary::create(
            builder,
            
                v0,
            
        )
    }
}

/// Reference to a deserialized [Binary].
#[derive(Copy, Clone)]
pub struct BinaryRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> BinaryRef<'a> {
    
        /// Getter for the [`nullable` field](Binary#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "Binary", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for BinaryRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("BinaryRef");
        f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<BinaryRef<'a>> for Binary {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: BinaryRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for BinaryRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for BinaryRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[BinaryRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Binary>> for Binary {
    type Value = ::planus::Offset<Binary>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Binary>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for BinaryRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[BinaryRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `Struct_`
/// 
/// Generated from these locations:
/// * Table `Struct_` in the file `flatbuffers/vortex-dtype/dtype.fbs:43`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Struct{
    
        /// The field `names` in the table `Struct_`
        pub names: ::core::option::Option<::planus::alloc::vec::Vec<::planus::alloc::string::String>>,
        /// The field `dtypes` in the table `Struct_`
        pub dtypes: ::core::option::Option<::planus::alloc::vec::Vec<self::DType>>,
        /// The field `nullable` in the table `Struct_`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Struct {
    fn default() -> Self {
        Self {
        names: ::core::default::Default::default(),
        dtypes: ::core::default::Default::default(),
        nullable: false,
        
        }
    }
}


impl Struct {
    /// Creates a [StructBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> StructBuilder<()> {
        StructBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_names: impl ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
        field_dtypes: impl ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::DType>]>>,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_names = field_names.prepare(builder);let prepared_dtypes = field_dtypes.prepare(builder);let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<10> = ::core::default::Default::default();
        if prepared_names.is_some() {table_writer.write_entry::<::planus::Offset<[::planus::Offset<str>]>>(0);}if prepared_dtypes.is_some() {table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::DType>]>>(1);}if prepared_nullable.is_some() {table_writer.write_entry::<bool>(2);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_names) = prepared_names {object_writer.write::<_, _, 4>(&prepared_names);}if let ::core::option::Option::Some(prepared_dtypes) = prepared_dtypes {object_writer.write::<_, _, 4>(&prepared_dtypes);}if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Struct>> for Struct {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Struct> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Struct>> for Struct {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Struct>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Struct> for Struct {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Struct> {
        Struct::create(
            builder,
        
            
            &self.names,
            
        
            
            &self.dtypes,
            
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Struct] type.
///
/// Can be created using the [Struct::builder] method.
#[derive(Debug)]
#[must_use]
pub struct StructBuilder<State>(State);

impl<
    
> StructBuilder
<
    (
    
    )
>
{
    /// Setter for the [`names` field](Struct#structfield.names).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn names<T0>(self, value: T0) -> StructBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>
    {
        StructBuilder((
            
            value,
        ))
    }

    

    
    /// Sets the [`names` field](Struct#structfield.names) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn names_as_null(self) -> StructBuilder<(
        
        (),
    )>
    {
        self.names(())
    }
    
}

impl<
    
        T0,
    
> StructBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`dtypes` field](Struct#structfield.dtypes).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn dtypes<T1>(self, value: T1) -> StructBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::DType>]>>
    {
        
        let (
            
                v0,
            
        ) = self.0;StructBuilder((
            
                v0,
            
            value,
        ))
    }

    

    
    /// Sets the [`dtypes` field](Struct#structfield.dtypes) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn dtypes_as_null(self) -> StructBuilder<(
        
            T0,
        
        (),
    )>
    {
        self.dtypes(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
> StructBuilder
<
    (
    
        T0,
    
        T1,
    
    )
>
{
    /// Setter for the [`nullable` field](Struct#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T2>(self, value: T2) -> StructBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
    )>
    where T2: ::planus::WriteAsDefault<bool, bool>
    {
        
        let (
            
                v0,
            
                v1,
            
        ) = self.0;StructBuilder((
            
                v0,
            
                v1,
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](Struct#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> StructBuilder<(
        
            T0,
        
            T1,
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
        T1,
    
        T2,
    
> StructBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Struct].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Struct>
        where Self: ::planus::WriteAsOffset<Struct>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::DType>]>>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<Struct>> for StructBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<Struct>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Struct> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::DType>]>>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<Struct>> for StructBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<Struct>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Struct>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::DType>]>>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<Struct> for StructBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Struct> {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
        ) = &self.0;Struct::create(
            builder,
            
                v0,
            
                v1,
            
                v2,
            
        )
    }
}

/// Reference to a deserialized [Struct].
#[derive(Copy, Clone)]
pub struct StructRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> StructRef<'a> {
    
        /// Getter for the [`names` field](Struct#structfield.names).
        #[inline]
        pub fn names(&self) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, ::planus::Result<&'a ::core::primitive::str>>>> {
            
            
              
              self.0.access(0, "Struct", "names")
              
            
            
            
        }
    
        /// Getter for the [`dtypes` field](Struct#structfield.dtypes).
        #[inline]
        pub fn dtypes(&self) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, ::planus::Result<self::DTypeRef<'a>>>>> {
            
            
              
              self.0.access(1, "Struct", "dtypes")
              
            
            
            
        }
    
        /// Getter for the [`nullable` field](Struct#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(2, "Struct", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for StructRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("StructRef");
        if let ::core::option::Option::Some(field_names) = self.names().transpose() {
                f.field("names", &field_names);
            }if let ::core::option::Option::Some(field_dtypes) = self.dtypes().transpose() {
                f.field("dtypes", &field_dtypes);
            }f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<StructRef<'a>> for Struct {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: StructRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            names: 
                            if let ::core::option::Option::Some(names) = value.names()? {
                                ::core::option::Option::Some(names.to_vec_result()?)
                            } else {
                                ::core::option::Option::None
                            }
                        ,dtypes: 
                            if let ::core::option::Option::Some(dtypes) = value.dtypes()? {
                                ::core::option::Option::Some(dtypes.to_vec_result()?)
                            } else {
                                ::core::option::Option::None
                            }
                        ,nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for StructRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for StructRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[StructRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Struct>> for Struct {
    type Value = ::planus::Offset<Struct>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Struct>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for StructRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[StructRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `List`
/// 
/// Generated from these locations:
/// * Table `List` in the file `flatbuffers/vortex-dtype/dtype.fbs:49`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct List{
    
        /// The field `element_type` in the table `List`
        pub element_type: ::core::option::Option<::planus::alloc::boxed::Box<self::DType>>,
        /// The field `nullable` in the table `List`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for List {
    fn default() -> Self {
        Self {
        element_type: ::core::default::Default::default(),
        nullable: false,
        
        }
    }
}


impl List {
    /// Creates a [ListBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> ListBuilder<()> {
        ListBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_element_type: impl ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_element_type = field_element_type.prepare(builder);let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<8> = ::core::default::Default::default();
        if prepared_element_type.is_some() {table_writer.write_entry::<::planus::Offset<self::DType>>(0);}if prepared_nullable.is_some() {table_writer.write_entry::<bool>(1);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_element_type) = prepared_element_type {object_writer.write::<_, _, 4>(&prepared_element_type);}if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<List>> for List {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<List> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<List>> for List {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<List>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<List> for List {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<List> {
        List::create(
            builder,
        
            
            &self.element_type,
            
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [List] type.
///
/// Can be created using the [List::builder] method.
#[derive(Debug)]
#[must_use]
pub struct ListBuilder<State>(State);

impl<
    
> ListBuilder
<
    (
    
    )
>
{
    /// Setter for the [`element_type` field](List#structfield.element_type).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn element_type<T0>(self, value: T0) -> ListBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>
    {
        ListBuilder((
            
            value,
        ))
    }

    

    
    /// Sets the [`element_type` field](List#structfield.element_type) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn element_type_as_null(self) -> ListBuilder<(
        
        (),
    )>
    {
        self.element_type(())
    }
    
}

impl<
    
        T0,
    
> ListBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`nullable` field](List#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T1>(self, value: T1) -> ListBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsDefault<bool, bool>
    {
        
        let (
            
                v0,
            
        ) = self.0;ListBuilder((
            
                v0,
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](List#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> ListBuilder<(
        
            T0,
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
        T1,
    
> ListBuilder<(
    
        T0,
    
        T1,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [List].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<List>
        where Self: ::planus::WriteAsOffset<List>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T1: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<List>> for ListBuilder<(
    
        T0,
    
        T1,
    
)> {
    type Prepared = ::planus::Offset<List>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<List> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T1: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<List>> for ListBuilder<(
    
        T0,
    
        T1,
    
)> {
    type Prepared = ::planus::Offset<List>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<List>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T1: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<List> for ListBuilder<(
    
        T0,
    
        T1,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<List> {
        
        let (
            
                v0,
            
                v1,
            
        ) = &self.0;List::create(
            builder,
            
                v0,
            
                v1,
            
        )
    }
}

/// Reference to a deserialized [List].
#[derive(Copy, Clone)]
pub struct ListRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> ListRef<'a> {
    
        /// Getter for the [`element_type` field](List#structfield.element_type).
        #[inline]
        pub fn element_type(&self) -> ::planus::Result<::core::option::Option<self::DTypeRef<'a>>> {
            
            
              
              self.0.access(0, "List", "element_type")
              
            
            
            
        }
    
        /// Getter for the [`nullable` field](List#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(1, "List", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for ListRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("ListRef");
        if let ::core::option::Option::Some(field_element_type) = self.element_type().transpose() {
                f.field("element_type", &field_element_type);
            }f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<ListRef<'a>> for List {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: ListRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            element_type: 
                                if let ::core::option::Option::Some(element_type) = value.element_type()? {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(element_type)?))
                                } else {
                                    ::core::option::Option::None
                                }
                            ,nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for ListRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for ListRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[ListRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<List>> for List {
    type Value = ::planus::Offset<List>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<List>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for ListRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[ListRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `FixedSizeList`
/// 
/// Generated from these locations:
/// * Table `FixedSizeList` in the file `flatbuffers/vortex-dtype/dtype.fbs:54`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct FixedSizeList{
    
        /// The field `element_type` in the table `FixedSizeList`
        pub element_type: ::core::option::Option<::planus::alloc::boxed::Box<self::DType>>,
        /// The field `size` in the table `FixedSizeList`
        pub size: u32,
        /// The field `nullable` in the table `FixedSizeList`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for FixedSizeList {
    fn default() -> Self {
        Self {
        element_type: ::core::default::Default::default(),
        size: 0,
        nullable: false,
        
        }
    }
}


impl FixedSizeList {
    /// Creates a [FixedSizeListBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> FixedSizeListBuilder<()> {
        FixedSizeListBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_element_type: impl ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        field_size: impl ::planus::WriteAsDefault<u32, u32>,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_element_type = field_element_type.prepare(builder);let prepared_size = field_size.prepare(builder, &0);let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<10> = ::core::default::Default::default();
        if prepared_element_type.is_some() {table_writer.write_entry::<::planus::Offset<self::DType>>(0);}if prepared_size.is_some() {table_writer.write_entry::<u32>(1);}if prepared_nullable.is_some() {table_writer.write_entry::<bool>(2);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_element_type) = prepared_element_type {object_writer.write::<_, _, 4>(&prepared_element_type);}if let ::core::option::Option::Some(prepared_size) = prepared_size {object_writer.write::<_, _, 4>(&prepared_size);}if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<FixedSizeList>> for FixedSizeList {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FixedSizeList> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<FixedSizeList>> for FixedSizeList {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<FixedSizeList>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<FixedSizeList> for FixedSizeList {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FixedSizeList> {
        FixedSizeList::create(
            builder,
        
            
            &self.element_type,
            
        
            
            self.size,
            
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [FixedSizeList] type.
///
/// Can be created using the [FixedSizeList::builder] method.
#[derive(Debug)]
#[must_use]
pub struct FixedSizeListBuilder<State>(State);

impl<
    
> FixedSizeListBuilder
<
    (
    
    )
>
{
    /// Setter for the [`element_type` field](FixedSizeList#structfield.element_type).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn element_type<T0>(self, value: T0) -> FixedSizeListBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>
    {
        FixedSizeListBuilder((
            
            value,
        ))
    }

    

    
    /// Sets the [`element_type` field](FixedSizeList#structfield.element_type) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn element_type_as_null(self) -> FixedSizeListBuilder<(
        
        (),
    )>
    {
        self.element_type(())
    }
    
}

impl<
    
        T0,
    
> FixedSizeListBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`size` field](FixedSizeList#structfield.size).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn size<T1>(self, value: T1) -> FixedSizeListBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsDefault<u32, u32>
    {
        
        let (
            
                v0,
            
        ) = self.0;FixedSizeListBuilder((
            
                v0,
            
            value,
        ))
    }

    
    /// Sets the [`size` field](FixedSizeList#structfield.size) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn size_as_default(self) -> FixedSizeListBuilder<(
        
            T0,
        
        ::planus::DefaultValue,
    )>
    {
        self.size(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
        T1,
    
> FixedSizeListBuilder
<
    (
    
        T0,
    
        T1,
    
    )
>
{
    /// Setter for the [`nullable` field](FixedSizeList#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T2>(self, value: T2) -> FixedSizeListBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
    )>
    where T2: ::planus::WriteAsDefault<bool, bool>
    {
        
        let (
            
                v0,
            
                v1,
            
        ) = self.0;FixedSizeListBuilder((
            
                v0,
            
                v1,
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](FixedSizeList#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> FixedSizeListBuilder<(
        
            T0,
        
            T1,
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
        T1,
    
        T2,
    
> FixedSizeListBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [FixedSizeList].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<FixedSizeList>
        where Self: ::planus::WriteAsOffset<FixedSizeList>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T1: ::planus::WriteAsDefault<u32, u32>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<FixedSizeList>> for FixedSizeListBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<FixedSizeList>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FixedSizeList> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T1: ::planus::WriteAsDefault<u32, u32>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<FixedSizeList>> for FixedSizeListBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<FixedSizeList>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<FixedSizeList>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T1: ::planus::WriteAsDefault<u32, u32>,
    
        T2: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<FixedSizeList> for FixedSizeListBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FixedSizeList> {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
        ) = &self.0;FixedSizeList::create(
            builder,
            
                v0,
            
                v1,
            
                v2,
            
        )
    }
}

/// Reference to a deserialized [FixedSizeList].
#[derive(Copy, Clone)]
pub struct FixedSizeListRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> FixedSizeListRef<'a> {
    
        /// Getter for the [`element_type` field](FixedSizeList#structfield.element_type).
        #[inline]
        pub fn element_type(&self) -> ::planus::Result<::core::option::Option<self::DTypeRef<'a>>> {
            
            
              
              self.0.access(0, "FixedSizeList", "element_type")
              
            
            
            
        }
    
        /// Getter for the [`size` field](FixedSizeList#structfield.size).
        #[inline]
        pub fn size(&self) -> ::planus::Result<u32> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(1, "FixedSizeList", "size")
              
            
            ?.unwrap_or(0))
            
        }
    
        /// Getter for the [`nullable` field](FixedSizeList#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(2, "FixedSizeList", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for FixedSizeListRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("FixedSizeListRef");
        if let ::core::option::Option::Some(field_element_type) = self.element_type().transpose() {
                f.field("element_type", &field_element_type);
            }f.field("size", &self.size());f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<FixedSizeListRef<'a>> for FixedSizeList {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: FixedSizeListRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            element_type: 
                                if let ::core::option::Option::Some(element_type) = value.element_type()? {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(element_type)?))
                                } else {
                                    ::core::option::Option::None
                                }
                            ,size: ::core::convert::TryInto::try_into(value.size()?)?,nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for FixedSizeListRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for FixedSizeListRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[FixedSizeListRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<FixedSizeList>> for FixedSizeList {
    type Value = ::planus::Offset<FixedSizeList>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<FixedSizeList>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for FixedSizeListRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[FixedSizeListRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `Extension`
/// 
/// Generated from these locations:
/// * Table `Extension` in the file `flatbuffers/vortex-dtype/dtype.fbs:60`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Extension{
    
        /// The field `id` in the table `Extension`
        pub id: ::core::option::Option<::planus::alloc::string::String>,
        /// The field `storage_dtype` in the table `Extension`
        pub storage_dtype: ::core::option::Option<::planus::alloc::boxed::Box<self::DType>>,
        /// The field `metadata` in the table `Extension`
        pub metadata: ::core::option::Option<::planus::alloc::vec::Vec<u8>>,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Extension {
    fn default() -> Self {
        Self {
        id: ::core::default::Default::default(),
        storage_dtype: ::core::default::Default::default(),
        metadata: ::core::default::Default::default(),
        
        }
    }
}


impl Extension {
    /// Creates a [ExtensionBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> ExtensionBuilder<()> {
        ExtensionBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_id: impl ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
        field_storage_dtype: impl ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        field_metadata: impl ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        
    ) -> ::planus::Offset<Self> {let prepared_id = field_id.prepare(builder);let prepared_storage_dtype = field_storage_dtype.prepare(builder);let prepared_metadata = field_metadata.prepare(builder);

        let mut table_writer: ::planus::table_writer::TableWriter::<10> = ::core::default::Default::default();
        if prepared_id.is_some() {table_writer.write_entry::<::planus::Offset<str>>(0);}if prepared_storage_dtype.is_some() {table_writer.write_entry::<::planus::Offset<self::DType>>(1);}if prepared_metadata.is_some() {table_writer.write_entry::<::planus::Offset<[u8]>>(2);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_id) = prepared_id {object_writer.write::<_, _, 4>(&prepared_id);}if let ::core::option::Option::Some(prepared_storage_dtype) = prepared_storage_dtype {object_writer.write::<_, _, 4>(&prepared_storage_dtype);}if let ::core::option::Option::Some(prepared_metadata) = prepared_metadata {object_writer.write::<_, _, 4>(&prepared_metadata);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Extension>> for Extension {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Extension> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Extension>> for Extension {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Extension>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Extension> for Extension {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Extension> {
        Extension::create(
            builder,
        
            
            &self.id,
            
        
            
            &self.storage_dtype,
            
        
            
            &self.metadata,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Extension] type.
///
/// Can be created using the [Extension::builder] method.
#[derive(Debug)]
#[must_use]
pub struct ExtensionBuilder<State>(State);

impl<
    
> ExtensionBuilder
<
    (
    
    )
>
{
    /// Setter for the [`id` field](Extension#structfield.id).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn id<T0>(self, value: T0) -> ExtensionBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>
    {
        ExtensionBuilder((
            
            value,
        ))
    }

    

    
    /// Sets the [`id` field](Extension#structfield.id) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn id_as_null(self) -> ExtensionBuilder<(
        
        (),
    )>
    {
        self.id(())
    }
    
}

impl<
    
        T0,
    
> ExtensionBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`storage_dtype` field](Extension#structfield.storage_dtype).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn storage_dtype<T1>(self, value: T1) -> ExtensionBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsOptional<::planus::Offset<self::DType>>
    {
        
        let (
            
                v0,
            
        ) = self.0;ExtensionBuilder((
            
                v0,
            
            value,
        ))
    }

    

    
    /// Sets the [`storage_dtype` field](Extension#structfield.storage_dtype) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn storage_dtype_as_null(self) -> ExtensionBuilder<(
        
            T0,
        
        (),
    )>
    {
        self.storage_dtype(())
    }
    
}

impl<
    
        T0,
    
        T1,
    
> ExtensionBuilder
<
    (
    
        T0,
    
        T1,
    
    )
>
{
    /// Setter for the [`metadata` field](Extension#structfield.metadata).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn metadata<T2>(self, value: T2) -> ExtensionBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
    )>
    where T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>
    {
        
        let (
            
                v0,
            
                v1,
            
        ) = self.0;ExtensionBuilder((
            
                v0,
            
                v1,
            
            value,
        ))
    }

    

    
    /// Sets the [`metadata` field](Extension#structfield.metadata) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn metadata_as_null(self) -> ExtensionBuilder<(
        
            T0,
        
            T1,
        
        (),
    )>
    {
        self.metadata(())
    }
    
}



impl<
    
        T0,
    
        T1,
    
        T2,
    
> ExtensionBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Extension].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Extension>
        where Self: ::planus::WriteAsOffset<Extension>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
> ::planus::WriteAs<::planus::Offset<Extension>> for ExtensionBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<Extension>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Extension> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
> ::planus::WriteAsOptional<::planus::Offset<Extension>> for ExtensionBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    type Prepared = ::planus::Offset<Extension>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Extension>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
    
        T1: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
> ::planus::WriteAsOffset<Extension> for ExtensionBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Extension> {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
        ) = &self.0;Extension::create(
            builder,
            
                v0,
            
                v1,
            
                v2,
            
        )
    }
}

/// Reference to a deserialized [Extension].
#[derive(Copy, Clone)]
pub struct ExtensionRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> ExtensionRef<'a> {
    
        /// Getter for the [`id` field](Extension#structfield.id).
        #[inline]
        pub fn id(&self) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>> {
            
            
              
              self.0.access(0, "Extension", "id")
              
            
            
            
        }
    
        /// Getter for the [`storage_dtype` field](Extension#structfield.storage_dtype).
        #[inline]
        pub fn storage_dtype(&self) -> ::planus::Result<::core::option::Option<self::DTypeRef<'a>>> {
            
            
              
              self.0.access(1, "Extension", "storage_dtype")
              
            
            
            
        }
    
        /// Getter for the [`metadata` field](Extension#structfield.metadata).
        #[inline]
        pub fn metadata(&self) -> ::planus::Result<::core::option::Option<&'a [u8]>> {
            
            
              
              self.0.access(2, "Extension", "metadata")
              
            
            
            
        }
    
}

impl<'a> ::core::fmt::Debug for ExtensionRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("ExtensionRef");
        if let ::core::option::Option::Some(field_id) = self.id().transpose() {
                f.field("id", &field_id);
            }if let ::core::option::Option::Some(field_storage_dtype) = self.storage_dtype().transpose() {
                f.field("storage_dtype", &field_storage_dtype);
            }if let ::core::option::Option::Some(field_metadata) = self.metadata().transpose() {
                f.field("metadata", &field_metadata);
            }
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<ExtensionRef<'a>> for Extension {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: ExtensionRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            id: 
                                value.id()?.map(::core::convert::Into::into)
                            ,storage_dtype: 
                                if let ::core::option::Option::Some(storage_dtype) = value.storage_dtype()? {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(::core::convert::TryInto::try_into(storage_dtype)?))
                                } else {
                                    ::core::option::Option::None
                                }
                            ,metadata: value.metadata()?.map(|v| v.to_vec()),
        })
    }
}

impl<'a> ::planus::TableRead<'a> for ExtensionRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for ExtensionRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[ExtensionRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Extension>> for Extension {
    type Value = ::planus::Offset<Extension>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Extension>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for ExtensionRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[ExtensionRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The table `Variant`
/// 
/// Generated from these locations:
/// * Table `Variant` in the file `flatbuffers/vortex-dtype/dtype.fbs:66`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Variant{
    
        /// The field `nullable` in the table `Variant`
        pub nullable: bool,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for Variant {
    fn default() -> Self {
        Self {
        nullable: false,
        
        }
    }
}


impl Variant {
    /// Creates a [VariantBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> VariantBuilder<()> {
        VariantBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        
    ) -> ::planus::Offset<Self> {let prepared_nullable = field_nullable.prepare(builder, &false);

        let mut table_writer: ::planus::table_writer::TableWriter::<6> = ::core::default::Default::default();
        if prepared_nullable.is_some() {table_writer.write_entry::<bool>(0);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {object_writer.write::<_, _, 1>(&prepared_nullable);}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<Variant>> for Variant {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Variant> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<Variant>> for Variant {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Variant>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<Variant> for Variant {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Variant> {
        Variant::create(
            builder,
        
            
            self.nullable,
            
        
        )
    }
}

/// Builder for serializing an instance of the [Variant] type.
///
/// Can be created using the [Variant::builder] method.
#[derive(Debug)]
#[must_use]
pub struct VariantBuilder<State>(State);

impl<
    
> VariantBuilder
<
    (
    
    )
>
{
    /// Setter for the [`nullable` field](Variant#structfield.nullable).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable<T0>(self, value: T0) -> VariantBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<bool, bool>
    {
        VariantBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`nullable` field](Variant#structfield.nullable) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn nullable_as_default(self) -> VariantBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.nullable(::planus::DefaultValue)
    }
    

    
}



impl<
    
        T0,
    
> VariantBuilder<(
    
        T0,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Variant].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Variant>
        where Self: ::planus::WriteAsOffset<Variant>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAs<::planus::Offset<Variant>> for VariantBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<Variant>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Variant> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOptional<::planus::Offset<Variant>> for VariantBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<Variant>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Variant>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<bool, bool>,
    
> ::planus::WriteAsOffset<Variant> for VariantBuilder<(
    
        T0,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Variant> {
        
        let (
            
                v0,
            
        ) = &self.0;Variant::create(
            builder,
            
                v0,
            
        )
    }
}

/// Reference to a deserialized [Variant].
#[derive(Copy, Clone)]
pub struct VariantRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> VariantRef<'a> {
    
        /// Getter for the [`nullable` field](Variant#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "Variant", "nullable")
              
            
            ?.unwrap_or(false))
            
        }
    
}

impl<'a> ::core::fmt::Debug for VariantRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("VariantRef");
        f.field("nullable", &self.nullable());
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<VariantRef<'a>> for Variant {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: VariantRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for VariantRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for VariantRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[VariantRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<Variant>> for Variant {
    type Value = ::planus::Offset<Variant>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<Variant>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for VariantRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[VariantRef]",
            "read_as_root",
            0,
        ))
    }
}

    
/// The union `Type`
/// 
/// Generated from these locations:
/// * Union `Type` in the file `flatbuffers/vortex-dtype/dtype.fbs:70`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize)]
pub enum Type{
    
        /// The variant of type `Null` in the union `Type`
        Null(::planus::alloc::boxed::Box<self::Null>),
    
        /// The variant of type `Bool` in the union `Type`
        Bool(::planus::alloc::boxed::Box<self::Bool>),
    
        /// The variant of type `Primitive` in the union `Type`
        Primitive(::planus::alloc::boxed::Box<self::Primitive>),
    
        /// The variant of type `Decimal` in the union `Type`
        Decimal(::planus::alloc::boxed::Box<self::Decimal>),
    
        /// The variant of type `Utf8` in the union `Type`
        Utf8(::planus::alloc::boxed::Box<self::Utf8>),
    
        /// The variant of type `Binary` in the union `Type`
        Binary(::planus::alloc::boxed::Box<self::Binary>),
    
        /// The variant of type `Struct_` in the union `Type`
        Struct(::planus::alloc::boxed::Box<self::Struct>),
    
        /// The variant of type `List` in the union `Type`
        List(::planus::alloc::boxed::Box<self::List>),
    
        /// The variant of type `Extension` in the union `Type`
        Extension(::planus::alloc::boxed::Box<self::Extension>),
    
        /// The variant of type `FixedSizeList` in the union `Type`
        FixedSizeList(::planus::alloc::boxed::Box<self::FixedSizeList>),
    
        /// The variant of type `Variant` in the union `Type`
        Variant(::planus::alloc::boxed::Box<self::Variant>),
    
}


impl Type {
    /// Creates a [TypeBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> TypeBuilder<::planus::Uninitialized> {
        TypeBuilder(::planus::Uninitialized)
    }

    #[inline]
    pub fn create_null(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Null>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(1, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_bool(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Bool>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(2, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_primitive(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Primitive>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(3, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_decimal(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Decimal>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(4, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_utf8(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Utf8>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(5, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_binary(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Binary>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(6, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_struct(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Struct>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(7, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_list(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::List>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(8, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_extension(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Extension>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(9, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_fixed_size_list(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::FixedSizeList>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(10, value.prepare(builder).downcast())
    }

    #[inline]
    pub fn create_variant(
      builder: &mut ::planus::Builder,
      value: impl ::planus::WriteAsOffset<self::Variant>,
    ) -> ::planus::UnionOffset<Self> {
        ::planus::UnionOffset::new(11, value.prepare(builder).downcast())
    }

    
}



impl ::planus::WriteAsUnion<Type> for Type {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Self> {
        match self {
            Self::Null(value) => Self::create_null(builder, value),
            Self::Bool(value) => Self::create_bool(builder, value),
            Self::Primitive(value) => Self::create_primitive(builder, value),
            Self::Decimal(value) => Self::create_decimal(builder, value),
            Self::Utf8(value) => Self::create_utf8(builder, value),
            Self::Binary(value) => Self::create_binary(builder, value),
            Self::Struct(value) => Self::create_struct(builder, value),
            Self::List(value) => Self::create_list(builder, value),
            Self::Extension(value) => Self::create_extension(builder, value),
            Self::FixedSizeList(value) => Self::create_fixed_size_list(builder, value),
            Self::Variant(value) => Self::create_variant(builder, value),
            
        }
    }
}


impl ::planus::WriteAsOptionalUnion<Type> for Type {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Self>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}

/// Builder for serializing an instance of the [Type] type.
///
/// Can be created using the [Type::builder] method.
#[derive(Debug)]
#[must_use]
pub struct TypeBuilder<T>(T);

impl TypeBuilder<::planus::Uninitialized> {
    /// Creates an instance of the [`Null` variant](Type#variant.Null).
    #[inline]
    pub fn null<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<1, T>>
        where T: ::planus::WriteAsOffset<self::Null> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`Bool` variant](Type#variant.Bool).
    #[inline]
    pub fn bool<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<2, T>>
        where T: ::planus::WriteAsOffset<self::Bool> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`Primitive` variant](Type#variant.Primitive).
    #[inline]
    pub fn primitive<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<3, T>>
        where T: ::planus::WriteAsOffset<self::Primitive> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`Decimal` variant](Type#variant.Decimal).
    #[inline]
    pub fn decimal<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<4, T>>
        where T: ::planus::WriteAsOffset<self::Decimal> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`Utf8` variant](Type#variant.Utf8).
    #[inline]
    pub fn utf8<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<5, T>>
        where T: ::planus::WriteAsOffset<self::Utf8> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`Binary` variant](Type#variant.Binary).
    #[inline]
    pub fn binary<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<6, T>>
        where T: ::planus::WriteAsOffset<self::Binary> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`Struct_` variant](Type#variant.Struct).
    #[inline]
    pub fn struct_<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<7, T>>
        where T: ::planus::WriteAsOffset<self::Struct> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`List` variant](Type#variant.List).
    #[inline]
    pub fn list<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<8, T>>
        where T: ::planus::WriteAsOffset<self::List> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`Extension` variant](Type#variant.Extension).
    #[inline]
    pub fn extension<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<9, T>>
        where T: ::planus::WriteAsOffset<self::Extension> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`FixedSizeList` variant](Type#variant.FixedSizeList).
    #[inline]
    pub fn fixed_size_list<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<10, T>>
        where T: ::planus::WriteAsOffset<self::FixedSizeList> {
        TypeBuilder(::planus::Initialized(value))
    }

    /// Creates an instance of the [`Variant` variant](Type#variant.Variant).
    #[inline]
    pub fn variant<T>(
        self,
        value: T,
    ) -> TypeBuilder<::planus::Initialized<11, T>>
        where T: ::planus::WriteAsOffset<self::Variant> {
        TypeBuilder(::planus::Initialized(value))
    }

    
}

impl<const N: u8, T> TypeBuilder<::planus::Initialized<N, T>> {
    /// Finish writing the builder to get an [UnionOffset](::planus::UnionOffset) to a serialized [Type].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type>
        where Self: ::planus::WriteAsUnion<Type>
    {
        ::planus::WriteAsUnion::prepare(&self, builder)
    }
}

impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<1, T>>
    where T: ::planus::WriteAsOffset<self::Null>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(1, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<1, T>>
    where T: ::planus::WriteAsOffset<self::Null>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<2, T>>
    where T: ::planus::WriteAsOffset<self::Bool>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(2, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<2, T>>
    where T: ::planus::WriteAsOffset<self::Bool>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<3, T>>
    where T: ::planus::WriteAsOffset<self::Primitive>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(3, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<3, T>>
    where T: ::planus::WriteAsOffset<self::Primitive>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<4, T>>
    where T: ::planus::WriteAsOffset<self::Decimal>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(4, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<4, T>>
    where T: ::planus::WriteAsOffset<self::Decimal>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<5, T>>
    where T: ::planus::WriteAsOffset<self::Utf8>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(5, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<5, T>>
    where T: ::planus::WriteAsOffset<self::Utf8>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<6, T>>
    where T: ::planus::WriteAsOffset<self::Binary>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(6, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<6, T>>
    where T: ::planus::WriteAsOffset<self::Binary>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<7, T>>
    where T: ::planus::WriteAsOffset<self::Struct>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(7, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<7, T>>
    where T: ::planus::WriteAsOffset<self::Struct>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<8, T>>
    where T: ::planus::WriteAsOffset<self::List>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(8, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<8, T>>
    where T: ::planus::WriteAsOffset<self::List>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<9, T>>
    where T: ::planus::WriteAsOffset<self::Extension>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(9, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<9, T>>
    where T: ::planus::WriteAsOffset<self::Extension>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<10, T>>
    where T: ::planus::WriteAsOffset<self::FixedSizeList>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(10, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<10, T>>
    where T: ::planus::WriteAsOffset<self::FixedSizeList>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}
impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<11, T>>
    where T: ::planus::WriteAsOffset<self::Variant>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
        ::planus::UnionOffset::new(11, (self.0).0.prepare(builder).downcast())
    }
}

impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<11, T>>
    where T: ::planus::WriteAsOffset<self::Variant>
{
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::UnionOffset<Type>> {
        ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
    }
}


/// Reference to a deserialized [Type].
#[derive(Copy, Clone, Debug)]
pub enum TypeRef<'a>{
    Null(self::NullRef<'a>),
    Bool(self::BoolRef<'a>),
    Primitive(self::PrimitiveRef<'a>),
    Decimal(self::DecimalRef<'a>),
    Utf8(self::Utf8Ref<'a>),
    Binary(self::BinaryRef<'a>),
    Struct(self::StructRef<'a>),
    List(self::ListRef<'a>),
    Extension(self::ExtensionRef<'a>),
    FixedSizeList(self::FixedSizeListRef<'a>),
    Variant(self::VariantRef<'a>),
    
}


impl<'a> ::core::convert::TryFrom<TypeRef<'a>> for Type {
    type Error = ::planus::Error;

    fn try_from(value: TypeRef<'a>) -> ::planus::Result<Self> {
        ::core::result::Result::Ok(match value {
            
                TypeRef::Null(value) => Self::Null(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::Bool(value) => Self::Bool(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::Primitive(value) => Self::Primitive(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::Decimal(value) => Self::Decimal(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::Utf8(value) => Self::Utf8(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::Binary(value) => Self::Binary(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::Struct(value) => Self::Struct(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::List(value) => Self::List(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::Extension(value) => Self::Extension(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::FixedSizeList(value) => Self::FixedSizeList(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
                TypeRef::Variant(value) => Self::Variant(::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?)),
                
            
        })
    }
}



impl<'a> ::planus::TableReadUnion<'a> for TypeRef<'a> {
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, tag: u8, field_offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        match tag {
            1 => ::core::result::Result::Ok(Self::Null(::planus::TableRead::from_buffer(buffer, field_offset)?)),2 => ::core::result::Result::Ok(Self::Bool(::planus::TableRead::from_buffer(buffer, field_offset)?)),3 => ::core::result::Result::Ok(Self::Primitive(::planus::TableRead::from_buffer(buffer, field_offset)?)),4 => ::core::result::Result::Ok(Self::Decimal(::planus::TableRead::from_buffer(buffer, field_offset)?)),5 => ::core::result::Result::Ok(Self::Utf8(::planus::TableRead::from_buffer(buffer, field_offset)?)),6 => ::core::result::Result::Ok(Self::Binary(::planus::TableRead::from_buffer(buffer, field_offset)?)),7 => ::core::result::Result::Ok(Self::Struct(::planus::TableRead::from_buffer(buffer, field_offset)?)),8 => ::core::result::Result::Ok(Self::List(::planus::TableRead::from_buffer(buffer, field_offset)?)),9 => ::core::result::Result::Ok(Self::Extension(::planus::TableRead::from_buffer(buffer, field_offset)?)),10 => ::core::result::Result::Ok(Self::FixedSizeList(::planus::TableRead::from_buffer(buffer, field_offset)?)),11 => ::core::result::Result::Ok(Self::Variant(::planus::TableRead::from_buffer(buffer, field_offset)?)),_ => ::core::result::Result::Err(::planus::errors::ErrorKind::UnknownUnionTag { tag }),
        }
    }
}



impl<'a> ::planus::VectorReadUnion<'a> for TypeRef<'a> {
    const VECTOR_NAME: &'static str = "[TypeRef]";
}


    
/// The table `DType`
/// 
/// Generated from these locations:
/// * Table `DType` in the file `flatbuffers/vortex-dtype/dtype.fbs:84`
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct DType{
    
        /// The field `type` in the table `DType`
        pub type_: ::core::option::Option<self::Type>,}


#[allow(clippy::derivable_impls)]
impl ::core::default::Default for DType {
    fn default() -> Self {
        Self {
        type_: ::core::default::Default::default(),
        
        }
    }
}


impl DType {
    /// Creates a [DTypeBuilder] for serializing an instance of this table.
    #[inline]
    pub fn builder() -> DTypeBuilder<()> {
        DTypeBuilder(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create(
        builder: &mut ::planus::Builder,
        field_type_: impl ::planus::WriteAsOptionalUnion<self::Type>,
        
    ) -> ::planus::Offset<Self> {let prepared_type_ = field_type_.prepare(builder);

        let mut table_writer: ::planus::table_writer::TableWriter::<8> = ::core::default::Default::default();
        if prepared_type_.is_some() {table_writer.write_entry::<::planus::Offset<self::Type>>(1);}if prepared_type_.is_some() {table_writer.write_entry::<u8>(0);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_type_) = prepared_type_ {object_writer.write::<_, _, 4>(&prepared_type_.offset());}if let ::core::option::Option::Some(prepared_type_) = prepared_type_ {object_writer.write::<_, _, 1>(&prepared_type_.tag());}
                
            });
        }builder.current_offset()
    }
}

impl ::planus::WriteAs<::planus::Offset<DType>> for DType {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl ::planus::WriteAsOptional<::planus::Offset<DType>> for DType {
    type Prepared = ::planus::Offset<Self>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<DType>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl ::planus::WriteAsOffset<DType> for DType {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType> {
        DType::create(
            builder,
        
            
            &self.type_,
            
        
        )
    }
}

/// Builder for serializing an instance of the [DType] type.
///
/// Can be created using the [DType::builder] method.
#[derive(Debug)]
#[must_use]
pub struct DTypeBuilder<State>(State);

impl<
    
> DTypeBuilder
<
    (
    
    )
>
{
    /// Setter for the [`type` field](DType#structfield.type_).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn type_<T0>(self, value: T0) -> DTypeBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsOptionalUnion<self::Type>
    {
        DTypeBuilder((
            
            value,
        ))
    }

    

    
    /// Sets the [`type` field](DType#structfield.type_) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn type_as_null(self) -> DTypeBuilder<(
        
        (),
    )>
    {
        self.type_(())
    }
    
}



impl<
    
        T0,
    
> DTypeBuilder<(
    
        T0,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [DType].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType>
        where Self: ::planus::WriteAsOffset<DType>
    {
        ::planus::WriteAsOffset::prepare(&self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptionalUnion<self::Type>,
    
> ::planus::WriteAs<::planus::Offset<DType>> for DTypeBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<DType>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType> {
        ::planus::WriteAsOffset::prepare(self, builder)
    }
}

impl<
    
        T0: ::planus::WriteAsOptionalUnion<self::Type>,
    
> ::planus::WriteAsOptional<::planus::Offset<DType>> for DTypeBuilder<(
    
        T0,
    
)> {
    type Prepared = ::planus::Offset<DType>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<DType>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsOptionalUnion<self::Type>,
    
> ::planus::WriteAsOffset<DType> for DTypeBuilder<(
    
        T0,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType> {
        
        let (
            
                v0,
            
        ) = &self.0;DType::create(
            builder,
            
                v0,
            
        )
    }
}

/// Reference to a deserialized [DType].
#[derive(Copy, Clone)]
pub struct DTypeRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> DTypeRef<'a> {
    
        /// Getter for the [`type` field](DType#structfield.type_).
        #[inline]
        pub fn type_(&self) -> ::planus::Result<::core::option::Option<self::TypeRef<'a>>> {
            
            
              
              self.0.access_union(0, "DType", "type_")
              
            
            
            
        }
    
}

impl<'a> ::core::fmt::Debug for DTypeRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("DTypeRef");
        if let ::core::option::Option::Some(field_type_) = self.type_().transpose() {
                f.field("type_", &field_type_);
            }
        f.finish()
    }
}

impl<'a> ::core::convert::TryFrom<DTypeRef<'a>> for DType {
    type Error = ::planus::Error;


    #[allow(unreachable_code)]
    fn try_from(value: DTypeRef<'a>) -> ::planus::Result<Self> {

        ::core::result::Result::Ok(Self {
            type_: 
                    if let ::core::option::Option::Some(type_) = value.type_()? {
                        ::core::option::Option::Some(::core::convert::TryInto::try_into(type_)?)
                    } else {
                        ::core::option::Option::None
                    }
                ,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for DTypeRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for DTypeRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[DTypeRef]",
            "get",
            buffer.offset_from_start,
        ))
    }
}

/// # Safety
/// The planus compiler generates implementations that initialize
/// the bytes in `write_values`.
unsafe impl ::planus::VectorWrite<::planus::Offset<DType>> for DType {
    type Value = ::planus::Offset<DType>;
    const STRIDE: usize = 4;
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
        ::planus::WriteAs::prepare(self, builder)
    }

    #[inline]
    unsafe fn write_values(
        values: &[::planus::Offset<DType>],
        bytes: *mut ::core::mem::MaybeUninit<u8>,
        buffer_position: u32,
    ) {
        let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
        for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
            ::planus::WriteAsPrimitive::write(
                v,
                ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                buffer_position - (Self::STRIDE * i) as u32,
            );
        }
    }
}

impl<'a> ::planus::ReadAsRoot<'a> for DTypeRef<'a> {
    fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[DTypeRef]",
            "read_as_root",
            0,
        ))
    }
}

    }
pub use root::*;

const _: () = ::planus::check_version_compatibility("planus-1.2.0");

/// The root namespace
///
/// Generated from these locations:
/// * File `flatbuffers/vortex-array/array.fbs`
#[no_implicit_prelude]
#[allow(dead_code, clippy::needless_lifetimes)]
mod root {
    ///  An Array describes the hierarchy of an array as well as the locations of the data buffers that appear
    ///  immediately after the message in the byte stream.
    ///
    /// Generated from these locations:
    /// * Table `Array` in the file `flatbuffers/vortex-array/array.fbs:6`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Array {
        ///  The array's hierarchical definition.
        pub root: ::core::option::Option<::planus::alloc::boxed::Box<self::ArrayNode>>,
        ///  The locations of the data buffers of the array
        pub buffers: ::core::option::Option<::planus::alloc::vec::Vec<self::Buffer>>,
    }

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
        ) -> ::planus::Offset<Self> {
            let prepared_root = field_root.prepare(builder);
            let prepared_buffers = field_buffers.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                ::core::default::Default::default();
            if prepared_root.is_some() {
                table_writer.write_entry::<::planus::Offset<self::ArrayNode>>(0);
            }
            if prepared_buffers.is_some() {
                table_writer.write_entry::<::planus::Offset<[self::Buffer]>>(1);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_root) = prepared_root {
                        object_writer.write::<_, _, 4>(&prepared_root);
                    }
                    if let ::core::option::Option::Some(prepared_buffers) = prepared_buffers {
                        object_writer.write::<_, _, 4>(&prepared_buffers);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Array>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Array> for Array {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array> {
            Array::create(builder, &self.root, &self.buffers)
        }
    }

    /// Builder for serializing an instance of the [Array] type.
    ///
    /// Can be created using the [Array::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct ArrayBuilder<State>(State);

    impl ArrayBuilder<()> {
        /// Setter for the [`root` field](Array#structfield.root).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn root<T0>(self, value: T0) -> ArrayBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>,
        {
            ArrayBuilder((value,))
        }

        /// Sets the [`root` field](Array#structfield.root) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn root_as_null(self) -> ArrayBuilder<((),)> {
            self.root(())
        }
    }

    impl<T0> ArrayBuilder<(T0,)> {
        /// Setter for the [`buffers` field](Array#structfield.buffers).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn buffers<T1>(self, value: T1) -> ArrayBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
        {
            let (v0,) = self.0;
            ArrayBuilder((v0, value))
        }

        /// Sets the [`buffers` field](Array#structfield.buffers) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn buffers_as_null(self) -> ArrayBuilder<(T0, ())> {
            self.buffers(())
        }
    }

    impl<T0, T1> ArrayBuilder<(T0, T1)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Array].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array>
        where
            Self: ::planus::WriteAsOffset<Array>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
    > ::planus::WriteAs<::planus::Offset<Array>> for ArrayBuilder<(T0, T1)>
    {
        type Prepared = ::planus::Offset<Array>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
    > ::planus::WriteAsOptional<::planus::Offset<Array>> for ArrayBuilder<(T0, T1)>
    {
        type Prepared = ::planus::Offset<Array>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Array>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::ArrayNode>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
    > ::planus::WriteAsOffset<Array> for ArrayBuilder<(T0, T1)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Array> {
            let (v0, v1) = &self.0;
            Array::create(builder, v0, v1)
        }
    }

    /// Reference to a deserialized [Array].
    #[derive(Copy, Clone)]
    pub struct ArrayRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> ArrayRef<'a> {
        /// Getter for the [`root` field](Array#structfield.root).
        #[inline]
        pub fn root(&self) -> ::planus::Result<::core::option::Option<self::ArrayNodeRef<'a>>> {
            self.0.access(0, "Array", "root")
        }

        /// Getter for the [`buffers` field](Array#structfield.buffers).
        #[inline]
        pub fn buffers(
            &self,
        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, self::BufferRef<'a>>>>
        {
            self.0.access(1, "Array", "buffers")
        }
    }

    impl<'a> ::core::fmt::Debug for ArrayRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("ArrayRef");
            if let ::core::option::Option::Some(field_root) = self.root().transpose() {
                f.field("root", &field_root);
            }
            if let ::core::option::Option::Some(field_buffers) = self.buffers().transpose() {
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
                root: if let ::core::option::Option::Some(root) = value.root()? {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(root)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
                buffers: if let ::core::option::Option::Some(buffers) = value.buffers()? {
                    ::core::option::Option::Some(buffers.to_vec()?)
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for ArrayRef<'a> {
        #[inline]
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                buffer, offset,
            )?))
        }
    }

    impl<'a> ::planus::VectorReadInner<'a> for ArrayRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[ArrayRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[ArrayRef]", "read_as_root", 0))
        }
    }

    ///  The compression mechanism used to compress the buffer.
    ///
    /// Generated from these locations:
    /// * Enum `Compression` in the file `flatbuffers/vortex-array/array.fbs:14`
    #[derive(
        Copy,
        Clone,
        Debug,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        ::serde::Serialize,
        ::serde::Deserialize,
    )]
    #[repr(u8)]
    pub enum Compression {
        /// The variant `None` in the enum `Compression`
        None = 0,

        /// The variant `LZ4` in the enum `Compression`
        Lz4 = 1,
    }

    impl Compression {
        /// Array containing all valid variants of Compression
        pub const ENUM_VALUES: [Self; 2] = [Self::None, Self::Lz4];
    }

    impl ::core::convert::TryFrom<u8> for Compression {
        type Error = ::planus::errors::UnknownEnumTagKind;
        #[inline]
        fn try_from(
            value: u8,
        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind> {
            #[allow(clippy::match_single_binding)]
            match value {
                0 => ::core::result::Result::Ok(Compression::None),
                1 => ::core::result::Result::Ok(Compression::Lz4),

                _ => ::core::result::Result::Err(::planus::errors::UnknownEnumTagKind {
                    tag: value as i128,
                }),
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
        fn prepare(
            &self,
            _builder: &mut ::planus::Builder,
            default: &Compression,
        ) -> ::core::option::Option<Compression> {
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
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'buf>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
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
        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag> {
            let value = unsafe { *buffer.buffer.get_unchecked(offset) };
            let value: ::core::result::Result<Self, _> = ::core::convert::TryInto::try_into(value);
            value.map_err(|error_kind| {
                error_kind.with_error_location(
                    "Compression",
                    "VectorRead::from_buffer",
                    buffer.offset_from_start,
                )
            })
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
    #[derive(
        Copy,
        Clone,
        Debug,
        PartialEq,
        PartialOrd,
        Eq,
        Ord,
        Hash,
        ::serde::Serialize,
        ::serde::Deserialize,
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
            let (cur, cursor) = cursor.split::<2, 6>();
            self.padding.write(cur, buffer_position - 0);
            let (cur, cursor) = cursor.split::<1, 5>();
            self.alignment_exponent.write(cur, buffer_position - 2);
            let (cur, cursor) = cursor.split::<1, 4>();
            self.compression.write(cur, buffer_position - 3);
            let (cur, cursor) = cursor.split::<4, 0>();
            self.length.write(cur, buffer_position - 4);
            cursor.finish([]);
        }
    }

    impl ::planus::WriteAsOffset<Buffer> for Buffer {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Buffer> {
            unsafe {
                builder.write_with(8, 3, |buffer_position, bytes| {
                    let bytes = bytes.as_mut_ptr();

                    ::planus::WriteAsPrimitive::write(
                        self,
                        ::planus::Cursor::new(
                            &mut *(bytes as *mut [::core::mem::MaybeUninit<u8>; 8]),
                        ),
                        buffer_position,
                    );
                });
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
        pub fn compression(
            &self,
        ) -> ::core::result::Result<self::Compression, ::planus::errors::UnknownEnumTag> {
            let buffer = self.0.advance_as_array::<1>(3).unwrap();

            let value: ::core::result::Result<self::Compression, _> =
                ::core::convert::TryInto::try_into(u8::from_le_bytes(*buffer.as_array()));
            value.map_err(|e| {
                e.with_error_location("BufferRef", "compression", buffer.offset_from_start)
            })
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
            f.field("padding", &self.padding());
            f.field("alignment_exponent", &self.alignment_exponent());
            f.field("compression", &self.compression());
            f.field("length", &self.length());
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
                padding: value.padding(),
                alignment_exponent: value.alignment_exponent(),
                compression: ::core::convert::TryInto::try_into(value.compression()?)?,
                length: value.length(),
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for BufferRef<'a> {
        #[inline]
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
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
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct ArrayNode {
        /// The field `encoding` in the table `ArrayNode`
        pub encoding: u16,
        /// The field `metadata` in the table `ArrayNode`
        pub metadata: ::core::option::Option<::planus::alloc::vec::Vec<u8>>,
        /// The field `children` in the table `ArrayNode`
        pub children: ::core::option::Option<::planus::alloc::vec::Vec<self::ArrayNode>>,
        /// The field `buffers` in the table `ArrayNode`
        pub buffers: ::core::option::Option<::planus::alloc::vec::Vec<u16>>,
        /// The field `stats` in the table `ArrayNode`
        pub stats: ::core::option::Option<::planus::alloc::boxed::Box<self::ArrayStats>>,
    }

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
            field_children: impl ::planus::WriteAsOptional<
                ::planus::Offset<[::planus::Offset<self::ArrayNode>]>,
            >,
            field_buffers: impl ::planus::WriteAsOptional<::planus::Offset<[u16]>>,
            field_stats: impl ::planus::WriteAsOptional<::planus::Offset<self::ArrayStats>>,
        ) -> ::planus::Offset<Self> {
            let prepared_encoding = field_encoding.prepare(builder, &0);
            let prepared_metadata = field_metadata.prepare(builder);
            let prepared_children = field_children.prepare(builder);
            let prepared_buffers = field_buffers.prepare(builder);
            let prepared_stats = field_stats.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                ::core::default::Default::default();
            if prepared_metadata.is_some() {
                table_writer.write_entry::<::planus::Offset<[u8]>>(1);
            }
            if prepared_children.is_some() {
                table_writer
                    .write_entry::<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>(2);
            }
            if prepared_buffers.is_some() {
                table_writer.write_entry::<::planus::Offset<[u16]>>(3);
            }
            if prepared_stats.is_some() {
                table_writer.write_entry::<::planus::Offset<self::ArrayStats>>(4);
            }
            if prepared_encoding.is_some() {
                table_writer.write_entry::<u16>(0);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_metadata) = prepared_metadata {
                        object_writer.write::<_, _, 4>(&prepared_metadata);
                    }
                    if let ::core::option::Option::Some(prepared_children) = prepared_children {
                        object_writer.write::<_, _, 4>(&prepared_children);
                    }
                    if let ::core::option::Option::Some(prepared_buffers) = prepared_buffers {
                        object_writer.write::<_, _, 4>(&prepared_buffers);
                    }
                    if let ::core::option::Option::Some(prepared_stats) = prepared_stats {
                        object_writer.write::<_, _, 4>(&prepared_stats);
                    }
                    if let ::core::option::Option::Some(prepared_encoding) = prepared_encoding {
                        object_writer.write::<_, _, 2>(&prepared_encoding);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<ArrayNode>> {
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

    impl ArrayNodeBuilder<()> {
        /// Setter for the [`encoding` field](ArrayNode#structfield.encoding).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn encoding<T0>(self, value: T0) -> ArrayNodeBuilder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<u16, u16>,
        {
            ArrayNodeBuilder((value,))
        }

        /// Sets the [`encoding` field](ArrayNode#structfield.encoding) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn encoding_as_default(self) -> ArrayNodeBuilder<(::planus::DefaultValue,)> {
            self.encoding(::planus::DefaultValue)
        }
    }

    impl<T0> ArrayNodeBuilder<(T0,)> {
        /// Setter for the [`metadata` field](ArrayNode#structfield.metadata).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn metadata<T1>(self, value: T1) -> ArrayNodeBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        {
            let (v0,) = self.0;
            ArrayNodeBuilder((v0, value))
        }

        /// Sets the [`metadata` field](ArrayNode#structfield.metadata) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn metadata_as_null(self) -> ArrayNodeBuilder<(T0, ())> {
            self.metadata(())
        }
    }

    impl<T0, T1> ArrayNodeBuilder<(T0, T1)> {
        /// Setter for the [`children` field](ArrayNode#structfield.children).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn children<T2>(self, value: T2) -> ArrayNodeBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>,
        {
            let (v0, v1) = self.0;
            ArrayNodeBuilder((v0, v1, value))
        }

        /// Sets the [`children` field](ArrayNode#structfield.children) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn children_as_null(self) -> ArrayNodeBuilder<(T0, T1, ())> {
            self.children(())
        }
    }

    impl<T0, T1, T2> ArrayNodeBuilder<(T0, T1, T2)> {
        /// Setter for the [`buffers` field](ArrayNode#structfield.buffers).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn buffers<T3>(self, value: T3) -> ArrayNodeBuilder<(T0, T1, T2, T3)>
        where
            T3: ::planus::WriteAsOptional<::planus::Offset<[u16]>>,
        {
            let (v0, v1, v2) = self.0;
            ArrayNodeBuilder((v0, v1, v2, value))
        }

        /// Sets the [`buffers` field](ArrayNode#structfield.buffers) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn buffers_as_null(self) -> ArrayNodeBuilder<(T0, T1, T2, ())> {
            self.buffers(())
        }
    }

    impl<T0, T1, T2, T3> ArrayNodeBuilder<(T0, T1, T2, T3)> {
        /// Setter for the [`stats` field](ArrayNode#structfield.stats).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn stats<T4>(self, value: T4) -> ArrayNodeBuilder<(T0, T1, T2, T3, T4)>
        where
            T4: ::planus::WriteAsOptional<::planus::Offset<self::ArrayStats>>,
        {
            let (v0, v1, v2, v3) = self.0;
            ArrayNodeBuilder((v0, v1, v2, v3, value))
        }

        /// Sets the [`stats` field](ArrayNode#structfield.stats) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn stats_as_null(self) -> ArrayNodeBuilder<(T0, T1, T2, T3, ())> {
            self.stats(())
        }
    }

    impl<T0, T1, T2, T3, T4> ArrayNodeBuilder<(T0, T1, T2, T3, T4)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [ArrayNode].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayNode>
        where
            Self: ::planus::WriteAsOffset<ArrayNode>,
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
    > ::planus::WriteAs<::planus::Offset<ArrayNode>> for ArrayNodeBuilder<(T0, T1, T2, T3, T4)>
    {
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
    > ::planus::WriteAsOptional<::planus::Offset<ArrayNode>>
        for ArrayNodeBuilder<(T0, T1, T2, T3, T4)>
    {
        type Prepared = ::planus::Offset<ArrayNode>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<ArrayNode>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u16, u16>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayNode>]>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<[u16]>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<self::ArrayStats>>,
    > ::planus::WriteAsOffset<ArrayNode> for ArrayNodeBuilder<(T0, T1, T2, T3, T4)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayNode> {
            let (v0, v1, v2, v3, v4) = &self.0;
            ArrayNode::create(builder, v0, v1, v2, v3, v4)
        }
    }

    /// Reference to a deserialized [ArrayNode].
    #[derive(Copy, Clone)]
    pub struct ArrayNodeRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> ArrayNodeRef<'a> {
        /// Getter for the [`encoding` field](ArrayNode#structfield.encoding).
        #[inline]
        pub fn encoding(&self) -> ::planus::Result<u16> {
            ::core::result::Result::Ok(self.0.access(0, "ArrayNode", "encoding")?.unwrap_or(0))
        }

        /// Getter for the [`metadata` field](ArrayNode#structfield.metadata).
        #[inline]
        pub fn metadata(&self) -> ::planus::Result<::core::option::Option<&'a [u8]>> {
            self.0.access(1, "ArrayNode", "metadata")
        }

        /// Getter for the [`children` field](ArrayNode#structfield.children).
        #[inline]
        pub fn children(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<::planus::Vector<'a, ::planus::Result<self::ArrayNodeRef<'a>>>>,
        > {
            self.0.access(2, "ArrayNode", "children")
        }

        /// Getter for the [`buffers` field](ArrayNode#structfield.buffers).
        #[inline]
        pub fn buffers(
            &self,
        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, u16>>> {
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
            f.field("encoding", &self.encoding());
            if let ::core::option::Option::Some(field_metadata) = self.metadata().transpose() {
                f.field("metadata", &field_metadata);
            }
            if let ::core::option::Option::Some(field_children) = self.children().transpose() {
                f.field("children", &field_children);
            }
            if let ::core::option::Option::Some(field_buffers) = self.buffers().transpose() {
                f.field("buffers", &field_buffers);
            }
            if let ::core::option::Option::Some(field_stats) = self.stats().transpose() {
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
                encoding: ::core::convert::TryInto::try_into(value.encoding()?)?,
                metadata: value.metadata()?.map(|v| v.to_vec()),
                children: if let ::core::option::Option::Some(children) = value.children()? {
                    ::core::option::Option::Some(children.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
                buffers: if let ::core::option::Option::Some(buffers) = value.buffers()? {
                    ::core::option::Option::Some(buffers.to_vec()?)
                } else {
                    ::core::option::Option::None
                },
                stats: if let ::core::option::Option::Some(stats) = value.stats()? {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(stats)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for ArrayNodeRef<'a> {
        #[inline]
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                buffer, offset,
            )?))
        }
    }

    impl<'a> ::planus::VectorReadInner<'a> for ArrayNodeRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[ArrayNodeRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[ArrayNodeRef]", "read_as_root", 0)
            })
        }
    }

    /// The enum `Precision`
    ///
    /// Generated from these locations:
    /// * Enum `Precision` in the file `flatbuffers/vortex-array/array.fbs:40`
    #[derive(
        Copy,
        Clone,
        Debug,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        Hash,
        ::serde::Serialize,
        ::serde::Deserialize,
    )]
    #[repr(u8)]
    pub enum Precision {
        /// The variant `Inexact` in the enum `Precision`
        Inexact = 0,

        /// The variant `Exact` in the enum `Precision`
        Exact = 1,
    }

    impl Precision {
        /// Array containing all valid variants of Precision
        pub const ENUM_VALUES: [Self; 2] = [Self::Inexact, Self::Exact];
    }

    impl ::core::convert::TryFrom<u8> for Precision {
        type Error = ::planus::errors::UnknownEnumTagKind;
        #[inline]
        fn try_from(
            value: u8,
        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind> {
            #[allow(clippy::match_single_binding)]
            match value {
                0 => ::core::result::Result::Ok(Precision::Inexact),
                1 => ::core::result::Result::Ok(Precision::Exact),

                _ => ::core::result::Result::Err(::planus::errors::UnknownEnumTagKind {
                    tag: value as i128,
                }),
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
        fn prepare(
            &self,
            _builder: &mut ::planus::Builder,
            default: &Precision,
        ) -> ::core::option::Option<Precision> {
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
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'buf>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
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
        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag> {
            let value = unsafe { *buffer.buffer.get_unchecked(offset) };
            let value: ::core::result::Result<Self, _> = ::core::convert::TryInto::try_into(value);
            value.map_err(|error_kind| {
                error_kind.with_error_location(
                    "Precision",
                    "VectorRead::from_buffer",
                    buffer.offset_from_start,
                )
            })
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
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct ArrayStats {
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
        pub nan_count: ::core::option::Option<u64>,
    }

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
        ) -> ::planus::Offset<Self> {
            let prepared_min = field_min.prepare(builder);
            let prepared_min_precision =
                field_min_precision.prepare(builder, &self::Precision::Inexact);
            let prepared_max = field_max.prepare(builder);
            let prepared_max_precision =
                field_max_precision.prepare(builder, &self::Precision::Inexact);
            let prepared_sum = field_sum.prepare(builder);
            let prepared_is_sorted = field_is_sorted.prepare(builder);
            let prepared_is_strict_sorted = field_is_strict_sorted.prepare(builder);
            let prepared_is_constant = field_is_constant.prepare(builder);
            let prepared_null_count = field_null_count.prepare(builder);
            let prepared_uncompressed_size_in_bytes =
                field_uncompressed_size_in_bytes.prepare(builder);
            let prepared_nan_count = field_nan_count.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<26> =
                ::core::default::Default::default();
            if prepared_null_count.is_some() {
                table_writer.write_entry::<u64>(8);
            }
            if prepared_uncompressed_size_in_bytes.is_some() {
                table_writer.write_entry::<u64>(9);
            }
            if prepared_nan_count.is_some() {
                table_writer.write_entry::<u64>(10);
            }
            if prepared_min.is_some() {
                table_writer.write_entry::<::planus::Offset<[u8]>>(0);
            }
            if prepared_max.is_some() {
                table_writer.write_entry::<::planus::Offset<[u8]>>(2);
            }
            if prepared_sum.is_some() {
                table_writer.write_entry::<::planus::Offset<[u8]>>(4);
            }
            if prepared_min_precision.is_some() {
                table_writer.write_entry::<self::Precision>(1);
            }
            if prepared_max_precision.is_some() {
                table_writer.write_entry::<self::Precision>(3);
            }
            if prepared_is_sorted.is_some() {
                table_writer.write_entry::<bool>(5);
            }
            if prepared_is_strict_sorted.is_some() {
                table_writer.write_entry::<bool>(6);
            }
            if prepared_is_constant.is_some() {
                table_writer.write_entry::<bool>(7);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_null_count) = prepared_null_count {
                        object_writer.write::<_, _, 8>(&prepared_null_count);
                    }
                    if let ::core::option::Option::Some(prepared_uncompressed_size_in_bytes) =
                        prepared_uncompressed_size_in_bytes
                    {
                        object_writer.write::<_, _, 8>(&prepared_uncompressed_size_in_bytes);
                    }
                    if let ::core::option::Option::Some(prepared_nan_count) = prepared_nan_count {
                        object_writer.write::<_, _, 8>(&prepared_nan_count);
                    }
                    if let ::core::option::Option::Some(prepared_min) = prepared_min {
                        object_writer.write::<_, _, 4>(&prepared_min);
                    }
                    if let ::core::option::Option::Some(prepared_max) = prepared_max {
                        object_writer.write::<_, _, 4>(&prepared_max);
                    }
                    if let ::core::option::Option::Some(prepared_sum) = prepared_sum {
                        object_writer.write::<_, _, 4>(&prepared_sum);
                    }
                    if let ::core::option::Option::Some(prepared_min_precision) =
                        prepared_min_precision
                    {
                        object_writer.write::<_, _, 1>(&prepared_min_precision);
                    }
                    if let ::core::option::Option::Some(prepared_max_precision) =
                        prepared_max_precision
                    {
                        object_writer.write::<_, _, 1>(&prepared_max_precision);
                    }
                    if let ::core::option::Option::Some(prepared_is_sorted) = prepared_is_sorted {
                        object_writer.write::<_, _, 1>(&prepared_is_sorted);
                    }
                    if let ::core::option::Option::Some(prepared_is_strict_sorted) =
                        prepared_is_strict_sorted
                    {
                        object_writer.write::<_, _, 1>(&prepared_is_strict_sorted);
                    }
                    if let ::core::option::Option::Some(prepared_is_constant) = prepared_is_constant
                    {
                        object_writer.write::<_, _, 1>(&prepared_is_constant);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<ArrayStats>> {
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

    impl ArrayStatsBuilder<()> {
        /// Setter for the [`min` field](ArrayStats#structfield.min).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn min<T0>(self, value: T0) -> ArrayStatsBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        {
            ArrayStatsBuilder((value,))
        }

        /// Sets the [`min` field](ArrayStats#structfield.min) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn min_as_null(self) -> ArrayStatsBuilder<((),)> {
            self.min(())
        }
    }

    impl<T0> ArrayStatsBuilder<(T0,)> {
        /// Setter for the [`min_precision` field](ArrayStats#structfield.min_precision).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn min_precision<T1>(self, value: T1) -> ArrayStatsBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsDefault<self::Precision, self::Precision>,
        {
            let (v0,) = self.0;
            ArrayStatsBuilder((v0, value))
        }

        /// Sets the [`min_precision` field](ArrayStats#structfield.min_precision) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn min_precision_as_default(self) -> ArrayStatsBuilder<(T0, ::planus::DefaultValue)> {
            self.min_precision(::planus::DefaultValue)
        }
    }

    impl<T0, T1> ArrayStatsBuilder<(T0, T1)> {
        /// Setter for the [`max` field](ArrayStats#structfield.max).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn max<T2>(self, value: T2) -> ArrayStatsBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        {
            let (v0, v1) = self.0;
            ArrayStatsBuilder((v0, v1, value))
        }

        /// Sets the [`max` field](ArrayStats#structfield.max) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn max_as_null(self) -> ArrayStatsBuilder<(T0, T1, ())> {
            self.max(())
        }
    }

    impl<T0, T1, T2> ArrayStatsBuilder<(T0, T1, T2)> {
        /// Setter for the [`max_precision` field](ArrayStats#structfield.max_precision).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn max_precision<T3>(self, value: T3) -> ArrayStatsBuilder<(T0, T1, T2, T3)>
        where
            T3: ::planus::WriteAsDefault<self::Precision, self::Precision>,
        {
            let (v0, v1, v2) = self.0;
            ArrayStatsBuilder((v0, v1, v2, value))
        }

        /// Sets the [`max_precision` field](ArrayStats#structfield.max_precision) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn max_precision_as_default(
            self,
        ) -> ArrayStatsBuilder<(T0, T1, T2, ::planus::DefaultValue)> {
            self.max_precision(::planus::DefaultValue)
        }
    }

    impl<T0, T1, T2, T3> ArrayStatsBuilder<(T0, T1, T2, T3)> {
        /// Setter for the [`sum` field](ArrayStats#structfield.sum).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn sum<T4>(self, value: T4) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4)>
        where
            T4: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        {
            let (v0, v1, v2, v3) = self.0;
            ArrayStatsBuilder((v0, v1, v2, v3, value))
        }

        /// Sets the [`sum` field](ArrayStats#structfield.sum) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn sum_as_null(self) -> ArrayStatsBuilder<(T0, T1, T2, T3, ())> {
            self.sum(())
        }
    }

    impl<T0, T1, T2, T3, T4> ArrayStatsBuilder<(T0, T1, T2, T3, T4)> {
        /// Setter for the [`is_sorted` field](ArrayStats#structfield.is_sorted).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn is_sorted<T5>(self, value: T5) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5)>
        where
            T5: ::planus::WriteAsOptional<bool>,
        {
            let (v0, v1, v2, v3, v4) = self.0;
            ArrayStatsBuilder((v0, v1, v2, v3, v4, value))
        }

        /// Sets the [`is_sorted` field](ArrayStats#structfield.is_sorted) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn is_sorted_as_null(self) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, ())> {
            self.is_sorted(())
        }
    }

    impl<T0, T1, T2, T3, T4, T5> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5)> {
        /// Setter for the [`is_strict_sorted` field](ArrayStats#structfield.is_strict_sorted).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn is_strict_sorted<T6>(
            self,
            value: T6,
        ) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6)>
        where
            T6: ::planus::WriteAsOptional<bool>,
        {
            let (v0, v1, v2, v3, v4, v5) = self.0;
            ArrayStatsBuilder((v0, v1, v2, v3, v4, v5, value))
        }

        /// Sets the [`is_strict_sorted` field](ArrayStats#structfield.is_strict_sorted) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn is_strict_sorted_as_null(self) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, ())> {
            self.is_strict_sorted(())
        }
    }

    impl<T0, T1, T2, T3, T4, T5, T6> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6)> {
        /// Setter for the [`is_constant` field](ArrayStats#structfield.is_constant).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn is_constant<T7>(
            self,
            value: T7,
        ) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7)>
        where
            T7: ::planus::WriteAsOptional<bool>,
        {
            let (v0, v1, v2, v3, v4, v5, v6) = self.0;
            ArrayStatsBuilder((v0, v1, v2, v3, v4, v5, v6, value))
        }

        /// Sets the [`is_constant` field](ArrayStats#structfield.is_constant) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn is_constant_as_null(self) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, ())> {
            self.is_constant(())
        }
    }

    impl<T0, T1, T2, T3, T4, T5, T6, T7> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7)> {
        /// Setter for the [`null_count` field](ArrayStats#structfield.null_count).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn null_count<T8>(
            self,
            value: T8,
        ) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8)>
        where
            T8: ::planus::WriteAsOptional<u64>,
        {
            let (v0, v1, v2, v3, v4, v5, v6, v7) = self.0;
            ArrayStatsBuilder((v0, v1, v2, v3, v4, v5, v6, v7, value))
        }

        /// Sets the [`null_count` field](ArrayStats#structfield.null_count) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn null_count_as_null(self) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, ())> {
            self.null_count(())
        }
    }

    impl<T0, T1, T2, T3, T4, T5, T6, T7, T8> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8)> {
        /// Setter for the [`uncompressed_size_in_bytes` field](ArrayStats#structfield.uncompressed_size_in_bytes).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn uncompressed_size_in_bytes<T9>(
            self,
            value: T9,
        ) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)>
        where
            T9: ::planus::WriteAsOptional<u64>,
        {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8) = self.0;
            ArrayStatsBuilder((v0, v1, v2, v3, v4, v5, v6, v7, v8, value))
        }

        /// Sets the [`uncompressed_size_in_bytes` field](ArrayStats#structfield.uncompressed_size_in_bytes) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn uncompressed_size_in_bytes_as_null(
            self,
        ) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, ())> {
            self.uncompressed_size_in_bytes(())
        }
    }

    impl<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
        ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)>
    {
        /// Setter for the [`nan_count` field](ArrayStats#structfield.nan_count).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nan_count<T10>(
            self,
            value: T10,
        ) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)>
        where
            T10: ::planus::WriteAsOptional<u64>,
        {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9) = self.0;
            ArrayStatsBuilder((v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, value))
        }

        /// Sets the [`nan_count` field](ArrayStats#structfield.nan_count) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nan_count_as_null(
            self,
        ) -> ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, ())> {
            self.nan_count(())
        }
    }

    impl<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>
        ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)>
    {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [ArrayStats].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayStats>
        where
            Self: ::planus::WriteAsOffset<ArrayStats>,
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
    > ::planus::WriteAs<::planus::Offset<ArrayStats>>
        for ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)>
    {
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
    > ::planus::WriteAsOptional<::planus::Offset<ArrayStats>>
        for ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)>
    {
        type Prepared = ::planus::Offset<ArrayStats>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<ArrayStats>> {
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
    > ::planus::WriteAsOffset<ArrayStats>
        for ArrayStatsBuilder<(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArrayStats> {
            let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10) = &self.0;
            ArrayStats::create(builder, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)
        }
    }

    /// Reference to a deserialized [ArrayStats].
    #[derive(Copy, Clone)]
    pub struct ArrayStatsRef<'a>(::planus::table_reader::Table<'a>);

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
                self.0
                    .access(1, "ArrayStats", "min_precision")?
                    .unwrap_or(self::Precision::Inexact),
            )
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
                self.0
                    .access(3, "ArrayStats", "max_precision")?
                    .unwrap_or(self::Precision::Inexact),
            )
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
            }
            f.field("min_precision", &self.min_precision());
            if let ::core::option::Option::Some(field_max) = self.max().transpose() {
                f.field("max", &field_max);
            }
            f.field("max_precision", &self.max_precision());
            if let ::core::option::Option::Some(field_sum) = self.sum().transpose() {
                f.field("sum", &field_sum);
            }
            if let ::core::option::Option::Some(field_is_sorted) = self.is_sorted().transpose() {
                f.field("is_sorted", &field_is_sorted);
            }
            if let ::core::option::Option::Some(field_is_strict_sorted) =
                self.is_strict_sorted().transpose()
            {
                f.field("is_strict_sorted", &field_is_strict_sorted);
            }
            if let ::core::option::Option::Some(field_is_constant) = self.is_constant().transpose()
            {
                f.field("is_constant", &field_is_constant);
            }
            if let ::core::option::Option::Some(field_null_count) = self.null_count().transpose() {
                f.field("null_count", &field_null_count);
            }
            if let ::core::option::Option::Some(field_uncompressed_size_in_bytes) =
                self.uncompressed_size_in_bytes().transpose()
            {
                f.field(
                    "uncompressed_size_in_bytes",
                    &field_uncompressed_size_in_bytes,
                );
            }
            if let ::core::option::Option::Some(field_nan_count) = self.nan_count().transpose() {
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
                min: value.min()?.map(|v| v.to_vec()),
                min_precision: ::core::convert::TryInto::try_into(value.min_precision()?)?,
                max: value.max()?.map(|v| v.to_vec()),
                max_precision: ::core::convert::TryInto::try_into(value.max_precision()?)?,
                sum: value.sum()?.map(|v| v.to_vec()),
                is_sorted: if let ::core::option::Option::Some(is_sorted) = value.is_sorted()? {
                    ::core::option::Option::Some(::core::convert::TryInto::try_into(is_sorted)?)
                } else {
                    ::core::option::Option::None
                },
                is_strict_sorted: if let ::core::option::Option::Some(is_strict_sorted) =
                    value.is_strict_sorted()?
                {
                    ::core::option::Option::Some(::core::convert::TryInto::try_into(
                        is_strict_sorted,
                    )?)
                } else {
                    ::core::option::Option::None
                },
                is_constant: if let ::core::option::Option::Some(is_constant) =
                    value.is_constant()?
                {
                    ::core::option::Option::Some(::core::convert::TryInto::try_into(is_constant)?)
                } else {
                    ::core::option::Option::None
                },
                null_count: if let ::core::option::Option::Some(null_count) = value.null_count()? {
                    ::core::option::Option::Some(::core::convert::TryInto::try_into(null_count)?)
                } else {
                    ::core::option::Option::None
                },
                uncompressed_size_in_bytes: if let ::core::option::Option::Some(
                    uncompressed_size_in_bytes,
                ) = value.uncompressed_size_in_bytes()?
                {
                    ::core::option::Option::Some(::core::convert::TryInto::try_into(
                        uncompressed_size_in_bytes,
                    )?)
                } else {
                    ::core::option::Option::None
                },
                nan_count: if let ::core::option::Option::Some(nan_count) = value.nan_count()? {
                    ::core::option::Option::Some(::core::convert::TryInto::try_into(nan_count)?)
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for ArrayStatsRef<'a> {
        #[inline]
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                buffer, offset,
            )?))
        }
    }

    impl<'a> ::planus::VectorReadInner<'a> for ArrayStatsRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[ArrayStatsRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[ArrayStatsRef]", "read_as_root", 0)
            })
        }
    }
}

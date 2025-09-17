pub use root::*;

const _: () = ::planus::check_version_compatibility("planus-1.2.0");

/// The root namespace
///
/// Generated from these locations:
/// * File `flatbuffers/vortex-file/footer.fbs`
/// * File `flatbuffers/vortex-array/array.fbs`
/// * File `flatbuffers/vortex-layout/layout.fbs`
#[no_implicit_prelude]
#[allow(dead_code, clippy::needless_lifetimes)]
mod root {
    ///  The `Postscript` is guaranteed by the file format to never exceed
    ///  65528 bytes (i.e., u16::MAX - 8 bytes) in length, and is immediately
    ///  followed by an 8-byte `EndOfFile` struct.
    ///
    ///  An initial read of a Vortex file defaults to at least 64KB (u16::MAX bytes) and therefore
    ///  is guaranteed to cover at least the Postscript.
    ///
    ///  The reason for a postscript at all is to ensure minimal but all necessary footer information
    ///  can be read in two round trips. Since the DType is optional and possibly large, it lives in
    ///  its own segment. If the footer were arbitrary size, with a pointer to the DType segment, then
    ///  in the worst case we would need one round trip to read the footer length, one to read the full
    ///  footer and parse the DType offset, and a third to fetch the DType segment.
    ///
    ///  The segments pointed to by the postscript have inline compression and encryption specs to avoid
    ///  the need to fetch encryption schemes up-front.
    ///
    /// Generated from these locations:
    /// * Table `Postscript` in the file `flatbuffers/vortex-file/footer.fbs:23`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Postscript {
        ///  Segment containing the root `DType` flatbuffer.
        pub dtype: ::core::option::Option<::planus::alloc::boxed::Box<self::PostscriptSegment>>,
        ///  Segment containing the root `Layout` flatbuffer (required).
        pub layout: ::core::option::Option<::planus::alloc::boxed::Box<self::PostscriptSegment>>,
        ///  Segment containing the file-level `Statistics` flatbuffer.
        pub statistics:
            ::core::option::Option<::planus::alloc::boxed::Box<self::PostscriptSegment>>,
        ///  Segment containing the 'Footer' flatbuffer (required)
        pub footer: ::core::option::Option<::planus::alloc::boxed::Box<self::PostscriptSegment>>,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for Postscript {
        fn default() -> Self {
            Self {
                dtype: ::core::default::Default::default(),
                layout: ::core::default::Default::default(),
                statistics: ::core::default::Default::default(),
                footer: ::core::default::Default::default(),
            }
        }
    }

    impl Postscript {
        /// Creates a [PostscriptBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> PostscriptBuilder<()> {
            PostscriptBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(
            builder: &mut ::planus::Builder,
            field_dtype: impl ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
            field_layout: impl ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
            field_statistics: impl ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
            field_footer: impl ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        ) -> ::planus::Offset<Self> {
            let prepared_dtype = field_dtype.prepare(builder);
            let prepared_layout = field_layout.prepare(builder);
            let prepared_statistics = field_statistics.prepare(builder);
            let prepared_footer = field_footer.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<12> =
                ::core::default::Default::default();
            if prepared_dtype.is_some() {
                table_writer.write_entry::<::planus::Offset<self::PostscriptSegment>>(0);
            }
            if prepared_layout.is_some() {
                table_writer.write_entry::<::planus::Offset<self::PostscriptSegment>>(1);
            }
            if prepared_statistics.is_some() {
                table_writer.write_entry::<::planus::Offset<self::PostscriptSegment>>(2);
            }
            if prepared_footer.is_some() {
                table_writer.write_entry::<::planus::Offset<self::PostscriptSegment>>(3);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_dtype) = prepared_dtype {
                        object_writer.write::<_, _, 4>(&prepared_dtype);
                    }
                    if let ::core::option::Option::Some(prepared_layout) = prepared_layout {
                        object_writer.write::<_, _, 4>(&prepared_layout);
                    }
                    if let ::core::option::Option::Some(prepared_statistics) = prepared_statistics {
                        object_writer.write::<_, _, 4>(&prepared_statistics);
                    }
                    if let ::core::option::Option::Some(prepared_footer) = prepared_footer {
                        object_writer.write::<_, _, 4>(&prepared_footer);
                    }
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<Postscript>> for Postscript {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Postscript> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<Postscript>> for Postscript {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Postscript>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Postscript> for Postscript {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Postscript> {
            Postscript::create(
                builder,
                &self.dtype,
                &self.layout,
                &self.statistics,
                &self.footer,
            )
        }
    }

    /// Builder for serializing an instance of the [Postscript] type.
    ///
    /// Can be created using the [Postscript::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct PostscriptBuilder<State>(State);

    impl PostscriptBuilder<()> {
        /// Setter for the [`dtype` field](Postscript#structfield.dtype).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn dtype<T0>(self, value: T0) -> PostscriptBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        {
            PostscriptBuilder((value,))
        }

        /// Sets the [`dtype` field](Postscript#structfield.dtype) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn dtype_as_null(self) -> PostscriptBuilder<((),)> {
            self.dtype(())
        }
    }

    impl<T0> PostscriptBuilder<(T0,)> {
        /// Setter for the [`layout` field](Postscript#structfield.layout).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn layout<T1>(self, value: T1) -> PostscriptBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        {
            let (v0,) = self.0;
            PostscriptBuilder((v0, value))
        }

        /// Sets the [`layout` field](Postscript#structfield.layout) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn layout_as_null(self) -> PostscriptBuilder<(T0, ())> {
            self.layout(())
        }
    }

    impl<T0, T1> PostscriptBuilder<(T0, T1)> {
        /// Setter for the [`statistics` field](Postscript#structfield.statistics).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn statistics<T2>(self, value: T2) -> PostscriptBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        {
            let (v0, v1) = self.0;
            PostscriptBuilder((v0, v1, value))
        }

        /// Sets the [`statistics` field](Postscript#structfield.statistics) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn statistics_as_null(self) -> PostscriptBuilder<(T0, T1, ())> {
            self.statistics(())
        }
    }

    impl<T0, T1, T2> PostscriptBuilder<(T0, T1, T2)> {
        /// Setter for the [`footer` field](Postscript#structfield.footer).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn footer<T3>(self, value: T3) -> PostscriptBuilder<(T0, T1, T2, T3)>
        where
            T3: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        {
            let (v0, v1, v2) = self.0;
            PostscriptBuilder((v0, v1, v2, value))
        }

        /// Sets the [`footer` field](Postscript#structfield.footer) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn footer_as_null(self) -> PostscriptBuilder<(T0, T1, T2, ())> {
            self.footer(())
        }
    }

    impl<T0, T1, T2, T3> PostscriptBuilder<(T0, T1, T2, T3)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Postscript].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Postscript>
        where
            Self: ::planus::WriteAsOffset<Postscript>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
    > ::planus::WriteAs<::planus::Offset<Postscript>> for PostscriptBuilder<(T0, T1, T2, T3)>
    {
        type Prepared = ::planus::Offset<Postscript>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Postscript> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
    > ::planus::WriteAsOptional<::planus::Offset<Postscript>>
        for PostscriptBuilder<(T0, T1, T2, T3)>
    {
        type Prepared = ::planus::Offset<Postscript>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Postscript>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<self::PostscriptSegment>>,
    > ::planus::WriteAsOffset<Postscript> for PostscriptBuilder<(T0, T1, T2, T3)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Postscript> {
            let (v0, v1, v2, v3) = &self.0;
            Postscript::create(builder, v0, v1, v2, v3)
        }
    }

    /// Reference to a deserialized [Postscript].
    #[derive(Copy, Clone)]
    pub struct PostscriptRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> PostscriptRef<'a> {
        /// Getter for the [`dtype` field](Postscript#structfield.dtype).
        #[inline]
        pub fn dtype(
            &self,
        ) -> ::planus::Result<::core::option::Option<self::PostscriptSegmentRef<'a>>> {
            self.0.access(0, "Postscript", "dtype")
        }

        /// Getter for the [`layout` field](Postscript#structfield.layout).
        #[inline]
        pub fn layout(
            &self,
        ) -> ::planus::Result<::core::option::Option<self::PostscriptSegmentRef<'a>>> {
            self.0.access(1, "Postscript", "layout")
        }

        /// Getter for the [`statistics` field](Postscript#structfield.statistics).
        #[inline]
        pub fn statistics(
            &self,
        ) -> ::planus::Result<::core::option::Option<self::PostscriptSegmentRef<'a>>> {
            self.0.access(2, "Postscript", "statistics")
        }

        /// Getter for the [`footer` field](Postscript#structfield.footer).
        #[inline]
        pub fn footer(
            &self,
        ) -> ::planus::Result<::core::option::Option<self::PostscriptSegmentRef<'a>>> {
            self.0.access(3, "Postscript", "footer")
        }
    }

    impl<'a> ::core::fmt::Debug for PostscriptRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("PostscriptRef");
            if let ::core::option::Option::Some(field_dtype) = self.dtype().transpose() {
                f.field("dtype", &field_dtype);
            }
            if let ::core::option::Option::Some(field_layout) = self.layout().transpose() {
                f.field("layout", &field_layout);
            }
            if let ::core::option::Option::Some(field_statistics) = self.statistics().transpose() {
                f.field("statistics", &field_statistics);
            }
            if let ::core::option::Option::Some(field_footer) = self.footer().transpose() {
                f.field("footer", &field_footer);
            }
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<PostscriptRef<'a>> for Postscript {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: PostscriptRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                dtype: if let ::core::option::Option::Some(dtype) = value.dtype()? {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(dtype)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
                layout: if let ::core::option::Option::Some(layout) = value.layout()? {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(layout)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
                statistics: if let ::core::option::Option::Some(statistics) = value.statistics()? {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(statistics)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
                footer: if let ::core::option::Option::Some(footer) = value.footer()? {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(footer)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for PostscriptRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for PostscriptRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[PostscriptRef]", "get", buffer.offset_from_start)
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<Postscript>> for Postscript {
        type Value = ::planus::Offset<Postscript>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<Postscript>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for PostscriptRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[PostscriptRef]", "read_as_root", 0)
            })
        }
    }

    ///  A `PostscriptSegment` describes the location of a segment in the file without referencing any
    ///  specification objects. That is, encryption and compression are defined inline.
    ///
    /// Generated from these locations:
    /// * Table `PostscriptSegment` in the file `flatbuffers/vortex-file/footer.fbs:36`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct PostscriptSegment {
        /// The field `offset` in the table `PostscriptSegment`
        pub offset: u64,
        /// The field `length` in the table `PostscriptSegment`
        pub length: u32,
        /// The field `alignment_exponent` in the table `PostscriptSegment`
        pub alignment_exponent: u8,
        /// The field `_compression` in the table `PostscriptSegment`
        pub compression: ::core::option::Option<::planus::alloc::boxed::Box<self::CompressionSpec>>,
        /// The field `_encryption` in the table `PostscriptSegment`
        pub encryption: ::core::option::Option<::planus::alloc::boxed::Box<self::EncryptionSpec>>,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for PostscriptSegment {
        fn default() -> Self {
            Self {
                offset: 0,
                length: 0,
                alignment_exponent: 0,
                compression: ::core::default::Default::default(),
                encryption: ::core::default::Default::default(),
            }
        }
    }

    impl PostscriptSegment {
        /// Creates a [PostscriptSegmentBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> PostscriptSegmentBuilder<()> {
            PostscriptSegmentBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(
            builder: &mut ::planus::Builder,
            field_offset: impl ::planus::WriteAsDefault<u64, u64>,
            field_length: impl ::planus::WriteAsDefault<u32, u32>,
            field_alignment_exponent: impl ::planus::WriteAsDefault<u8, u8>,
            field_compression: impl ::planus::WriteAsOptional<::planus::Offset<self::CompressionSpec>>,
            field_encryption: impl ::planus::WriteAsOptional<::planus::Offset<self::EncryptionSpec>>,
        ) -> ::planus::Offset<Self> {
            let prepared_offset = field_offset.prepare(builder, &0);
            let prepared_length = field_length.prepare(builder, &0);
            let prepared_alignment_exponent = field_alignment_exponent.prepare(builder, &0);
            let prepared_compression = field_compression.prepare(builder);
            let prepared_encryption = field_encryption.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                ::core::default::Default::default();
            if prepared_offset.is_some() {
                table_writer.write_entry::<u64>(0);
            }
            if prepared_length.is_some() {
                table_writer.write_entry::<u32>(1);
            }
            if prepared_compression.is_some() {
                table_writer.write_entry::<::planus::Offset<self::CompressionSpec>>(3);
            }
            if prepared_encryption.is_some() {
                table_writer.write_entry::<::planus::Offset<self::EncryptionSpec>>(4);
            }
            if prepared_alignment_exponent.is_some() {
                table_writer.write_entry::<u8>(2);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_offset) = prepared_offset {
                        object_writer.write::<_, _, 8>(&prepared_offset);
                    }
                    if let ::core::option::Option::Some(prepared_length) = prepared_length {
                        object_writer.write::<_, _, 4>(&prepared_length);
                    }
                    if let ::core::option::Option::Some(prepared_compression) = prepared_compression
                    {
                        object_writer.write::<_, _, 4>(&prepared_compression);
                    }
                    if let ::core::option::Option::Some(prepared_encryption) = prepared_encryption {
                        object_writer.write::<_, _, 4>(&prepared_encryption);
                    }
                    if let ::core::option::Option::Some(prepared_alignment_exponent) =
                        prepared_alignment_exponent
                    {
                        object_writer.write::<_, _, 1>(&prepared_alignment_exponent);
                    }
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<PostscriptSegment>> for PostscriptSegment {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<PostscriptSegment> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<PostscriptSegment>> for PostscriptSegment {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<PostscriptSegment>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<PostscriptSegment> for PostscriptSegment {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<PostscriptSegment> {
            PostscriptSegment::create(
                builder,
                self.offset,
                self.length,
                self.alignment_exponent,
                &self.compression,
                &self.encryption,
            )
        }
    }

    /// Builder for serializing an instance of the [PostscriptSegment] type.
    ///
    /// Can be created using the [PostscriptSegment::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct PostscriptSegmentBuilder<State>(State);

    impl PostscriptSegmentBuilder<()> {
        /// Setter for the [`offset` field](PostscriptSegment#structfield.offset).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn offset<T0>(self, value: T0) -> PostscriptSegmentBuilder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<u64, u64>,
        {
            PostscriptSegmentBuilder((value,))
        }

        /// Sets the [`offset` field](PostscriptSegment#structfield.offset) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn offset_as_default(self) -> PostscriptSegmentBuilder<(::planus::DefaultValue,)> {
            self.offset(::planus::DefaultValue)
        }
    }

    impl<T0> PostscriptSegmentBuilder<(T0,)> {
        /// Setter for the [`length` field](PostscriptSegment#structfield.length).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn length<T1>(self, value: T1) -> PostscriptSegmentBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsDefault<u32, u32>,
        {
            let (v0,) = self.0;
            PostscriptSegmentBuilder((v0, value))
        }

        /// Sets the [`length` field](PostscriptSegment#structfield.length) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn length_as_default(self) -> PostscriptSegmentBuilder<(T0, ::planus::DefaultValue)> {
            self.length(::planus::DefaultValue)
        }
    }

    impl<T0, T1> PostscriptSegmentBuilder<(T0, T1)> {
        /// Setter for the [`alignment_exponent` field](PostscriptSegment#structfield.alignment_exponent).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn alignment_exponent<T2>(self, value: T2) -> PostscriptSegmentBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsDefault<u8, u8>,
        {
            let (v0, v1) = self.0;
            PostscriptSegmentBuilder((v0, v1, value))
        }

        /// Sets the [`alignment_exponent` field](PostscriptSegment#structfield.alignment_exponent) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn alignment_exponent_as_default(
            self,
        ) -> PostscriptSegmentBuilder<(T0, T1, ::planus::DefaultValue)> {
            self.alignment_exponent(::planus::DefaultValue)
        }
    }

    impl<T0, T1, T2> PostscriptSegmentBuilder<(T0, T1, T2)> {
        /// Setter for the [`_compression` field](PostscriptSegment#structfield.compression).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn compression<T3>(self, value: T3) -> PostscriptSegmentBuilder<(T0, T1, T2, T3)>
        where
            T3: ::planus::WriteAsOptional<::planus::Offset<self::CompressionSpec>>,
        {
            let (v0, v1, v2) = self.0;
            PostscriptSegmentBuilder((v0, v1, v2, value))
        }

        /// Sets the [`_compression` field](PostscriptSegment#structfield.compression) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn compression_as_null(self) -> PostscriptSegmentBuilder<(T0, T1, T2, ())> {
            self.compression(())
        }
    }

    impl<T0, T1, T2, T3> PostscriptSegmentBuilder<(T0, T1, T2, T3)> {
        /// Setter for the [`_encryption` field](PostscriptSegment#structfield.encryption).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn encryption<T4>(self, value: T4) -> PostscriptSegmentBuilder<(T0, T1, T2, T3, T4)>
        where
            T4: ::planus::WriteAsOptional<::planus::Offset<self::EncryptionSpec>>,
        {
            let (v0, v1, v2, v3) = self.0;
            PostscriptSegmentBuilder((v0, v1, v2, v3, value))
        }

        /// Sets the [`_encryption` field](PostscriptSegment#structfield.encryption) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn encryption_as_null(self) -> PostscriptSegmentBuilder<(T0, T1, T2, T3, ())> {
            self.encryption(())
        }
    }

    impl<T0, T1, T2, T3, T4> PostscriptSegmentBuilder<(T0, T1, T2, T3, T4)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [PostscriptSegment].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<PostscriptSegment>
        where
            Self: ::planus::WriteAsOffset<PostscriptSegment>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u64, u64>,
        T1: ::planus::WriteAsDefault<u32, u32>,
        T2: ::planus::WriteAsDefault<u8, u8>,
        T3: ::planus::WriteAsOptional<::planus::Offset<self::CompressionSpec>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<self::EncryptionSpec>>,
    > ::planus::WriteAs<::planus::Offset<PostscriptSegment>>
        for PostscriptSegmentBuilder<(T0, T1, T2, T3, T4)>
    {
        type Prepared = ::planus::Offset<PostscriptSegment>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<PostscriptSegment> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u64, u64>,
        T1: ::planus::WriteAsDefault<u32, u32>,
        T2: ::planus::WriteAsDefault<u8, u8>,
        T3: ::planus::WriteAsOptional<::planus::Offset<self::CompressionSpec>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<self::EncryptionSpec>>,
    > ::planus::WriteAsOptional<::planus::Offset<PostscriptSegment>>
        for PostscriptSegmentBuilder<(T0, T1, T2, T3, T4)>
    {
        type Prepared = ::planus::Offset<PostscriptSegment>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<PostscriptSegment>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u64, u64>,
        T1: ::planus::WriteAsDefault<u32, u32>,
        T2: ::planus::WriteAsDefault<u8, u8>,
        T3: ::planus::WriteAsOptional<::planus::Offset<self::CompressionSpec>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<self::EncryptionSpec>>,
    > ::planus::WriteAsOffset<PostscriptSegment>
        for PostscriptSegmentBuilder<(T0, T1, T2, T3, T4)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<PostscriptSegment> {
            let (v0, v1, v2, v3, v4) = &self.0;
            PostscriptSegment::create(builder, v0, v1, v2, v3, v4)
        }
    }

    /// Reference to a deserialized [PostscriptSegment].
    #[derive(Copy, Clone)]
    pub struct PostscriptSegmentRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> PostscriptSegmentRef<'a> {
        /// Getter for the [`offset` field](PostscriptSegment#structfield.offset).
        #[inline]
        pub fn offset(&self) -> ::planus::Result<u64> {
            ::core::result::Result::Ok(
                self.0
                    .access(0, "PostscriptSegment", "offset")?
                    .unwrap_or(0),
            )
        }

        /// Getter for the [`length` field](PostscriptSegment#structfield.length).
        #[inline]
        pub fn length(&self) -> ::planus::Result<u32> {
            ::core::result::Result::Ok(
                self.0
                    .access(1, "PostscriptSegment", "length")?
                    .unwrap_or(0),
            )
        }

        /// Getter for the [`alignment_exponent` field](PostscriptSegment#structfield.alignment_exponent).
        #[inline]
        pub fn alignment_exponent(&self) -> ::planus::Result<u8> {
            ::core::result::Result::Ok(
                self.0
                    .access(2, "PostscriptSegment", "alignment_exponent")?
                    .unwrap_or(0),
            )
        }

        /// Getter for the [`_compression` field](PostscriptSegment#structfield.compression).
        #[inline]
        pub fn compression(
            &self,
        ) -> ::planus::Result<::core::option::Option<self::CompressionSpecRef<'a>>> {
            self.0.access(3, "PostscriptSegment", "compression")
        }

        /// Getter for the [`_encryption` field](PostscriptSegment#structfield.encryption).
        #[inline]
        pub fn encryption(
            &self,
        ) -> ::planus::Result<::core::option::Option<self::EncryptionSpecRef<'a>>> {
            self.0.access(4, "PostscriptSegment", "encryption")
        }
    }

    impl<'a> ::core::fmt::Debug for PostscriptSegmentRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("PostscriptSegmentRef");
            f.field("offset", &self.offset());
            f.field("length", &self.length());
            f.field("alignment_exponent", &self.alignment_exponent());
            if let ::core::option::Option::Some(field_compression) = self.compression().transpose()
            {
                f.field("compression", &field_compression);
            }
            if let ::core::option::Option::Some(field_encryption) = self.encryption().transpose() {
                f.field("encryption", &field_encryption);
            }
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<PostscriptSegmentRef<'a>> for PostscriptSegment {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: PostscriptSegmentRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                offset: ::core::convert::TryInto::try_into(value.offset()?)?,
                length: ::core::convert::TryInto::try_into(value.length()?)?,
                alignment_exponent: ::core::convert::TryInto::try_into(
                    value.alignment_exponent()?,
                )?,
                compression: if let ::core::option::Option::Some(compression) =
                    value.compression()?
                {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(compression)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
                encryption: if let ::core::option::Option::Some(encryption) = value.encryption()? {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(encryption)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for PostscriptSegmentRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for PostscriptSegmentRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location(
                    "[PostscriptSegmentRef]",
                    "get",
                    buffer.offset_from_start,
                )
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<PostscriptSegment>> for PostscriptSegment {
        type Value = ::planus::Offset<PostscriptSegment>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<PostscriptSegment>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for PostscriptSegmentRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[PostscriptSegmentRef]", "read_as_root", 0)
            })
        }
    }

    ///  The `FileStatistics` object contains file-level statistics for the Vortex file.
    ///
    /// Generated from these locations:
    /// * Table `FileStatistics` in the file `flatbuffers/vortex-file/footer.fbs:47`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct FileStatistics {
        ///  Statistics for each field in the root schema. If the root schema is not a struct, there will
        ///  be a single entry in this array.
        pub field_stats: ::core::option::Option<::planus::alloc::vec::Vec<self::ArrayStats>>,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for FileStatistics {
        fn default() -> Self {
            Self {
                field_stats: ::core::default::Default::default(),
            }
        }
    }

    impl FileStatistics {
        /// Creates a [FileStatisticsBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> FileStatisticsBuilder<()> {
            FileStatisticsBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(
            builder: &mut ::planus::Builder,
            field_field_stats: impl ::planus::WriteAsOptional<
                ::planus::Offset<[::planus::Offset<self::ArrayStats>]>,
            >,
        ) -> ::planus::Offset<Self> {
            let prepared_field_stats = field_field_stats.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                ::core::default::Default::default();
            if prepared_field_stats.is_some() {
                table_writer
                    .write_entry::<::planus::Offset<[::planus::Offset<self::ArrayStats>]>>(0);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_field_stats) = prepared_field_stats
                    {
                        object_writer.write::<_, _, 4>(&prepared_field_stats);
                    }
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<FileStatistics>> for FileStatistics {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FileStatistics> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<FileStatistics>> for FileStatistics {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<FileStatistics>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<FileStatistics> for FileStatistics {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FileStatistics> {
            FileStatistics::create(builder, &self.field_stats)
        }
    }

    /// Builder for serializing an instance of the [FileStatistics] type.
    ///
    /// Can be created using the [FileStatistics::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct FileStatisticsBuilder<State>(State);

    impl FileStatisticsBuilder<()> {
        /// Setter for the [`field_stats` field](FileStatistics#structfield.field_stats).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn field_stats<T0>(self, value: T0) -> FileStatisticsBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayStats>]>>,
        {
            FileStatisticsBuilder((value,))
        }

        /// Sets the [`field_stats` field](FileStatistics#structfield.field_stats) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn field_stats_as_null(self) -> FileStatisticsBuilder<((),)> {
            self.field_stats(())
        }
    }

    impl<T0> FileStatisticsBuilder<(T0,)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [FileStatistics].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<FileStatistics>
        where
            Self: ::planus::WriteAsOffset<FileStatistics>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayStats>]>>>
        ::planus::WriteAs<::planus::Offset<FileStatistics>> for FileStatisticsBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<FileStatistics>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FileStatistics> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayStats>]>>>
        ::planus::WriteAsOptional<::planus::Offset<FileStatistics>>
        for FileStatisticsBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<FileStatistics>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<FileStatistics>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArrayStats>]>>>
        ::planus::WriteAsOffset<FileStatistics> for FileStatisticsBuilder<(T0,)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FileStatistics> {
            let (v0,) = &self.0;
            FileStatistics::create(builder, v0)
        }
    }

    /// Reference to a deserialized [FileStatistics].
    #[derive(Copy, Clone)]
    pub struct FileStatisticsRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> FileStatisticsRef<'a> {
        /// Getter for the [`field_stats` field](FileStatistics#structfield.field_stats).
        #[inline]
        pub fn field_stats(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<::planus::Vector<'a, ::planus::Result<self::ArrayStatsRef<'a>>>>,
        > {
            self.0.access(0, "FileStatistics", "field_stats")
        }
    }

    impl<'a> ::core::fmt::Debug for FileStatisticsRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("FileStatisticsRef");
            if let ::core::option::Option::Some(field_field_stats) = self.field_stats().transpose()
            {
                f.field("field_stats", &field_field_stats);
            }
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<FileStatisticsRef<'a>> for FileStatistics {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: FileStatisticsRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                field_stats: if let ::core::option::Option::Some(field_stats) =
                    value.field_stats()?
                {
                    ::core::option::Option::Some(field_stats.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for FileStatisticsRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for FileStatisticsRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location(
                    "[FileStatisticsRef]",
                    "get",
                    buffer.offset_from_start,
                )
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<FileStatistics>> for FileStatistics {
        type Value = ::planus::Offset<FileStatistics>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<FileStatistics>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for FileStatisticsRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[FileStatisticsRef]", "read_as_root", 0)
            })
        }
    }

    ///  The `Registry` object stores dictionary-encoded configuration for segments,
    ///  compression schemes, encryption schemes, etc.
    ///
    /// Generated from these locations:
    /// * Table `Footer` in the file `flatbuffers/vortex-file/footer.fbs:55`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Footer {
        /// The field `array_specs` in the table `Footer`
        pub array_specs: ::core::option::Option<::planus::alloc::vec::Vec<self::ArraySpec>>,
        /// The field `layout_specs` in the table `Footer`
        pub layout_specs: ::core::option::Option<::planus::alloc::vec::Vec<self::LayoutSpec>>,
        /// The field `segment_specs` in the table `Footer`
        pub segment_specs: ::core::option::Option<::planus::alloc::vec::Vec<self::SegmentSpec>>,
        /// The field `compression_specs` in the table `Footer`
        pub compression_specs:
            ::core::option::Option<::planus::alloc::vec::Vec<self::CompressionSpec>>,
        /// The field `encryption_specs` in the table `Footer`
        pub encryption_specs:
            ::core::option::Option<::planus::alloc::vec::Vec<self::EncryptionSpec>>,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for Footer {
        fn default() -> Self {
            Self {
                array_specs: ::core::default::Default::default(),
                layout_specs: ::core::default::Default::default(),
                segment_specs: ::core::default::Default::default(),
                compression_specs: ::core::default::Default::default(),
                encryption_specs: ::core::default::Default::default(),
            }
        }
    }

    impl Footer {
        /// Creates a [FooterBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> FooterBuilder<()> {
            FooterBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(
            builder: &mut ::planus::Builder,
            field_array_specs: impl ::planus::WriteAsOptional<
                ::planus::Offset<[::planus::Offset<self::ArraySpec>]>,
            >,
            field_layout_specs: impl ::planus::WriteAsOptional<
                ::planus::Offset<[::planus::Offset<self::LayoutSpec>]>,
            >,
            field_segment_specs: impl ::planus::WriteAsOptional<::planus::Offset<[self::SegmentSpec]>>,
            field_compression_specs: impl ::planus::WriteAsOptional<
                ::planus::Offset<[::planus::Offset<self::CompressionSpec>]>,
            >,
            field_encryption_specs: impl ::planus::WriteAsOptional<
                ::planus::Offset<[::planus::Offset<self::EncryptionSpec>]>,
            >,
        ) -> ::planus::Offset<Self> {
            let prepared_array_specs = field_array_specs.prepare(builder);
            let prepared_layout_specs = field_layout_specs.prepare(builder);
            let prepared_segment_specs = field_segment_specs.prepare(builder);
            let prepared_compression_specs = field_compression_specs.prepare(builder);
            let prepared_encryption_specs = field_encryption_specs.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                ::core::default::Default::default();
            if prepared_array_specs.is_some() {
                table_writer
                    .write_entry::<::planus::Offset<[::planus::Offset<self::ArraySpec>]>>(0);
            }
            if prepared_layout_specs.is_some() {
                table_writer
                    .write_entry::<::planus::Offset<[::planus::Offset<self::LayoutSpec>]>>(1);
            }
            if prepared_segment_specs.is_some() {
                table_writer.write_entry::<::planus::Offset<[self::SegmentSpec]>>(2);
            }
            if prepared_compression_specs.is_some() {
                table_writer
                    .write_entry::<::planus::Offset<[::planus::Offset<self::CompressionSpec>]>>(3);
            }
            if prepared_encryption_specs.is_some() {
                table_writer
                    .write_entry::<::planus::Offset<[::planus::Offset<self::EncryptionSpec>]>>(4);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_array_specs) = prepared_array_specs
                    {
                        object_writer.write::<_, _, 4>(&prepared_array_specs);
                    }
                    if let ::core::option::Option::Some(prepared_layout_specs) =
                        prepared_layout_specs
                    {
                        object_writer.write::<_, _, 4>(&prepared_layout_specs);
                    }
                    if let ::core::option::Option::Some(prepared_segment_specs) =
                        prepared_segment_specs
                    {
                        object_writer.write::<_, _, 4>(&prepared_segment_specs);
                    }
                    if let ::core::option::Option::Some(prepared_compression_specs) =
                        prepared_compression_specs
                    {
                        object_writer.write::<_, _, 4>(&prepared_compression_specs);
                    }
                    if let ::core::option::Option::Some(prepared_encryption_specs) =
                        prepared_encryption_specs
                    {
                        object_writer.write::<_, _, 4>(&prepared_encryption_specs);
                    }
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<Footer>> for Footer {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Footer> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<Footer>> for Footer {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Footer>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Footer> for Footer {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Footer> {
            Footer::create(
                builder,
                &self.array_specs,
                &self.layout_specs,
                &self.segment_specs,
                &self.compression_specs,
                &self.encryption_specs,
            )
        }
    }

    /// Builder for serializing an instance of the [Footer] type.
    ///
    /// Can be created using the [Footer::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct FooterBuilder<State>(State);

    impl FooterBuilder<()> {
        /// Setter for the [`array_specs` field](Footer#structfield.array_specs).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn array_specs<T0>(self, value: T0) -> FooterBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArraySpec>]>>,
        {
            FooterBuilder((value,))
        }

        /// Sets the [`array_specs` field](Footer#structfield.array_specs) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn array_specs_as_null(self) -> FooterBuilder<((),)> {
            self.array_specs(())
        }
    }

    impl<T0> FooterBuilder<(T0,)> {
        /// Setter for the [`layout_specs` field](Footer#structfield.layout_specs).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn layout_specs<T1>(self, value: T1) -> FooterBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::LayoutSpec>]>>,
        {
            let (v0,) = self.0;
            FooterBuilder((v0, value))
        }

        /// Sets the [`layout_specs` field](Footer#structfield.layout_specs) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn layout_specs_as_null(self) -> FooterBuilder<(T0, ())> {
            self.layout_specs(())
        }
    }

    impl<T0, T1> FooterBuilder<(T0, T1)> {
        /// Setter for the [`segment_specs` field](Footer#structfield.segment_specs).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn segment_specs<T2>(self, value: T2) -> FooterBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsOptional<::planus::Offset<[self::SegmentSpec]>>,
        {
            let (v0, v1) = self.0;
            FooterBuilder((v0, v1, value))
        }

        /// Sets the [`segment_specs` field](Footer#structfield.segment_specs) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn segment_specs_as_null(self) -> FooterBuilder<(T0, T1, ())> {
            self.segment_specs(())
        }
    }

    impl<T0, T1, T2> FooterBuilder<(T0, T1, T2)> {
        /// Setter for the [`compression_specs` field](Footer#structfield.compression_specs).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn compression_specs<T3>(self, value: T3) -> FooterBuilder<(T0, T1, T2, T3)>
        where
            T3: ::planus::WriteAsOptional<
                    ::planus::Offset<[::planus::Offset<self::CompressionSpec>]>,
                >,
        {
            let (v0, v1, v2) = self.0;
            FooterBuilder((v0, v1, v2, value))
        }

        /// Sets the [`compression_specs` field](Footer#structfield.compression_specs) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn compression_specs_as_null(self) -> FooterBuilder<(T0, T1, T2, ())> {
            self.compression_specs(())
        }
    }

    impl<T0, T1, T2, T3> FooterBuilder<(T0, T1, T2, T3)> {
        /// Setter for the [`encryption_specs` field](Footer#structfield.encryption_specs).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn encryption_specs<T4>(self, value: T4) -> FooterBuilder<(T0, T1, T2, T3, T4)>
        where
            T4: ::planus::WriteAsOptional<
                    ::planus::Offset<[::planus::Offset<self::EncryptionSpec>]>,
                >,
        {
            let (v0, v1, v2, v3) = self.0;
            FooterBuilder((v0, v1, v2, v3, value))
        }

        /// Sets the [`encryption_specs` field](Footer#structfield.encryption_specs) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn encryption_specs_as_null(self) -> FooterBuilder<(T0, T1, T2, T3, ())> {
            self.encryption_specs(())
        }
    }

    impl<T0, T1, T2, T3, T4> FooterBuilder<(T0, T1, T2, T3, T4)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Footer].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Footer>
        where
            Self: ::planus::WriteAsOffset<Footer>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArraySpec>]>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::LayoutSpec>]>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[self::SegmentSpec]>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::CompressionSpec>]>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::EncryptionSpec>]>>,
    > ::planus::WriteAs<::planus::Offset<Footer>> for FooterBuilder<(T0, T1, T2, T3, T4)>
    {
        type Prepared = ::planus::Offset<Footer>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Footer> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArraySpec>]>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::LayoutSpec>]>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[self::SegmentSpec]>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::CompressionSpec>]>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::EncryptionSpec>]>>,
    > ::planus::WriteAsOptional<::planus::Offset<Footer>> for FooterBuilder<(T0, T1, T2, T3, T4)>
    {
        type Prepared = ::planus::Offset<Footer>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Footer>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::ArraySpec>]>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::LayoutSpec>]>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[self::SegmentSpec]>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::CompressionSpec>]>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::EncryptionSpec>]>>,
    > ::planus::WriteAsOffset<Footer> for FooterBuilder<(T0, T1, T2, T3, T4)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Footer> {
            let (v0, v1, v2, v3, v4) = &self.0;
            Footer::create(builder, v0, v1, v2, v3, v4)
        }
    }

    /// Reference to a deserialized [Footer].
    #[derive(Copy, Clone)]
    pub struct FooterRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> FooterRef<'a> {
        /// Getter for the [`array_specs` field](Footer#structfield.array_specs).
        #[inline]
        pub fn array_specs(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<::planus::Vector<'a, ::planus::Result<self::ArraySpecRef<'a>>>>,
        > {
            self.0.access(0, "Footer", "array_specs")
        }

        /// Getter for the [`layout_specs` field](Footer#structfield.layout_specs).
        #[inline]
        pub fn layout_specs(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<::planus::Vector<'a, ::planus::Result<self::LayoutSpecRef<'a>>>>,
        > {
            self.0.access(1, "Footer", "layout_specs")
        }

        /// Getter for the [`segment_specs` field](Footer#structfield.segment_specs).
        #[inline]
        pub fn segment_specs(
            &self,
        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, self::SegmentSpecRef<'a>>>>
        {
            self.0.access(2, "Footer", "segment_specs")
        }

        /// Getter for the [`compression_specs` field](Footer#structfield.compression_specs).
        #[inline]
        pub fn compression_specs(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<
                ::planus::Vector<'a, ::planus::Result<self::CompressionSpecRef<'a>>>,
            >,
        > {
            self.0.access(3, "Footer", "compression_specs")
        }

        /// Getter for the [`encryption_specs` field](Footer#structfield.encryption_specs).
        #[inline]
        pub fn encryption_specs(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<
                ::planus::Vector<'a, ::planus::Result<self::EncryptionSpecRef<'a>>>,
            >,
        > {
            self.0.access(4, "Footer", "encryption_specs")
        }
    }

    impl<'a> ::core::fmt::Debug for FooterRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("FooterRef");
            if let ::core::option::Option::Some(field_array_specs) = self.array_specs().transpose()
            {
                f.field("array_specs", &field_array_specs);
            }
            if let ::core::option::Option::Some(field_layout_specs) =
                self.layout_specs().transpose()
            {
                f.field("layout_specs", &field_layout_specs);
            }
            if let ::core::option::Option::Some(field_segment_specs) =
                self.segment_specs().transpose()
            {
                f.field("segment_specs", &field_segment_specs);
            }
            if let ::core::option::Option::Some(field_compression_specs) =
                self.compression_specs().transpose()
            {
                f.field("compression_specs", &field_compression_specs);
            }
            if let ::core::option::Option::Some(field_encryption_specs) =
                self.encryption_specs().transpose()
            {
                f.field("encryption_specs", &field_encryption_specs);
            }
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<FooterRef<'a>> for Footer {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: FooterRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                array_specs: if let ::core::option::Option::Some(array_specs) =
                    value.array_specs()?
                {
                    ::core::option::Option::Some(array_specs.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
                layout_specs: if let ::core::option::Option::Some(layout_specs) =
                    value.layout_specs()?
                {
                    ::core::option::Option::Some(layout_specs.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
                segment_specs: if let ::core::option::Option::Some(segment_specs) =
                    value.segment_specs()?
                {
                    ::core::option::Option::Some(segment_specs.to_vec()?)
                } else {
                    ::core::option::Option::None
                },
                compression_specs: if let ::core::option::Option::Some(compression_specs) =
                    value.compression_specs()?
                {
                    ::core::option::Option::Some(compression_specs.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
                encryption_specs: if let ::core::option::Option::Some(encryption_specs) =
                    value.encryption_specs()?
                {
                    ::core::option::Option::Some(encryption_specs.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for FooterRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for FooterRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[FooterRef]", "get", buffer.offset_from_start)
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<Footer>> for Footer {
        type Value = ::planus::Offset<Footer>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<Footer>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for FooterRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[FooterRef]", "read_as_root", 0))
        }
    }

    ///  An `ArraySpec` describes the type of a particular array.
    ///
    ///  These are identified by a globally unique string identifier, and looked up in the Vortex registry
    ///  at read-time.
    ///
    /// Generated from these locations:
    /// * Table `ArraySpec` in the file `flatbuffers/vortex-file/footer.fbs:72`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct ArraySpec {
        /// The field `id` in the table `ArraySpec`
        pub id: ::planus::alloc::string::String,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for ArraySpec {
        fn default() -> Self {
            Self {
                id: ::core::default::Default::default(),
            }
        }
    }

    impl ArraySpec {
        /// Creates a [ArraySpecBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> ArraySpecBuilder<()> {
            ArraySpecBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(
            builder: &mut ::planus::Builder,
            field_id: impl ::planus::WriteAs<::planus::Offset<str>>,
        ) -> ::planus::Offset<Self> {
            let prepared_id = field_id.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                ::core::default::Default::default();
            table_writer.write_entry::<::planus::Offset<str>>(0);

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    object_writer.write::<_, _, 4>(&prepared_id);
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<ArraySpec>> for ArraySpec {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArraySpec> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<ArraySpec>> for ArraySpec {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<ArraySpec>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<ArraySpec> for ArraySpec {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArraySpec> {
            ArraySpec::create(builder, &self.id)
        }
    }

    /// Builder for serializing an instance of the [ArraySpec] type.
    ///
    /// Can be created using the [ArraySpec::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct ArraySpecBuilder<State>(State);

    impl ArraySpecBuilder<()> {
        /// Setter for the [`id` field](ArraySpec#structfield.id).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn id<T0>(self, value: T0) -> ArraySpecBuilder<(T0,)>
        where
            T0: ::planus::WriteAs<::planus::Offset<str>>,
        {
            ArraySpecBuilder((value,))
        }
    }

    impl<T0> ArraySpecBuilder<(T0,)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [ArraySpec].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArraySpec>
        where
            Self: ::planus::WriteAsOffset<ArraySpec>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<T0: ::planus::WriteAs<::planus::Offset<str>>>
        ::planus::WriteAs<::planus::Offset<ArraySpec>> for ArraySpecBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<ArraySpec>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArraySpec> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<T0: ::planus::WriteAs<::planus::Offset<str>>>
        ::planus::WriteAsOptional<::planus::Offset<ArraySpec>> for ArraySpecBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<ArraySpec>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<ArraySpec>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<T0: ::planus::WriteAs<::planus::Offset<str>>> ::planus::WriteAsOffset<ArraySpec>
        for ArraySpecBuilder<(T0,)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<ArraySpec> {
            let (v0,) = &self.0;
            ArraySpec::create(builder, v0)
        }
    }

    /// Reference to a deserialized [ArraySpec].
    #[derive(Copy, Clone)]
    pub struct ArraySpecRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> ArraySpecRef<'a> {
        /// Getter for the [`id` field](ArraySpec#structfield.id).
        #[inline]
        pub fn id(&self) -> ::planus::Result<&'a ::core::primitive::str> {
            self.0.access_required(0, "ArraySpec", "id")
        }
    }

    impl<'a> ::core::fmt::Debug for ArraySpecRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("ArraySpecRef");
            f.field("id", &self.id());
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<ArraySpecRef<'a>> for ArraySpec {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: ArraySpecRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                id: ::core::convert::Into::into(value.id()?),
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for ArraySpecRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for ArraySpecRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[ArraySpecRef]", "get", buffer.offset_from_start)
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<ArraySpec>> for ArraySpec {
        type Value = ::planus::Offset<ArraySpec>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<ArraySpec>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for ArraySpecRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[ArraySpecRef]", "read_as_root", 0)
            })
        }
    }

    ///  A `LayoutSpec` describes the type of a particular layout.
    ///
    ///  These are identified by a globally unique string identifier, and looked up in the Vortex registry
    ///  at read-time.
    ///
    /// Generated from these locations:
    /// * Table `LayoutSpec` in the file `flatbuffers/vortex-file/footer.fbs:80`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct LayoutSpec {
        /// The field `id` in the table `LayoutSpec`
        pub id: ::planus::alloc::string::String,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for LayoutSpec {
        fn default() -> Self {
            Self {
                id: ::core::default::Default::default(),
            }
        }
    }

    impl LayoutSpec {
        /// Creates a [LayoutSpecBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> LayoutSpecBuilder<()> {
            LayoutSpecBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(
            builder: &mut ::planus::Builder,
            field_id: impl ::planus::WriteAs<::planus::Offset<str>>,
        ) -> ::planus::Offset<Self> {
            let prepared_id = field_id.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                ::core::default::Default::default();
            table_writer.write_entry::<::planus::Offset<str>>(0);

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    object_writer.write::<_, _, 4>(&prepared_id);
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<LayoutSpec>> for LayoutSpec {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<LayoutSpec> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<LayoutSpec>> for LayoutSpec {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<LayoutSpec>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<LayoutSpec> for LayoutSpec {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<LayoutSpec> {
            LayoutSpec::create(builder, &self.id)
        }
    }

    /// Builder for serializing an instance of the [LayoutSpec] type.
    ///
    /// Can be created using the [LayoutSpec::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct LayoutSpecBuilder<State>(State);

    impl LayoutSpecBuilder<()> {
        /// Setter for the [`id` field](LayoutSpec#structfield.id).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn id<T0>(self, value: T0) -> LayoutSpecBuilder<(T0,)>
        where
            T0: ::planus::WriteAs<::planus::Offset<str>>,
        {
            LayoutSpecBuilder((value,))
        }
    }

    impl<T0> LayoutSpecBuilder<(T0,)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [LayoutSpec].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<LayoutSpec>
        where
            Self: ::planus::WriteAsOffset<LayoutSpec>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<T0: ::planus::WriteAs<::planus::Offset<str>>>
        ::planus::WriteAs<::planus::Offset<LayoutSpec>> for LayoutSpecBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<LayoutSpec>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<LayoutSpec> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<T0: ::planus::WriteAs<::planus::Offset<str>>>
        ::planus::WriteAsOptional<::planus::Offset<LayoutSpec>> for LayoutSpecBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<LayoutSpec>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<LayoutSpec>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<T0: ::planus::WriteAs<::planus::Offset<str>>> ::planus::WriteAsOffset<LayoutSpec>
        for LayoutSpecBuilder<(T0,)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<LayoutSpec> {
            let (v0,) = &self.0;
            LayoutSpec::create(builder, v0)
        }
    }

    /// Reference to a deserialized [LayoutSpec].
    #[derive(Copy, Clone)]
    pub struct LayoutSpecRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> LayoutSpecRef<'a> {
        /// Getter for the [`id` field](LayoutSpec#structfield.id).
        #[inline]
        pub fn id(&self) -> ::planus::Result<&'a ::core::primitive::str> {
            self.0.access_required(0, "LayoutSpec", "id")
        }
    }

    impl<'a> ::core::fmt::Debug for LayoutSpecRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("LayoutSpecRef");
            f.field("id", &self.id());
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<LayoutSpecRef<'a>> for LayoutSpec {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: LayoutSpecRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                id: ::core::convert::Into::into(value.id()?),
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for LayoutSpecRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for LayoutSpecRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[LayoutSpecRef]", "get", buffer.offset_from_start)
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<LayoutSpec>> for LayoutSpec {
        type Value = ::planus::Offset<LayoutSpec>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<LayoutSpec>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for LayoutSpecRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[LayoutSpecRef]", "read_as_root", 0)
            })
        }
    }

    ///  A `SegmentSpec` acts as the locator for a buffer within the file.
    ///
    /// Generated from these locations:
    /// * Struct `SegmentSpec` in the file `flatbuffers/vortex-file/footer.fbs:85`
    #[derive(
        Copy,
        Clone,
        Debug,
        PartialEq,
        PartialOrd,
        Eq,
        Ord,
        Hash,
        Default,
        ::serde::Serialize,
        ::serde::Deserialize,
    )]
    pub struct SegmentSpec {
        ///  Offset relative to the start of the file.
        pub offset: u64,

        ///  Length in bytes of the segment.
        pub length: u32,

        ///  Base-2 exponent of the alignment of the segment.
        pub alignment_exponent: u8,

        /// The field `_compression` in the struct `SegmentSpec`
        pub compression: u8,

        /// The field `_encryption` in the struct `SegmentSpec`
        pub encryption: u16,
    }

    /// # Safety
    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
    unsafe impl ::planus::Primitive for SegmentSpec {
        const ALIGNMENT: usize = 8;
        const SIZE: usize = 16;
    }

    #[allow(clippy::identity_op)]
    impl ::planus::WriteAsPrimitive<SegmentSpec> for SegmentSpec {
        #[inline]
        fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
            let (cur, cursor) = cursor.split::<8, 8>();
            self.offset.write(cur, buffer_position - 0);
            let (cur, cursor) = cursor.split::<4, 4>();
            self.length.write(cur, buffer_position - 8);
            let (cur, cursor) = cursor.split::<1, 3>();
            self.alignment_exponent.write(cur, buffer_position - 12);
            let (cur, cursor) = cursor.split::<1, 2>();
            self.compression.write(cur, buffer_position - 13);
            let (cur, cursor) = cursor.split::<2, 0>();
            self.encryption.write(cur, buffer_position - 14);
            cursor.finish([]);
        }
    }

    impl ::planus::WriteAsOffset<SegmentSpec> for SegmentSpec {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<SegmentSpec> {
            unsafe {
                builder.write_with(16, 7, |buffer_position, bytes| {
                    let bytes = bytes.as_mut_ptr();

                    ::planus::WriteAsPrimitive::write(
                        self,
                        ::planus::Cursor::new(
                            &mut *(bytes as *mut [::core::mem::MaybeUninit<u8>; 16]),
                        ),
                        buffer_position,
                    );
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<SegmentSpec> for SegmentSpec {
        type Prepared = Self;
        #[inline]
        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
            *self
        }
    }

    impl ::planus::WriteAsOptional<SegmentSpec> for SegmentSpec {
        type Prepared = Self;
        #[inline]
        fn prepare(&self, _builder: &mut ::planus::Builder) -> ::core::option::Option<Self> {
            ::core::option::Option::Some(*self)
        }
    }

    /// Reference to a deserialized [SegmentSpec].
    #[derive(Copy, Clone)]
    pub struct SegmentSpecRef<'a>(::planus::ArrayWithStartOffset<'a, 16>);

    impl<'a> SegmentSpecRef<'a> {
        /// Getter for the [`offset` field](SegmentSpec#structfield.offset).
        pub fn offset(&self) -> u64 {
            let buffer = self.0.advance_as_array::<8>(0).unwrap();

            u64::from_le_bytes(*buffer.as_array())
        }

        /// Getter for the [`length` field](SegmentSpec#structfield.length).
        pub fn length(&self) -> u32 {
            let buffer = self.0.advance_as_array::<4>(8).unwrap();

            u32::from_le_bytes(*buffer.as_array())
        }

        /// Getter for the [`alignment_exponent` field](SegmentSpec#structfield.alignment_exponent).
        pub fn alignment_exponent(&self) -> u8 {
            let buffer = self.0.advance_as_array::<1>(12).unwrap();

            u8::from_le_bytes(*buffer.as_array())
        }

        /// Getter for the [`_compression` field](SegmentSpec#structfield.compression).
        pub fn compression(&self) -> u8 {
            let buffer = self.0.advance_as_array::<1>(13).unwrap();

            u8::from_le_bytes(*buffer.as_array())
        }

        /// Getter for the [`_encryption` field](SegmentSpec#structfield.encryption).
        pub fn encryption(&self) -> u16 {
            let buffer = self.0.advance_as_array::<2>(14).unwrap();

            u16::from_le_bytes(*buffer.as_array())
        }
    }

    impl<'a> ::core::fmt::Debug for SegmentSpecRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("SegmentSpecRef");
            f.field("offset", &self.offset());
            f.field("length", &self.length());
            f.field("alignment_exponent", &self.alignment_exponent());
            f.field("compression", &self.compression());
            f.field("encryption", &self.encryption());
            f.finish()
        }
    }

    impl<'a> ::core::convert::From<::planus::ArrayWithStartOffset<'a, 16>> for SegmentSpecRef<'a> {
        fn from(array: ::planus::ArrayWithStartOffset<'a, 16>) -> Self {
            Self(array)
        }
    }

    impl<'a> ::core::convert::From<SegmentSpecRef<'a>> for SegmentSpec {
        #[allow(unreachable_code)]
        fn from(value: SegmentSpecRef<'a>) -> Self {
            Self {
                offset: value.offset(),
                length: value.length(),
                alignment_exponent: value.alignment_exponent(),
                compression: value.compression(),
                encryption: value.encryption(),
            }
        }
    }

    impl<'a, 'b> ::core::cmp::PartialEq<SegmentSpecRef<'a>> for SegmentSpecRef<'b> {
        fn eq(&self, other: &SegmentSpecRef<'_>) -> bool {
            self.offset() == other.offset()
                && self.length() == other.length()
                && self.alignment_exponent() == other.alignment_exponent()
                && self.compression() == other.compression()
                && self.encryption() == other.encryption()
        }
    }

    impl<'a> ::core::cmp::Eq for SegmentSpecRef<'a> {}
    impl<'a, 'b> ::core::cmp::PartialOrd<SegmentSpecRef<'a>> for SegmentSpecRef<'b> {
        fn partial_cmp(
            &self,
            other: &SegmentSpecRef<'_>,
        ) -> ::core::option::Option<::core::cmp::Ordering> {
            ::core::option::Option::Some(::core::cmp::Ord::cmp(self, other))
        }
    }

    impl<'a> ::core::cmp::Ord for SegmentSpecRef<'a> {
        fn cmp(&self, other: &SegmentSpecRef<'_>) -> ::core::cmp::Ordering {
            self.offset()
                .cmp(&other.offset())
                .then_with(|| self.length().cmp(&other.length()))
                .then_with(|| self.alignment_exponent().cmp(&other.alignment_exponent()))
                .then_with(|| self.compression().cmp(&other.compression()))
                .then_with(|| self.encryption().cmp(&other.encryption()))
        }
    }

    impl<'a> ::core::hash::Hash for SegmentSpecRef<'a> {
        fn hash<H: ::core::hash::Hasher>(&self, state: &mut H) {
            self.offset().hash(state);
            self.length().hash(state);
            self.alignment_exponent().hash(state);
            self.compression().hash(state);
            self.encryption().hash(state);
        }
    }

    impl<'a> ::planus::TableRead<'a> for SegmentSpecRef<'a> {
        #[inline]
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            let buffer = buffer.advance_as_array::<16>(offset)?;
            ::core::result::Result::Ok(Self(buffer))
        }
    }

    impl<'a> ::planus::VectorRead<'a> for SegmentSpecRef<'a> {
        const STRIDE: usize = 16;

        #[inline]
        unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> Self {
            Self(unsafe { buffer.unchecked_advance_as_array(offset) })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<SegmentSpec> for SegmentSpec {
        const STRIDE: usize = 16;

        type Value = SegmentSpec;

        #[inline]
        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
            *self
        }

        #[inline]
        unsafe fn write_values(
            values: &[SegmentSpec],
            bytes: *mut ::core::mem::MaybeUninit<u8>,
            buffer_position: u32,
        ) {
            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 16];
            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                ::planus::WriteAsPrimitive::write(
                    v,
                    ::planus::Cursor::new(unsafe { &mut *bytes.add(i) }),
                    buffer_position - (16 * i) as u32,
                );
            }
        }
    }

    /// The enum `CompressionScheme`
    ///
    /// Generated from these locations:
    /// * Enum `CompressionScheme` in the file `flatbuffers/vortex-file/footer.fbs:99`
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
    pub enum CompressionScheme {
        /// The variant `None` in the enum `CompressionScheme`
        None = 0,

        /// The variant `LZ4` in the enum `CompressionScheme`
        Lz4 = 1,

        /// The variant `ZLib` in the enum `CompressionScheme`
        ZLib = 2,

        /// The variant `ZStd` in the enum `CompressionScheme`
        ZStd = 3,
    }

    impl CompressionScheme {
        /// Array containing all valid variants of CompressionScheme
        pub const ENUM_VALUES: [Self; 4] = [Self::None, Self::Lz4, Self::ZLib, Self::ZStd];
    }

    impl ::core::convert::TryFrom<u8> for CompressionScheme {
        type Error = ::planus::errors::UnknownEnumTagKind;
        #[inline]
        fn try_from(
            value: u8,
        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind> {
            #[allow(clippy::match_single_binding)]
            match value {
                0 => ::core::result::Result::Ok(CompressionScheme::None),
                1 => ::core::result::Result::Ok(CompressionScheme::Lz4),
                2 => ::core::result::Result::Ok(CompressionScheme::ZLib),
                3 => ::core::result::Result::Ok(CompressionScheme::ZStd),

                _ => ::core::result::Result::Err(::planus::errors::UnknownEnumTagKind {
                    tag: value as i128,
                }),
            }
        }
    }

    impl ::core::convert::From<CompressionScheme> for u8 {
        #[inline]
        fn from(value: CompressionScheme) -> Self {
            value as u8
        }
    }

    /// # Safety
    /// The Planus compiler correctly calculates `ALIGNMENT` and `SIZE`.
    unsafe impl ::planus::Primitive for CompressionScheme {
        const ALIGNMENT: usize = 1;
        const SIZE: usize = 1;
    }

    impl ::planus::WriteAsPrimitive<CompressionScheme> for CompressionScheme {
        #[inline]
        fn write<const N: usize>(&self, cursor: ::planus::Cursor<'_, N>, buffer_position: u32) {
            (*self as u8).write(cursor, buffer_position);
        }
    }

    impl ::planus::WriteAs<CompressionScheme> for CompressionScheme {
        type Prepared = Self;

        #[inline]
        fn prepare(&self, _builder: &mut ::planus::Builder) -> CompressionScheme {
            *self
        }
    }

    impl ::planus::WriteAsDefault<CompressionScheme, CompressionScheme> for CompressionScheme {
        type Prepared = Self;

        #[inline]
        fn prepare(
            &self,
            _builder: &mut ::planus::Builder,
            default: &CompressionScheme,
        ) -> ::core::option::Option<CompressionScheme> {
            if self == default {
                ::core::option::Option::None
            } else {
                ::core::option::Option::Some(*self)
            }
        }
    }

    impl ::planus::WriteAsOptional<CompressionScheme> for CompressionScheme {
        type Prepared = Self;

        #[inline]
        fn prepare(
            &self,
            _builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<CompressionScheme> {
            ::core::option::Option::Some(*self)
        }
    }

    impl<'buf> ::planus::TableRead<'buf> for CompressionScheme {
        #[inline]
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'buf>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            let n: u8 = ::planus::TableRead::from_buffer(buffer, offset)?;
            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
        }
    }

    impl<'buf> ::planus::VectorReadInner<'buf> for CompressionScheme {
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
                    "CompressionScheme",
                    "VectorRead::from_buffer",
                    buffer.offset_from_start,
                )
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<CompressionScheme> for CompressionScheme {
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

    ///  Definition of a compression scheme.
    ///
    /// Generated from these locations:
    /// * Table `CompressionSpec` in the file `flatbuffers/vortex-file/footer.fbs:107`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct CompressionSpec {
        /// The field `scheme` in the table `CompressionSpec`
        pub scheme: self::CompressionScheme,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for CompressionSpec {
        fn default() -> Self {
            Self {
                scheme: self::CompressionScheme::None,
            }
        }
    }

    impl CompressionSpec {
        /// Creates a [CompressionSpecBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> CompressionSpecBuilder<()> {
            CompressionSpecBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(
            builder: &mut ::planus::Builder,
            field_scheme: impl ::planus::WriteAsDefault<
                self::CompressionScheme,
                self::CompressionScheme,
            >,
        ) -> ::planus::Offset<Self> {
            let prepared_scheme = field_scheme.prepare(builder, &self::CompressionScheme::None);

            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                ::core::default::Default::default();
            if prepared_scheme.is_some() {
                table_writer.write_entry::<self::CompressionScheme>(0);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_scheme) = prepared_scheme {
                        object_writer.write::<_, _, 1>(&prepared_scheme);
                    }
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<CompressionSpec>> for CompressionSpec {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<CompressionSpec> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<CompressionSpec>> for CompressionSpec {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<CompressionSpec>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<CompressionSpec> for CompressionSpec {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<CompressionSpec> {
            CompressionSpec::create(builder, self.scheme)
        }
    }

    /// Builder for serializing an instance of the [CompressionSpec] type.
    ///
    /// Can be created using the [CompressionSpec::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct CompressionSpecBuilder<State>(State);

    impl CompressionSpecBuilder<()> {
        /// Setter for the [`scheme` field](CompressionSpec#structfield.scheme).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn scheme<T0>(self, value: T0) -> CompressionSpecBuilder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<self::CompressionScheme, self::CompressionScheme>,
        {
            CompressionSpecBuilder((value,))
        }

        /// Sets the [`scheme` field](CompressionSpec#structfield.scheme) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn scheme_as_default(self) -> CompressionSpecBuilder<(::planus::DefaultValue,)> {
            self.scheme(::planus::DefaultValue)
        }
    }

    impl<T0> CompressionSpecBuilder<(T0,)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [CompressionSpec].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<CompressionSpec>
        where
            Self: ::planus::WriteAsOffset<CompressionSpec>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<T0: ::planus::WriteAsDefault<self::CompressionScheme, self::CompressionScheme>>
        ::planus::WriteAs<::planus::Offset<CompressionSpec>> for CompressionSpecBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<CompressionSpec>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<CompressionSpec> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<T0: ::planus::WriteAsDefault<self::CompressionScheme, self::CompressionScheme>>
        ::planus::WriteAsOptional<::planus::Offset<CompressionSpec>>
        for CompressionSpecBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<CompressionSpec>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<CompressionSpec>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<T0: ::planus::WriteAsDefault<self::CompressionScheme, self::CompressionScheme>>
        ::planus::WriteAsOffset<CompressionSpec> for CompressionSpecBuilder<(T0,)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<CompressionSpec> {
            let (v0,) = &self.0;
            CompressionSpec::create(builder, v0)
        }
    }

    /// Reference to a deserialized [CompressionSpec].
    #[derive(Copy, Clone)]
    pub struct CompressionSpecRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> CompressionSpecRef<'a> {
        /// Getter for the [`scheme` field](CompressionSpec#structfield.scheme).
        #[inline]
        pub fn scheme(&self) -> ::planus::Result<self::CompressionScheme> {
            ::core::result::Result::Ok(
                self.0
                    .access(0, "CompressionSpec", "scheme")?
                    .unwrap_or(self::CompressionScheme::None),
            )
        }
    }

    impl<'a> ::core::fmt::Debug for CompressionSpecRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("CompressionSpecRef");
            f.field("scheme", &self.scheme());
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<CompressionSpecRef<'a>> for CompressionSpec {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: CompressionSpecRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                scheme: ::core::convert::TryInto::try_into(value.scheme()?)?,
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for CompressionSpecRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for CompressionSpecRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location(
                    "[CompressionSpecRef]",
                    "get",
                    buffer.offset_from_start,
                )
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<CompressionSpec>> for CompressionSpec {
        type Value = ::planus::Offset<CompressionSpec>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<CompressionSpec>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for CompressionSpecRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[CompressionSpecRef]", "read_as_root", 0)
            })
        }
    }

    /// The table `EncryptionSpec`
    ///
    /// Generated from these locations:
    /// * Table `EncryptionSpec` in the file `flatbuffers/vortex-file/footer.fbs:111`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct EncryptionSpec {}

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for EncryptionSpec {
        fn default() -> Self {
            Self {}
        }
    }

    impl EncryptionSpec {
        /// Creates a [EncryptionSpecBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> EncryptionSpecBuilder<()> {
            EncryptionSpecBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
            let table_writer: ::planus::table_writer::TableWriter<4> =
                ::core::default::Default::default();
            unsafe {
                table_writer.finish(builder, |_table_writer| {});
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<EncryptionSpec>> for EncryptionSpec {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<EncryptionSpec> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<EncryptionSpec>> for EncryptionSpec {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<EncryptionSpec>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<EncryptionSpec> for EncryptionSpec {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<EncryptionSpec> {
            EncryptionSpec::create(builder)
        }
    }

    /// Builder for serializing an instance of the [EncryptionSpec] type.
    ///
    /// Can be created using the [EncryptionSpec::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct EncryptionSpecBuilder<State>(State);

    impl EncryptionSpecBuilder<()> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [EncryptionSpec].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<EncryptionSpec>
        where
            Self: ::planus::WriteAsOffset<EncryptionSpec>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl ::planus::WriteAs<::planus::Offset<EncryptionSpec>> for EncryptionSpecBuilder<()> {
        type Prepared = ::planus::Offset<EncryptionSpec>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<EncryptionSpec> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<EncryptionSpec>> for EncryptionSpecBuilder<()> {
        type Prepared = ::planus::Offset<EncryptionSpec>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<EncryptionSpec>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<EncryptionSpec> for EncryptionSpecBuilder<()> {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<EncryptionSpec> {
            EncryptionSpec::create(builder)
        }
    }

    /// Reference to a deserialized [EncryptionSpec].
    #[derive(Copy, Clone)]
    pub struct EncryptionSpecRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> EncryptionSpecRef<'a> {}

    impl<'a> ::core::fmt::Debug for EncryptionSpecRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("EncryptionSpecRef");

            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<EncryptionSpecRef<'a>> for EncryptionSpec {
        type Error = ::planus::Error;

        fn try_from(_value: EncryptionSpecRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {})
        }
    }

    impl<'a> ::planus::TableRead<'a> for EncryptionSpecRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for EncryptionSpecRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location(
                    "[EncryptionSpecRef]",
                    "get",
                    buffer.offset_from_start,
                )
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<EncryptionSpec>> for EncryptionSpec {
        type Value = ::planus::Offset<EncryptionSpec>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<EncryptionSpec>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for EncryptionSpecRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[EncryptionSpecRef]", "read_as_root", 0)
            })
        }
    }

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

    ///  A `Layout` is a recursive data structure describing the physical layout of Vortex arrays in random access storage.
    ///  As a starting, concrete example, the first three Layout encodings are defined as:
    ///
    ///  1. encoding == 1, `Flat` -> one buffer, zero child Layouts
    ///  2. encoding == 2, `Chunked` -> zero buffers, one or more child Layouts (used for chunks of rows)
    ///  3. encoding == 3, `Columnar` -> zero buffers, one or more child Layouts (used for columns of structs)
    ///
    ///  The `row_count` represents the number of rows represented by this Layout. This is very useful for
    ///  pruning the Layout tree based on row filters.
    ///
    ///  The `metadata` field is fully opaque at this layer, and allows the Layout implementation corresponding to
    ///  `encoding` to embed additional information that may be useful for the reader. For example, the `ChunkedLayout`
    ///  uses the first byte of the `metadata` array as a boolean to indicate whether the first child Layout represents
    ///  the statistics table for the other chunks.
    ///
    /// Generated from these locations:
    /// * Table `Layout` in the file `flatbuffers/vortex-layout/layout.fbs:18`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Layout {
        ///  The ID of the encoding used for this Layout.
        pub encoding: u16,
        ///  The number of rows of data represented by this Layout.
        pub row_count: u64,
        ///  Any additional metadata this layout needs to interpret its children.
        ///  This does not include data-specific metadata, which the layout should store in a segment.
        pub metadata: ::core::option::Option<::planus::alloc::vec::Vec<u8>>,
        ///  The children of this Layout.
        pub children: ::core::option::Option<::planus::alloc::vec::Vec<self::Layout>>,
        ///  Identifiers for each `SegmentSpec` of data required by this layout.
        pub segments: ::core::option::Option<::planus::alloc::vec::Vec<u32>>,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for Layout {
        fn default() -> Self {
            Self {
                encoding: 0,
                row_count: 0,
                metadata: ::core::default::Default::default(),
                children: ::core::default::Default::default(),
                segments: ::core::default::Default::default(),
            }
        }
    }

    impl Layout {
        /// Creates a [LayoutBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> LayoutBuilder<()> {
            LayoutBuilder(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn create(
            builder: &mut ::planus::Builder,
            field_encoding: impl ::planus::WriteAsDefault<u16, u16>,
            field_row_count: impl ::planus::WriteAsDefault<u64, u64>,
            field_metadata: impl ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
            field_children: impl ::planus::WriteAsOptional<
                ::planus::Offset<[::planus::Offset<self::Layout>]>,
            >,
            field_segments: impl ::planus::WriteAsOptional<::planus::Offset<[u32]>>,
        ) -> ::planus::Offset<Self> {
            let prepared_encoding = field_encoding.prepare(builder, &0);
            let prepared_row_count = field_row_count.prepare(builder, &0);
            let prepared_metadata = field_metadata.prepare(builder);
            let prepared_children = field_children.prepare(builder);
            let prepared_segments = field_segments.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<14> =
                ::core::default::Default::default();
            if prepared_row_count.is_some() {
                table_writer.write_entry::<u64>(1);
            }
            if prepared_metadata.is_some() {
                table_writer.write_entry::<::planus::Offset<[u8]>>(2);
            }
            if prepared_children.is_some() {
                table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::Layout>]>>(3);
            }
            if prepared_segments.is_some() {
                table_writer.write_entry::<::planus::Offset<[u32]>>(4);
            }
            if prepared_encoding.is_some() {
                table_writer.write_entry::<u16>(0);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_row_count) = prepared_row_count {
                        object_writer.write::<_, _, 8>(&prepared_row_count);
                    }
                    if let ::core::option::Option::Some(prepared_metadata) = prepared_metadata {
                        object_writer.write::<_, _, 4>(&prepared_metadata);
                    }
                    if let ::core::option::Option::Some(prepared_children) = prepared_children {
                        object_writer.write::<_, _, 4>(&prepared_children);
                    }
                    if let ::core::option::Option::Some(prepared_segments) = prepared_segments {
                        object_writer.write::<_, _, 4>(&prepared_segments);
                    }
                    if let ::core::option::Option::Some(prepared_encoding) = prepared_encoding {
                        object_writer.write::<_, _, 2>(&prepared_encoding);
                    }
                });
            }
            builder.current_offset()
        }
    }

    impl ::planus::WriteAs<::planus::Offset<Layout>> for Layout {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Layout> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<Layout>> for Layout {
        type Prepared = ::planus::Offset<Self>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Layout>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Layout> for Layout {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Layout> {
            Layout::create(
                builder,
                self.encoding,
                self.row_count,
                &self.metadata,
                &self.children,
                &self.segments,
            )
        }
    }

    /// Builder for serializing an instance of the [Layout] type.
    ///
    /// Can be created using the [Layout::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct LayoutBuilder<State>(State);

    impl LayoutBuilder<()> {
        /// Setter for the [`encoding` field](Layout#structfield.encoding).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn encoding<T0>(self, value: T0) -> LayoutBuilder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<u16, u16>,
        {
            LayoutBuilder((value,))
        }

        /// Sets the [`encoding` field](Layout#structfield.encoding) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn encoding_as_default(self) -> LayoutBuilder<(::planus::DefaultValue,)> {
            self.encoding(::planus::DefaultValue)
        }
    }

    impl<T0> LayoutBuilder<(T0,)> {
        /// Setter for the [`row_count` field](Layout#structfield.row_count).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn row_count<T1>(self, value: T1) -> LayoutBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsDefault<u64, u64>,
        {
            let (v0,) = self.0;
            LayoutBuilder((v0, value))
        }

        /// Sets the [`row_count` field](Layout#structfield.row_count) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn row_count_as_default(self) -> LayoutBuilder<(T0, ::planus::DefaultValue)> {
            self.row_count(::planus::DefaultValue)
        }
    }

    impl<T0, T1> LayoutBuilder<(T0, T1)> {
        /// Setter for the [`metadata` field](Layout#structfield.metadata).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn metadata<T2>(self, value: T2) -> LayoutBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        {
            let (v0, v1) = self.0;
            LayoutBuilder((v0, v1, value))
        }

        /// Sets the [`metadata` field](Layout#structfield.metadata) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn metadata_as_null(self) -> LayoutBuilder<(T0, T1, ())> {
            self.metadata(())
        }
    }

    impl<T0, T1, T2> LayoutBuilder<(T0, T1, T2)> {
        /// Setter for the [`children` field](Layout#structfield.children).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn children<T3>(self, value: T3) -> LayoutBuilder<(T0, T1, T2, T3)>
        where
            T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::Layout>]>>,
        {
            let (v0, v1, v2) = self.0;
            LayoutBuilder((v0, v1, v2, value))
        }

        /// Sets the [`children` field](Layout#structfield.children) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn children_as_null(self) -> LayoutBuilder<(T0, T1, T2, ())> {
            self.children(())
        }
    }

    impl<T0, T1, T2, T3> LayoutBuilder<(T0, T1, T2, T3)> {
        /// Setter for the [`segments` field](Layout#structfield.segments).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn segments<T4>(self, value: T4) -> LayoutBuilder<(T0, T1, T2, T3, T4)>
        where
            T4: ::planus::WriteAsOptional<::planus::Offset<[u32]>>,
        {
            let (v0, v1, v2, v3) = self.0;
            LayoutBuilder((v0, v1, v2, v3, value))
        }

        /// Sets the [`segments` field](Layout#structfield.segments) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn segments_as_null(self) -> LayoutBuilder<(T0, T1, T2, T3, ())> {
            self.segments(())
        }
    }

    impl<T0, T1, T2, T3, T4> LayoutBuilder<(T0, T1, T2, T3, T4)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Layout].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Layout>
        where
            Self: ::planus::WriteAsOffset<Layout>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u16, u16>,
        T1: ::planus::WriteAsDefault<u64, u64>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::Layout>]>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<[u32]>>,
    > ::planus::WriteAs<::planus::Offset<Layout>> for LayoutBuilder<(T0, T1, T2, T3, T4)>
    {
        type Prepared = ::planus::Offset<Layout>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Layout> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u16, u16>,
        T1: ::planus::WriteAsDefault<u64, u64>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::Layout>]>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<[u32]>>,
    > ::planus::WriteAsOptional<::planus::Offset<Layout>> for LayoutBuilder<(T0, T1, T2, T3, T4)>
    {
        type Prepared = ::planus::Offset<Layout>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Layout>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u16, u16>,
        T1: ::planus::WriteAsDefault<u64, u64>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::Layout>]>>,
        T4: ::planus::WriteAsOptional<::planus::Offset<[u32]>>,
    > ::planus::WriteAsOffset<Layout> for LayoutBuilder<(T0, T1, T2, T3, T4)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Layout> {
            let (v0, v1, v2, v3, v4) = &self.0;
            Layout::create(builder, v0, v1, v2, v3, v4)
        }
    }

    /// Reference to a deserialized [Layout].
    #[derive(Copy, Clone)]
    pub struct LayoutRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> LayoutRef<'a> {
        /// Getter for the [`encoding` field](Layout#structfield.encoding).
        #[inline]
        pub fn encoding(&self) -> ::planus::Result<u16> {
            ::core::result::Result::Ok(self.0.access(0, "Layout", "encoding")?.unwrap_or(0))
        }

        /// Getter for the [`row_count` field](Layout#structfield.row_count).
        #[inline]
        pub fn row_count(&self) -> ::planus::Result<u64> {
            ::core::result::Result::Ok(self.0.access(1, "Layout", "row_count")?.unwrap_or(0))
        }

        /// Getter for the [`metadata` field](Layout#structfield.metadata).
        #[inline]
        pub fn metadata(&self) -> ::planus::Result<::core::option::Option<&'a [u8]>> {
            self.0.access(2, "Layout", "metadata")
        }

        /// Getter for the [`children` field](Layout#structfield.children).
        #[inline]
        pub fn children(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<::planus::Vector<'a, ::planus::Result<self::LayoutRef<'a>>>>,
        > {
            self.0.access(3, "Layout", "children")
        }

        /// Getter for the [`segments` field](Layout#structfield.segments).
        #[inline]
        pub fn segments(
            &self,
        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, u32>>> {
            self.0.access(4, "Layout", "segments")
        }
    }

    impl<'a> ::core::fmt::Debug for LayoutRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("LayoutRef");
            f.field("encoding", &self.encoding());
            f.field("row_count", &self.row_count());
            if let ::core::option::Option::Some(field_metadata) = self.metadata().transpose() {
                f.field("metadata", &field_metadata);
            }
            if let ::core::option::Option::Some(field_children) = self.children().transpose() {
                f.field("children", &field_children);
            }
            if let ::core::option::Option::Some(field_segments) = self.segments().transpose() {
                f.field("segments", &field_segments);
            }
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<LayoutRef<'a>> for Layout {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: LayoutRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                encoding: ::core::convert::TryInto::try_into(value.encoding()?)?,
                row_count: ::core::convert::TryInto::try_into(value.row_count()?)?,
                metadata: value.metadata()?.map(|v| v.to_vec()),
                children: if let ::core::option::Option::Some(children) = value.children()? {
                    ::core::option::Option::Some(children.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
                segments: if let ::core::option::Option::Some(segments) = value.segments()? {
                    ::core::option::Option::Some(segments.to_vec()?)
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for LayoutRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for LayoutRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[LayoutRef]", "get", buffer.offset_from_start)
            })
        }
    }

    /// # Safety
    /// The planus compiler generates implementations that initialize
    /// the bytes in `write_values`.
    unsafe impl ::planus::VectorWrite<::planus::Offset<Layout>> for Layout {
        type Value = ::planus::Offset<Layout>;
        const STRIDE: usize = 4;
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
            ::planus::WriteAs::prepare(self, builder)
        }

        #[inline]
        unsafe fn write_values(
            values: &[::planus::Offset<Layout>],
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

    impl<'a> ::planus::ReadAsRoot<'a> for LayoutRef<'a> {
        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[LayoutRef]", "read_as_root", 0))
        }
    }
}

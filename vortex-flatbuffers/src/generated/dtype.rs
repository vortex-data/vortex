pub use root::*;

const _: () = ::planus::check_version_compatibility("planus-1.2.0");

/// The root namespace
///
/// Generated from these locations:
/// * File `flatbuffers/vortex-dtype/dtype.fbs`
#[no_implicit_prelude]
#[allow(dead_code, clippy::needless_lifetimes)]
mod root {
    /// The enum `PType`
    ///
    /// Generated from these locations:
    /// * Enum `PType` in the file `flatbuffers/vortex-dtype/dtype.fbs:4`
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
    pub enum PType {
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
        pub const ENUM_VALUES: [Self; 11] = [
            Self::U8,
            Self::U16,
            Self::U32,
            Self::U64,
            Self::I8,
            Self::I16,
            Self::I32,
            Self::I64,
            Self::F16,
            Self::F32,
            Self::F64,
        ];
    }

    impl ::core::convert::TryFrom<u8> for PType {
        type Error = ::planus::errors::UnknownEnumTagKind;
        #[inline]
        fn try_from(
            value: u8,
        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind> {
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

                _ => ::core::result::Result::Err(::planus::errors::UnknownEnumTagKind {
                    tag: value as i128,
                }),
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
        fn prepare(
            &self,
            _builder: &mut ::planus::Builder,
            default: &PType,
        ) -> ::core::option::Option<PType> {
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
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'buf>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
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
        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag> {
            let value = unsafe { *buffer.buffer.get_unchecked(offset) };
            let value: ::core::result::Result<Self, _> = ::core::convert::TryInto::try_into(value);
            value.map_err(|error_kind| {
                error_kind.with_error_location(
                    "PType",
                    "VectorRead::from_buffer",
                    buffer.offset_from_start,
                )
            })
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
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Null {}

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for Null {
        fn default() -> Self {
            Self {}
        }
    }

    impl Null {
        /// Creates a [NullBuilder] for serializing an instance of this table.
        #[inline]
        pub fn builder() -> NullBuilder<()> {
            NullBuilder(())
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Null>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Null> for Null {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null> {
            Null::create(builder)
        }
    }

    /// Builder for serializing an instance of the [Null] type.
    ///
    /// Can be created using the [Null::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct NullBuilder<State>(State);

    impl NullBuilder<()> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Null].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null>
        where
            Self: ::planus::WriteAsOffset<Null>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl ::planus::WriteAs<::planus::Offset<Null>> for NullBuilder<()> {
        type Prepared = ::planus::Offset<Null>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl ::planus::WriteAsOptional<::planus::Offset<Null>> for NullBuilder<()> {
        type Prepared = ::planus::Offset<Null>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Null>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Null> for NullBuilder<()> {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Null> {
            Null::create(builder)
        }
    }

    /// Reference to a deserialized [Null].
    #[derive(Copy, Clone)]
    pub struct NullRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> NullRef<'a> {}

    impl<'a> ::core::fmt::Debug for NullRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("NullRef");

            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<NullRef<'a>> for Null {
        type Error = ::planus::Error;

        fn try_from(_value: NullRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {})
        }
    }

    impl<'a> ::planus::TableRead<'a> for NullRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for NullRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[NullRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[NullRef]", "read_as_root", 0))
        }
    }

    /// The table `Bool`
    ///
    /// Generated from these locations:
    /// * Table `Bool` in the file `flatbuffers/vortex-dtype/dtype.fbs:20`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Bool {
        /// The field `nullable` in the table `Bool`
        pub nullable: bool,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for Bool {
        fn default() -> Self {
            Self { nullable: false }
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
        ) -> ::planus::Offset<Self> {
            let prepared_nullable = field_nullable.prepare(builder, &false);

            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                ::core::default::Default::default();
            if prepared_nullable.is_some() {
                table_writer.write_entry::<bool>(0);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {
                        object_writer.write::<_, _, 1>(&prepared_nullable);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Bool>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Bool> for Bool {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool> {
            Bool::create(builder, self.nullable)
        }
    }

    /// Builder for serializing an instance of the [Bool] type.
    ///
    /// Can be created using the [Bool::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct BoolBuilder<State>(State);

    impl BoolBuilder<()> {
        /// Setter for the [`nullable` field](Bool#structfield.nullable).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable<T0>(self, value: T0) -> BoolBuilder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<bool, bool>,
        {
            BoolBuilder((value,))
        }

        /// Sets the [`nullable` field](Bool#structfield.nullable) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable_as_default(self) -> BoolBuilder<(::planus::DefaultValue,)> {
            self.nullable(::planus::DefaultValue)
        }
    }

    impl<T0> BoolBuilder<(T0,)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Bool].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool>
        where
            Self: ::planus::WriteAsOffset<Bool>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAs<::planus::Offset<Bool>>
        for BoolBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<Bool>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAsOptional<::planus::Offset<Bool>>
        for BoolBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<Bool>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Bool>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAsOffset<Bool>
        for BoolBuilder<(T0,)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Bool> {
            let (v0,) = &self.0;
            Bool::create(builder, v0)
        }
    }

    /// Reference to a deserialized [Bool].
    #[derive(Copy, Clone)]
    pub struct BoolRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> BoolRef<'a> {
        /// Getter for the [`nullable` field](Bool#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
            ::core::result::Result::Ok(self.0.access(0, "Bool", "nullable")?.unwrap_or(false))
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
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                buffer, offset,
            )?))
        }
    }

    impl<'a> ::planus::VectorReadInner<'a> for BoolRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[BoolRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[BoolRef]", "read_as_root", 0))
        }
    }

    /// The table `Primitive`
    ///
    /// Generated from these locations:
    /// * Table `Primitive` in the file `flatbuffers/vortex-dtype/dtype.fbs:24`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Primitive {
        /// The field `ptype` in the table `Primitive`
        pub ptype: self::PType,
        /// The field `nullable` in the table `Primitive`
        pub nullable: bool,
    }

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
        ) -> ::planus::Offset<Self> {
            let prepared_ptype = field_ptype.prepare(builder, &self::PType::U8);
            let prepared_nullable = field_nullable.prepare(builder, &false);

            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                ::core::default::Default::default();
            if prepared_ptype.is_some() {
                table_writer.write_entry::<self::PType>(0);
            }
            if prepared_nullable.is_some() {
                table_writer.write_entry::<bool>(1);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_ptype) = prepared_ptype {
                        object_writer.write::<_, _, 1>(&prepared_ptype);
                    }
                    if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {
                        object_writer.write::<_, _, 1>(&prepared_nullable);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Primitive>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Primitive> for Primitive {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive> {
            Primitive::create(builder, self.ptype, self.nullable)
        }
    }

    /// Builder for serializing an instance of the [Primitive] type.
    ///
    /// Can be created using the [Primitive::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct PrimitiveBuilder<State>(State);

    impl PrimitiveBuilder<()> {
        /// Setter for the [`ptype` field](Primitive#structfield.ptype).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn ptype<T0>(self, value: T0) -> PrimitiveBuilder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<self::PType, self::PType>,
        {
            PrimitiveBuilder((value,))
        }

        /// Sets the [`ptype` field](Primitive#structfield.ptype) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn ptype_as_default(self) -> PrimitiveBuilder<(::planus::DefaultValue,)> {
            self.ptype(::planus::DefaultValue)
        }
    }

    impl<T0> PrimitiveBuilder<(T0,)> {
        /// Setter for the [`nullable` field](Primitive#structfield.nullable).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable<T1>(self, value: T1) -> PrimitiveBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsDefault<bool, bool>,
        {
            let (v0,) = self.0;
            PrimitiveBuilder((v0, value))
        }

        /// Sets the [`nullable` field](Primitive#structfield.nullable) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable_as_default(self) -> PrimitiveBuilder<(T0, ::planus::DefaultValue)> {
            self.nullable(::planus::DefaultValue)
        }
    }

    impl<T0, T1> PrimitiveBuilder<(T0, T1)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Primitive].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive>
        where
            Self: ::planus::WriteAsOffset<Primitive>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<self::PType, self::PType>,
        T1: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAs<::planus::Offset<Primitive>> for PrimitiveBuilder<(T0, T1)>
    {
        type Prepared = ::planus::Offset<Primitive>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<self::PType, self::PType>,
        T1: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAsOptional<::planus::Offset<Primitive>> for PrimitiveBuilder<(T0, T1)>
    {
        type Prepared = ::planus::Offset<Primitive>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Primitive>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<self::PType, self::PType>,
        T1: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAsOffset<Primitive> for PrimitiveBuilder<(T0, T1)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Primitive> {
            let (v0, v1) = &self.0;
            Primitive::create(builder, v0, v1)
        }
    }

    /// Reference to a deserialized [Primitive].
    #[derive(Copy, Clone)]
    pub struct PrimitiveRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> PrimitiveRef<'a> {
        /// Getter for the [`ptype` field](Primitive#structfield.ptype).
        #[inline]
        pub fn ptype(&self) -> ::planus::Result<self::PType> {
            ::core::result::Result::Ok(
                self.0
                    .access(0, "Primitive", "ptype")?
                    .unwrap_or(self::PType::U8),
            )
        }

        /// Getter for the [`nullable` field](Primitive#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
            ::core::result::Result::Ok(self.0.access(1, "Primitive", "nullable")?.unwrap_or(false))
        }
    }

    impl<'a> ::core::fmt::Debug for PrimitiveRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("PrimitiveRef");
            f.field("ptype", &self.ptype());
            f.field("nullable", &self.nullable());
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<PrimitiveRef<'a>> for Primitive {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: PrimitiveRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                ptype: ::core::convert::TryInto::try_into(value.ptype()?)?,
                nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for PrimitiveRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for PrimitiveRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[PrimitiveRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[PrimitiveRef]", "read_as_root", 0)
            })
        }
    }

    /// The table `Decimal`
    ///
    /// Generated from these locations:
    /// * Table `Decimal` in the file `flatbuffers/vortex-dtype/dtype.fbs:29`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Decimal {
        /// The field `precision` in the table `Decimal`
        pub precision: u8,
        /// The field `scale` in the table `Decimal`
        pub scale: i8,
        /// The field `nullable` in the table `Decimal`
        pub nullable: bool,
    }

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
        ) -> ::planus::Offset<Self> {
            let prepared_precision = field_precision.prepare(builder, &0);
            let prepared_scale = field_scale.prepare(builder, &0);
            let prepared_nullable = field_nullable.prepare(builder, &false);

            let mut table_writer: ::planus::table_writer::TableWriter<10> =
                ::core::default::Default::default();
            if prepared_precision.is_some() {
                table_writer.write_entry::<u8>(0);
            }
            if prepared_scale.is_some() {
                table_writer.write_entry::<i8>(1);
            }
            if prepared_nullable.is_some() {
                table_writer.write_entry::<bool>(2);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_precision) = prepared_precision {
                        object_writer.write::<_, _, 1>(&prepared_precision);
                    }
                    if let ::core::option::Option::Some(prepared_scale) = prepared_scale {
                        object_writer.write::<_, _, 1>(&prepared_scale);
                    }
                    if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {
                        object_writer.write::<_, _, 1>(&prepared_nullable);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Decimal>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Decimal> for Decimal {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Decimal> {
            Decimal::create(builder, self.precision, self.scale, self.nullable)
        }
    }

    /// Builder for serializing an instance of the [Decimal] type.
    ///
    /// Can be created using the [Decimal::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct DecimalBuilder<State>(State);

    impl DecimalBuilder<()> {
        /// Setter for the [`precision` field](Decimal#structfield.precision).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn precision<T0>(self, value: T0) -> DecimalBuilder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<u8, u8>,
        {
            DecimalBuilder((value,))
        }

        /// Sets the [`precision` field](Decimal#structfield.precision) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn precision_as_default(self) -> DecimalBuilder<(::planus::DefaultValue,)> {
            self.precision(::planus::DefaultValue)
        }
    }

    impl<T0> DecimalBuilder<(T0,)> {
        /// Setter for the [`scale` field](Decimal#structfield.scale).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn scale<T1>(self, value: T1) -> DecimalBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsDefault<i8, i8>,
        {
            let (v0,) = self.0;
            DecimalBuilder((v0, value))
        }

        /// Sets the [`scale` field](Decimal#structfield.scale) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn scale_as_default(self) -> DecimalBuilder<(T0, ::planus::DefaultValue)> {
            self.scale(::planus::DefaultValue)
        }
    }

    impl<T0, T1> DecimalBuilder<(T0, T1)> {
        /// Setter for the [`nullable` field](Decimal#structfield.nullable).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable<T2>(self, value: T2) -> DecimalBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsDefault<bool, bool>,
        {
            let (v0, v1) = self.0;
            DecimalBuilder((v0, v1, value))
        }

        /// Sets the [`nullable` field](Decimal#structfield.nullable) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable_as_default(self) -> DecimalBuilder<(T0, T1, ::planus::DefaultValue)> {
            self.nullable(::planus::DefaultValue)
        }
    }

    impl<T0, T1, T2> DecimalBuilder<(T0, T1, T2)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Decimal].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Decimal>
        where
            Self: ::planus::WriteAsOffset<Decimal>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u8, u8>,
        T1: ::planus::WriteAsDefault<i8, i8>,
        T2: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAs<::planus::Offset<Decimal>> for DecimalBuilder<(T0, T1, T2)>
    {
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
    > ::planus::WriteAsOptional<::planus::Offset<Decimal>> for DecimalBuilder<(T0, T1, T2)>
    {
        type Prepared = ::planus::Offset<Decimal>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Decimal>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsDefault<u8, u8>,
        T1: ::planus::WriteAsDefault<i8, i8>,
        T2: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAsOffset<Decimal> for DecimalBuilder<(T0, T1, T2)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Decimal> {
            let (v0, v1, v2) = &self.0;
            Decimal::create(builder, v0, v1, v2)
        }
    }

    /// Reference to a deserialized [Decimal].
    #[derive(Copy, Clone)]
    pub struct DecimalRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> DecimalRef<'a> {
        /// Getter for the [`precision` field](Decimal#structfield.precision).
        #[inline]
        pub fn precision(&self) -> ::planus::Result<u8> {
            ::core::result::Result::Ok(self.0.access(0, "Decimal", "precision")?.unwrap_or(0))
        }

        /// Getter for the [`scale` field](Decimal#structfield.scale).
        #[inline]
        pub fn scale(&self) -> ::planus::Result<i8> {
            ::core::result::Result::Ok(self.0.access(1, "Decimal", "scale")?.unwrap_or(0))
        }

        /// Getter for the [`nullable` field](Decimal#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
            ::core::result::Result::Ok(self.0.access(2, "Decimal", "nullable")?.unwrap_or(false))
        }
    }

    impl<'a> ::core::fmt::Debug for DecimalRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("DecimalRef");
            f.field("precision", &self.precision());
            f.field("scale", &self.scale());
            f.field("nullable", &self.nullable());
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<DecimalRef<'a>> for Decimal {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: DecimalRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                precision: ::core::convert::TryInto::try_into(value.precision()?)?,
                scale: ::core::convert::TryInto::try_into(value.scale()?)?,
                nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for DecimalRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for DecimalRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[DecimalRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[DecimalRef]", "read_as_root", 0))
        }
    }

    /// The table `Utf8`
    ///
    /// Generated from these locations:
    /// * Table `Utf8` in the file `flatbuffers/vortex-dtype/dtype.fbs:35`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Utf8 {
        /// The field `nullable` in the table `Utf8`
        pub nullable: bool,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for Utf8 {
        fn default() -> Self {
            Self { nullable: false }
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
        ) -> ::planus::Offset<Self> {
            let prepared_nullable = field_nullable.prepare(builder, &false);

            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                ::core::default::Default::default();
            if prepared_nullable.is_some() {
                table_writer.write_entry::<bool>(0);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {
                        object_writer.write::<_, _, 1>(&prepared_nullable);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Utf8>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Utf8> for Utf8 {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8> {
            Utf8::create(builder, self.nullable)
        }
    }

    /// Builder for serializing an instance of the [Utf8] type.
    ///
    /// Can be created using the [Utf8::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct Utf8Builder<State>(State);

    impl Utf8Builder<()> {
        /// Setter for the [`nullable` field](Utf8#structfield.nullable).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable<T0>(self, value: T0) -> Utf8Builder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<bool, bool>,
        {
            Utf8Builder((value,))
        }

        /// Sets the [`nullable` field](Utf8#structfield.nullable) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable_as_default(self) -> Utf8Builder<(::planus::DefaultValue,)> {
            self.nullable(::planus::DefaultValue)
        }
    }

    impl<T0> Utf8Builder<(T0,)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Utf8].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8>
        where
            Self: ::planus::WriteAsOffset<Utf8>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAs<::planus::Offset<Utf8>>
        for Utf8Builder<(T0,)>
    {
        type Prepared = ::planus::Offset<Utf8>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAsOptional<::planus::Offset<Utf8>>
        for Utf8Builder<(T0,)>
    {
        type Prepared = ::planus::Offset<Utf8>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Utf8>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAsOffset<Utf8>
        for Utf8Builder<(T0,)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Utf8> {
            let (v0,) = &self.0;
            Utf8::create(builder, v0)
        }
    }

    /// Reference to a deserialized [Utf8].
    #[derive(Copy, Clone)]
    pub struct Utf8Ref<'a>(::planus::table_reader::Table<'a>);

    impl<'a> Utf8Ref<'a> {
        /// Getter for the [`nullable` field](Utf8#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
            ::core::result::Result::Ok(self.0.access(0, "Utf8", "nullable")?.unwrap_or(false))
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
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                buffer, offset,
            )?))
        }
    }

    impl<'a> ::planus::VectorReadInner<'a> for Utf8Ref<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[Utf8Ref]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[Utf8Ref]", "read_as_root", 0))
        }
    }

    /// The table `Binary`
    ///
    /// Generated from these locations:
    /// * Table `Binary` in the file `flatbuffers/vortex-dtype/dtype.fbs:39`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Binary {
        /// The field `nullable` in the table `Binary`
        pub nullable: bool,
    }

    #[allow(clippy::derivable_impls)]
    impl ::core::default::Default for Binary {
        fn default() -> Self {
            Self { nullable: false }
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
        ) -> ::planus::Offset<Self> {
            let prepared_nullable = field_nullable.prepare(builder, &false);

            let mut table_writer: ::planus::table_writer::TableWriter<6> =
                ::core::default::Default::default();
            if prepared_nullable.is_some() {
                table_writer.write_entry::<bool>(0);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {
                        object_writer.write::<_, _, 1>(&prepared_nullable);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Binary>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Binary> for Binary {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary> {
            Binary::create(builder, self.nullable)
        }
    }

    /// Builder for serializing an instance of the [Binary] type.
    ///
    /// Can be created using the [Binary::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct BinaryBuilder<State>(State);

    impl BinaryBuilder<()> {
        /// Setter for the [`nullable` field](Binary#structfield.nullable).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable<T0>(self, value: T0) -> BinaryBuilder<(T0,)>
        where
            T0: ::planus::WriteAsDefault<bool, bool>,
        {
            BinaryBuilder((value,))
        }

        /// Sets the [`nullable` field](Binary#structfield.nullable) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable_as_default(self) -> BinaryBuilder<(::planus::DefaultValue,)> {
            self.nullable(::planus::DefaultValue)
        }
    }

    impl<T0> BinaryBuilder<(T0,)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Binary].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary>
        where
            Self: ::planus::WriteAsOffset<Binary>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAs<::planus::Offset<Binary>>
        for BinaryBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<Binary>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>>
        ::planus::WriteAsOptional<::planus::Offset<Binary>> for BinaryBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<Binary>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Binary>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<T0: ::planus::WriteAsDefault<bool, bool>> ::planus::WriteAsOffset<Binary>
        for BinaryBuilder<(T0,)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Binary> {
            let (v0,) = &self.0;
            Binary::create(builder, v0)
        }
    }

    /// Reference to a deserialized [Binary].
    #[derive(Copy, Clone)]
    pub struct BinaryRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> BinaryRef<'a> {
        /// Getter for the [`nullable` field](Binary#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
            ::core::result::Result::Ok(self.0.access(0, "Binary", "nullable")?.unwrap_or(false))
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
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(
                buffer, offset,
            )?))
        }
    }

    impl<'a> ::planus::VectorReadInner<'a> for BinaryRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[BinaryRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[BinaryRef]", "read_as_root", 0))
        }
    }

    /// The table `Struct_`
    ///
    /// Generated from these locations:
    /// * Table `Struct_` in the file `flatbuffers/vortex-dtype/dtype.fbs:43`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Struct {
        /// The field `names` in the table `Struct_`
        pub names:
            ::core::option::Option<::planus::alloc::vec::Vec<::planus::alloc::string::String>>,
        /// The field `dtypes` in the table `Struct_`
        pub dtypes: ::core::option::Option<::planus::alloc::vec::Vec<self::DType>>,
        /// The field `nullable` in the table `Struct_`
        pub nullable: bool,
    }

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
            field_dtypes: impl ::planus::WriteAsOptional<
                ::planus::Offset<[::planus::Offset<self::DType>]>,
            >,
            field_nullable: impl ::planus::WriteAsDefault<bool, bool>,
        ) -> ::planus::Offset<Self> {
            let prepared_names = field_names.prepare(builder);
            let prepared_dtypes = field_dtypes.prepare(builder);
            let prepared_nullable = field_nullable.prepare(builder, &false);

            let mut table_writer: ::planus::table_writer::TableWriter<10> =
                ::core::default::Default::default();
            if prepared_names.is_some() {
                table_writer.write_entry::<::planus::Offset<[::planus::Offset<str>]>>(0);
            }
            if prepared_dtypes.is_some() {
                table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::DType>]>>(1);
            }
            if prepared_nullable.is_some() {
                table_writer.write_entry::<bool>(2);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_names) = prepared_names {
                        object_writer.write::<_, _, 4>(&prepared_names);
                    }
                    if let ::core::option::Option::Some(prepared_dtypes) = prepared_dtypes {
                        object_writer.write::<_, _, 4>(&prepared_dtypes);
                    }
                    if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {
                        object_writer.write::<_, _, 1>(&prepared_nullable);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Struct>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Struct> for Struct {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Struct> {
            Struct::create(builder, &self.names, &self.dtypes, self.nullable)
        }
    }

    /// Builder for serializing an instance of the [Struct] type.
    ///
    /// Can be created using the [Struct::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct StructBuilder<State>(State);

    impl StructBuilder<()> {
        /// Setter for the [`names` field](Struct#structfield.names).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn names<T0>(self, value: T0) -> StructBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
        {
            StructBuilder((value,))
        }

        /// Sets the [`names` field](Struct#structfield.names) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn names_as_null(self) -> StructBuilder<((),)> {
            self.names(())
        }
    }

    impl<T0> StructBuilder<(T0,)> {
        /// Setter for the [`dtypes` field](Struct#structfield.dtypes).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn dtypes<T1>(self, value: T1) -> StructBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::DType>]>>,
        {
            let (v0,) = self.0;
            StructBuilder((v0, value))
        }

        /// Sets the [`dtypes` field](Struct#structfield.dtypes) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn dtypes_as_null(self) -> StructBuilder<(T0, ())> {
            self.dtypes(())
        }
    }

    impl<T0, T1> StructBuilder<(T0, T1)> {
        /// Setter for the [`nullable` field](Struct#structfield.nullable).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable<T2>(self, value: T2) -> StructBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsDefault<bool, bool>,
        {
            let (v0, v1) = self.0;
            StructBuilder((v0, v1, value))
        }

        /// Sets the [`nullable` field](Struct#structfield.nullable) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable_as_default(self) -> StructBuilder<(T0, T1, ::planus::DefaultValue)> {
            self.nullable(::planus::DefaultValue)
        }
    }

    impl<T0, T1, T2> StructBuilder<(T0, T1, T2)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Struct].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Struct>
        where
            Self: ::planus::WriteAsOffset<Struct>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::DType>]>>,
        T2: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAs<::planus::Offset<Struct>> for StructBuilder<(T0, T1, T2)>
    {
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
    > ::planus::WriteAsOptional<::planus::Offset<Struct>> for StructBuilder<(T0, T1, T2)>
    {
        type Prepared = ::planus::Offset<Struct>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Struct>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<str>]>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::DType>]>>,
        T2: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAsOffset<Struct> for StructBuilder<(T0, T1, T2)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Struct> {
            let (v0, v1, v2) = &self.0;
            Struct::create(builder, v0, v1, v2)
        }
    }

    /// Reference to a deserialized [Struct].
    #[derive(Copy, Clone)]
    pub struct StructRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> StructRef<'a> {
        /// Getter for the [`names` field](Struct#structfield.names).
        #[inline]
        pub fn names(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<
                ::planus::Vector<'a, ::planus::Result<&'a ::core::primitive::str>>,
            >,
        > {
            self.0.access(0, "Struct", "names")
        }

        /// Getter for the [`dtypes` field](Struct#structfield.dtypes).
        #[inline]
        pub fn dtypes(
            &self,
        ) -> ::planus::Result<
            ::core::option::Option<::planus::Vector<'a, ::planus::Result<self::DTypeRef<'a>>>>,
        > {
            self.0.access(1, "Struct", "dtypes")
        }

        /// Getter for the [`nullable` field](Struct#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
            ::core::result::Result::Ok(self.0.access(2, "Struct", "nullable")?.unwrap_or(false))
        }
    }

    impl<'a> ::core::fmt::Debug for StructRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("StructRef");
            if let ::core::option::Option::Some(field_names) = self.names().transpose() {
                f.field("names", &field_names);
            }
            if let ::core::option::Option::Some(field_dtypes) = self.dtypes().transpose() {
                f.field("dtypes", &field_dtypes);
            }
            f.field("nullable", &self.nullable());
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<StructRef<'a>> for Struct {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: StructRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                names: if let ::core::option::Option::Some(names) = value.names()? {
                    ::core::option::Option::Some(names.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
                dtypes: if let ::core::option::Option::Some(dtypes) = value.dtypes()? {
                    ::core::option::Option::Some(dtypes.to_vec_result()?)
                } else {
                    ::core::option::Option::None
                },
                nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for StructRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for StructRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[StructRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[StructRef]", "read_as_root", 0))
        }
    }

    /// The table `List`
    ///
    /// Generated from these locations:
    /// * Table `List` in the file `flatbuffers/vortex-dtype/dtype.fbs:49`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct List {
        /// The field `element_type` in the table `List`
        pub element_type: ::core::option::Option<::planus::alloc::boxed::Box<self::DType>>,
        /// The field `nullable` in the table `List`
        pub nullable: bool,
    }

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
        ) -> ::planus::Offset<Self> {
            let prepared_element_type = field_element_type.prepare(builder);
            let prepared_nullable = field_nullable.prepare(builder, &false);

            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                ::core::default::Default::default();
            if prepared_element_type.is_some() {
                table_writer.write_entry::<::planus::Offset<self::DType>>(0);
            }
            if prepared_nullable.is_some() {
                table_writer.write_entry::<bool>(1);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_element_type) =
                        prepared_element_type
                    {
                        object_writer.write::<_, _, 4>(&prepared_element_type);
                    }
                    if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {
                        object_writer.write::<_, _, 1>(&prepared_nullable);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<List>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<List> for List {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<List> {
            List::create(builder, &self.element_type, self.nullable)
        }
    }

    /// Builder for serializing an instance of the [List] type.
    ///
    /// Can be created using the [List::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct ListBuilder<State>(State);

    impl ListBuilder<()> {
        /// Setter for the [`element_type` field](List#structfield.element_type).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn element_type<T0>(self, value: T0) -> ListBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        {
            ListBuilder((value,))
        }

        /// Sets the [`element_type` field](List#structfield.element_type) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn element_type_as_null(self) -> ListBuilder<((),)> {
            self.element_type(())
        }
    }

    impl<T0> ListBuilder<(T0,)> {
        /// Setter for the [`nullable` field](List#structfield.nullable).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable<T1>(self, value: T1) -> ListBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsDefault<bool, bool>,
        {
            let (v0,) = self.0;
            ListBuilder((v0, value))
        }

        /// Sets the [`nullable` field](List#structfield.nullable) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable_as_default(self) -> ListBuilder<(T0, ::planus::DefaultValue)> {
            self.nullable(::planus::DefaultValue)
        }
    }

    impl<T0, T1> ListBuilder<(T0, T1)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [List].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<List>
        where
            Self: ::planus::WriteAsOffset<List>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        T1: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAs<::planus::Offset<List>> for ListBuilder<(T0, T1)>
    {
        type Prepared = ::planus::Offset<List>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<List> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        T1: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAsOptional<::planus::Offset<List>> for ListBuilder<(T0, T1)>
    {
        type Prepared = ::planus::Offset<List>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<List>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        T1: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAsOffset<List> for ListBuilder<(T0, T1)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<List> {
            let (v0, v1) = &self.0;
            List::create(builder, v0, v1)
        }
    }

    /// Reference to a deserialized [List].
    #[derive(Copy, Clone)]
    pub struct ListRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> ListRef<'a> {
        /// Getter for the [`element_type` field](List#structfield.element_type).
        #[inline]
        pub fn element_type(&self) -> ::planus::Result<::core::option::Option<self::DTypeRef<'a>>> {
            self.0.access(0, "List", "element_type")
        }

        /// Getter for the [`nullable` field](List#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
            ::core::result::Result::Ok(self.0.access(1, "List", "nullable")?.unwrap_or(false))
        }
    }

    impl<'a> ::core::fmt::Debug for ListRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("ListRef");
            if let ::core::option::Option::Some(field_element_type) =
                self.element_type().transpose()
            {
                f.field("element_type", &field_element_type);
            }
            f.field("nullable", &self.nullable());
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<ListRef<'a>> for List {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: ListRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                element_type: if let ::core::option::Option::Some(element_type) =
                    value.element_type()?
                {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(element_type)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
                nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for ListRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for ListRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[ListRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[ListRef]", "read_as_root", 0))
        }
    }

    /// The table `FixedSizeList`
    ///
    /// Generated from these locations:
    /// * Table `FixedSizeList` in the file `flatbuffers/vortex-dtype/dtype.fbs:54`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct FixedSizeList {
        /// The field `element_type` in the table `FixedSizeList`
        pub element_type: ::core::option::Option<::planus::alloc::boxed::Box<self::DType>>,
        /// The field `size` in the table `FixedSizeList`
        pub size: u32,
        /// The field `nullable` in the table `FixedSizeList`
        pub nullable: bool,
    }

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
        ) -> ::planus::Offset<Self> {
            let prepared_element_type = field_element_type.prepare(builder);
            let prepared_size = field_size.prepare(builder, &0);
            let prepared_nullable = field_nullable.prepare(builder, &false);

            let mut table_writer: ::planus::table_writer::TableWriter<10> =
                ::core::default::Default::default();
            if prepared_element_type.is_some() {
                table_writer.write_entry::<::planus::Offset<self::DType>>(0);
            }
            if prepared_size.is_some() {
                table_writer.write_entry::<u32>(1);
            }
            if prepared_nullable.is_some() {
                table_writer.write_entry::<bool>(2);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_element_type) =
                        prepared_element_type
                    {
                        object_writer.write::<_, _, 4>(&prepared_element_type);
                    }
                    if let ::core::option::Option::Some(prepared_size) = prepared_size {
                        object_writer.write::<_, _, 4>(&prepared_size);
                    }
                    if let ::core::option::Option::Some(prepared_nullable) = prepared_nullable {
                        object_writer.write::<_, _, 1>(&prepared_nullable);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<FixedSizeList>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<FixedSizeList> for FixedSizeList {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FixedSizeList> {
            FixedSizeList::create(builder, &self.element_type, self.size, self.nullable)
        }
    }

    /// Builder for serializing an instance of the [FixedSizeList] type.
    ///
    /// Can be created using the [FixedSizeList::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct FixedSizeListBuilder<State>(State);

    impl FixedSizeListBuilder<()> {
        /// Setter for the [`element_type` field](FixedSizeList#structfield.element_type).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn element_type<T0>(self, value: T0) -> FixedSizeListBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        {
            FixedSizeListBuilder((value,))
        }

        /// Sets the [`element_type` field](FixedSizeList#structfield.element_type) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn element_type_as_null(self) -> FixedSizeListBuilder<((),)> {
            self.element_type(())
        }
    }

    impl<T0> FixedSizeListBuilder<(T0,)> {
        /// Setter for the [`size` field](FixedSizeList#structfield.size).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn size<T1>(self, value: T1) -> FixedSizeListBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsDefault<u32, u32>,
        {
            let (v0,) = self.0;
            FixedSizeListBuilder((v0, value))
        }

        /// Sets the [`size` field](FixedSizeList#structfield.size) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn size_as_default(self) -> FixedSizeListBuilder<(T0, ::planus::DefaultValue)> {
            self.size(::planus::DefaultValue)
        }
    }

    impl<T0, T1> FixedSizeListBuilder<(T0, T1)> {
        /// Setter for the [`nullable` field](FixedSizeList#structfield.nullable).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable<T2>(self, value: T2) -> FixedSizeListBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsDefault<bool, bool>,
        {
            let (v0, v1) = self.0;
            FixedSizeListBuilder((v0, v1, value))
        }

        /// Sets the [`nullable` field](FixedSizeList#structfield.nullable) to the default value.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn nullable_as_default(self) -> FixedSizeListBuilder<(T0, T1, ::planus::DefaultValue)> {
            self.nullable(::planus::DefaultValue)
        }
    }

    impl<T0, T1, T2> FixedSizeListBuilder<(T0, T1, T2)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [FixedSizeList].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<FixedSizeList>
        where
            Self: ::planus::WriteAsOffset<FixedSizeList>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        T1: ::planus::WriteAsDefault<u32, u32>,
        T2: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAs<::planus::Offset<FixedSizeList>> for FixedSizeListBuilder<(T0, T1, T2)>
    {
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
    > ::planus::WriteAsOptional<::planus::Offset<FixedSizeList>>
        for FixedSizeListBuilder<(T0, T1, T2)>
    {
        type Prepared = ::planus::Offset<FixedSizeList>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<FixedSizeList>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        T1: ::planus::WriteAsDefault<u32, u32>,
        T2: ::planus::WriteAsDefault<bool, bool>,
    > ::planus::WriteAsOffset<FixedSizeList> for FixedSizeListBuilder<(T0, T1, T2)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<FixedSizeList> {
            let (v0, v1, v2) = &self.0;
            FixedSizeList::create(builder, v0, v1, v2)
        }
    }

    /// Reference to a deserialized [FixedSizeList].
    #[derive(Copy, Clone)]
    pub struct FixedSizeListRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> FixedSizeListRef<'a> {
        /// Getter for the [`element_type` field](FixedSizeList#structfield.element_type).
        #[inline]
        pub fn element_type(&self) -> ::planus::Result<::core::option::Option<self::DTypeRef<'a>>> {
            self.0.access(0, "FixedSizeList", "element_type")
        }

        /// Getter for the [`size` field](FixedSizeList#structfield.size).
        #[inline]
        pub fn size(&self) -> ::planus::Result<u32> {
            ::core::result::Result::Ok(self.0.access(1, "FixedSizeList", "size")?.unwrap_or(0))
        }

        /// Getter for the [`nullable` field](FixedSizeList#structfield.nullable).
        #[inline]
        pub fn nullable(&self) -> ::planus::Result<bool> {
            ::core::result::Result::Ok(
                self.0
                    .access(2, "FixedSizeList", "nullable")?
                    .unwrap_or(false),
            )
        }
    }

    impl<'a> ::core::fmt::Debug for FixedSizeListRef<'a> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
            let mut f = f.debug_struct("FixedSizeListRef");
            if let ::core::option::Option::Some(field_element_type) =
                self.element_type().transpose()
            {
                f.field("element_type", &field_element_type);
            }
            f.field("size", &self.size());
            f.field("nullable", &self.nullable());
            f.finish()
        }
    }

    impl<'a> ::core::convert::TryFrom<FixedSizeListRef<'a>> for FixedSizeList {
        type Error = ::planus::Error;

        #[allow(unreachable_code)]
        fn try_from(value: FixedSizeListRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(Self {
                element_type: if let ::core::option::Option::Some(element_type) =
                    value.element_type()?
                {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(element_type)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
                size: ::core::convert::TryInto::try_into(value.size()?)?,
                nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for FixedSizeListRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for FixedSizeListRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location(
                    "[FixedSizeListRef]",
                    "get",
                    buffer.offset_from_start,
                )
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[FixedSizeListRef]", "read_as_root", 0)
            })
        }
    }

    /// The table `Extension`
    ///
    /// Generated from these locations:
    /// * Table `Extension` in the file `flatbuffers/vortex-dtype/dtype.fbs:60`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct Extension {
        /// The field `id` in the table `Extension`
        pub id: ::core::option::Option<::planus::alloc::string::String>,
        /// The field `storage_dtype` in the table `Extension`
        pub storage_dtype: ::core::option::Option<::planus::alloc::boxed::Box<self::DType>>,
        /// The field `metadata` in the table `Extension`
        pub metadata: ::core::option::Option<::planus::alloc::vec::Vec<u8>>,
    }

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
        ) -> ::planus::Offset<Self> {
            let prepared_id = field_id.prepare(builder);
            let prepared_storage_dtype = field_storage_dtype.prepare(builder);
            let prepared_metadata = field_metadata.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<10> =
                ::core::default::Default::default();
            if prepared_id.is_some() {
                table_writer.write_entry::<::planus::Offset<str>>(0);
            }
            if prepared_storage_dtype.is_some() {
                table_writer.write_entry::<::planus::Offset<self::DType>>(1);
            }
            if prepared_metadata.is_some() {
                table_writer.write_entry::<::planus::Offset<[u8]>>(2);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_id) = prepared_id {
                        object_writer.write::<_, _, 4>(&prepared_id);
                    }
                    if let ::core::option::Option::Some(prepared_storage_dtype) =
                        prepared_storage_dtype
                    {
                        object_writer.write::<_, _, 4>(&prepared_storage_dtype);
                    }
                    if let ::core::option::Option::Some(prepared_metadata) = prepared_metadata {
                        object_writer.write::<_, _, 4>(&prepared_metadata);
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Extension>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<Extension> for Extension {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Extension> {
            Extension::create(builder, &self.id, &self.storage_dtype, &self.metadata)
        }
    }

    /// Builder for serializing an instance of the [Extension] type.
    ///
    /// Can be created using the [Extension::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct ExtensionBuilder<State>(State);

    impl ExtensionBuilder<()> {
        /// Setter for the [`id` field](Extension#structfield.id).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn id<T0>(self, value: T0) -> ExtensionBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
        {
            ExtensionBuilder((value,))
        }

        /// Sets the [`id` field](Extension#structfield.id) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn id_as_null(self) -> ExtensionBuilder<((),)> {
            self.id(())
        }
    }

    impl<T0> ExtensionBuilder<(T0,)> {
        /// Setter for the [`storage_dtype` field](Extension#structfield.storage_dtype).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn storage_dtype<T1>(self, value: T1) -> ExtensionBuilder<(T0, T1)>
        where
            T1: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        {
            let (v0,) = self.0;
            ExtensionBuilder((v0, value))
        }

        /// Sets the [`storage_dtype` field](Extension#structfield.storage_dtype) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn storage_dtype_as_null(self) -> ExtensionBuilder<(T0, ())> {
            self.storage_dtype(())
        }
    }

    impl<T0, T1> ExtensionBuilder<(T0, T1)> {
        /// Setter for the [`metadata` field](Extension#structfield.metadata).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn metadata<T2>(self, value: T2) -> ExtensionBuilder<(T0, T1, T2)>
        where
            T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
        {
            let (v0, v1) = self.0;
            ExtensionBuilder((v0, v1, value))
        }

        /// Sets the [`metadata` field](Extension#structfield.metadata) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn metadata_as_null(self) -> ExtensionBuilder<(T0, T1, ())> {
            self.metadata(())
        }
    }

    impl<T0, T1, T2> ExtensionBuilder<(T0, T1, T2)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Extension].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Extension>
        where
            Self: ::planus::WriteAsOffset<Extension>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    > ::planus::WriteAs<::planus::Offset<Extension>> for ExtensionBuilder<(T0, T1, T2)>
    {
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
    > ::planus::WriteAsOptional<::planus::Offset<Extension>> for ExtensionBuilder<(T0, T1, T2)>
    {
        type Prepared = ::planus::Offset<Extension>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<Extension>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<
        T0: ::planus::WriteAsOptional<::planus::Offset<::core::primitive::str>>,
        T1: ::planus::WriteAsOptional<::planus::Offset<self::DType>>,
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    > ::planus::WriteAsOffset<Extension> for ExtensionBuilder<(T0, T1, T2)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Extension> {
            let (v0, v1, v2) = &self.0;
            Extension::create(builder, v0, v1, v2)
        }
    }

    /// Reference to a deserialized [Extension].
    #[derive(Copy, Clone)]
    pub struct ExtensionRef<'a>(::planus::table_reader::Table<'a>);

    impl<'a> ExtensionRef<'a> {
        /// Getter for the [`id` field](Extension#structfield.id).
        #[inline]
        pub fn id(&self) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>> {
            self.0.access(0, "Extension", "id")
        }

        /// Getter for the [`storage_dtype` field](Extension#structfield.storage_dtype).
        #[inline]
        pub fn storage_dtype(
            &self,
        ) -> ::planus::Result<::core::option::Option<self::DTypeRef<'a>>> {
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
            }
            if let ::core::option::Option::Some(field_storage_dtype) =
                self.storage_dtype().transpose()
            {
                f.field("storage_dtype", &field_storage_dtype);
            }
            if let ::core::option::Option::Some(field_metadata) = self.metadata().transpose() {
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
                id: value.id()?.map(::core::convert::Into::into),
                storage_dtype: if let ::core::option::Option::Some(storage_dtype) =
                    value.storage_dtype()?
                {
                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                        ::core::convert::TryInto::try_into(storage_dtype)?,
                    ))
                } else {
                    ::core::option::Option::None
                },
                metadata: value.metadata()?.map(|v| v.to_vec()),
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for ExtensionRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for ExtensionRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[ExtensionRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| {
                error_kind.with_error_location("[ExtensionRef]", "read_as_root", 0)
            })
        }
    }

    /// The union `Type`
    ///
    /// Generated from these locations:
    /// * Union `Type` in the file `flatbuffers/vortex-dtype/dtype.fbs:66`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub enum Type {
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
            }
        }
    }

    impl ::planus::WriteAsOptionalUnion<Type> for Type {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Self>> {
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
        pub fn null<T>(self, value: T) -> TypeBuilder<::planus::Initialized<1, T>>
        where
            T: ::planus::WriteAsOffset<self::Null>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`Bool` variant](Type#variant.Bool).
        #[inline]
        pub fn bool<T>(self, value: T) -> TypeBuilder<::planus::Initialized<2, T>>
        where
            T: ::planus::WriteAsOffset<self::Bool>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`Primitive` variant](Type#variant.Primitive).
        #[inline]
        pub fn primitive<T>(self, value: T) -> TypeBuilder<::planus::Initialized<3, T>>
        where
            T: ::planus::WriteAsOffset<self::Primitive>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`Decimal` variant](Type#variant.Decimal).
        #[inline]
        pub fn decimal<T>(self, value: T) -> TypeBuilder<::planus::Initialized<4, T>>
        where
            T: ::planus::WriteAsOffset<self::Decimal>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`Utf8` variant](Type#variant.Utf8).
        #[inline]
        pub fn utf8<T>(self, value: T) -> TypeBuilder<::planus::Initialized<5, T>>
        where
            T: ::planus::WriteAsOffset<self::Utf8>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`Binary` variant](Type#variant.Binary).
        #[inline]
        pub fn binary<T>(self, value: T) -> TypeBuilder<::planus::Initialized<6, T>>
        where
            T: ::planus::WriteAsOffset<self::Binary>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`Struct_` variant](Type#variant.Struct).
        #[inline]
        pub fn struct_<T>(self, value: T) -> TypeBuilder<::planus::Initialized<7, T>>
        where
            T: ::planus::WriteAsOffset<self::Struct>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`List` variant](Type#variant.List).
        #[inline]
        pub fn list<T>(self, value: T) -> TypeBuilder<::planus::Initialized<8, T>>
        where
            T: ::planus::WriteAsOffset<self::List>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`Extension` variant](Type#variant.Extension).
        #[inline]
        pub fn extension<T>(self, value: T) -> TypeBuilder<::planus::Initialized<9, T>>
        where
            T: ::planus::WriteAsOffset<self::Extension>,
        {
            TypeBuilder(::planus::Initialized(value))
        }

        /// Creates an instance of the [`FixedSizeList` variant](Type#variant.FixedSizeList).
        #[inline]
        pub fn fixed_size_list<T>(self, value: T) -> TypeBuilder<::planus::Initialized<10, T>>
        where
            T: ::planus::WriteAsOffset<self::FixedSizeList>,
        {
            TypeBuilder(::planus::Initialized(value))
        }
    }

    impl<const N: u8, T> TypeBuilder<::planus::Initialized<N, T>> {
        /// Finish writing the builder to get an [UnionOffset](::planus::UnionOffset) to a serialized [Type].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type>
        where
            Self: ::planus::WriteAsUnion<Type>,
        {
            ::planus::WriteAsUnion::prepare(&self, builder)
        }
    }

    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<1, T>>
    where
        T: ::planus::WriteAsOffset<self::Null>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(1, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<1, T>>
    where
        T: ::planus::WriteAsOffset<self::Null>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<2, T>>
    where
        T: ::planus::WriteAsOffset<self::Bool>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(2, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<2, T>>
    where
        T: ::planus::WriteAsOffset<self::Bool>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<3, T>>
    where
        T: ::planus::WriteAsOffset<self::Primitive>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(3, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<3, T>>
    where
        T: ::planus::WriteAsOffset<self::Primitive>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<4, T>>
    where
        T: ::planus::WriteAsOffset<self::Decimal>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(4, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<4, T>>
    where
        T: ::planus::WriteAsOffset<self::Decimal>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<5, T>>
    where
        T: ::planus::WriteAsOffset<self::Utf8>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(5, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<5, T>>
    where
        T: ::planus::WriteAsOffset<self::Utf8>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<6, T>>
    where
        T: ::planus::WriteAsOffset<self::Binary>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(6, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<6, T>>
    where
        T: ::planus::WriteAsOffset<self::Binary>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<7, T>>
    where
        T: ::planus::WriteAsOffset<self::Struct>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(7, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<7, T>>
    where
        T: ::planus::WriteAsOffset<self::Struct>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<8, T>>
    where
        T: ::planus::WriteAsOffset<self::List>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(8, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<8, T>>
    where
        T: ::planus::WriteAsOffset<self::List>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<9, T>>
    where
        T: ::planus::WriteAsOffset<self::Extension>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(9, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<9, T>>
    where
        T: ::planus::WriteAsOffset<self::Extension>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }
    impl<T> ::planus::WriteAsUnion<Type> for TypeBuilder<::planus::Initialized<10, T>>
    where
        T: ::planus::WriteAsOffset<self::FixedSizeList>,
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::UnionOffset<Type> {
            ::planus::UnionOffset::new(10, (self.0).0.prepare(builder).downcast())
        }
    }

    impl<T> ::planus::WriteAsOptionalUnion<Type> for TypeBuilder<::planus::Initialized<10, T>>
    where
        T: ::planus::WriteAsOffset<self::FixedSizeList>,
    {
        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::UnionOffset<Type>> {
            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(self, builder))
        }
    }

    /// Reference to a deserialized [Type].
    #[derive(Copy, Clone, Debug)]
    pub enum TypeRef<'a> {
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
    }

    impl<'a> ::core::convert::TryFrom<TypeRef<'a>> for Type {
        type Error = ::planus::Error;

        fn try_from(value: TypeRef<'a>) -> ::planus::Result<Self> {
            ::core::result::Result::Ok(match value {
                TypeRef::Null(value) => Self::Null(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::Bool(value) => Self::Bool(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::Primitive(value) => Self::Primitive(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::Decimal(value) => Self::Decimal(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::Utf8(value) => Self::Utf8(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::Binary(value) => Self::Binary(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::Struct(value) => Self::Struct(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::List(value) => Self::List(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::Extension(value) => Self::Extension(::planus::alloc::boxed::Box::new(
                    ::core::convert::TryFrom::try_from(value)?,
                )),

                TypeRef::FixedSizeList(value) => Self::FixedSizeList(
                    ::planus::alloc::boxed::Box::new(::core::convert::TryFrom::try_from(value)?),
                ),
            })
        }
    }

    impl<'a> ::planus::TableReadUnion<'a> for TypeRef<'a> {
        fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            tag: u8,
            field_offset: usize,
        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
            match tag {
                1 => ::core::result::Result::Ok(Self::Null(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                2 => ::core::result::Result::Ok(Self::Bool(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                3 => ::core::result::Result::Ok(Self::Primitive(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                4 => ::core::result::Result::Ok(Self::Decimal(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                5 => ::core::result::Result::Ok(Self::Utf8(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                6 => ::core::result::Result::Ok(Self::Binary(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                7 => ::core::result::Result::Ok(Self::Struct(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                8 => ::core::result::Result::Ok(Self::List(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                9 => ::core::result::Result::Ok(Self::Extension(::planus::TableRead::from_buffer(
                    buffer,
                    field_offset,
                )?)),
                10 => ::core::result::Result::Ok(Self::FixedSizeList(
                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                )),
                _ => ::core::result::Result::Err(::planus::errors::ErrorKind::UnknownUnionTag {
                    tag,
                }),
            }
        }
    }

    impl<'a> ::planus::VectorReadUnion<'a> for TypeRef<'a> {
        const VECTOR_NAME: &'static str = "[TypeRef]";
    }

    /// The table `DType`
    ///
    /// Generated from these locations:
    /// * Table `DType` in the file `flatbuffers/vortex-dtype/dtype.fbs:79`
    #[derive(
        Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash, ::serde::Serialize, ::serde::Deserialize,
    )]
    pub struct DType {
        /// The field `type` in the table `DType`
        pub type_: ::core::option::Option<self::Type>,
    }

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
        ) -> ::planus::Offset<Self> {
            let prepared_type_ = field_type_.prepare(builder);

            let mut table_writer: ::planus::table_writer::TableWriter<8> =
                ::core::default::Default::default();
            if prepared_type_.is_some() {
                table_writer.write_entry::<::planus::Offset<self::Type>>(1);
            }
            if prepared_type_.is_some() {
                table_writer.write_entry::<u8>(0);
            }

            unsafe {
                table_writer.finish(builder, |object_writer| {
                    if let ::core::option::Option::Some(prepared_type_) = prepared_type_ {
                        object_writer.write::<_, _, 4>(&prepared_type_.offset());
                    }
                    if let ::core::option::Option::Some(prepared_type_) = prepared_type_ {
                        object_writer.write::<_, _, 1>(&prepared_type_.tag());
                    }
                });
            }
            builder.current_offset()
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
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<DType>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl ::planus::WriteAsOffset<DType> for DType {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType> {
            DType::create(builder, &self.type_)
        }
    }

    /// Builder for serializing an instance of the [DType] type.
    ///
    /// Can be created using the [DType::builder] method.
    #[derive(Debug)]
    #[must_use]
    pub struct DTypeBuilder<State>(State);

    impl DTypeBuilder<()> {
        /// Setter for the [`type` field](DType#structfield.type_).
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn type_<T0>(self, value: T0) -> DTypeBuilder<(T0,)>
        where
            T0: ::planus::WriteAsOptionalUnion<self::Type>,
        {
            DTypeBuilder((value,))
        }

        /// Sets the [`type` field](DType#structfield.type_) to null.
        #[inline]
        #[allow(clippy::type_complexity)]
        pub fn type_as_null(self) -> DTypeBuilder<((),)> {
            self.type_(())
        }
    }

    impl<T0> DTypeBuilder<(T0,)> {
        /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [DType].
        #[inline]
        pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType>
        where
            Self: ::planus::WriteAsOffset<DType>,
        {
            ::planus::WriteAsOffset::prepare(&self, builder)
        }
    }

    impl<T0: ::planus::WriteAsOptionalUnion<self::Type>> ::planus::WriteAs<::planus::Offset<DType>>
        for DTypeBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<DType>;

        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType> {
            ::planus::WriteAsOffset::prepare(self, builder)
        }
    }

    impl<T0: ::planus::WriteAsOptionalUnion<self::Type>>
        ::planus::WriteAsOptional<::planus::Offset<DType>> for DTypeBuilder<(T0,)>
    {
        type Prepared = ::planus::Offset<DType>;

        #[inline]
        fn prepare(
            &self,
            builder: &mut ::planus::Builder,
        ) -> ::core::option::Option<::planus::Offset<DType>> {
            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
        }
    }

    impl<T0: ::planus::WriteAsOptionalUnion<self::Type>> ::planus::WriteAsOffset<DType>
        for DTypeBuilder<(T0,)>
    {
        #[inline]
        fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<DType> {
            let (v0,) = &self.0;
            DType::create(builder, v0)
        }
    }

    /// Reference to a deserialized [DType].
    #[derive(Copy, Clone)]
    pub struct DTypeRef<'a>(::planus::table_reader::Table<'a>);

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
                type_: if let ::core::option::Option::Some(type_) = value.type_()? {
                    ::core::option::Option::Some(::core::convert::TryInto::try_into(type_)?)
                } else {
                    ::core::option::Option::None
                },
            })
        }
    }

    impl<'a> ::planus::TableRead<'a> for DTypeRef<'a> {
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

    impl<'a> ::planus::VectorReadInner<'a> for DTypeRef<'a> {
        type Error = ::planus::Error;
        const STRIDE: usize = 4;

        unsafe fn from_buffer(
            buffer: ::planus::SliceWithStartOffset<'a>,
            offset: usize,
        ) -> ::planus::Result<Self> {
            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                error_kind.with_error_location("[DTypeRef]", "get", buffer.offset_from_start)
            })
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
            ::planus::TableRead::from_buffer(
                ::planus::SliceWithStartOffset {
                    buffer: slice,
                    offset_from_start: 0,
                },
                0,
            )
            .map_err(|error_kind| error_kind.with_error_location("[DTypeRef]", "read_as_root", 0))
        }
    }
}

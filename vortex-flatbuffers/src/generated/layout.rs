pub use root::*;

const _: () = ::planus::check_version_compatibility("planus-1.3.0");


/// The root namespace
/// 
/// Generated from these locations:
/// * File `flatbuffers/vortex-layout/layout.fbs`
#[no_implicit_prelude]
#[allow(clippy::needless_lifetimes)]
mod root {
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
#[derive(Clone, Debug, PartialEq, PartialOrd,
Eq, Ord, Hash,
::serde::Serialize, ::serde::Deserialize
)]
pub struct Layout{
    
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
        pub segments: ::core::option::Option<::planus::alloc::vec::Vec<u32>>,}


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
        field_children: impl ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::Layout>]>>,
        field_segments: impl ::planus::WriteAsOptional<::planus::Offset<[u32]>>,
        
    ) -> ::planus::Offset<Self> {let prepared_encoding = field_encoding.prepare(builder, &0);let prepared_row_count = field_row_count.prepare(builder, &0);let prepared_metadata = field_metadata.prepare(builder);let prepared_children = field_children.prepare(builder);let prepared_segments = field_segments.prepare(builder);

        let mut table_writer: ::planus::table_writer::TableWriter::<14> = ::core::default::Default::default();
        if prepared_row_count.is_some() {table_writer.write_entry::<u64>(1);}if prepared_metadata.is_some() {table_writer.write_entry::<::planus::Offset<[u8]>>(2);}if prepared_children.is_some() {table_writer.write_entry::<::planus::Offset<[::planus::Offset<self::Layout>]>>(3);}if prepared_segments.is_some() {table_writer.write_entry::<::planus::Offset<[u32]>>(4);}if prepared_encoding.is_some() {table_writer.write_entry::<u16>(0);}

        unsafe {
            table_writer.finish(builder, |object_writer| {if let ::core::option::Option::Some(prepared_row_count) = prepared_row_count {object_writer.write::<_, _, 8>(&prepared_row_count);}if let ::core::option::Option::Some(prepared_metadata) = prepared_metadata {object_writer.write::<_, _, 4>(&prepared_metadata);}if let ::core::option::Option::Some(prepared_children) = prepared_children {object_writer.write::<_, _, 4>(&prepared_children);}if let ::core::option::Option::Some(prepared_segments) = prepared_segments {object_writer.write::<_, _, 4>(&prepared_segments);}if let ::core::option::Option::Some(prepared_encoding) = prepared_encoding {object_writer.write::<_, _, 2>(&prepared_encoding);}
                
            });
        }builder.current_offset()
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
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Layout>> {
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

impl<
    
> LayoutBuilder
<
    (
    
    )
>
{
    /// Setter for the [`encoding` field](Layout#structfield.encoding).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn encoding<T0>(self, value: T0) -> LayoutBuilder<(
    
        T0,
    
    )>
    where T0: ::planus::WriteAsDefault<u16, u16>
    {
        LayoutBuilder((
            
            value,
        ))
    }

    
    /// Sets the [`encoding` field](Layout#structfield.encoding) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn encoding_as_default(self) -> LayoutBuilder<(
        
        ::planus::DefaultValue,
    )>
    {
        self.encoding(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
> LayoutBuilder
<
    (
    
        T0,
    
    )
>
{
    /// Setter for the [`row_count` field](Layout#structfield.row_count).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn row_count<T1>(self, value: T1) -> LayoutBuilder<(
    
        T0,
    
        T1,
    
    )>
    where T1: ::planus::WriteAsDefault<u64, u64>
    {
        
        let (
            
                v0,
            
        ) = self.0;LayoutBuilder((
            
                v0,
            
            value,
        ))
    }

    
    /// Sets the [`row_count` field](Layout#structfield.row_count) to the default value.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn row_count_as_default(self) -> LayoutBuilder<(
        
            T0,
        
        ::planus::DefaultValue,
    )>
    {
        self.row_count(::planus::DefaultValue)
    }
    

    
}

impl<
    
        T0,
    
        T1,
    
> LayoutBuilder
<
    (
    
        T0,
    
        T1,
    
    )
>
{
    /// Setter for the [`metadata` field](Layout#structfield.metadata).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn metadata<T2>(self, value: T2) -> LayoutBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
    )>
    where T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>
    {
        
        let (
            
                v0,
            
                v1,
            
        ) = self.0;LayoutBuilder((
            
                v0,
            
                v1,
            
            value,
        ))
    }

    

    
    /// Sets the [`metadata` field](Layout#structfield.metadata) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn metadata_as_null(self) -> LayoutBuilder<(
        
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
    
> LayoutBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
    )
>
{
    /// Setter for the [`children` field](Layout#structfield.children).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn children<T3>(self, value: T3) -> LayoutBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
    )>
    where T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::Layout>]>>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
        ) = self.0;LayoutBuilder((
            
                v0,
            
                v1,
            
                v2,
            
            value,
        ))
    }

    

    
    /// Sets the [`children` field](Layout#structfield.children) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn children_as_null(self) -> LayoutBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
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
    
        T3,
    
> LayoutBuilder
<
    (
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
    )
>
{
    /// Setter for the [`segments` field](Layout#structfield.segments).
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn segments<T4>(self, value: T4) -> LayoutBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
    )>
    where T4: ::planus::WriteAsOptional<::planus::Offset<[u32]>>
    {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
        ) = self.0;LayoutBuilder((
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
            value,
        ))
    }

    

    
    /// Sets the [`segments` field](Layout#structfield.segments) to null.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn segments_as_null(self) -> LayoutBuilder<(
        
            T0,
        
            T1,
        
            T2,
        
            T3,
        
        (),
    )>
    {
        self.segments(())
    }
    
}



impl<
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
> LayoutBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
)> {
    /// Finish writing the builder to get an [Offset](::planus::Offset) to a serialized [Layout].
    #[inline]
    pub fn finish(self, builder: &mut ::planus::Builder) -> ::planus::Offset<Layout>
        where Self: ::planus::WriteAsOffset<Layout>
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
    
> ::planus::WriteAs<::planus::Offset<Layout>> for LayoutBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
)> {
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
    
> ::planus::WriteAsOptional<::planus::Offset<Layout>> for LayoutBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
)> {
    type Prepared = ::planus::Offset<Layout>;

    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::core::option::Option<::planus::Offset<Layout>> {
        ::core::option::Option::Some(::planus::WriteAsOffset::prepare(self, builder))
    }
}

impl<
    
        T0: ::planus::WriteAsDefault<u16, u16>,
    
        T1: ::planus::WriteAsDefault<u64, u64>,
    
        T2: ::planus::WriteAsOptional<::planus::Offset<[u8]>>,
    
        T3: ::planus::WriteAsOptional<::planus::Offset<[::planus::Offset<self::Layout>]>>,
    
        T4: ::planus::WriteAsOptional<::planus::Offset<[u32]>>,
    
> ::planus::WriteAsOffset<Layout> for LayoutBuilder<(
    
        T0,
    
        T1,
    
        T2,
    
        T3,
    
        T4,
    
)> {
    #[inline]
    fn prepare(&self, builder: &mut ::planus::Builder) -> ::planus::Offset<Layout> {
        
        let (
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
        ) = &self.0;Layout::create(
            builder,
            
                v0,
            
                v1,
            
                v2,
            
                v3,
            
                v4,
            
        )
    }
}

/// Reference to a deserialized [Layout].
#[derive(Copy, Clone)]
pub struct LayoutRef<'a>(
    #[allow(dead_code)]
    ::planus::table_reader::Table<'a>
);

impl<'a> LayoutRef<'a> {
    
        /// Getter for the [`encoding` field](Layout#structfield.encoding).
        #[inline]
        pub fn encoding(&self) -> ::planus::Result<u16> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(0, "Layout", "encoding")
              
            
            ?.unwrap_or(0))
            
        }
    
        /// Getter for the [`row_count` field](Layout#structfield.row_count).
        #[inline]
        pub fn row_count(&self) -> ::planus::Result<u64> {
             ::core::result::Result::Ok( 
            
              
              self.0.access(1, "Layout", "row_count")
              
            
            ?.unwrap_or(0))
            
        }
    
        /// Getter for the [`metadata` field](Layout#structfield.metadata).
        #[inline]
        pub fn metadata(&self) -> ::planus::Result<::core::option::Option<&'a [u8]>> {
            
            
              
              self.0.access(2, "Layout", "metadata")
              
            
            
            
        }
    
        /// Getter for the [`children` field](Layout#structfield.children).
        #[inline]
        pub fn children(&self) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, ::planus::Result<self::LayoutRef<'a>>>>> {
            
            
              
              self.0.access(3, "Layout", "children")
              
            
            
            
        }
    
        /// Getter for the [`segments` field](Layout#structfield.segments).
        #[inline]
        pub fn segments(&self) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, u32>>> {
            
            
              
              self.0.access(4, "Layout", "segments")
              
            
            
            
        }
    
}

impl<'a> ::core::fmt::Debug for LayoutRef<'a> {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        let mut f = f.debug_struct("LayoutRef");
        f.field("encoding", &self.encoding());f.field("row_count", &self.row_count());if let ::core::option::Option::Some(field_metadata) = self.metadata().transpose() {
                f.field("metadata", &field_metadata);
            }if let ::core::option::Option::Some(field_children) = self.children().transpose() {
                f.field("children", &field_children);
            }if let ::core::option::Option::Some(field_segments) = self.segments().transpose() {
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
            encoding: ::core::convert::TryInto::try_into(value.encoding()?)?,row_count: ::core::convert::TryInto::try_into(value.row_count()?)?,metadata: value.metadata()?.map(|v| v.to_vec()),children: 
                            if let ::core::option::Option::Some(children) = value.children()? {
                                ::core::option::Option::Some(children.to_vec_result()?)
                            } else {
                                ::core::option::Option::None
                            }
                        ,segments: 
                            if let ::core::option::Option::Some(segments) = value.segments()? {
                                ::core::option::Option::Some(segments.to_vec()?)
                            } else {
                                ::core::option::Option::None
                            }
                        ,
        })
    }
}

impl<'a> ::planus::TableRead<'a> for LayoutRef<'a> {
    #[inline]
    fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::core::result::Result<Self, ::planus::errors::ErrorKind> {
        ::core::result::Result::Ok(Self(::planus::table_reader::Table::from_buffer(buffer, offset)?))
    }
}

impl<'a> ::planus::VectorReadInner<'a> for LayoutRef<'a> {
    type Error = ::planus::Error;
    const STRIDE: usize = 4;

    unsafe fn from_buffer(buffer: ::planus::SliceWithStartOffset<'a>, offset: usize) -> ::planus::Result<Self> {
        ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| error_kind.with_error_location(
            "[LayoutRef]",
            "get",
            buffer.offset_from_start,
        ))
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
        ::planus::TableRead::from_buffer(::planus::SliceWithStartOffset {
            buffer: slice,
            offset_from_start: 0,
        }, 0).map_err(|error_kind| error_kind.with_error_location(
            "[LayoutRef]",
            "read_as_root",
            0,
        ))
    }
}

    }
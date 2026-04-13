// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encodings that enable zero-copy sharing of data with Arrow.

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::Executable;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::child_to_validity;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::bool::BoolDataParts;
use crate::arrays::decimal::DecimalDataParts;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::listview::ListViewDataParts;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrays::primitive::PrimitiveDataParts;
use crate::arrays::struct_::StructDataParts;
use crate::arrays::varbinview::VarBinViewDataParts;
use crate::arrays::variant::VariantArrayExt;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::matcher::Matcher;
use crate::validity::Validity;

/// An enum capturing the default uncompressed encodings for each [Vortex type](DType).
///
/// Any array can be decoded into canonical form via the `to_canonical`
/// trait method. This is the simplest encoding for a type, and will not be compressed but may
/// contain compressed child arrays.
///
/// Canonical form is useful for doing type-specific compute where you need to know that all
/// elements are laid out decompressed and contiguous in memory.
///
/// Each `Canonical` variant has a corresponding [`DType`] variant, with the notable exception of
/// [`Canonical::VarBinView`], which is the canonical encoding for both [`DType::Utf8`] and
/// [`DType::Binary`].
///
/// # Laziness
///
/// Canonical form is not recursive, so while a `StructArray` is the canonical format for any
/// `Struct` type, individual column child arrays may still be compressed. This allows
/// compute over Vortex arrays to push decoding as late as possible, and ideally many child arrays
/// never need to be decoded into canonical form at all depending on the compute.
///
/// # Arrow interoperability
///
/// All of the Vortex canonical encodings have an equivalent Arrow encoding that can be built
/// zero-copy, and the corresponding Arrow array types can also be built directly.
///
/// The full list of canonical types and their equivalent Arrow array types are:
///
/// * `NullArray`: [`arrow_array::NullArray`]
/// * `BoolArray`: [`arrow_array::BooleanArray`]
/// * `PrimitiveArray`: [`arrow_array::PrimitiveArray`]
/// * `DecimalArray`: [`arrow_array::Decimal128Array`] and [`arrow_array::Decimal256Array`]
/// * `VarBinViewArray`: [`arrow_array::GenericByteViewArray`]
/// * `ListViewArray`: [`arrow_array::ListViewArray`]
/// * `FixedSizeListArray`: [`arrow_array::FixedSizeListArray`]
/// * `StructArray`: [`arrow_array::StructArray`]
///
/// Vortex uses a logical type system, unlike Arrow which uses physical encodings for its types.
/// As an example, there are at least six valid physical encodings for a `Utf8` array. This can
/// create ambiguity.
/// Thus, if you receive an Arrow array, compress it using Vortex, and then
/// decompress it later to pass to a compute kernel, there are multiple suitable Arrow array
/// variants to hold the data.
///
/// To disambiguate, we choose a canonical physical encoding for every Vortex [`DType`], which
/// will correspond to an arrow-rs [`arrow_schema::DataType`].
///
/// # Views support
///
/// Binary and String views, also known as "German strings" are a better encoding format for
/// nearly all use-cases. Variable-length binary views are part of the Apache Arrow spec, and are
/// fully supported by the Datafusion query engine. We use them as our canonical string encoding
/// for all `Utf8` and `Binary` typed arrays in Vortex. They provide considerably faster filter
/// execution than the core `StringArray` and `BinaryArray` types, at the expense of potentially
/// needing [garbage collection][arrow_array::GenericByteViewArray::gc] to clear unreferenced items
/// from memory.
///
/// # For Developers
///
/// If you add another variant to this enum, make sure to update `dyn Array::is_canonical`,
/// and the fuzzer in `fuzz/fuzz_targets/array_ops.rs`.
#[derive(Debug, Clone)]
pub enum Canonical {
    Null(NullArray),
    Bool(BoolArray),
    Primitive(PrimitiveArray),
    Decimal(DecimalArray),
    VarBinView(VarBinViewArray),
    List(ListViewArray),
    FixedSizeList(FixedSizeListArray),
    Struct(StructArray),
    Extension(ExtensionArray),
    Variant(VariantArray),
}

/// Match on every canonical variant and evaluate a code block on all variants
macro_rules! match_each_canonical {
    ($self:expr, | $ident:ident | $eval:expr) => {{
        match $self {
            Canonical::Null($ident) => $eval,
            Canonical::Bool($ident) => $eval,
            Canonical::Primitive($ident) => $eval,
            Canonical::Decimal($ident) => $eval,
            Canonical::VarBinView($ident) => $eval,
            Canonical::List($ident) => $eval,
            Canonical::FixedSizeList($ident) => $eval,
            Canonical::Struct($ident) => $eval,
            Canonical::Variant($ident) => $eval,
            Canonical::Extension($ident) => $eval,
        }
    }};
}

impl Canonical {
    /// Create an empty canonical array of the given dtype.
    pub fn empty(dtype: &DType) -> Canonical {
        match dtype {
            DType::Null => Canonical::Null(NullArray::new(0)),
            DType::Bool(n) => Canonical::Bool(unsafe {
                BoolArray::new_unchecked(BitBuffer::empty(), Validity::from(n))
            }),
            DType::Primitive(ptype, n) => {
                match_each_native_ptype!(ptype, |P| {
                    Canonical::Primitive(unsafe {
                        PrimitiveArray::new_unchecked(Buffer::<P>::empty(), Validity::from(n))
                    })
                })
            }
            DType::Decimal(decimal_type, n) => {
                match_each_decimal_value_type!(
                    DecimalType::smallest_decimal_value_type(decimal_type),
                    |D| {
                        Canonical::Decimal(unsafe {
                            DecimalArray::new_unchecked::<D>(
                                Buffer::empty(),
                                *decimal_type,
                                Validity::from(n),
                            )
                        })
                    }
                )
            }
            DType::Utf8(n) => Canonical::VarBinView(unsafe {
                VarBinViewArray::new_unchecked(
                    Buffer::empty(),
                    Arc::new([]),
                    dtype.clone(),
                    Validity::from(n),
                )
            }),
            DType::Binary(n) => Canonical::VarBinView(unsafe {
                VarBinViewArray::new_unchecked(
                    Buffer::empty(),
                    Arc::new([]),
                    dtype.clone(),
                    Validity::from(n),
                )
            }),
            DType::Struct(struct_dtype, n) => Canonical::Struct(unsafe {
                StructArray::new_unchecked(
                    struct_dtype
                        .fields()
                        .map(|f| Canonical::empty(&f).into_array())
                        .collect::<Arc<[_]>>(),
                    struct_dtype.clone(),
                    0,
                    Validity::from(n),
                )
            }),
            DType::List(dtype, n) => Canonical::List(unsafe {
                ListViewArray::new_unchecked(
                    Canonical::empty(dtype).into_array(),
                    Canonical::empty(&DType::Primitive(PType::U8, Nullability::NonNullable))
                        .into_array(),
                    Canonical::empty(&DType::Primitive(PType::U8, Nullability::NonNullable))
                        .into_array(),
                    Validity::from(n),
                )
                // An empty list view is trivially copyable to a list.
                .with_zero_copy_to_list(true)
            }),
            DType::FixedSizeList(elem_dtype, list_size, null) => Canonical::FixedSizeList(unsafe {
                FixedSizeListArray::new_unchecked(
                    Canonical::empty(elem_dtype).into_array(),
                    *list_size,
                    Validity::from(null),
                    0,
                )
            }),
            DType::Extension(ext_dtype) => Canonical::Extension(ExtensionArray::new(
                ext_dtype.clone(),
                Canonical::empty(ext_dtype.storage_dtype()).into_array(),
            )),
            DType::Variant(_) => {
                vortex_panic!(InvalidArgument: "Canonical empty is not supported for Variant")
            }
        }
    }

    pub fn len(&self) -> usize {
        match_each_canonical!(self, |arr| arr.len())
    }

    pub fn dtype(&self) -> &DType {
        match_each_canonical!(self, |arr| arr.dtype())
    }

    pub fn is_empty(&self) -> bool {
        match_each_canonical!(self, |arr| arr.is_empty())
    }
}

impl Canonical {
    /// Performs a (potentially expensive) compaction operation on the array before it is complete.
    ///
    /// This is mostly relevant for the variable-length types such as Utf8, Binary or List where
    /// they can accumulate wasted space after slicing and taking operations.
    ///
    /// This operation is very expensive and can result in things like allocations, full-scans
    /// and copy operations.
    pub fn compact(&self) -> VortexResult<Canonical> {
        match self {
            Canonical::VarBinView(array) => Ok(Canonical::VarBinView(array.compact_buffers()?)),
            Canonical::List(array) => Ok(Canonical::List(
                array.rebuild(ListViewRebuildMode::TrimElements)?,
            )),
            _ => Ok(self.clone()),
        }
    }
}

// Unwrap canonical type back down to specialized type.
impl Canonical {
    pub fn as_null(&self) -> &NullArray {
        if let Canonical::Null(a) = self {
            a
        } else {
            vortex_panic!("Cannot get NullArray from {:?}", &self)
        }
    }

    pub fn into_null(self) -> NullArray {
        if let Canonical::Null(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap NullArray from {:?}", &self)
        }
    }

    pub fn as_bool(&self) -> &BoolArray {
        if let Canonical::Bool(a) = self {
            a
        } else {
            vortex_panic!("Cannot get BoolArray from {:?}", &self)
        }
    }

    pub fn into_bool(self) -> BoolArray {
        if let Canonical::Bool(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap BoolArray from {:?}", &self)
        }
    }

    pub fn as_primitive(&self) -> &PrimitiveArray {
        if let Canonical::Primitive(a) = self {
            a
        } else {
            vortex_panic!("Cannot get PrimitiveArray from {:?}", &self)
        }
    }

    pub fn into_primitive(self) -> PrimitiveArray {
        if let Canonical::Primitive(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap PrimitiveArray from {:?}", &self)
        }
    }

    pub fn as_decimal(&self) -> &DecimalArray {
        if let Canonical::Decimal(a) = self {
            a
        } else {
            vortex_panic!("Cannot get DecimalArray from {:?}", &self)
        }
    }

    pub fn into_decimal(self) -> DecimalArray {
        if let Canonical::Decimal(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap DecimalArray from {:?}", &self)
        }
    }

    pub fn as_varbinview(&self) -> &VarBinViewArray {
        if let Canonical::VarBinView(a) = self {
            a
        } else {
            vortex_panic!("Cannot get VarBinViewArray from {:?}", &self)
        }
    }

    pub fn into_varbinview(self) -> VarBinViewArray {
        if let Canonical::VarBinView(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap VarBinViewArray from {:?}", &self)
        }
    }

    pub fn as_listview(&self) -> &ListViewArray {
        if let Canonical::List(a) = self {
            a
        } else {
            vortex_panic!("Cannot get ListArray from {:?}", &self)
        }
    }

    pub fn into_listview(self) -> ListViewArray {
        if let Canonical::List(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap ListArray from {:?}", &self)
        }
    }

    pub fn as_fixed_size_list(&self) -> &FixedSizeListArray {
        if let Canonical::FixedSizeList(a) = self {
            a
        } else {
            vortex_panic!("Cannot get FixedSizeListArray from {:?}", &self)
        }
    }

    pub fn into_fixed_size_list(self) -> FixedSizeListArray {
        if let Canonical::FixedSizeList(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap FixedSizeListArray from {:?}", &self)
        }
    }

    pub fn as_struct(&self) -> &StructArray {
        if let Canonical::Struct(a) = self {
            a
        } else {
            vortex_panic!("Cannot get StructArray from {:?}", &self)
        }
    }

    pub fn into_struct(self) -> StructArray {
        if let Canonical::Struct(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap StructArray from {:?}", &self)
        }
    }

    pub fn as_extension(&self) -> &ExtensionArray {
        if let Canonical::Extension(a) = self {
            a
        } else {
            vortex_panic!("Cannot get ExtensionArray from {:?}", &self)
        }
    }

    pub fn into_extension(self) -> ExtensionArray {
        if let Canonical::Extension(a) = self {
            a
        } else {
            vortex_panic!("Cannot unwrap ExtensionArray from {:?}", &self)
        }
    }
}

impl IntoArray for Canonical {
    fn into_array(self) -> ArrayRef {
        match_each_canonical!(self, |arr| arr.into_array())
    }
}

/// Trait for types that can be converted from an owned type into an owned array variant.
///
/// # Canonicalization
///
/// This trait has a blanket implementation for all types implementing [ToCanonical].
pub trait ToCanonical {
    /// Canonicalize into a [`NullArray`] if the target is [`Null`](DType::Null) typed.
    fn to_null(&self) -> NullArray;

    /// Canonicalize into a [`BoolArray`] if the target is [`Bool`](DType::Bool) typed.
    fn to_bool(&self) -> BoolArray;

    /// Canonicalize into a [`PrimitiveArray`] if the target is [`Primitive`](DType::Primitive)
    /// typed.
    fn to_primitive(&self) -> PrimitiveArray;

    /// Canonicalize into a [`DecimalArray`] if the target is [`Decimal`](DType::Decimal)
    /// typed.
    fn to_decimal(&self) -> DecimalArray;

    /// Canonicalize into a [`StructArray`] if the target is [`Struct`](DType::Struct) typed.
    fn to_struct(&self) -> StructArray;

    /// Canonicalize into a [`ListViewArray`] if the target is [`List`](DType::List) typed.
    fn to_listview(&self) -> ListViewArray;

    /// Canonicalize into a [`FixedSizeListArray`] if the target is [`List`](DType::FixedSizeList)
    /// typed.
    fn to_fixed_size_list(&self) -> FixedSizeListArray;

    /// Canonicalize into a [`VarBinViewArray`] if the target is [`Utf8`](DType::Utf8)
    /// or [`Binary`](DType::Binary) typed.
    fn to_varbinview(&self) -> VarBinViewArray;

    /// Canonicalize into an [`ExtensionArray`] if the array is [`Extension`](DType::Extension)
    /// typed.
    fn to_extension(&self) -> ExtensionArray;
}

// Blanket impl for all Array encodings.
impl ToCanonical for ArrayRef {
    fn to_null(&self) -> NullArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_null()
    }

    fn to_bool(&self) -> BoolArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_bool()
    }

    fn to_primitive(&self) -> PrimitiveArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_primitive()
    }

    fn to_decimal(&self) -> DecimalArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_decimal()
    }

    fn to_struct(&self) -> StructArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_struct()
    }

    fn to_listview(&self) -> ListViewArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_listview()
    }

    fn to_fixed_size_list(&self) -> FixedSizeListArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_fixed_size_list()
    }

    fn to_varbinview(&self) -> VarBinViewArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_varbinview()
    }

    fn to_extension(&self) -> ExtensionArray {
        self.to_canonical()
            .vortex_expect("to_canonical failed")
            .into_extension()
    }
}

impl From<Canonical> for ArrayRef {
    fn from(value: Canonical) -> Self {
        match_each_canonical!(value, |arr| arr.into_array())
    }
}

/// Execute into [`Canonical`] by running `execute_until` with the [`AnyCanonical`] matcher.
///
/// Unlike executing into [`crate::Columnar`], this will fully expand constant arrays into their
/// canonical form. Callers should prefer to execute into `Columnar` if they are able to optimize
/// their use for constant arrays.
impl Executable for Canonical {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let result = array.execute_until::<AnyCanonical>(ctx)?;
        Ok(result
            .as_opt::<AnyCanonical>()
            .map(Canonical::from)
            .vortex_expect("execute_until::<AnyCanonical> must return a canonical array"))
    }
}

/// Recursively execute the array until it reaches canonical form along with its validity.
///
/// Callers should prefer to execute into `Columnar` instead of this specific target.
/// This target is useful when preparing arrays for writing.
pub struct CanonicalValidity(pub Canonical);

impl Executable for CanonicalValidity {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.execute::<Canonical>(ctx)? {
            n @ Canonical::Null(_) => Ok(CanonicalValidity(n)),
            Canonical::Bool(b) => {
                let validity = child_to_validity(&b.slots()[0], b.dtype().nullability());
                let len = b.len();
                let BoolDataParts { bits, offset, len } = b.into_data().into_parts(len);
                Ok(CanonicalValidity(Canonical::Bool(
                    BoolArray::try_new_from_handle(bits, offset, len, validity.execute(ctx)?)?,
                )))
            }
            Canonical::Primitive(p) => {
                let PrimitiveDataParts {
                    ptype,
                    buffer,
                    validity,
                } = p.into_data_parts();
                Ok(CanonicalValidity(Canonical::Primitive(unsafe {
                    PrimitiveArray::new_unchecked_from_handle(buffer, ptype, validity.execute(ctx)?)
                })))
            }
            Canonical::Decimal(d) => {
                let DecimalDataParts {
                    decimal_dtype,
                    values,
                    values_type,
                    validity,
                } = d.into_data_parts();
                Ok(CanonicalValidity(Canonical::Decimal(unsafe {
                    DecimalArray::new_unchecked_handle(
                        values,
                        values_type,
                        decimal_dtype,
                        validity.execute(ctx)?,
                    )
                })))
            }
            Canonical::VarBinView(vbv) => {
                let VarBinViewDataParts {
                    dtype,
                    buffers,
                    views,
                    validity,
                } = vbv.into_data_parts();
                Ok(CanonicalValidity(Canonical::VarBinView(unsafe {
                    VarBinViewArray::new_handle_unchecked(
                        views,
                        buffers,
                        dtype,
                        validity.execute(ctx)?,
                    )
                })))
            }
            Canonical::List(l) => {
                let zctl = l.is_zero_copy_to_list();
                let ListViewDataParts {
                    elements,
                    offsets,
                    sizes,
                    validity,
                    ..
                } = l.into_data_parts();
                Ok(CanonicalValidity(Canonical::List(unsafe {
                    ListViewArray::new_unchecked(elements, offsets, sizes, validity.execute(ctx)?)
                        .with_zero_copy_to_list(zctl)
                })))
            }
            Canonical::FixedSizeList(fsl) => {
                let list_size = fsl.list_size();
                let len = fsl.len();
                let parts = fsl.into_data_parts();
                let elements = parts.elements;
                let validity = parts.validity;
                Ok(CanonicalValidity(Canonical::FixedSizeList(
                    FixedSizeListArray::new(elements, list_size, validity.execute(ctx)?, len),
                )))
            }
            Canonical::Struct(st) => {
                let len = st.len();
                let StructDataParts {
                    struct_fields,
                    fields,
                    validity,
                } = st.into_data_parts();
                Ok(CanonicalValidity(Canonical::Struct(unsafe {
                    StructArray::new_unchecked(fields, struct_fields, len, validity.execute(ctx)?)
                })))
            }
            Canonical::Extension(ext) => Ok(CanonicalValidity(Canonical::Extension(
                ExtensionArray::new(
                    ext.ext_dtype().clone(),
                    ext.storage_array()
                        .clone()
                        .execute::<CanonicalValidity>(ctx)?
                        .0
                        .into_array(),
                ),
            ))),
            Canonical::Variant(variant) => {
                Ok(CanonicalValidity(Canonical::Variant(VariantArray::new(
                    variant
                        .child()
                        .clone()
                        .execute::<CanonicalValidity>(ctx)?
                        .0
                        .into_array(),
                ))))
            }
        }
    }
}

/// Recursively execute the array until all of its children are canonical.
///
/// This method is useful to guarantee that all operators are fully executed,
/// callers should prefer an execution target that's suitable for their use case instead of this one.
pub struct RecursiveCanonical(pub Canonical);

impl Executable for RecursiveCanonical {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.execute::<Canonical>(ctx)? {
            n @ Canonical::Null(_) => Ok(RecursiveCanonical(n)),
            Canonical::Bool(b) => {
                let validity = child_to_validity(&b.slots()[0], b.dtype().nullability());
                let len = b.len();
                let BoolDataParts { bits, offset, len } = b.into_data().into_parts(len);
                Ok(RecursiveCanonical(Canonical::Bool(
                    BoolArray::try_new_from_handle(bits, offset, len, validity.execute(ctx)?)?,
                )))
            }
            Canonical::Primitive(p) => {
                let PrimitiveDataParts {
                    ptype,
                    buffer,
                    validity,
                } = p.into_data_parts();
                Ok(RecursiveCanonical(Canonical::Primitive(unsafe {
                    PrimitiveArray::new_unchecked_from_handle(buffer, ptype, validity.execute(ctx)?)
                })))
            }
            Canonical::Decimal(d) => {
                let DecimalDataParts {
                    decimal_dtype,
                    values,
                    values_type,
                    validity,
                } = d.into_data_parts();
                Ok(RecursiveCanonical(Canonical::Decimal(unsafe {
                    DecimalArray::new_unchecked_handle(
                        values,
                        values_type,
                        decimal_dtype,
                        validity.execute(ctx)?,
                    )
                })))
            }
            Canonical::VarBinView(vbv) => {
                let VarBinViewDataParts {
                    dtype,
                    buffers,
                    views,
                    validity,
                } = vbv.into_data_parts();
                Ok(RecursiveCanonical(Canonical::VarBinView(unsafe {
                    VarBinViewArray::new_handle_unchecked(
                        views,
                        buffers,
                        dtype,
                        validity.execute(ctx)?,
                    )
                })))
            }
            Canonical::List(l) => {
                let zctl = l.is_zero_copy_to_list();
                let ListViewDataParts {
                    elements,
                    offsets,
                    sizes,
                    validity,
                    ..
                } = l.into_data_parts();
                Ok(RecursiveCanonical(Canonical::List(unsafe {
                    ListViewArray::new_unchecked(
                        elements.execute::<RecursiveCanonical>(ctx)?.0.into_array(),
                        offsets.execute::<RecursiveCanonical>(ctx)?.0.into_array(),
                        sizes.execute::<RecursiveCanonical>(ctx)?.0.into_array(),
                        validity.execute(ctx)?,
                    )
                    .with_zero_copy_to_list(zctl)
                })))
            }
            Canonical::FixedSizeList(fsl) => {
                let list_size = fsl.list_size();
                let len = fsl.len();
                let parts = fsl.into_data_parts();
                let elements = parts.elements;
                let validity = parts.validity;
                Ok(RecursiveCanonical(Canonical::FixedSizeList(
                    FixedSizeListArray::new(
                        elements.execute::<RecursiveCanonical>(ctx)?.0.into_array(),
                        list_size,
                        validity.execute(ctx)?,
                        len,
                    ),
                )))
            }
            Canonical::Struct(st) => {
                let len = st.len();
                let StructDataParts {
                    struct_fields,
                    fields,
                    validity,
                } = st.into_data_parts();
                let executed_fields = fields
                    .iter()
                    .map(|f| Ok(f.clone().execute::<RecursiveCanonical>(ctx)?.0.into_array()))
                    .collect::<VortexResult<Arc<[_]>>>()?;

                Ok(RecursiveCanonical(Canonical::Struct(unsafe {
                    StructArray::new_unchecked(
                        executed_fields,
                        struct_fields,
                        len,
                        validity.execute(ctx)?,
                    )
                })))
            }
            Canonical::Extension(ext) => Ok(RecursiveCanonical(Canonical::Extension(
                ExtensionArray::new(
                    ext.ext_dtype().clone(),
                    ext.storage_array()
                        .clone()
                        .execute::<RecursiveCanonical>(ctx)?
                        .0
                        .into_array(),
                ),
            ))),
            Canonical::Variant(variant) => {
                Ok(RecursiveCanonical(Canonical::Variant(VariantArray::new(
                    variant
                        .child()
                        .clone()
                        .execute::<RecursiveCanonical>(ctx)?
                        .0
                        .into_array(),
                ))))
            }
        }
    }
}

/// Execute a primitive typed array into a buffer of native values, assuming all values are valid.
///
/// # Errors
///
/// Returns a `VortexError` if the array is not all-valid (has any nulls).
impl<T: NativePType> Executable for Buffer<T> {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let array = PrimitiveArray::execute(array, ctx)?;
        vortex_ensure!(
            matches!(
                array.validity()?,
                Validity::NonNullable | Validity::AllValid
            ),
            "Cannot execute to native buffer: array is not all-valid."
        );
        Ok(array.into_buffer())
    }
}

/// Execute the array to canonical form and unwrap as a [`PrimitiveArray`].
///
/// This will panic if the array's dtype is not primitive.
impl Executable for PrimitiveArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<Primitive>() {
            Ok(primitive) => Ok(primitive),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_primitive()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`BoolArray`].
///
/// This will panic if the array's dtype is not bool.
impl Executable for BoolArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<Bool>() {
            Ok(bool_array) => Ok(bool_array),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_bool()),
        }
    }
}

/// Execute the array to a [`BitBuffer`], aka a non-nullable  [`BoolArray`].
///
/// This will panic if the array's dtype is not non-nullable bool.
impl Executable for BitBuffer {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let bool = BoolArray::execute(array, ctx)?;
        assert!(
            !bool.dtype().is_nullable(),
            "bit buffer execute only works with non-nullable bool arrays"
        );
        Ok(bool.into_bit_buffer())
    }
}

/// Execute the array to canonical form and unwrap as a [`NullArray`].
///
/// This will panic if the array's dtype is not null.
impl Executable for NullArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<Null>() {
            Ok(null_array) => Ok(null_array),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_null()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`VarBinViewArray`].
///
/// This will panic if the array's dtype is not utf8 or binary.
impl Executable for VarBinViewArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<VarBinView>() {
            Ok(varbinview) => Ok(varbinview),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_varbinview()),
        }
    }
}

/// Execute the array to canonical form and unwrap as an [`ExtensionArray`].
///
/// This will panic if the array's dtype is not an extension type.
impl Executable for ExtensionArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<Extension>() {
            Ok(ext_array) => Ok(ext_array),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_extension()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`DecimalArray`].
///
/// This will panic if the array's dtype is not decimal.
impl Executable for DecimalArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<Decimal>() {
            Ok(decimal) => Ok(decimal),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_decimal()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`ListViewArray`].
///
/// This will panic if the array's dtype is not list.
impl Executable for ListViewArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<ListView>() {
            Ok(list) => Ok(list),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_listview()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`FixedSizeListArray`].
///
/// This will panic if the array's dtype is not fixed size list.
impl Executable for FixedSizeListArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<FixedSizeList>() {
            Ok(fsl) => Ok(fsl),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_fixed_size_list()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`StructArray`].
///
/// This will panic if the array's dtype is not struct.
impl Executable for StructArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_downcast::<Struct>() {
            Ok(struct_array) => Ok(struct_array),
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_struct()),
        }
    }
}

/// A view into a canonical array type.
///
/// Uses `ArrayView<V>` because these are obtained by
/// downcasting through the `Matcher` trait which returns `ArrayView<V>`.
#[derive(Debug, Clone, Copy)]
pub enum CanonicalView<'a> {
    Null(ArrayView<'a, Null>),
    Bool(ArrayView<'a, Bool>),
    Primitive(ArrayView<'a, Primitive>),
    Decimal(ArrayView<'a, Decimal>),
    VarBinView(ArrayView<'a, VarBinView>),
    List(ArrayView<'a, ListView>),
    FixedSizeList(ArrayView<'a, FixedSizeList>),
    Struct(ArrayView<'a, Struct>),
    Extension(ArrayView<'a, Extension>),
    Variant(ArrayView<'a, Variant>),
}

impl From<CanonicalView<'_>> for Canonical {
    fn from(value: CanonicalView<'_>) -> Self {
        match value {
            CanonicalView::Null(a) => Canonical::Null(a.into_owned()),
            CanonicalView::Bool(a) => Canonical::Bool(a.into_owned()),
            CanonicalView::Primitive(a) => Canonical::Primitive(a.into_owned()),
            CanonicalView::Decimal(a) => Canonical::Decimal(a.into_owned()),
            CanonicalView::VarBinView(a) => Canonical::VarBinView(a.into_owned()),
            CanonicalView::List(a) => Canonical::List(a.into_owned()),
            CanonicalView::FixedSizeList(a) => Canonical::FixedSizeList(a.into_owned()),
            CanonicalView::Struct(a) => Canonical::Struct(a.into_owned()),
            CanonicalView::Extension(a) => Canonical::Extension(a.into_owned()),
            CanonicalView::Variant(a) => Canonical::Variant(a.into_owned()),
        }
    }
}

impl CanonicalView<'_> {
    /// Convert to a type-erased [`ArrayRef`].
    pub fn to_array_ref(&self) -> ArrayRef {
        match self {
            CanonicalView::Null(a) => a.array().clone(),
            CanonicalView::Bool(a) => a.array().clone(),
            CanonicalView::Primitive(a) => a.array().clone(),
            CanonicalView::Decimal(a) => a.array().clone(),
            CanonicalView::VarBinView(a) => a.array().clone(),
            CanonicalView::List(a) => a.array().clone(),
            CanonicalView::FixedSizeList(a) => a.array().clone(),
            CanonicalView::Struct(a) => a.array().clone(),
            CanonicalView::Extension(a) => a.array().clone(),
            CanonicalView::Variant(a) => a.array().clone(),
        }
    }
}

/// A matcher for any canonical array type.
pub struct AnyCanonical;
impl Matcher for AnyCanonical {
    type Match<'a> = CanonicalView<'a>;

    fn matches(array: &ArrayRef) -> bool {
        array.is::<Null>()
            || array.is::<Bool>()
            || array.is::<Primitive>()
            || array.is::<Decimal>()
            || array.is::<Struct>()
            || array.is::<ListView>()
            || array.is::<FixedSizeList>()
            || array.is::<VarBinView>()
            || array.is::<Variant>()
            || array.is::<Extension>()
            || array.is::<Variant>()
    }

    fn try_match<'a>(array: &'a ArrayRef) -> Option<Self::Match<'a>> {
        if let Some(a) = array.as_opt::<Null>() {
            Some(CanonicalView::Null(a))
        } else if let Some(a) = array.as_opt::<Bool>() {
            Some(CanonicalView::Bool(a))
        } else if let Some(a) = array.as_opt::<Primitive>() {
            Some(CanonicalView::Primitive(a))
        } else if let Some(a) = array.as_opt::<Decimal>() {
            Some(CanonicalView::Decimal(a))
        } else if let Some(a) = array.as_opt::<Struct>() {
            Some(CanonicalView::Struct(a))
        } else if let Some(a) = array.as_opt::<ListView>() {
            Some(CanonicalView::List(a))
        } else if let Some(a) = array.as_opt::<FixedSizeList>() {
            Some(CanonicalView::FixedSizeList(a))
        } else if let Some(a) = array.as_opt::<VarBinView>() {
            Some(CanonicalView::VarBinView(a))
        } else if let Some(a) = array.as_opt::<Variant>() {
            Some(CanonicalView::Variant(a))
        } else {
            array.as_opt::<Extension>().map(CanonicalView::Extension)
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::Array as ArrowArray;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::ListArray as ArrowListArray;
    use arrow_array::PrimitiveArray as ArrowPrimitiveArray;
    use arrow_array::StringArray;
    use arrow_array::StringViewArray;
    use arrow_array::StructArray as ArrowStructArray;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::types::Int64Type;
    use arrow_array::types::UInt64Type;
    use arrow_buffer::NullBufferBuilder;
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use vortex_buffer::buffer;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrow::FromArrowArray;
    use crate::arrow::IntoArrowArray;
    use crate::canonical::StructArray;

    #[test]
    fn test_canonicalize_nested_struct() {
        // Create a struct array with multiple internal components.
        let nested_struct_array = StructArray::from_fields(&[
            ("a", buffer![1u64].into_array()),
            (
                "b",
                StructArray::from_fields(&[(
                    "inner_a",
                    // The nested struct contains a ConstantArray representing the primitive array
                    //   [100i64]
                    // ConstantArray is not a canonical type, so converting `into_arrow()` should
                    // map this to the nearest canonical type (PrimitiveArray).
                    ConstantArray::new(100i64, 1).into_array(),
                )])
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap();

        let arrow_struct = nested_struct_array
            .into_array()
            .into_arrow_preferred()
            .unwrap()
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .cloned()
            .unwrap();

        assert!(
            arrow_struct
                .column(0)
                .as_any()
                .downcast_ref::<ArrowPrimitiveArray<UInt64Type>>()
                .is_some()
        );

        let inner_struct = Arc::clone(arrow_struct.column(1))
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .cloned()
            .unwrap();

        let inner_a = inner_struct
            .column(0)
            .as_any()
            .downcast_ref::<ArrowPrimitiveArray<Int64Type>>();
        assert!(inner_a.is_some());

        assert_eq!(
            inner_a.cloned().unwrap(),
            ArrowPrimitiveArray::from_iter([100i64])
        );
    }

    #[test]
    fn roundtrip_struct() {
        let mut nulls = NullBufferBuilder::new(6);
        nulls.append_n_non_nulls(4);
        nulls.append_null();
        nulls.append_non_null();
        let names = Arc::new(StringViewArray::from_iter(vec![
            Some("Joseph"),
            None,
            Some("Angela"),
            Some("Mikhail"),
            None,
            None,
        ]));
        let ages = Arc::new(ArrowPrimitiveArray::<Int32Type>::from(vec![
            Some(25),
            Some(31),
            None,
            Some(57),
            None,
            None,
        ]));

        let arrow_struct = ArrowStructArray::new(
            vec![
                Arc::new(Field::new("name", DataType::Utf8View, true)),
                Arc::new(Field::new("age", DataType::Int32, true)),
            ]
            .into(),
            vec![names, ages],
            nulls.finish(),
        );

        let vortex_struct = ArrayRef::from_arrow(&arrow_struct, true).unwrap();

        assert_eq!(
            &arrow_struct,
            vortex_struct.into_arrow_preferred().unwrap().as_struct()
        );
    }

    #[test]
    fn roundtrip_list() {
        let names = Arc::new(StringArray::from_iter(vec![
            Some("Joseph"),
            Some("Angela"),
            Some("Mikhail"),
        ]));

        let arrow_list = ArrowListArray::new(
            Arc::new(Field::new_list_field(DataType::Utf8, true)),
            OffsetBuffer::from_lengths(vec![0, 2, 1]),
            names,
            None,
        );
        let list_data_type = arrow_list.data_type();

        let vortex_list = ArrayRef::from_arrow(&arrow_list, true).unwrap();

        let rt_arrow_list = vortex_list.into_arrow(list_data_type).unwrap();

        assert_eq!(
            (Arc::new(arrow_list.clone()) as ArrowArrayRef).as_ref(),
            rt_arrow_list.as_ref()
        );
    }
}

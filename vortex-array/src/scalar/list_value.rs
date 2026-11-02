// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ListValue`], the array-backed value of a list or fixed-size list scalar.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::LazyLock;

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::BufferString;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::array::ArrayView;
use crate::VortexSessionExecute;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinView;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinViewBuilder;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::Nullability;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::validity::Validity;

/// The session used to read nested element values when the caller has no execution context.
///
/// Reading an element of a recursively canonical array never needs a kernel, so an empty session
/// is enough.
static ELEMENT_SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// An execution context for reading or copying the elements of a [`ListValue`].
///
/// The elements are recursively canonical, so no kernel runs and an empty session suffices.
pub(crate) fn element_ctx() -> ExecutionCtx {
    ELEMENT_SESSION.create_execution_ctx()
}

/// The value of a non-null list or fixed-size list scalar.
///
/// The elements are held as a single array of the list's element dtype, so a list row of an array
/// becomes a scalar by slicing, and a list scalar becomes an array by reusing that slice.
///
/// Invariant: the elements array is *recursively canonical*: it and all of its descendants are
/// canonical encodings, so element values can be read without an [`ExecutionCtx`]. The one
/// exception is an element dtype with no canonical builder (unions and variants), whose elements
/// built from scalars are held as constant chunks.
///
/// [`PartialEq`], [`Hash`] and ordering compare element values, never physical buffers, so two
/// equal lists compare equal whatever their elements' encodings.
#[derive(Clone)]
pub struct ListValue {
    elements: ArrayRef,
    /// Whether every element is non-null, computed once so the fast paths need not ask the
    /// elements for their validity.
    all_valid: bool,
}

impl ListValue {
    /// Creates a list value from an elements array, executing it to recursively canonical form if
    /// it is not already.
    pub fn try_from_array(elements: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if is_recursively_canonical(&elements) {
            return Ok(Self::wrap(elements));
        }
        let elements = elements.execute::<RecursiveCanonical>(ctx)?.0.into_array();
        Ok(Self::wrap(elements))
    }

    /// Creates a list value from an elements array that is already recursively canonical.
    ///
    /// # Safety
    ///
    /// The caller must ensure `elements` and all of its descendants are canonical encodings.
    pub unsafe fn new_unchecked(elements: ArrayRef) -> Self {
        debug_assert!(
            is_recursively_canonical(&elements) || !has_builder(elements.dtype()),
            "ListValue elements must be recursively canonical"
        );
        Self::wrap(elements)
    }

    fn wrap(elements: ArrayRef) -> Self {
        let all_valid = all_valid(&elements);
        Self {
            elements,
            all_valid,
        }
    }

    /// Creates a list value from element values of `element_dtype`.
    ///
    /// The caller must ensure each value is valid for `element_dtype`, as [`Scalar::validate`]
    /// checks.
    pub fn from_values<I>(element_dtype: &DType, values: I) -> Self
    where
        I: IntoIterator<Item = Option<ScalarValue>>,
        I::IntoIter: ExactSizeIterator,
    {
        let values: Vec<Option<ScalarValue>> = values.into_iter().collect();
        let elements = build_elements(element_dtype, &values, BufferAllocatorRef::static_ref());
        if is_flat(element_dtype) {
            // Flat elements are written straight into canonical arrays.
            return Self {
                elements,
                all_valid: values.iter().all(Option::is_some),
            };
        }
        Self::from_built(elements)
    }

    /// Creates a list value from element scalars of `element_dtype`.
    pub fn from_scalars(element_dtype: &DType, scalars: &[Scalar]) -> Self {
        if !has_builder(element_dtype) {
            return Self::from_built(constant_chunks(element_dtype, scalars.iter().cloned()));
        }
        if !is_flat(element_dtype) {
            let mut builder = builder_with_capacity_in(
                element_dtype,
                scalars.len(),
                BufferAllocatorRef::static_ref(),
            );
            for scalar in scalars {
                builder
                    .append_scalar(scalar)
                    .vortex_expect("list element scalar matches the element dtype");
            }
            return Self::from_built(builder.finish());
        }
        Self::from_values(
            element_dtype,
            scalars.iter().map(|scalar| scalar.value().cloned()),
        )
    }

    /// Wraps an array produced by a canonical builder.
    fn from_built(elements: ArrayRef) -> Self {
        if is_recursively_canonical(&elements) || !has_builder(elements.dtype()) {
            return Self::wrap(elements);
        }
        let mut ctx = element_ctx();
        Self::try_from_array(elements, &mut ctx)
            .vortex_expect("builder output executes to canonical form")
    }

    /// The elements of the list, as a recursively canonical array of the element dtype.
    #[inline]
    pub fn elements(&self) -> &ArrayRef {
        &self.elements
    }

    /// Consumes the value, returning the elements array.
    #[inline]
    pub fn into_elements(self) -> ArrayRef {
        self.elements
    }

    /// The dtype of the elements.
    #[inline]
    pub fn element_dtype(&self) -> &DType {
        self.elements.dtype()
    }

    /// The number of elements in the list.
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns `true` if the list has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// The elements as a slice of `T`, if they are non-null primitives of type `T`.
    pub fn as_slice<T: NativePType>(&self) -> Option<&[T]> {
        let primitive = self.elements.as_opt::<Primitive>()?;
        (primitive.ptype() == T::PTYPE && self.all_valid)
            .then(|| primitive.data().as_slice::<T>())
    }

    /// The same list with elements of `element_dtype`, a flat dtype equal to the current element
    /// dtype up to nullability.
    ///
    /// Fails if the list holds a null and `element_dtype` is non-nullable.
    pub(crate) fn with_flat_element_nullability(&self, element_dtype: &DType) -> VortexResult<Self> {
        if !element_dtype.is_nullable() && !self.all_valid {
            vortex_bail!(
                "Cannot cast a list with null elements to non-nullable elements {element_dtype}"
            );
        }
        if self.element_dtype() == element_dtype {
            return Ok(self.clone());
        }
        Ok(match self.primitive_tile() {
            // Non-null primitives only swap their validity, keeping the values buffer.
            Some(primitive) => {
                // SAFETY: the buffer holds `len` values of `ptype`, and the validity carries no
                // nulls.
                let elements = unsafe {
                    PrimitiveArray::new_unchecked_from_handle(
                        primitive.buffer_handle().clone(),
                        primitive.ptype(),
                        Validity::from(element_dtype.nullability()),
                    )
                };
                Self {
                    elements: elements.into_array(),
                    all_valid: true,
                }
            }
            None => Self::from_values(element_dtype, self.values()),
        })
    }

    /// The elements as a primitive array, if they are non-null primitives.
    pub(crate) fn primitive_tile(&self) -> Option<ArrayView<'_, Primitive>> {
        self.all_valid
            .then(|| self.elements.as_opt::<Primitive>())
            .flatten()
    }

    /// Returns `true` if any element of the list is null.
    pub fn has_null_element(&self) -> bool {
        !self.all_valid
    }

    /// The value of the element at `index`, or [`None`] if the element is null.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn value_at(&self, index: usize) -> Option<ScalarValue> {
        ValueReader::new(&self.elements).value_at(index)
    }

    /// The element at `index` as a scalar.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn scalar_at(&self, index: usize) -> Scalar {
        // SAFETY: element values read from the elements array match its dtype.
        unsafe { Scalar::new_unchecked(self.element_dtype().clone(), self.value_at(index)) }
    }

    /// Iterates the values of the elements, [`None`] for a null element.
    pub fn values(&self) -> impl ExactSizeIterator<Item = Option<ScalarValue>> + '_ {
        let mut reader = ValueReader::new(&self.elements);
        (0..self.len()).map(move |index| reader.value_at(index))
    }

    /// Compares two lists element by element, then by length.
    ///
    /// Returns [`None`] if any pair of elements is incomparable.
    pub(crate) fn partial_cmp_with(
        &self,
        other: &Self,
        mut cmp: impl FnMut(Option<&ScalarValue>, Option<&ScalarValue>) -> Option<Ordering>,
    ) -> Option<Ordering> {
        if let Some(ordering) = fast_cmp(self, other) {
            return Some(ordering);
        }
        for (lhs, rhs) in self.values().zip(other.values()) {
            match cmp(lhs.as_ref(), rhs.as_ref())? {
                Ordering::Equal => continue,
                ordering => return Some(ordering),
            }
        }
        Some(self.len().cmp(&other.len()))
    }
}

impl Debug for ListValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.values()).finish()
    }
}

impl PartialEq for ListValue {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        if ArrayRef::ptr_eq(&self.elements, &other.elements) {
            return true;
        }
        if let Some(eq) = fast_eq(self, other) {
            return eq;
        }
        self.values().eq(other.values())
    }
}

impl Eq for ListValue {}

impl Hash for ListValue {
    /// Hashes the same way as the element values in a `Vec`, so equal lists hash equally
    /// whatever their elements' physical buffers.
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        if self.all_valid
            && let Some(primitive) = self.elements.as_opt::<Primitive>()
        {
            // The same hashes as the generic loop below, without a dispatch per element.
            match_each_native_ptype!(primitive.ptype(), |T| {
                for value in primitive.as_slice::<T>() {
                    hash_primitive_element(PValue::from(*value), state);
                }
            });
            return;
        }
        for value in self.values() {
            match value {
                Some(ScalarValue::Primitive(value)) => hash_primitive_element(value, state),
                value => value.hash(state),
            }
        }
    }
}

/// Hashes a non-null primitive element, distinct from the hash of any other element value.
#[inline]
fn hash_primitive_element<H: Hasher>(value: PValue, state: &mut H) {
    state.write_u8(u8::MAX);
    value.hash(state);
}

/// Compares two lists of non-null primitive elements of the same type directly on their buffers.
///
/// Returns [`None`] when the fast path does not apply.
fn fast_eq(lhs: &ListValue, rhs: &ListValue) -> Option<bool> {
    let lhs_prim = lhs.elements.as_opt::<Primitive>()?;
    let rhs_prim = rhs.elements.as_opt::<Primitive>()?;
    if lhs_prim.ptype() != rhs_prim.ptype() || !lhs.all_valid || !rhs.all_valid {
        return None;
    }
    // Integer equality is bit equality. Floats compare by total order in `PValue`, which is also
    // bit equality for values of the same type.
    Some(
        lhs_prim.buffer_handle().as_host().as_slice()
            == rhs_prim.buffer_handle().as_host().as_slice(),
    )
}

fn fast_cmp(lhs: &ListValue, rhs: &ListValue) -> Option<Ordering> {
    let lhs_prim = lhs.elements.as_opt::<Primitive>()?;
    let rhs_prim = rhs.elements.as_opt::<Primitive>()?;
    if lhs_prim.ptype() != rhs_prim.ptype() || !lhs.all_valid || !rhs.all_valid {
        return None;
    }
    match_each_native_ptype!(lhs_prim.ptype(), |T| {
        let l = lhs_prim.as_slice::<T>();
        let r = rhs_prim.as_slice::<T>();
        for (a, b) in l.iter().zip(r.iter()) {
            match NativePType::total_compare(*a, *b) {
                Ordering::Equal => continue,
                ordering => return Some(ordering),
            }
        }
        Some(l.len().cmp(&r.len()))
    })
}

fn all_valid(array: &ArrayRef) -> bool {
    if array.is::<Null>() {
        return array.is_empty();
    }
    match array.validity() {
        Ok(Validity::NonNullable | Validity::AllValid) => true,
        Ok(Validity::Array(validity)) => validity
            .as_opt::<Bool>()
            .is_some_and(|bits| bits.to_bit_buffer().true_count() == bits.len()),
        _ => false,
    }
}

/// Rewrites the list values under `value` to hold the element dtypes of `dtype`, which equals the
/// value's dtype up to nullability.
///
/// Values carry no nullability except in a list's elements array, so only lists whose element
/// nullability changes are rebuilt. Fails if a null element would land in a non-nullable dtype.
pub(crate) fn retype_list_values(dtype: &DType, value: ScalarValue) -> VortexResult<ScalarValue> {
    match (dtype, value) {
        (
            DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..),
            ScalarValue::List(list),
        ) => {
            if list.element_dtype() == element_dtype.as_ref() {
                return Ok(ScalarValue::List(list));
            }
            if is_flat(element_dtype) {
                return Ok(ScalarValue::List(
                    list.with_flat_element_nullability(element_dtype)?,
                ));
            }
            let values = list
                .values()
                .map(|value| match value {
                    Some(value) => retype_list_values(element_dtype, value).map(Some),
                    None if element_dtype.is_nullable() => Ok(None),
                    None => vortex_bail!(
                        "Cannot cast a list with null elements to non-nullable elements \
                         {element_dtype}"
                    ),
                })
                .collect::<VortexResult<Vec<_>>>()?;
            Ok(ScalarValue::List(ListValue::from_values(element_dtype, values)))
        }
        (DType::Struct(fields, _), ScalarValue::Tuple(values)) => Ok(ScalarValue::Tuple(
            fields
                .field_dtypes()
                .zip(values)
                .map(|(field_dtype, value)| {
                    value
                        .map(|value| retype_list_values(field_dtype, value))
                        .transpose()
                })
                .collect::<VortexResult<_>>()?,
        )),
        (DType::Map(map_dtype, _), ScalarValue::Tuple(entries)) => {
            let entry_dtype = map_dtype.entries_dtype();
            Ok(ScalarValue::Tuple(
                entries
                    .into_iter()
                    .map(|entry| {
                        entry
                            .map(|entry| retype_list_values(&entry_dtype, entry))
                            .transpose()
                    })
                    .collect::<VortexResult<_>>()?,
            ))
        }
        (DType::Extension(ext_dtype), value) => {
            retype_list_values(ext_dtype.storage_dtype(), value)
        }
        (_, value) => Ok(value),
    }
}

/// Returns `true` if a canonical builder exists for `dtype`.
///
/// Unions and variants have none yet, and neither does any dtype nesting them.
fn has_builder(dtype: &DType) -> bool {
    match dtype {
        DType::Union(..) | DType::Variant(_) => false,
        DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..) => {
            has_builder(element_dtype)
        }
        DType::Struct(fields, _) => fields.field_dtypes().all(has_builder),
        DType::Map(map_dtype, _) => {
            has_builder(&map_dtype.key_dtype()) && has_builder(&map_dtype.value_dtype())
        }
        DType::Extension(ext_dtype) => has_builder(ext_dtype.storage_dtype()),
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
    }
}

/// Holds element scalars of a dtype with no builder as one constant chunk per element.
///
/// Such elements are the exception to the recursively canonical invariant. Every read, equality
/// and hash of a [`ListValue`] goes through element values, so the representation does not affect
/// them.
fn constant_chunks(element_dtype: &DType, scalars: impl Iterator<Item = Scalar>) -> ArrayRef {
    let chunks = scalars
        .map(|scalar| ConstantArray::new(scalar, 1).into_array())
        .collect::<Vec<_>>();
    // SAFETY: every chunk holds a scalar of `element_dtype`.
    unsafe { ChunkedArray::new_unchecked(chunks, element_dtype.clone()) }.into_array()
}

/// Returns `true` if values of `dtype` are written straight to buffers by [`build_elements`].
pub(crate) fn is_flat(dtype: &DType) -> bool {
    match dtype {
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        DType::Extension(ext_dtype) => is_flat(ext_dtype.storage_dtype()),
        _ => false,
    }
}

/// The array of `elements`, the values of a list of `element_dtype`.
///
/// Flat elements are written straight from their values, since a scalar per element costs more
/// than a probe over the result does. Nested elements go through the canonical builder.
fn build_elements(
    element_dtype: &DType,
    elements: &[Option<ScalarValue>],
    allocator: &BufferAllocatorRef,
) -> ArrayRef {
    match element_dtype {
        DType::Null => NullArray::new(elements.len()).into_array(),
        DType::Bool(nullability) => {
            let bits = BitBuffer::from_iter(elements.iter().map(|element| match element {
                Some(ScalarValue::Bool(value)) => *value,
                Some(_) => vortex_panic!("bool list holds a non-bool element"),
                None => false,
            }));
            BoolArray::new(bits, element_validity(elements, *nullability)).into_array()
        }
        DType::Primitive(ptype, nullability) => match_each_native_ptype!(ptype, |T| {
            let mut buffer = BufferMut::<T>::with_capacity_in(elements.len(), allocator.clone());
            buffer.extend(elements.iter().map(|element| {
                match element {
                    Some(ScalarValue::Primitive(value)) => value
                        .cast::<T>()
                        .vortex_expect("list element of the list's element ptype"),
                    Some(_) => vortex_panic!("primitive list holds a non-primitive element"),
                    None => T::default(),
                }
            }));
            PrimitiveArray::new(buffer.freeze(), element_validity(elements, *nullability))
                .into_array()
        }),
        DType::Decimal(decimal, nullability) => {
            let values_type = decimal.canonical_values_type();
            match_each_decimal_value_type!(values_type, |T| {
                let mut buffer =
                    BufferMut::<T>::with_capacity_in(elements.len(), allocator.clone());
                buffer.extend(elements.iter().map(|element| {
                    match element {
                        Some(ScalarValue::Decimal(value)) => value
                            .cast::<T>()
                            .vortex_expect("list element fits the list's decimal type"),
                        Some(_) => vortex_panic!("decimal list holds a non-decimal element"),
                        None => T::default(),
                    }
                }));
                DecimalArray::new(
                    buffer.freeze(),
                    *decimal,
                    element_validity(elements, *nullability),
                )
                .into_array()
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                element_dtype.clone(),
                elements.len(),
                allocator.clone(),
            );
            for element in elements {
                match element {
                    None => builder.append_null(),
                    Some(ScalarValue::Utf8(value)) => builder.append_value(value.as_bytes()),
                    Some(ScalarValue::Binary(value)) => builder.append_value(value.as_slice()),
                    Some(_) => vortex_panic!("byte list holds a non-byte element"),
                }
            }
            builder.finish()
        }
        // An extension value is its storage value.
        DType::Extension(ext_dtype) => ExtensionArray::new(
            ext_dtype.clone(),
            build_elements(ext_dtype.storage_dtype(), elements, allocator),
        )
        .into_array(),
        _ if !has_builder(element_dtype) => constant_chunks(
            element_dtype,
            elements.iter().map(|element| {
                // SAFETY: the caller guarantees each value matches `element_dtype`.
                unsafe { Scalar::new_unchecked(element_dtype.clone(), element.clone()) }
            }),
        ),
        _ => {
            let mut builder = builder_with_capacity_in(element_dtype, elements.len(), allocator);
            for element in elements {
                match element {
                    Some(element) => {
                        builder
                            .append_scalar(&unsafe {
                                Scalar::new_unchecked(element_dtype.clone(), Some(element.clone()))
                            })
                            .vortex_expect("list element scalar was invalid");
                    }
                    None => {
                        builder.append_null();
                    }
                }
            }
            builder.finish()
        }
    }
}

/// The validity of a list's elements, a null element being a `None` value.
fn element_validity(elements: &[Option<ScalarValue>], nullability: Nullability) -> Validity {
    match nullability {
        Nullability::NonNullable => Validity::NonNullable,
        Nullability::Nullable if elements.iter().all(Option::is_some) => Validity::AllValid,
        Nullability::Nullable => {
            Validity::from(BitBuffer::from_iter(elements.iter().map(Option::is_some)))
        }
    }
}

/// Converts every list or fixed-size list value under `dtype` that is still in the legacy
/// [`ScalarValue::Tuple`] form into a [`ScalarValue::List`], in place.
///
/// Values that hold no lists are walked without allocating. The caller must have validated
/// `value` against `dtype`.
pub(crate) fn normalize_list_values(dtype: &DType, value: &mut ScalarValue) {
    match dtype {
        DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..) => {
            if let ScalarValue::Tuple(elements) = value {
                let elements = std::mem::take(elements);
                *value = ScalarValue::List(ListValue::from_values(
                    element_dtype,
                    elements.into_iter().map(|mut element| {
                        if let Some(element) = element.as_mut() {
                            normalize_list_values(element_dtype, element);
                        }
                        element
                    }),
                ));
            }
        }
        DType::Struct(fields, _) => {
            if let ScalarValue::Tuple(values) = value {
                for (field_dtype, field_value) in fields.field_dtypes().zip(values.iter_mut()) {
                    if let Some(field_value) = field_value.as_mut() {
                        normalize_list_values(field_dtype, field_value);
                    }
                }
            }
        }
        DType::Map(map_dtype, _) => {
            if let ScalarValue::Tuple(entries) = value {
                let key_dtype = map_dtype.key_dtype();
                let value_dtype = map_dtype.value_dtype();
                for entry in entries.iter_mut().flatten() {
                    if let ScalarValue::Tuple(kv) = entry {
                        if let Some(Some(key)) = kv.get_mut(0) {
                            normalize_list_values(&key_dtype, key);
                        }
                        if let Some(Some(value)) = kv.get_mut(1) {
                            normalize_list_values(&value_dtype, value);
                        }
                    }
                }
            }
        }
        DType::Union(variants, _) => {
            if let ScalarValue::Union(union_value) = value
                && let Some(child_dtype) = variants
                    .tag_to_child_index(union_value.type_id())
                    .and_then(|index| variants.variant_by_index(index))
                && let Some(child) = union_value.child_value_mut()
            {
                normalize_list_values(&child_dtype, child);
            }
        }
        DType::Extension(ext_dtype) => normalize_list_values(ext_dtype.storage_dtype(), value),
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_)
        | DType::Variant(_) => {}
    }
}

/// Returns `true` if `array` and all of its descendants are canonical encodings.
pub(crate) fn is_recursively_canonical(array: &ArrayRef) -> bool {
    array.is_canonical()
        && array
            .slots()
            .iter()
            .flatten()
            .all(is_recursively_canonical)
}

/// Reads element values out of a recursively canonical array.
///
/// Flat encodings are read straight from their buffers. Nested ones go through
/// [`ArrayRef::execute_scalar`], which on canonical input runs no kernels.
pub(crate) struct ValueReader<'a> {
    array: &'a ArrayRef,
    validity: ValidityBits,
    kind: ReaderKind<'a>,
}

enum ValidityBits {
    AllValid,
    AllInvalid,
    Bits(BitBuffer),
}

enum ReaderKind<'a> {
    Null,
    Bool,
    /// The values buffer and its type, so a read is one load with no downcast.
    Primitive(PType, &'a [u8]),
    Decimal,
    Utf8,
    Binary,
    /// The storage of an extension array, which holds the extension's values.
    Extension(ArrayRef),
    Nested(Box<ExecutionCtx>),
}

impl<'a> ValueReader<'a> {
    pub(crate) fn new(array: &'a ArrayRef) -> Self {
        let kind = if let Some(primitive) = array.as_opt::<Primitive>() {
            ReaderKind::Primitive(
                primitive.ptype(),
                primitive.data().buffer_handle().as_host().as_slice(),
            )
        } else if array.is::<Null>() {
            ReaderKind::Null
        } else if array.is::<Bool>() {
            ReaderKind::Bool
        } else if array.is::<Decimal>() {
            ReaderKind::Decimal
        } else if array.is::<VarBinView>() {
            match array.dtype() {
                DType::Utf8(_) => ReaderKind::Utf8,
                _ => ReaderKind::Binary,
            }
        } else if let Some(ext) = array.as_opt::<Extension>() {
            ReaderKind::Extension(ext.storage_array().clone())
        } else {
            ReaderKind::Nested(Box::new(element_ctx()))
        };

        let validity = match &kind {
            // Extension and nested reads take nullness from the storage or the scalar read.
            ReaderKind::Null | ReaderKind::Extension(_) | ReaderKind::Nested(_) => {
                ValidityBits::AllValid
            }
            _ => match array
                .validity()
                .vortex_expect("canonical arrays report their validity")
            {
                Validity::NonNullable | Validity::AllValid => ValidityBits::AllValid,
                Validity::AllInvalid => ValidityBits::AllInvalid,
                Validity::Array(validity) => match validity.as_opt::<Bool>() {
                    Some(bits) => ValidityBits::Bits(bits.to_bit_buffer()),
                    None => vortex_panic!("validity of a canonical array must be a bool array"),
                },
            },
        };

        Self {
            array,
            validity,
            kind,
        }
    }

    pub(crate) fn value_at(&mut self, index: usize) -> Option<ScalarValue> {
        match &self.validity {
            ValidityBits::AllValid => {}
            ValidityBits::AllInvalid => return None,
            ValidityBits::Bits(bits) => {
                if !bits.value(index) {
                    return None;
                }
            }
        }

        let array = self.array;
        match &mut self.kind {
            ReaderKind::Null => None,
            ReaderKind::Bool => Some(ScalarValue::Bool(
                array.as_::<Bool>().bit_buffer_view().value(index),
            )),
            ReaderKind::Primitive(ptype, bytes) => {
                Some(match_each_native_ptype!(*ptype, |T| {
                    let width = size_of::<T>();
                    let value = &bytes[index * width..(index + 1) * width];
                    // SAFETY: `value` is exactly `size_of::<T>()` bytes of a buffer of `T`, read
                    // unaligned so its offset in the buffer does not matter.
                    let value = unsafe { std::ptr::read_unaligned(value.as_ptr().cast::<T>()) };
                    ScalarValue::Primitive(value.into())
                }))
            }
            ReaderKind::Decimal => {
                let decimal = array.as_::<Decimal>();
                Some(match_each_decimal_value_type!(decimal.values_type(), |D| {
                    ScalarValue::Decimal(DecimalValue::from(decimal.buffer::<D>()[index]))
                }))
            }
            ReaderKind::Utf8 => {
                let bytes = array.as_::<VarBinView>().bytes_at(index);
                // SAFETY: a Utf8 VarBinView array only holds valid UTF-8.
                Some(ScalarValue::Utf8(unsafe { BufferString::new_unchecked(bytes) }))
            }
            ReaderKind::Binary => Some(ScalarValue::Binary(
                array.as_::<VarBinView>().bytes_at(index),
            )),
            ReaderKind::Extension(storage) => ValueReader::new(storage).value_at(index),
            ReaderKind::Nested(ctx) => array
                .execute_scalar(index, ctx)
                .vortex_expect("reading a canonical list element")
                .into_value(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::listview::ListViewArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn hash_of(scalar: &Scalar) -> u64 {
        let mut hasher = DefaultHasher::new();
        scalar.hash(&mut hasher);
        hasher.finish()
    }

    fn i32_list(values: &[Option<i32>]) -> Scalar {
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            values
                .iter()
                .map(|value| match value {
                    Some(value) => Scalar::primitive(*value, Nullability::Nullable),
                    None => Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
                })
                .collect(),
            Nullability::NonNullable,
        )
    }

    /// A list read out of an array and the same list built from scalars are one value.
    #[test]
    fn list_from_array_equals_list_from_scalars() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements =
            PrimitiveArray::from_option_iter([Some(9i32), Some(1), None, Some(3), Some(9)])
                .into_array();
        let offsets = PrimitiveArray::from_iter([1u32]).into_array();
        let sizes = PrimitiveArray::from_iter([3u32]).into_array();
        let array =
            ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable, &mut ctx)?
                .into_array();

        let read = array.execute_scalar(0, &mut ctx)?;
        let built = i32_list(&[Some(1), None, Some(3)]);

        assert_eq!(read, built);
        assert_eq!(hash_of(&read), hash_of(&built));
        assert_eq!(read.as_list().element(1), built.as_list().element(1));
        assert!(read.as_list().list_value().unwrap().has_null_element());
        Ok(())
    }

    /// The fast primitive comparison agrees with the per-element one.
    #[test]
    fn primitive_lists_order_and_hash_like_values() {
        let lhs = i32_list(&[Some(1), Some(2)]);
        let rhs = i32_list(&[Some(1), Some(3)]);
        let prefix = i32_list(&[Some(1)]);
        let with_null = i32_list(&[Some(1), None]);

        assert!(lhs < rhs);
        assert!(prefix < lhs);
        assert!(with_null < lhs);
        assert_ne!(lhs, with_null);

        // The fast path hashes like the generic per-value loop.
        let list = lhs.as_list().list_value().unwrap();
        let mut fast = DefaultHasher::new();
        list.hash(&mut fast);
        let mut expected = DefaultHasher::new();
        expected.write_usize(2);
        Some(ScalarValue::Primitive(1i32.into())).hash(&mut expected);
        Some(ScalarValue::Primitive(2i32.into())).hash(&mut expected);
        assert_eq!(fast.finish(), expected.finish());
    }

    /// Lists given as tuples, at the top level and nested in a struct, become array-backed.
    #[test]
    fn tuple_values_normalize_to_list_values() -> VortexResult<()> {
        let list_dtype = DType::List(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            Nullability::NonNullable,
        );
        let struct_dtype = DType::struct_([("tags", list_dtype.clone())], Nullability::NonNullable);
        let scalar = Scalar::try_new(
            struct_dtype,
            Some(ScalarValue::Tuple(vec![Some(ScalarValue::Tuple(vec![
                Some(ScalarValue::Utf8("a".into())),
                Some(ScalarValue::Utf8("b".into())),
            ]))])),
        )?;

        let tags = scalar.as_struct().field_by_idx(0).unwrap();
        assert!(matches!(tags.value(), Some(ScalarValue::List(_))));
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            tags.as_list().elements_array().unwrap().clone(),
            VarBinViewArray::from_iter_str(["a", "b"]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn proto_round_trip_keeps_nested_lists() -> VortexResult<()> {
        let inner = |values: &[i32]| {
            Scalar::list(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                values
                    .iter()
                    .map(|v| Scalar::primitive(*v, Nullability::NonNullable))
                    .collect(),
                Nullability::NonNullable,
            )
        };
        let outer = Scalar::list(
            Arc::new(inner(&[]).dtype().clone()),
            vec![inner(&[1, 2]), inner(&[]), inner(&[3])],
            Nullability::Nullable,
        );

        let bytes = ScalarValue::to_proto_bytes::<Vec<u8>>(outer.value());
        let value = ScalarValue::from_proto_bytes(&bytes, outer.dtype(), &SESSION)?;
        let decoded = Scalar::try_new(outer.dtype().clone(), value)?;

        assert_eq!(decoded, outer);
        assert_eq!(decoded.as_list().element(0), Some(inner(&[1, 2])));
        Ok(())
    }

    /// A constant list array expands to its scalar's elements without copying them.
    #[test]
    fn constant_list_reuses_scalar_elements() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let scalar = i32_list(&[Some(4), None, Some(6)]);
        let elements = scalar.as_list().elements_array().unwrap().clone();

        let canonical = ConstantArray::new(scalar.clone(), 3)
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_listview();
        assert!(ArrayRef::ptr_eq(canonical.elements(), &elements));
        for row in 0..3 {
            assert_eq!(canonical.clone().into_array().execute_scalar(row, &mut ctx)?, scalar);
        }
        Ok(())
    }

    #[test]
    fn fixed_size_list_scalar_round_trips_through_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = FixedSizeListArray::new(
            PrimitiveArray::from_iter([1.0f32, 2.0, 3.0, 4.0]).into_array(),
            2,
            Validity::NonNullable,
            2,
        )
        .into_array();

        let row = array.execute_scalar(1, &mut ctx)?;
        assert_eq!(
            row,
            Scalar::fixed_size_list(
                DType::Primitive(PType::F32, Nullability::NonNullable),
                vec![3.0f32.into(), 4.0f32.into()],
                Nullability::NonNullable,
            )
        );

        let expanded = ConstantArray::new(row, 2)
            .into_array()
            .execute::<Canonical>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(
            expanded,
            FixedSizeListArray::new(
                PrimitiveArray::from_iter([3.0f32, 4.0, 3.0, 4.0]).into_array(),
                2,
                Validity::NonNullable,
                2,
            ),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_to_nullable_elements_keeps_values() -> VortexResult<()> {
        let list = i32_list(&[Some(1), Some(2)]);
        let non_null = list.cast(&DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        ))?;
        assert_eq!(non_null.as_list().element(1), Some(2i32.into()));

        let with_null = i32_list(&[Some(1), None]);
        assert!(
            with_null
                .cast(&DType::List(
                    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                    Nullability::Nullable,
                ))
                .is_err()
        );
        Ok(())
    }
}

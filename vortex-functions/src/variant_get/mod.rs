// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod variant;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Masked;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::masked::MaskedArrayExt;
use vortex_array::arrays::masked::MaskedArraySlotsExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::builders::VariantBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::proto::expr as pb;
use vortex_array::proto::expr::variant_path_element;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::mask::Mask;
use vortex_array::scalar_fn::display::ExprDisplay;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::StringEscape;

/// Registers [`VariantGet`] and its kernel for canonical variant arrays.
pub(crate) fn initialize(session: &VortexSession) {
    session.scalar_fns().register(VariantGet);
    variant::initialize(session);
}

/// Extracts a field/index path from Variant values.
///
/// Missing paths, type mismatches while traversing, and failed casts produce nulls. Without a
/// requested dtype, results are returned as nullable Variant values; with one, results are cast to
/// that dtype with nullable nullability. Encodings may serve perfectly shredded paths directly,
/// but must fall back to the core Variant value for paths not represented by shredded storage.
#[derive(Clone)]
pub struct VariantGet;

impl VariantGet {
    /// Creates a lazy extraction from Variant `input`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is not Variant data.
    pub fn try_new(input: ArrayRef, options: VariantGetOptions) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(VariantGet.bind(options), [input])
    }
}

impl ScalarFnVTable for VariantGet {
    type Options = VariantGetOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.variant_get");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let path = options
            .path()
            .elements()
            .iter()
            .map(VariantPathElement::to_proto)
            .collect();
        let dtype = options.dtype().map(TryInto::try_into).transpose()?;

        Ok(Some(pb::VariantGetOpts { path, dtype }.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let opts = pb::VariantGetOpts::decode(metadata)?;
        let path = opts
            .path
            .into_iter()
            .map(VariantPathElement::from_proto)
            .collect::<VortexResult<_>>()?;
        let dtype = opts
            .dtype
            .as_ref()
            .map(|dtype| DType::from_proto(dtype, session))
            .transpose()?;

        Ok(VariantGetOptions::new(path, dtype))
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for VariantGet expression"),
        }
    }

    fn fmt_sql(
        &self,
        options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "variant_get(")?;
        Display::fmt(expr.display_child(0), f)?;
        let path = options.path().to_string();
        write!(f, ", \"{}\"", StringEscape(&path))?;
        if let Some(dtype) = options.dtype() {
            write!(f, ", {dtype}")?;
        }
        write!(f, ")")
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input_dtype = &arg_dtypes[0];
        vortex_ensure!(
            matches!(input_dtype, DType::Variant(_)),
            "VariantGet input must be Variant, found {input_dtype}"
        );

        // Missing paths, traversal mismatches, and cast failures all produce nulls.
        Ok(options
            .dtype()
            .map_or(DType::Variant(Nullability::Nullable), DType::as_nullable))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        // Missing paths, traversal mismatches, and cast failures all produce nulls.
        let dtype = options
            .dtype()
            .map_or(DType::Variant(Nullability::Nullable), DType::as_nullable);

        // The kernels that extract in bulk are keyed on the Variant encodings. A lazy input, such
        // as a filtered or chunked Variant read from a file, reaches this point without one, so
        // bring it to those encodings before falling back to extracting row by row.
        if let Some(chunked) = input.as_opt::<Chunked>() {
            let chunks = chunked
                .iter_chunks()
                .map(|chunk| {
                    VariantGet::try_new(chunk.clone(), options.clone())?
                        .into_array()
                        .execute::<ArrayRef>(ctx)
                })
                .collect::<VortexResult<Vec<_>>>()?;
            return Ok(ChunkedArray::try_new(chunks, dtype)?.into_array());
        }
        // A mask only nulls rows out, so it commutes with extraction.
        if let Some(masked) = input.as_opt::<ScalarFn>()
            && masked.scalar_fn().id() == Mask.id()
        {
            return VariantGet::try_new(masked.child_at(0).clone(), options.clone())?
                .into_array()
                .execute::<ArrayRef>(ctx)?
                .mask(masked.child_at(1).clone())?
                .execute::<ArrayRef>(ctx);
        }

        if let Some(masked) = input.as_opt::<Masked>() {
            return VariantGet::try_new(masked.child().clone(), options.clone())?
                .into_array()
                .execute::<ArrayRef>(ctx)?
                .mask(masked.masked_validity().to_array(input.len()))?
                .execute::<ArrayRef>(ctx);
        }
        // Each canonicalization unwraps one lazy layer, such as a filter over a slice over a
        // shared array.
        if !input.is::<Variant>() {
            let canonical = input.clone().execute::<VariantArray>(ctx)?;
            if !wraps_input(&canonical, &input) {
                return VariantGet::try_new(canonical.into_array(), options.clone())?
                    .into_array()
                    .execute::<ArrayRef>(ctx);
            }
        }

        if !dtype.is_variant() {
            let mut builder = builder_with_capacity_in(&dtype, input.len(), ctx.allocator());
            for idx in 0..input.len() {
                let scalar = input.execute_scalar(idx, ctx)?;
                let output = variant_get_scalar(&scalar, options, &dtype)?;
                builder.append_scalar(&output)?;
            }

            return Ok(builder.finish());
        }

        let mut builder = VariantBuilder::new(dtype.nullability());
        for idx in 0..input.len() {
            let scalar = input.execute_scalar(idx, ctx)?;
            builder.append_n_scalars(&variant_get_scalar(&scalar, options, &dtype)?, 1)?;
        }
        builder.finish_into_variant().map(IntoArray::into_array)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Returns whether `canonical` only wraps `input` again, as a constant canonicalizes to a Variant
/// array over itself.
///
/// The kernel of such an array extracts from its core storage, which is `input`. That builds the
/// same extraction again, so the caller must extract row by row.
fn wraps_input(canonical: &VariantArray, input: &ArrayRef) -> bool {
    canonical.shredded().is_none() && ArrayRef::ptr_eq(canonical.core_storage(), input)
}

fn variant_get_scalar(
    scalar: &Scalar,
    options: &VariantGetOptions,
    output_dtype: &DType,
) -> VortexResult<Scalar> {
    let Some(value) = variant_path_scalar(scalar, options.path().elements())? else {
        return Ok(Scalar::null(output_dtype.clone()));
    };

    if options.dtype().is_none_or(DType::is_variant) {
        return Scalar::variant(value).cast(output_dtype);
    }

    if value.is_null() {
        return Ok(Scalar::null(output_dtype.clone()));
    }

    Ok(cast_variant_scalar(value, output_dtype)?
        .unwrap_or_else(|| Scalar::null(output_dtype.clone())))
}

fn cast_variant_scalar(value: Scalar, output_dtype: &DType) -> VortexResult<Option<Scalar>> {
    if value.is_null() {
        return Ok(None);
    }

    if output_dtype.is_variant() {
        let variant = if value.dtype().is_variant() {
            value
        } else {
            Scalar::variant(value)
        };
        return variant.cast(output_dtype).map(Some);
    }

    let Some(value) = variant_payload(value) else {
        return Ok(None);
    };

    if value.is_null() {
        return Ok(None);
    }

    if let DType::Struct(target_fields, _) = output_dtype {
        let Some(struct_scalar) = value.as_struct_opt() else {
            return Ok(None);
        };
        if struct_scalar.is_null() {
            return Ok(None);
        }
        let mut fields = Vec::with_capacity(target_fields.nfields());
        for (name, field_dtype) in target_fields.names().iter().zip(target_fields.fields()) {
            let field = match struct_scalar.field(name.as_ref()) {
                Some(field) => match cast_variant_scalar(field, &field_dtype)? {
                    Some(cast_field) => cast_field,
                    None if field_dtype.is_nullable() => Scalar::null(field_dtype),
                    None => return Ok(None),
                },
                None if field_dtype.is_nullable() => Scalar::null(field_dtype),
                None => return Ok(None),
            };
            fields.push(field);
        }
        return Ok(Some(Scalar::struct_(output_dtype.clone(), fields)));
    }

    if let DType::List(elem_dtype, _) = output_dtype {
        let Some(list_scalar) = value.as_list_opt() else {
            return Ok(None);
        };
        let Some(elements) = list_scalar.elements() else {
            return Ok(None);
        };
        let mut cast_elements = Vec::with_capacity(elements.len());
        for element in elements {
            let cast_element = match cast_variant_scalar(element, elem_dtype)? {
                Some(cast_element) => cast_element,
                None if elem_dtype.is_nullable() => Scalar::null(elem_dtype.as_ref().clone()),
                None => return Ok(None),
            };
            cast_elements.push(cast_element);
        }
        return Ok(Some(Scalar::list(
            Arc::clone(elem_dtype),
            cast_elements,
            output_dtype.nullability(),
        )));
    }

    if let DType::FixedSizeList(elem_dtype, size, _) = output_dtype {
        let Some(list_scalar) = value.as_list_opt() else {
            return Ok(None);
        };
        let Some(elements) = list_scalar.elements() else {
            return Ok(None);
        };
        if elements.len() != *size as usize {
            return Ok(None);
        }
        let mut cast_elements = Vec::with_capacity(elements.len());
        for element in elements {
            let cast_element = match cast_variant_scalar(element, elem_dtype)? {
                Some(cast_element) => cast_element,
                None if elem_dtype.is_nullable() => Scalar::null(elem_dtype.as_ref().clone()),
                None => return Ok(None),
            };
            cast_elements.push(cast_element);
        }
        return Ok(Some(Scalar::fixed_size_list(
            Arc::clone(elem_dtype),
            cast_elements,
            output_dtype.nullability(),
        )));
    }

    Ok(value
        .cast(output_dtype)
        .ok()
        .or_else(|| convert_to_bool(&value, output_dtype))
        .or_else(|| convert_to_decimal(&value, output_dtype)))
}

/// Converts a number or a string to a Boolean as Arrow's Variant conversion does. A Vortex cast
/// converts neither to a Boolean.
///
/// Zero is false and any other number is true. A string converts as [`string_as_bool`] reads it.
/// The value can be in a Variant, as a partially shredded field holds its values.
pub(super) fn convert_to_bool(scalar: &Scalar, dtype: &DType) -> Option<Scalar> {
    if !dtype.is_boolean() {
        return None;
    }

    let value = if scalar.dtype().is_variant() {
        scalar.as_variant().value()?
    } else {
        scalar
    };

    let boolean = match value.dtype() {
        DType::Primitive(..) => !value.is_zero()?,
        DType::Utf8(_) => string_as_bool(value.as_utf8().value()?.as_str())?,
        _ => return None,
    };
    Some(Scalar::bool(boolean, dtype.nullability()))
}

/// Converts a float or a decimal to a decimal as Arrow's Variant conversion does, where a Vortex
/// cast fails: both round half away from zero. Returns `None` if the result does not fit.
pub(super) fn convert_to_decimal(scalar: &Scalar, dtype: &DType) -> Option<Scalar> {
    let DType::Decimal(target, nullability) = dtype else {
        return None;
    };

    let value = if scalar.dtype().is_variant() {
        scalar.as_variant().value()?
    } else {
        scalar
    };

    let unscaled = match value.dtype() {
        DType::Primitive(ptype, _) if ptype.is_float() => {
            let float = value.as_primitive().as_::<f64>()?;
            let scaled = (10f64.powi(i32::from(target.scale())) * float).round();
            i128_from_f64(scaled)?
        }
        DType::Decimal(source, _) => {
            let dropped_digits =
                u32::try_from(i16::from(source.scale()) - i16::from(target.scale())).ok()?;
            let unscaled = value.as_decimal().decimal_value()?.cast::<i128>()?;
            let divisor = 10i128.checked_pow(dropped_digits)?;

            let mut rounded = unscaled / divisor;
            if (unscaled % divisor).unsigned_abs() * 2 >= divisor.unsigned_abs() {
                rounded += unscaled.signum();
            }
            rounded
        }
        _ => return None,
    };

    Scalar::try_decimal(DecimalValue::I128(unscaled), *target, *nullability).ok()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is checked to be in range"
)]
fn i128_from_f64(float: f64) -> Option<i128> {
    // `i128::MAX as f64` rounds up to 2^127, so the upper bound is exclusive.
    (float >= i128::MIN as f64 && float < i128::MAX as f64).then_some(float as i128)
}

/// Reads a string as a Boolean with the spellings that Arrow's string cast accepts, ignoring case
/// and the whitespace around the string. Returns `None` for any other string.
fn string_as_bool(string: &str) -> Option<bool> {
    match string.trim().to_ascii_lowercase().as_str() {
        // spellchecker:off
        "t" | "tr" | "tru" | "true" | "y" | "ye" | "yes" | "on" | "1" => Some(true),
        "f" | "fa" | "fal" | "fals" | "false" | "n" | "no" | "of" | "off" | "0" => Some(false),
        // spellchecker:on
        _ => None,
    }
}

fn variant_path_scalar(
    scalar: &Scalar,
    path: &[VariantPathElement],
) -> VortexResult<Option<Scalar>> {
    let mut current = match variant_payload(scalar.clone()) {
        Some(value) => value,
        None => return Ok(None),
    };

    for element in path {
        current = match variant_payload(current) {
            Some(value) => value,
            None => return Ok(None),
        };

        if current.is_null() {
            return Ok(None);
        }

        current = match element {
            VariantPathElement::Field(name) => {
                let Some(struct_scalar) = current.as_struct_opt() else {
                    return Ok(None);
                };
                if struct_scalar.is_null() {
                    return Ok(None);
                }
                let Some(field) = struct_scalar.field(name.as_ref()) else {
                    return Ok(None);
                };
                field
            }
            VariantPathElement::Index(index) => {
                let Ok(index) = usize::try_from(*index) else {
                    return Ok(None);
                };
                let Some(list_scalar) = current.as_list_opt() else {
                    return Ok(None);
                };
                let Some(element) = list_scalar.element(index) else {
                    return Ok(None);
                };
                element
            }
        };
    }

    Ok(variant_payload(current))
}

fn variant_payload(scalar: Scalar) -> Option<Scalar> {
    if scalar.dtype().is_variant() {
        scalar.as_variant().value().cloned()
    } else {
        Some(scalar)
    }
}

/// Options for [`VariantGet`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VariantGetOptions {
    path: VariantPath,
    dtype: Option<DType>,
}

impl VariantGetOptions {
    /// Creates options for extracting `path`, returning Variant values when `dtype` is `None`.
    pub fn new(path: VariantPath, dtype: Option<DType>) -> Self {
        Self { path, dtype }
    }

    /// Returns the path to extract.
    pub fn path(&self) -> &VariantPath {
        &self.path
    }

    /// Returns the requested output dtype, if any.
    pub fn dtype(&self) -> Option<&DType> {
        self.dtype.as_ref()
    }
}

impl Display for VariantGetOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)?;
        if let Some(dtype) = &self.dtype {
            write!(f, " as {dtype}")?;
        }
        Ok(())
    }
}

/// A strict Variant path made from object fields and list indexes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct VariantPath(Vec<VariantPathElement>);

impl VariantPath {
    /// Creates a path from explicit elements.
    pub fn new(elements: impl IntoIterator<Item = VariantPathElement>) -> Self {
        Self(elements.into_iter().collect())
    }

    /// Creates the root path.
    pub fn root() -> Self {
        Self::default()
    }

    /// Creates a path containing one object field.
    pub fn field(field: impl Into<FieldName>) -> Self {
        Self(vec![VariantPathElement::field(field)])
    }

    /// Returns the path elements.
    pub fn elements(&self) -> &[VariantPathElement] {
        &self.0
    }

    /// Returns whether this path references the root Variant value.
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<VariantPathElement> for VariantPath {
    fn from_iter<T: IntoIterator<Item = VariantPathElement>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl From<VariantPathElement> for VariantPath {
    fn from(value: VariantPathElement) -> Self {
        Self(vec![value])
    }
}

impl Display for VariantPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "$")?;
        for element in self.elements() {
            match element {
                VariantPathElement::Field(name) => write!(f, ".{name}")?,
                VariantPathElement::Index(index) => write!(f, "[{index}]")?,
            }
        }
        Ok(())
    }
}

/// A single field or index step in a [`VariantPath`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VariantPathElement {
    /// Select an object field by name.
    Field(FieldName),
    /// Select a list element by zero-based index.
    Index(u64),
}

impl VariantPathElement {
    /// Creates an object-field path element.
    pub fn field(field: impl Into<FieldName>) -> Self {
        Self::Field(field.into())
    }

    /// Creates a list-index path element.
    pub fn index(index: u64) -> Self {
        Self::Index(index)
    }

    /// Decodes a path element from its protobuf representation.
    pub fn from_proto(value: pb::VariantPathElement) -> VortexResult<Self> {
        match value
            .element
            .ok_or_else(|| vortex_err!("Variant path element missing value"))?
        {
            variant_path_element::Element::Field(field) => Ok(Self::field(field)),
            variant_path_element::Element::Index(index) => Ok(Self::index(index)),
        }
    }

    /// Encodes this path element into its protobuf representation.
    pub fn to_proto(&self) -> pb::VariantPathElement {
        match self {
            VariantPathElement::Field(name) => pb::VariantPathElement {
                element: Some(variant_path_element::Element::Field(
                    name.as_ref().to_string(),
                )),
            },
            VariantPathElement::Index(index) => pb::VariantPathElement {
                element: Some(variant_path_element::Element::Index(*index)),
            },
        }
    }
}

impl From<FieldName> for VariantPathElement {
    fn from(value: FieldName) -> Self {
        Self::field(value)
    }
}

impl From<&str> for VariantPathElement {
    fn from(value: &str) -> Self {
        Self::field(value)
    }
}

impl From<u64> for VariantPathElement {
    fn from(value: u64) -> Self {
        Self::index(value)
    }
}

#[cfg(test)]
mod tests;

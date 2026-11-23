// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod variant;

use std::cell::Cell;
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
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::builders::VariantBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::expr::display::ExprDisplay;
use vortex_array::proto::expr as pb;
use vortex_array::proto::expr::variant_path_element;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ReduceNode;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::scalar_fn::builtin::mask::Mask;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::StringEscape;

use crate::variant_pack::VariantPack;

/// How many lazy layers `variant_get` unwraps from its input before it extracts row by row.
const MAX_CANONICALIZE_DEPTH: usize = 8;

thread_local! {
    /// How many inputs the executions of `variant_get` on this thread's stack have canonicalized.
    static CANONICALIZE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

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
        // Each canonicalization unwraps one lazy layer, such as a filter over a slice over a
        // shared array. An input that canonicalizes to itself, such as a constant, would recurse
        // forever, so the depth is bounded.
        if !input.is::<Variant>() && CANONICALIZE_DEPTH.get() < MAX_CANONICALIZE_DEPTH {
            let canonical = input.clone().execute::<VariantArray>(ctx)?;
            CANONICALIZE_DEPTH.set(CANONICALIZE_DEPTH.get() + 1);
            let result = VariantGet::try_new(canonical.into_array(), options.clone())
                .and_then(|array| array.into_array().execute::<ArrayRef>(ctx));
            CANONICALIZE_DEPTH.set(CANONICALIZE_DEPTH.get() - 1);
            return result;
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

    /// Reads a shredded path from the shredded child of a [`VariantPack`] input, leaving its
    /// core storage unread.
    fn reduce<T: ReduceNode>(&self, options: &Self::Options, node: &T) -> VortexResult<Option<T>> {
        let input = node.child(0);
        let Some(pack) = input
            .scalar_fn()
            .and_then(|scalar_fn| scalar_fn.as_opt::<VariantPack>())
        else {
            return Ok(None);
        };
        let core = input.child(0);
        let mut current = input.child(1);
        let mut path = options.path().elements();
        // Whether `current` is below the shredded child itself. Only the shredded child can
        // have left values in the core storage, so only reading it needs it to be complete.
        let mut descended = false;

        loop {
            let dtype = current.node_dtype()?;
            match (&dtype, path.split_first()) {
                // A shredded Variant holds the rest of the path itself.
                (DType::Variant(_), _) if descended => {
                    let rest = VariantGetOptions::new(
                        VariantPath::new(path.iter().cloned()),
                        options.dtype().cloned(),
                    );
                    return node.new_node(VariantGet.bind(rest), &[current]).map(Some);
                }
                (DType::Struct(fields, _), Some((VariantPathElement::Field(name), rest))) => {
                    if fields.find(name.as_ref()).is_none() {
                        // A field that was not shredded is only in the core storage.
                        return if descended {
                            Ok(None)
                        } else {
                            node.new_node(VariantGet.bind(options.clone()), &[core])
                                .map(Some)
                        };
                    }
                    if !descended && !pack.complete() {
                        return Ok(None);
                    }
                    current = current.new_node(GetItem.bind(name.clone()), &[current.clone()])?;
                    path = rest;
                    descended = true;
                }
                // The end of the path at a typed value of the requested dtype.
                (dtype, None)
                    if options.dtype().is_some_and(|requested| {
                        !requested.is_variant() && dtype.eq_ignore_nullability(requested)
                    }) && (descended || (pack.complete() && !pack.shredded_validity())) =>
                {
                    return node
                        .new_node(Cast.bind(node.node_dtype()?), &[current])
                        .map(Some);
                }
                _ => return Ok(None),
            }
        }
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
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
        .or_else(|| convert_to_bool(&value, output_dtype)))
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
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Chunked;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::arrays::variant::VariantArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar_is_null;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::proto::ExprSerializeProtoExt;
    use vortex_array::expr::root;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_ensure;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::expr::variant_get;
    use crate::variant_get::VariantGet;
    use crate::variant_get::VariantGetOptions;
    use crate::variant_get::VariantPath;
    use crate::variant_get::VariantPathElement;
    use crate::variant_pack::VariantPack;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();

        vortex_array::initialize(&session);

        crate::initialize(&session);

        session
    });

    fn variant_object(fields: impl IntoIterator<Item = (&'static str, Scalar)>) -> Scalar {
        let fields = fields.into_iter().collect::<Vec<_>>();
        let names = FieldNames::from_iter(fields.iter().map(|(name, _)| FieldName::from(*name)));
        let dtypes = vec![DType::Variant(Nullability::NonNullable); fields.len()];
        let values = fields
            .into_iter()
            .map(|(_, value)| Scalar::variant(value).into_value())
            .collect();
        Scalar::try_new(
            DType::Struct(StructFields::new(names, dtypes), Nullability::NonNullable),
            Some(ScalarValue::Tuple(values)),
        )
        .unwrap()
    }

    fn variant_rows(rows: impl IntoIterator<Item = Scalar>) -> VortexResult<ArrayRef> {
        let dtype = DType::Variant(Nullability::Nullable);
        let chunks = rows
            .into_iter()
            .map(|row| ConstantArray::new(row.cast(&dtype).unwrap(), 1).into_array());
        ChunkedArray::try_new(chunks, dtype.clone()).map(|array| array.into_array())
    }

    /// Test-only syntax for keeping `variant_get` cases compact without committing
    /// to a public string grammar yet.
    fn parse_path(path: &str) -> VortexResult<VariantPath> {
        if path.is_empty() || path == "$" {
            return Ok(VariantPath::root());
        }

        let mut elements = Vec::new();
        let mut pos = usize::from(path.as_bytes().first() == Some(&b'$'));
        if pos == 1
            && path
                .as_bytes()
                .get(pos)
                .is_some_and(|byte| !matches!(byte, b'.' | b'['))
        {
            vortex_bail!("Invalid Variant path {path:?}: expected '.' or '[' after '$'");
        }

        while pos < path.len() {
            match path.as_bytes()[pos] {
                b'.' => {
                    pos += 1;
                    let (field, next_pos) = parse_field(path, pos)?;
                    elements.push(VariantPathElement::field(field));
                    pos = next_pos;
                }
                b'[' => {
                    let (index, next_pos) = parse_index(path, pos + 1)?;
                    elements.push(VariantPathElement::index(index));
                    pos = next_pos;
                }
                _ if pos == 0 => {
                    let (field, next_pos) = parse_field(path, pos)?;
                    elements.push(VariantPathElement::field(field));
                    pos = next_pos;
                }
                _ => {
                    vortex_bail!("Invalid Variant path {path:?}: expected '.', '[', or end of path")
                }
            }
        }

        Ok(VariantPath::new(elements))
    }

    fn parse_field(path: &str, start: usize) -> VortexResult<(&str, usize)> {
        let mut pos = start;
        while path
            .as_bytes()
            .get(pos)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            pos += 1;
        }
        vortex_ensure!(
            pos > start,
            "Invalid Variant path {path:?}: expected field name"
        );
        Ok((&path[start..pos], pos))
    }

    fn parse_index(path: &str, start: usize) -> VortexResult<(u64, usize)> {
        let mut pos = start;
        while path
            .as_bytes()
            .get(pos)
            .is_some_and(|byte| byte.is_ascii_digit())
        {
            pos += 1;
        }
        vortex_ensure!(
            pos > start,
            "Invalid Variant path {path:?}: expected list index"
        );
        vortex_ensure!(
            path.as_bytes().get(pos) == Some(&b']'),
            "Invalid Variant path {path:?}: expected closing ']'"
        );
        let index = path[start..pos]
            .parse()
            .map_err(|_| vortex_err!("Invalid Variant path {path:?}: list index is too large"))?;
        Ok((index, pos + 1))
    }

    fn execute_variant_get(
        array: ArrayRef,
        path: &str,
        dtype: Option<DType>,
    ) -> VortexResult<ArrayRef> {
        let expr = variant_get(root(), parse_path(path)?, dtype);
        array
            .apply(&expr)?
            .optimize_ctx(&SESSION)?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
    }

    #[test]
    fn variant_get_path_parse_and_display() {
        let path = parse_path("$.data[1].a").unwrap();
        assert_eq!(
            path.elements(),
            &[
                VariantPathElement::field("data"),
                VariantPathElement::index(1),
                VariantPathElement::field("a")
            ]
        );
        assert_eq!(path.to_string(), "$.data[1].a");

        let bare_path = parse_path("data[2]").unwrap();
        assert_eq!(bare_path.to_string(), "$.data[2]");
    }

    #[test]
    fn variant_get_return_dtype_is_nullable_variant_without_requested_dtype() {
        let expr = variant_get(root(), VariantPath::field("data"), None);
        let dtype = expr
            .return_dtype(&DType::Variant(Nullability::NonNullable))
            .unwrap();

        assert_eq!(dtype, DType::Variant(Nullability::Nullable));
    }

    #[test]
    fn variant_get_return_dtype_makes_requested_dtype_nullable() {
        let requested = DType::Primitive(PType::I64, Nullability::NonNullable);
        let expr = variant_get(root(), VariantPath::field("data"), Some(requested));
        let dtype = expr
            .return_dtype(&DType::Variant(Nullability::NonNullable))
            .unwrap();

        assert_eq!(dtype, DType::Primitive(PType::I64, Nullability::Nullable));
    }

    #[test]
    fn variant_get_rejects_non_variant_input() {
        let expr = variant_get(root(), VariantPath::field("data"), None);
        let err = expr
            .return_dtype(&DType::Utf8(Nullability::NonNullable))
            .unwrap_err();

        assert!(err.to_string().contains("VariantGet input must be Variant"));
    }

    #[test]
    fn variant_get_formats_sql() {
        let expr = variant_get(
            root(),
            parse_path("$.data[1].a").unwrap(),
            Some(DType::Utf8(Nullability::NonNullable)),
        );

        assert_eq!(expr.to_string(), "variant_get($, \"$.data[1].a\", utf8)");
    }

    #[test]
    fn variant_get_options_roundtrip_serialization() {
        let options = VariantGetOptions::new(
            VariantPath::new([
                VariantPathElement::field("data"),
                VariantPathElement::index(1),
                VariantPathElement::field("a"),
            ]),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let metadata = VariantGet.serialize(&options).unwrap().unwrap();
        let actual = VariantGet.deserialize(&metadata, &SESSION).unwrap();

        assert_eq!(actual, options);
    }

    #[test]
    fn variant_get_expression_roundtrip_serialization() {
        let expr: Expression = variant_get(
            root(),
            parse_path("$.data[1].a").unwrap(),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let proto = expr.serialize_proto().unwrap();
        let actual = Expression::from_proto(&proto, &SESSION).unwrap();

        assert_eq!(actual, expr);
    }

    #[test]
    fn variant_get_generic_fallback_extracts_field_and_list_index() -> VortexResult<()> {
        let items = Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![
                Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
                Scalar::variant(Scalar::primitive(20i32, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        let array = variant_rows([
            Scalar::variant(variant_object([("items", items)])),
            Scalar::variant(variant_object([(
                "items",
                Scalar::list_empty(
                    DType::Variant(Nullability::NonNullable).into(),
                    Nullability::NonNullable,
                ),
            )])),
            Scalar::variant(variant_object([(
                "items",
                Scalar::list(
                    DType::Variant(Nullability::NonNullable),
                    vec![
                        Scalar::variant(Scalar::utf8("x", Nullability::NonNullable)),
                        Scalar::variant(Scalar::utf8("wrong", Nullability::NonNullable)),
                    ],
                    Nullability::NonNullable,
                ),
            )])),
        ])?;

        let result = execute_variant_get(
            array,
            "$.items[1]",
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;
        let mut ctx = SESSION.create_execution_ctx();

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(20i32), None, None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_reads_chunked_variant_input() -> VortexResult<()> {
        let array = variant_rows([
            Scalar::variant(variant_object([(
                "a",
                Scalar::primitive(10i32, Nullability::NonNullable),
            )])),
            Scalar::variant(variant_object([(
                "b",
                Scalar::primitive(20i32, Nullability::NonNullable),
            )])),
            Scalar::variant(variant_object([(
                "a",
                Scalar::primitive(30i32, Nullability::NonNullable),
            )])),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?;
        assert!(array.is::<Chunked>());

        let result = execute_variant_get(
            array,
            "$.a",
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;
        let mut ctx = SESSION.create_execution_ctx();

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(30), None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_fallback_typed_output_is_contiguous() -> VortexResult<()> {
        let array = variant_rows([
            Scalar::variant(variant_object([(
                "a",
                Scalar::primitive(10i32, Nullability::NonNullable),
            )])),
            Scalar::variant(variant_object([(
                "a",
                Scalar::primitive(20i32, Nullability::NonNullable),
            )])),
            Scalar::variant(variant_object([(
                "b",
                Scalar::primitive(30i32, Nullability::NonNullable),
            )])),
        ])?;

        let result = execute_variant_get(
            array,
            "$.a",
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        )?;

        assert!(!result.is::<Chunked>());
        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_generic_fallback_preserves_variant_null() -> VortexResult<()> {
        let array = variant_rows([
            Scalar::variant(variant_object([(
                "a",
                Scalar::utf8("ok", Nullability::NonNullable),
            )])),
            Scalar::null(DType::Variant(Nullability::Nullable)),
            Scalar::variant(variant_object([("a", Scalar::null(DType::Null))])),
            Scalar::variant(variant_object([(
                "b",
                Scalar::primitive(2i32, Nullability::NonNullable),
            )])),
        ])?;

        let result = execute_variant_get(array, "$.a", None)?;

        let mut ctx = SESSION.create_execution_ctx();
        let row0 = result.execute_scalar(0, &mut ctx)?;
        assert_eq!(
            row0.as_variant()
                .value()
                .and_then(|value| value.as_utf8().value())
                .map(|value| value.as_str()),
            Some("ok")
        );
        assert_nth_scalar_is_null!(result, 1, &mut ctx);
        assert_eq!(
            result
                .execute_scalar(2, &mut ctx)?
                .as_variant()
                .is_variant_null(),
            Some(true)
        );
        assert_nth_scalar_is_null!(result, 3, &mut ctx);
        Ok(())
    }

    #[test]
    fn variant_get_fallback_variant_output_canonicalizes() -> VortexResult<()> {
        let array = variant_rows([
            Scalar::variant(variant_object([(
                "a",
                Scalar::primitive(10i32, Nullability::NonNullable),
            )])),
            Scalar::variant(variant_object([(
                "a",
                Scalar::primitive(20i32, Nullability::NonNullable),
            )])),
        ])?;

        let result = execute_variant_get(array, "$.a", None)?;
        let variant = result
            .clone()
            .execute::<VariantArray>(&mut SESSION.create_execution_ctx())?;
        let canonical = result.execute::<Canonical>(&mut SESSION.create_execution_ctx())?;
        let Canonical::Variant(canonical_variant) = canonical else {
            vortex_bail!("expected Variant canonical array");
        };

        assert_eq!(variant.len(), 2);
        assert_eq!(canonical_variant.len(), 2);
        assert_eq!(variant.core_storage().dtype(), variant.dtype());
        assert_eq!(variant.core_storage().len(), variant.len());

        let mut ctx = SESSION.create_execution_ctx();
        for array in [variant, canonical_variant] {
            assert_eq!(array.dtype(), &DType::Variant(Nullability::Nullable));
            for (idx, expected) in [10i32, 20].into_iter().enumerate() {
                let scalar = array.execute_scalar(idx, &mut ctx)?;
                let actual = scalar
                    .as_variant()
                    .value()
                    .and_then(|value| value.as_primitive().as_::<i32>());
                assert_eq!(actual, Some(expected), "row {idx}");
            }
        }
        Ok(())
    }

    fn row_storage(values: impl IntoIterator<Item = i32>) -> VortexResult<ArrayRef> {
        Ok(ChunkedArray::try_new(
            values.into_iter().map(|value| {
                ConstantArray::new(
                    Scalar::variant(Scalar::primitive(value, Nullability::NonNullable)),
                    1,
                )
                .into_array()
            }),
            DType::Variant(Nullability::NonNullable),
        )?
        .into_array())
    }

    #[test]
    fn variant_get_keeps_valid_shredded_rows_for_matching_dtype() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core_storage = row_storage([1, 2, 3])?;
        let shredded = StructArray::try_from_iter([(
            "a",
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
        )])?;
        let variant = VariantArray::try_new(core_storage, Some(shredded.into_array()))?;
        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );

        let result = variant.into_array().apply(&expr)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_treats_value_and_typed_value_as_logical_field_names() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core_storage = row_storage([1, 2, 3])?;
        let shredded = StructArray::try_from_iter([
            (
                "value",
                PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            ),
            (
                "typed_value",
                PrimitiveArray::from_iter([40i32, 50, 60]).into_array(),
            ),
        ])?;
        let variant = VariantArray::try_new(core_storage, Some(shredded.into_array()))?;

        let value_expr = variant_get(
            root(),
            VariantPath::field("value"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let value_result = variant.clone().into_array().apply(&value_expr)?;
        assert_arrays_eq!(
            value_result,
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30)]),
            &mut ctx
        );

        let typed_value_expr = variant_get(
            root(),
            VariantPath::field("typed_value"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let typed_value_result = variant.into_array().apply(&typed_value_expr)?;
        assert_arrays_eq!(
            typed_value_result,
            PrimitiveArray::from_option_iter([Some(40i32), Some(50), Some(60)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_non_nullable_nested_target_returns_null_without_panicking() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let row0 = Scalar::variant(Scalar::struct_(
            DType::Struct(
                StructFields::new(
                    FieldNames::from(["x"]),
                    vec![DType::Variant(Nullability::NonNullable)],
                ),
                Nullability::NonNullable,
            ),
            vec![Scalar::variant(Scalar::utf8(
                "not_an_int",
                Nullability::NonNullable,
            ))],
        ));
        let row1 = Scalar::variant(Scalar::struct_(
            DType::Struct(
                StructFields::new(
                    FieldNames::from(["x"]),
                    vec![DType::Variant(Nullability::NonNullable)],
                ),
                Nullability::NonNullable,
            ),
            vec![Scalar::variant(Scalar::primitive(
                7i32,
                Nullability::NonNullable,
            ))],
        ));
        let input = ChunkedArray::try_new(
            vec![
                ConstantArray::new(row0, 1).into_array(),
                ConstantArray::new(row1, 1).into_array(),
            ],
            DType::Variant(Nullability::NonNullable),
        )?
        .into_array();

        let struct_target = DType::Struct(
            StructFields::new(
                FieldNames::from(["x"]),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::Nullable,
        );
        let expr = variant_get(root(), VariantPath::root(), Some(struct_target));
        let result = input.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is_invalid(0, &mut ctx)?);
        assert_eq!(
            result
                .execute_scalar(1, &mut ctx)?
                .as_struct()
                .field("x")
                .and_then(|s| s.as_primitive().typed_value::<i32>()),
            Some(7)
        );

        let list_row0 = Scalar::variant(Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![Scalar::variant(Scalar::utf8(
                "bad",
                Nullability::NonNullable,
            ))],
            Nullability::NonNullable,
        ));
        let list_row1 = Scalar::variant(Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![Scalar::variant(Scalar::primitive(
                9i32,
                Nullability::NonNullable,
            ))],
            Nullability::NonNullable,
        ));
        let list_input = ChunkedArray::try_new(
            vec![
                ConstantArray::new(list_row0, 1).into_array(),
                ConstantArray::new(list_row1, 1).into_array(),
            ],
            DType::Variant(Nullability::NonNullable),
        )?
        .into_array();

        let list_target = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        let list_expr = variant_get(root(), VariantPath::root(), Some(list_target));
        let list_result = list_input
            .apply(&list_expr)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(list_result.is_invalid(0, &mut ctx)?);
        assert_eq!(
            list_result
                .execute_scalar(1, &mut ctx)?
                .as_list()
                .element(0)
                .and_then(|s| s.as_primitive().typed_value::<i32>()),
            Some(9)
        );
        Ok(())
    }

    #[test]
    fn variant_get_fixed_size_list_target() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let row0 = Scalar::variant(Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![
                Scalar::variant(Scalar::primitive(1i64, Nullability::NonNullable)),
                Scalar::variant(Scalar::primitive(2i64, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        ));
        let row1 = Scalar::variant(Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![Scalar::variant(Scalar::primitive(
                3i64,
                Nullability::NonNullable,
            ))],
            Nullability::NonNullable,
        ));
        let input = ChunkedArray::try_new(
            vec![
                ConstantArray::new(row0, 1).into_array(),
                ConstantArray::new(row1, 1).into_array(),
            ],
            DType::Variant(Nullability::NonNullable),
        )?
        .into_array();

        let target = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::Nullable,
        );
        let expr = variant_get(root(), VariantPath::root(), Some(target));
        let result = input.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;
        let scalar0 = result.execute_scalar(0, &mut ctx)?;
        assert_eq!(
            scalar0.as_list().elements().map(|elems| elems
                .into_iter()
                .map(|e| e.as_primitive().typed_value::<i32>())
                .collect::<Vec<_>>()),
            Some(vec![Some(1), Some(2)])
        );
        assert!(result.is_invalid(1, &mut ctx)?);
        Ok(())
    }

    #[test]
    fn variant_get_fixed_size_list_preserves_variant_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let variant_null = Scalar::variant(Scalar::null(DType::Null));
        let row = Scalar::variant(Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![
                Scalar::variant(Scalar::primitive(7i32, Nullability::NonNullable)),
                variant_null,
            ],
            Nullability::NonNullable,
        ));
        let input = ConstantArray::new(row, 1).into_array();

        for elem_nullability in [Nullability::NonNullable, Nullability::Nullable] {
            let target = DType::FixedSizeList(
                Arc::new(DType::Variant(elem_nullability)),
                2,
                Nullability::Nullable,
            );
            let expr = variant_get(root(), VariantPath::root(), Some(target));
            let result = input.clone().apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;
            assert!(result.is_valid(0, &mut ctx)?);
            let list = result.execute_scalar(0, &mut ctx)?;
            let elems = list
                .as_list()
                .elements()
                .ok_or_else(|| vortex_error::vortex_err!("expected non-null fixed-size list"))?;
            assert_eq!(elems.len(), 2);
            assert_eq!(
                elems[0]
                    .as_variant()
                    .value()
                    .and_then(|v| v.as_primitive().typed_value::<i32>()),
                Some(7)
            );
            assert!(!elems[1].is_null());
            assert_eq!(elems[1].as_variant().is_variant_null(), Some(true));
        }
        Ok(())
    }

    #[test]
    fn variant_get_root_struct_respects_variant_pack_shredded_validity() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core = variant_rows([
            Scalar::variant(variant_object([])),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?;
        let shredded = StructArray::try_new(
            FieldNames::from_iter(["a"]),
            vec![PrimitiveArray::from_option_iter([Some(10i32), None]).into_array()],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        let validity = BoolArray::from_iter([true, false]).into_array();
        let packed =
            VariantPack::try_new_with_validity(core, shredded, validity, true)?.into_array();

        let target = DType::struct_(
            [("a", DType::Primitive(PType::I32, Nullability::Nullable))],
            Nullability::Nullable,
        );
        let expr = variant_get(root(), VariantPath::root(), Some(target));
        let result = packed
            .apply(&expr)?
            .optimize_ctx(&ctx)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is_valid(0, &mut ctx)?);
        assert_eq!(
            result
                .execute_scalar(0, &mut ctx)?
                .as_struct()
                .field("a")
                .and_then(|s| s.as_primitive().typed_value::<i32>()),
            Some(10),
        );
        assert!(result.is_invalid(1, &mut ctx)?);
        Ok(())
    }
}

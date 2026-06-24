// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_proto::expr::variant_path_element;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::StringEscape;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::VariantArray;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Extracts a field/index path from Variant values.
///
/// Missing paths, type mismatches while traversing, and failed casts produce nulls. Without a
/// requested dtype, results are returned as nullable Variant values; with one, results are cast to
/// that dtype with nullable nullability. Encodings may serve perfectly shredded paths directly,
/// but must fall back to the core Variant value for paths not represented by shredded storage.
#[derive(Clone)]
pub struct VariantGet;

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
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "variant_get(")?;
        expr.child(0).fmt_sql(f)?;
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

        if !dtype.is_variant() {
            let mut builder = builder_with_capacity_in(ctx.allocator(), &dtype, input.len());
            for idx in 0..input.len() {
                let scalar = input.execute_scalar(idx, ctx)?;
                let output = variant_get_scalar(&scalar, options, &dtype)?;
                builder.append_scalar(&output)?;
            }

            return Ok(builder.finish_into_canonical().into_array());
        }

        // TODO(variant): replace this with a Variant builder once one exists.
        // Chunked<Variant> canonicalizes to VariantArray, so this row-wise fallback is safe.
        let mut chunks = Vec::with_capacity(input.len());

        for idx in 0..input.len() {
            let scalar = input.execute_scalar(idx, ctx)?;
            let output = variant_get_scalar(&scalar, options, &dtype)?;
            chunks.push(ConstantArray::new(output, 1).into_array());
        }

        let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
        VariantArray::try_new(array, None).map(|array| array.into_array())
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

    value
        .cast(output_dtype)
        .or_else(|_| Ok(Scalar::null(output_dtype.clone())))
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
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_ensure;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VariantArray;
    use crate::arrays::variant::VariantArrayExt;
    use crate::assert_arrays_eq;
    use crate::assert_nth_scalar_is_null;
    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::expr::variant_get;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::variant_get::VariantGet;
    use crate::scalar_fn::fns::variant_get::VariantGetOptions;
    use crate::scalar_fn::fns::variant_get::VariantPath;
    use crate::scalar_fn::fns::variant_get::VariantPathElement;

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
            .map(|row| ConstantArray::new(row.cast(&dtype).unwrap(), 1).into_array())
            .collect();
        ChunkedArray::try_new(chunks, dtype).map(|array| array.into_array())
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
            .execute::<ArrayRef>(&mut array_session().create_execution_ctx())
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
        assert!(parse_path("$.").is_err());
        assert!(parse_path("$data").is_err());
        assert!(parse_path("$.data[-1]").is_err());
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
        let actual = VariantGet
            .deserialize(&metadata, &VortexSession::empty())
            .unwrap();

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
        let actual = Expression::from_proto(&proto, &array_session()).unwrap();

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
        let mut ctx = array_session().create_execution_ctx();

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
        let mut ctx = array_session().create_execution_ctx();

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
        let mut ctx = array_session().create_execution_ctx();
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

        let mut ctx = array_session().create_execution_ctx();
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
            .execute::<VariantArray>(&mut array_session().create_execution_ctx())?;
        let canonical = result.execute::<Canonical>(&mut array_session().create_execution_ctx())?;
        let Canonical::Variant(canonical_variant) = canonical else {
            vortex_bail!("expected Variant canonical array");
        };

        assert_eq!(variant.len(), 2);
        assert_eq!(canonical_variant.len(), 2);
        assert_eq!(variant.core_storage().dtype(), variant.dtype());
        assert_eq!(variant.core_storage().len(), variant.len());

        let mut ctx = array_session().create_execution_ctx();
        for (idx, expected) in [10i32, 20].into_iter().enumerate() {
            let scalar = variant.execute_scalar(idx, &mut ctx)?;
            let actual = scalar
                .as_variant()
                .value()
                .and_then(|value| value.as_primitive().as_::<i32>());
            assert_eq!(actual, Some(expected), "row {idx}");
        }
        Ok(())
    }
}

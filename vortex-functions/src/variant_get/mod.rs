// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod variant;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::builders::VariantBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::proto::expr as pb;
use vortex_array::proto::expr::variant_path_element;
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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Chunked;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
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
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_ensure;
    use vortex_error::vortex_err;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::Expression;
    use vortex_expr::proto::ExprSerializeProtoExt;
    use vortex_expr::root;
    use vortex_session::VortexSession;

    use crate::expr::variant_get;
    use crate::variant_get::VariantGet;
    use crate::variant_get::VariantGetOptions;
    use crate::variant_get::VariantPath;
    use crate::variant_get::VariantPathElement;

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

    fn obj(name: &'static str, value: Scalar) -> Scalar {
        Scalar::variant(variant_object([(name, value)]))
    }

    fn obj_a(value: i32) -> Scalar {
        obj("a", Scalar::primitive(value, Nullability::NonNullable))
    }

    #[test]
    fn variant_get_propagates_null_nested_shredded_struct() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core_storage = variant_rows([
            obj(
                "nested",
                variant_object([("a", Scalar::primitive(100i32, Nullability::NonNullable))]),
            ),
            obj(
                "nested",
                variant_object([("a", Scalar::primitive(200i32, Nullability::NonNullable))]),
            ),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?;
        let nested = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![PrimitiveArray::from_iter([10i32, 20, 30]).into_array()],
            3,
            Validity::from_iter([true, false, true]),
        )?
        .into_array();
        let shredded = StructArray::try_from_iter([("nested", nested)])?.into_array();
        let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

        let expr = variant_get(
            root(),
            parse_path("$.nested.a")?,
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let result = variant.apply(&expr)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10i32), Some(200), None]),
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    #[case::missing_struct_field("$.missing", true, [Some(100i32), Some(200), None])]
    #[case::list_index_path("$.items[1]", true, [Some(20i32), None, None])]
    #[case::field_through_non_struct("$.a.b", true, [None, None, None])]
    #[case::no_shredded_child("$.missing", false, [Some(100i32), Some(200), None])]
    fn variant_get_shredded_none_path_falls_back_to_core(
        #[case] path: &str,
        #[case] with_shredded: bool,
        #[case] expected: [Option<i32>; 3],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let list_two = Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![
                Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
                Scalar::variant(Scalar::primitive(20i32, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        let list_one = Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![Scalar::variant(Scalar::primitive(
                30i32,
                Nullability::NonNullable,
            ))],
            Nullability::NonNullable,
        );
        let core_storage = variant_rows([
            Scalar::variant(variant_object([
                ("a", Scalar::primitive(1i32, Nullability::NonNullable)),
                (
                    "missing",
                    Scalar::primitive(100i32, Nullability::NonNullable),
                ),
                ("items", list_two),
            ])),
            Scalar::variant(variant_object([
                ("a", Scalar::primitive(2i32, Nullability::NonNullable)),
                (
                    "missing",
                    Scalar::primitive(200i32, Nullability::NonNullable),
                ),
                ("items", list_one),
            ])),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?;
        let shredded = with_shredded
            .then(|| {
                let items = ListArray::try_new(
                    buffer![10i32, 20, 30].into_array(),
                    buffer![0u32, 2, 3, 3].into_array(),
                    Validity::NonNullable,
                    &mut ctx,
                )?
                .into_array();
                Ok::<_, vortex_error::VortexError>(
                    StructArray::try_from_iter([
                        ("a", buffer![1i32, 2, 3].into_array()),
                        ("items", items),
                    ])?
                    .into_array(),
                )
            })
            .transpose()?;
        let variant = VariantArray::try_new(core_storage, shredded)?.into_array();

        let expr = variant_get(
            root(),
            parse_path(path)?,
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let result = variant.apply(&expr)?;

        assert_arrays_eq!(result, PrimitiveArray::from_option_iter(expected), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::all_valid_variant_child(
        vec![
            Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::utf8("not_an_int", Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(30i32, Nullability::NonNullable)),
        ],
        [Some(10i32), None, Some(30)],
    )]
    #[case::null_variant_child_falls_back_to_core(
        vec![
            Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
            Scalar::null(DType::Variant(Nullability::Nullable)),
            Scalar::variant(Scalar::primitive(30i32, Nullability::NonNullable)),
        ],
        [Some(10i32), Some(200), Some(30)],
    )]
    #[case::all_null_variant_child_falls_back_to_core(
        vec![
            Scalar::null(DType::Variant(Nullability::Nullable)),
            Scalar::null(DType::Variant(Nullability::Nullable)),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ],
        [Some(100i32), Some(200), Some(300)],
    )]
    fn variant_get_shredded_variant_child_casts_to_concrete_dtype(
        #[case] shredded_rows: Vec<Scalar>,
        #[case] expected: [Option<i32>; 3],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core_storage = variant_rows([obj_a(100), obj_a(200), obj_a(300)])?;
        let shredded_variant = VariantArray::try_new(variant_rows(shredded_rows)?, None)?;
        let shredded =
            StructArray::try_from_iter([("a", shredded_variant.into_array())])?.into_array();
        let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let result = variant.apply(&expr)?;

        assert_arrays_eq!(result, PrimitiveArray::from_option_iter(expected), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::none_dtype(None)]
    #[case::explicit_variant_dtype(Some(DType::Variant(Nullability::NonNullable)))]
    fn variant_get_untyped_merges_shredded_struct_with_fallback(
        #[case] requested_dtype: Option<DType>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core_storage = variant_rows([
            obj(
                "obj",
                variant_object([
                    ("a", Scalar::primitive(1i32, Nullability::NonNullable)),
                    ("b", Scalar::utf8("raw0", Nullability::NonNullable)),
                ]),
            ),
            obj(
                "obj",
                variant_object([("b", Scalar::utf8("raw1", Nullability::NonNullable))]),
            ),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?;
        let shredded_obj = StructArray::try_from_iter([(
            "a",
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array(),
        )])?
        .into_array();
        let shredded = StructArray::try_from_iter([("obj", shredded_obj)])?.into_array();
        let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

        let expr = variant_get(root(), VariantPath::field("obj"), requested_dtype);
        let result = variant.apply(&expr)?;

        assert_arrays_eq!(
            result,
            variant_rows([
                Scalar::variant(variant_object([
                    ("a", Scalar::primitive(10i32, Nullability::NonNullable)),
                    ("b", Scalar::utf8("raw0", Nullability::NonNullable)),
                ])),
                obj("b", Scalar::utf8("raw1", Nullability::NonNullable)),
                Scalar::null(DType::Variant(Nullability::Nullable)),
            ])?,
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    #[case::all_valid_primitive(
        PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
        vec![obj_a(1), obj_a(2), obj_a(3)],
        vec![
            Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(20i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::primitive(30i32, Nullability::NonNullable)),
        ],
    )]
    #[case::partial_null_primitive(
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array(),
        vec![
            obj_a(1),
            obj("a", Scalar::utf8("fallback", Nullability::NonNullable)),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ],
        vec![
            Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
            Scalar::variant(Scalar::utf8("fallback", Nullability::NonNullable)),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ],
    )]
    fn variant_get_untyped_merges_shredded_leaf_values(
        #[case] shredded_a: ArrayRef,
        #[case] core_rows: Vec<Scalar>,
        #[case] expected: Vec<Scalar>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core_storage = variant_rows(core_rows)?;
        let shredded = StructArray::try_from_iter([("a", shredded_a)])?.into_array();
        let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

        let expr = variant_get(root(), VariantPath::field("a"), None);
        let result = variant.apply(&expr)?;

        assert_arrays_eq!(result, variant_rows(expected)?, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::list(false)]
    #[case::fixed_size_list(true)]
    fn variant_get_untyped_list_falls_back_to_core(#[case] fixed_size: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let row0_list = Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![
                Scalar::variant(Scalar::primitive(10i32, Nullability::NonNullable)),
                Scalar::variant(Scalar::primitive(20i32, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        let row1_list = Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![
                Scalar::variant(Scalar::primitive(30i32, Nullability::NonNullable)),
                Scalar::variant(Scalar::primitive(40i32, Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );
        let core_storage = variant_rows([
            Scalar::variant(row0_list.clone()),
            Scalar::variant(row1_list.clone()),
        ])?;
        let shredded_items = if fixed_size {
            FixedSizeListArray::try_new(
                buffer![1i32, 2, 3, 4].into_array(),
                2,
                Validity::NonNullable,
                2,
            )?
            .into_array()
        } else {
            ListArray::try_new(
                buffer![1i32, 2, 3, 4].into_array(),
                buffer![0u32, 2, 4].into_array(),
                Validity::NonNullable,
                &mut ctx,
            )?
            .into_array()
        };
        let variant = VariantArray::try_new(core_storage, Some(shredded_items))?.into_array();

        let expr = variant_get(root(), VariantPath::root(), None);
        let result = variant.apply(&expr)?;

        assert_arrays_eq!(
            result,
            variant_rows([Scalar::variant(row0_list), Scalar::variant(row1_list)])?,
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_dtype_mismatch_falls_back_to_core() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core_storage = variant_rows([
            obj_a(10),
            obj_a(20),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?;
        let shredded = StructArray::try_from_iter([(
            "a",
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
        )])?
        .into_array();
        let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

        let widened = variant.clone().apply(&variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Primitive(PType::I64, Nullability::NonNullable)),
        ))?;
        assert_arrays_eq!(
            widened,
            PrimitiveArray::from_option_iter([Some(10i64), Some(20), None]),
            &mut ctx
        );

        let incompatible = variant.apply(&variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Utf8(Nullability::NonNullable)),
        ))?;
        assert_arrays_eq!(
            incompatible,
            VarBinViewArray::from_iter_nullable_str([None::<&str>, None, None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_partial_null_shredded_rows_zip_with_fallback() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let core_storage = variant_rows([
            obj_a(100),
            obj_a(200),
            obj("a", Scalar::utf8("not_int", Nullability::NonNullable)),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?;
        let shredded = StructArray::try_from_iter([(
            "a",
            PrimitiveArray::from_option_iter([Some(10i32), None, None, Some(40)]).into_array(),
        )])?
        .into_array();
        let variant = VariantArray::try_new(core_storage, Some(shredded))?.into_array();

        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let result = variant.apply(&expr)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10i32), Some(200), None, None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn variant_get_unwraps_nested_variant_core_storage() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let inner_core = variant_rows([obj_a(10), obj_a(20)])?;
        let inner_variant = VariantArray::try_new(inner_core, None)?.into_array();
        let outer_variant = VariantArray::try_new(inner_variant, None)?.into_array();

        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let result = outer_variant.apply(&expr)?;

        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(10i32), Some(20)]),
            &mut ctx
        );
        Ok(())
    }
}

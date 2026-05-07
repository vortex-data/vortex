// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::str::FromStr;

use prost::Message;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_proto::expr::variant_path_element;
use vortex_session::VortexSession;
use vortex_utils::aliases::StringEscape;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::Nullability;
use crate::expr::Expression;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Extracts a path from a Variant expression, optionally as a typed result.
#[derive(Clone)]
pub struct VariantGet;

impl ScalarFnVTable for VariantGet {
    type Options = VariantGetOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.variant_get")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let path = options
            .path()
            .elements()
            .iter()
            .map(|element| match element {
                VariantPathElement::Field(name) => pb::VariantPathElement {
                    element: Some(variant_path_element::Element::Field(
                        name.as_ref().to_string(),
                    )),
                },
                VariantPathElement::Index(index) => pb::VariantPathElement {
                    element: Some(variant_path_element::Element::Index(*index)),
                },
            })
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

        Ok(options
            .dtype()
            .map_or(DType::Variant(Nullability::Nullable), DType::as_nullable))
    }

    fn execute(
        &self,
        options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "VariantGet execution is not implemented yet for path {}",
            options.path()
        )
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

    /// Parses the strict path grammar supported by this skeleton.
    ///
    /// The accepted grammar is an optional leading `$`, dot-separated ASCII identifier fields,
    /// and non-negative decimal list indexes in brackets, for example `$.data[1].a` or `data[1]`.
    pub fn parse(path: &str) -> VortexResult<Self> {
        Self::from_str(path)
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

impl FromStr for VariantPath {
    type Err = VortexError;

    fn from_str(path: &str) -> Result<Self, Self::Err> {
        if path.is_empty() || path == "$" {
            return Ok(Self::root());
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

        Ok(Self(elements))
    }
}

impl TryFrom<&str> for VariantPath {
    type Error = VortexError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
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

    fn from_proto(value: pb::VariantPathElement) -> VortexResult<Self> {
        match value
            .element
            .ok_or_else(|| vortex_err!("VariantGet path element missing value"))?
        {
            variant_path_element::Element::Field(field) => Ok(Self::field(field)),
            variant_path_element::Element::Index(index) => Ok(Self::index(index)),
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

fn parse_field(path: &str, start: usize) -> VortexResult<(FieldName, usize)> {
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
    Ok((FieldName::from(&path[start..pos]), pos))
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

#[cfg(test)]
mod tests {
    use vortex_session::VortexSession;

    use crate::LEGACY_SESSION;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::proto::ExprSerializeProtoExt;
    use crate::expr::root;
    use crate::expr::variant_get;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::variant_get::VariantGet;
    use crate::scalar_fn::fns::variant_get::VariantGetOptions;
    use crate::scalar_fn::fns::variant_get::VariantPath;
    use crate::scalar_fn::fns::variant_get::VariantPathElement;

    #[test]
    fn variant_get_path_parse_and_display() {
        let path = VariantPath::parse("$.data[1].a").unwrap();
        assert_eq!(
            path.elements(),
            &[
                VariantPathElement::field("data"),
                VariantPathElement::index(1),
                VariantPathElement::field("a")
            ]
        );
        assert_eq!(path.to_string(), "$.data[1].a");

        let bare_path = VariantPath::parse("data[2]").unwrap();
        assert_eq!(bare_path.to_string(), "$.data[2]");
        assert!(VariantPath::parse("$.").is_err());
        assert!(VariantPath::parse("$data").is_err());
        assert!(VariantPath::parse("$.data[-1]").is_err());
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
            VariantPath::parse("$.data[1].a").unwrap(),
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
            VariantPath::parse("$.data[1].a").unwrap(),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable)),
        );
        let proto = expr.serialize_proto().unwrap();
        let actual = Expression::from_proto(&proto, &LEGACY_SESSION).unwrap();

        assert_eq!(actual, expr);
    }
}

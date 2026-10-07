// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rewriting expressions over a Variant column into expressions over its Parquet storage struct.

use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::cast;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::pack;
use vortex_array::expr::bound::select;
use vortex_array::expr::display::ExprDisplay;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantGetOptions;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::ShreddedPath;
use crate::ParquetVariant;

const METADATA: &str = "metadata";
const VALUE: &str = "value";
const TYPED_VALUE: &str = "typed_value";

/// Assembles Parquet Variant values from a `{metadata, value?, typed_value?}` storage struct.
///
/// The struct's validity becomes the Variant validity. This only exists to evaluate rewritten
/// layout expressions; it is never serialized.
#[derive(Clone)]
pub(crate) struct ParquetVariantFromStorage;

impl ScalarFnVTable for ParquetVariantFromStorage {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.parquet_variant.from_storage");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("parquet_variant_from_storage is not serializable")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("storage"),
            _ => unreachable!("Invalid child index {child_idx} for parquet_variant_from_storage"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "parquet_variant_from_storage({})", expr.display_child(0))
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let storage = &arg_dtypes[0];
        let fields = storage.as_struct_fields_opt().ok_or_else(|| {
            vortex_err!("Parquet Variant storage must be a struct, found {storage}")
        })?;
        vortex_ensure!(
            fields.field(METADATA).is_some(),
            "Parquet Variant storage must have a metadata field, found {storage}"
        );
        Ok(DType::Variant(storage.nullability()))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let storage = args.get(0)?.execute::<StructArray>(ctx)?;
        let field = |name: &str| storage.unmasked_field_by_name_opt(name).cloned();
        let metadata = field(METADATA)
            .ok_or_else(|| vortex_err!("Parquet Variant storage is missing metadata"))?;
        let validity = match storage.dtype().nullability() {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable => storage.validity()?,
        };
        Ok(
            ParquetVariant::try_new(validity, metadata, field(VALUE), field(TYPED_VALUE))?
                .into_array(),
        )
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Rewrites `expr`, bound over a Variant root, into an expression over the storage struct root.
pub(crate) fn rewrite_variant_expr(
    expr: &BoundExpression,
    storage_dtype: &DType,
    typed_paths: &[ShreddedPath],
) -> VortexResult<BoundExpression> {
    let storage_root = BoundExpression::new_root(storage_dtype.clone());
    rewrite_node(expr, &storage_root, typed_paths)
}

fn rewrite_node(
    expr: &BoundExpression,
    storage_root: &BoundExpression,
    typed_paths: &[ShreddedPath],
) -> VortexResult<BoundExpression> {
    if expr.is_root() {
        return assemble_variant(storage_root.clone());
    }

    if let Some(options) = expr.as_opt::<VariantGet>()
        && expr.child(0).is_root()
    {
        return rewrite_variant_get(options, storage_root, typed_paths);
    }

    let children = expr
        .children()
        .iter()
        .map(|child| rewrite_node(child, storage_root, typed_paths))
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

fn assemble_variant(storage: BoundExpression) -> VortexResult<BoundExpression> {
    ParquetVariantFromStorage.try_new_bound_expr(EmptyOptions, [storage])
}

/// Rewrites `variant_get(root, path, dtype)` to read as little of the storage as possible.
///
/// The longest prefix of `path` that is shredded as an object field resolves to that field's
/// `{value, typed_value}` wrapper, which fully represents the Variant value at the prefix. When the
/// whole path is shredded, its residual `value` is known to be empty, and its typed column has the
/// requested dtype, the result is the typed column itself. Otherwise the rest of the path is
/// extracted from a Variant assembled from the shared `metadata` and the wrapper alone.
///
/// When the rest of the path starts with an object field that is not shredded, only the wrapper's
/// residual `value` can hold it: shredded object fields never appear in the residual, and a value
/// that is not an object has no fields. Its `typed_value` is then left unread.
fn rewrite_variant_get(
    options: &VariantGetOptions,
    storage_root: &BoundExpression,
    typed_paths: &[ShreddedPath],
) -> VortexResult<BoundExpression> {
    let elements = options.path().elements();

    // Walk the shredded `typed_value` tree as far along the path as it goes.
    let mut wrapper = storage_root.clone();
    let mut consumed: ShreddedPath = Vec::new();
    for element in elements {
        let VariantPathElement::Field(name) = element else {
            break;
        };
        let Some(child) = shredded_field(&wrapper, name)? else {
            break;
        };
        wrapper = child;
        consumed.push(name.clone());
    }
    let remaining = VariantPath::new(elements[consumed.len()..].iter().cloned());

    if remaining.is_root()
        && !consumed.is_empty()
        && let Some(dtype) = options.dtype()
        && !dtype.is_variant()
        && let Some(typed) = fully_typed_leaf(&wrapper, &consumed, typed_paths)?
        && typed.dtype().eq_ignore_nullability(dtype)
    {
        let nullable = dtype.as_nullable();
        return Ok(if typed.dtype() == &nullable {
            typed
        } else {
            cast(typed, nullable)
        });
    }

    let residual_only = residual_holds_rest(wrapper.dtype(), &remaining);
    let variant = if consumed.is_empty() {
        if residual_only {
            // Keep the storage validity: null Variant rows stay null.
            assemble_variant(select(
                FieldNames::from_iter([FieldName::from(METADATA), FieldName::from(VALUE)]),
                storage_root.clone(),
            ))?
        } else {
            assemble_variant(storage_root.clone())?
        }
    } else {
        // A shredded object field shares the top-level metadata with its own wrapper columns.
        let metadata = get_item(METADATA, storage_root.clone());
        let mut fields = vec![(FieldName::from(METADATA), metadata)];
        for name in [VALUE, TYPED_VALUE] {
            if has_field(wrapper.dtype(), name) && !(residual_only && name == TYPED_VALUE) {
                fields.push((FieldName::from(name), get_item(name, wrapper.clone())));
            }
        }
        assemble_variant(pack(fields, Nullability::NonNullable))?
    };
    VariantGet.try_new_bound_expr(
        VariantGetOptions::new(remaining, options.dtype().cloned()),
        [variant],
    )
}

/// Whether only the residual `value` of `wrapper` can hold the rest of a path that the shredded
/// tree could not follow.
fn residual_holds_rest(wrapper: &DType, remaining: &VariantPath) -> bool {
    matches!(
        remaining.elements().first(),
        Some(VariantPathElement::Field(_))
    ) && has_field(wrapper, VALUE)
}

/// Returns the wrapper `{value, typed_value}` of object field `name` shredded under `wrapper`.
fn shredded_field(
    wrapper: &BoundExpression,
    name: &FieldName,
) -> VortexResult<Option<BoundExpression>> {
    let Some(typed_value) = struct_field(wrapper.dtype(), TYPED_VALUE) else {
        return Ok(None);
    };
    let Some(field) = typed_value
        .as_struct_fields_opt()
        .and_then(|fields| fields.field(name))
    else {
        return Ok(None);
    };
    if !is_wrapper(&field) {
        return Ok(None);
    }
    Ok(Some(get_item(
        name.clone(),
        get_item(TYPED_VALUE, wrapper.clone()),
    )))
}

/// The wrapper's typed column, if its values are all represented by it alone.
fn fully_typed_leaf(
    wrapper: &BoundExpression,
    path: &ShreddedPath,
    typed_paths: &[ShreddedPath],
) -> VortexResult<Option<BoundExpression>> {
    let Some(typed_value) = struct_field(wrapper.dtype(), TYPED_VALUE) else {
        return Ok(None);
    };
    if typed_value.is_struct() || typed_value.is_list() {
        return Ok(None);
    }
    let residual_empty =
        !has_field(wrapper.dtype(), VALUE) || typed_paths.iter().any(|typed| typed == path);
    Ok(residual_empty.then(|| get_item(TYPED_VALUE, wrapper.clone())))
}

/// Whether `dtype` is a Parquet shredded field wrapper: a struct of `value` and/or `typed_value`.
pub(crate) fn is_wrapper(dtype: &DType) -> bool {
    dtype.as_struct_fields_opt().is_some_and(|fields| {
        fields.nfields() > 0
            && fields
                .names()
                .iter()
                .all(|name| matches!(name.as_ref(), VALUE | TYPED_VALUE))
    })
}

fn struct_field(dtype: &DType, name: &str) -> Option<DType> {
    dtype
        .as_struct_fields_opt()
        .and_then(|fields: &StructFields| fields.field(name))
}

fn has_field(dtype: &DType, name: &str) -> bool {
    struct_field(dtype, name).is_some()
}

/// Translates field masks over a Variant column into field masks over its storage struct.
///
/// A mask's field path below the Variant column is the object path a `variant_get` extracts (see
/// `referenced_field_paths`). It selects the same storage columns that [`rewrite_variant_expr`]
/// reads for that path: the typed column of a fully typed shredded path, or the shared metadata
/// and the wrapper of the longest shredded prefix.
pub(crate) fn storage_field_masks(
    masks: &[FieldMask],
    storage_dtype: &DType,
    typed_paths: &[ShreddedPath],
) -> Vec<FieldMask> {
    let mut storage_masks = Vec::with_capacity(masks.len());
    for mask in masks {
        let path = match mask {
            FieldMask::All => return vec![FieldMask::All],
            FieldMask::Prefix(path) | FieldMask::Exact(path) => path,
        };

        let mut storage_path = FieldPath::root();
        let mut wrapper = storage_dtype.clone();
        let mut consumed: ShreddedPath = Vec::new();
        for field in path.parts() {
            let Field::Name(name) = field else {
                break;
            };
            let Some(child) = struct_field(&wrapper, TYPED_VALUE)
                .and_then(|typed_value| struct_field(&typed_value, name.as_ref()))
                .filter(is_wrapper)
            else {
                break;
            };
            wrapper = child;
            storage_path = storage_path.push(TYPED_VALUE).push(name.clone());
            consumed.push(name.clone());
        }

        let rest_is_field = consumed.len() < path.parts().len()
            && matches!(path.parts()[consumed.len()], Field::Name(_))
            && has_field(&wrapper, VALUE);
        if rest_is_field {
            storage_masks.push(FieldMask::Prefix(FieldPath::from_name(METADATA)));
            storage_masks.push(FieldMask::Prefix(storage_path.push(VALUE)));
            continue;
        }
        if consumed.is_empty() {
            return vec![FieldMask::All];
        }
        let fully_typed = consumed.len() == path.parts().len()
            && struct_field(&wrapper, TYPED_VALUE)
                .is_some_and(|typed| !typed.is_struct() && !typed.is_list())
            && (!has_field(&wrapper, VALUE) || typed_paths.contains(&consumed));
        if fully_typed {
            storage_masks.push(FieldMask::Prefix(storage_path.push(TYPED_VALUE)));
        } else {
            storage_masks.push(FieldMask::Prefix(FieldPath::from_name(METADATA)));
            storage_masks.push(FieldMask::Prefix(storage_path));
        }
    }
    storage_masks
}

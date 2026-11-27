// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Up-front verification of a whole `DType` flatbuffer.
//!
//! [`DType::from_flatbuffer`] parses eagerly only down to the first lazy boundary: struct fields
//! and union variants are kept as [`ViewedDType`]s and parsed on first access, by infallible
//! accessors such as [`StructFields::fields`](crate::dtype::StructFields::fields) that panic if
//! the bytes don't describe a valid `DType`. The `flatbuffers` verifier run by
//! [`flatbuffers::root`] only proves the buffer is structurally sound (offsets in bounds, valid
//! UTF-8, nesting depth), not that it describes a valid `DType`.
//!
//! [`verify_dtype`] closes that gap. It walks the whole tree once and checks every condition that
//! `TryFrom<ViewedDType> for DType` can fail on, without building the `DType`s, so any subtree of
//! a verified buffer parses without error.
//!
//! When adding a check to `TryFrom<ViewedDType> for DType`, add the same check here.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use super::ViewedDType;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::dtype::PType;
use crate::dtype::UnionVariants;
use crate::dtype::flatbuffers as fb;
use crate::flatbuffers::FlatBuffer;
use crate::flatbuffers::dtype as fbd;

/// Verifies that every `DType` table reachable from `root` describes a valid [`DType`].
///
/// `root` must come from a buffer already checked by the `flatbuffers` verifier, which bounds the
/// nesting depth and therefore the recursion depth here.
pub(super) fn verify_dtype(
    root: fbd::DType<'_>,
    buffer: &FlatBuffer,
    session: &VortexSession,
) -> VortexResult<()> {
    DTypeVerifier {
        buffer,
        session,
        verified: HashSet::default(),
    }
    .verify(root)
}

struct DTypeVerifier<'a> {
    buffer: &'a FlatBuffer,
    session: &'a VortexSession,
    /// Locations of the `DType` tables already verified.
    ///
    /// The writer interns equal subtrees, so a schema is a DAG in which one table can be referenced
    /// from many parents. Verifying each table once keeps the walk linear in the size of the buffer
    /// rather than the size of the expanded tree.
    verified: HashSet<usize>,
}

impl DTypeVerifier<'_> {
    fn verify(&mut self, dtype: fbd::DType<'_>) -> VortexResult<()> {
        // Flatbuffer offsets are unsigned and point forward, so the graph is acyclic and a table
        // can be marked before its children are verified. On error the whole walk is abandoned.
        if !self.verified.insert(dtype._tab.loc()) {
            return Ok(());
        }

        match dtype.type_type() {
            fb::Type::Null => {}
            fb::Type::Bool => {
                required(dtype.type__as_bool(), "bool")?;
            }
            fb::Type::Primitive => {
                let primitive = required(dtype.type__as_primitive(), "primitive")?;
                PType::try_from(primitive.ptype())?;
            }
            fb::Type::Decimal => {
                let decimal = required(dtype.type__as_decimal(), "decimal dtype")?;
                DecimalDType::try_new(decimal.precision(), decimal.scale())?;
            }
            fb::Type::Utf8 => {
                required(dtype.type__as_utf_8(), "utf-8")?;
            }
            fb::Type::Binary => {
                required(dtype.type__as_binary(), "binary")?;
            }
            fb::Type::Variant => {
                required(dtype.type__as_variant(), "variant")?;
            }
            fb::Type::List => {
                let list = required(dtype.type__as_list(), "list")?;
                self.verify(required(list.element_type(), "list element type")?)?;
            }
            fb::Type::FixedSizeList => {
                let list = required(dtype.type__as_fixed_size_list(), "fixed-size list")?;
                self.verify(required(list.element_type(), "list element type")?)?;
            }
            fb::Type::Map => {
                let map = required(dtype.type__as_map(), "map")?;
                let key = required(map.key_type(), "map key type")?;
                let value = required(map.value_type(), "map value type")?;
                self.verify(key)?;
                self.verify(value)?;
                // `MapDType::try_from_fields` parses the key eagerly to check its nullability.
                vortex_ensure!(
                    !self.parse(key)?.is_nullable(),
                    "map key dtype must be non-nullable"
                );
            }
            fb::Type::Struct_ => {
                let struct_ = required(dtype.type__as_struct_(), "struct")?;
                let names = required(struct_.names(), "struct names")?;
                let dtypes = required(struct_.dtypes(), "struct dtypes")?;
                vortex_ensure_eq!(
                    names.len(),
                    dtypes.len(),
                    "length mismatch between struct names and dtypes"
                );
                for field in dtypes.iter() {
                    self.verify(field)?;
                }
            }
            fb::Type::Union => {
                let union = required(dtype.type__as_union(), "union")?;
                let names: FieldNames = required(union.names(), "union names")?.iter().collect();
                let dtypes = required(union.dtypes(), "union dtypes")?;
                let type_ids: Vec<u8> = required(union.type_ids(), "union type_ids")?
                    .iter()
                    .map(i8::cast_unsigned)
                    .collect();
                UnionVariants::validate_shape(&names, dtypes.len(), &type_ids)?;
                for variant in dtypes.iter() {
                    self.verify(variant)?;
                }
            }
            fb::Type::Extension => {
                let ext = required(dtype.type__as_extension(), "extension")?;
                self.verify(required(ext.storage_dtype(), "extension storage dtype")?)?;
                // Resolving the extension ID and deserializing its metadata needs the parsed storage
                // dtype, which `TryFrom` parses eagerly anyway, so parse the extension itself. Its
                // storage was verified above, so any lazy children it holds are valid too.
                self.parse(dtype)?;
            }
            _ => vortex_bail!("Unknown DType variant {}", dtype.type_type().0),
        }

        Ok(())
    }

    /// Parses the table at `dtype` down to its first lazy boundary.
    fn parse(&self, dtype: fbd::DType<'_>) -> VortexResult<DType> {
        DType::try_from(ViewedDType::from_fb_loc(
            dtype._tab.loc(),
            self.buffer.clone(),
            self.session.clone(),
        ))
    }
}

fn required<T>(value: Option<T>, what: &str) -> VortexResult<T> {
    value.ok_or_else(|| vortex_err!("failed to parse {what} from flatbuffer"))
}

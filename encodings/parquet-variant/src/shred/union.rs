// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shreds values into the members of a union.
//!
//! A union node of the schema takes a value whole, or not at all: the value goes to the first
//! member that holds all of it, and the residual of the node keeps a value that no member holds.
//! So a member has no residual of its own, and its typed tree is the value.
//!
//! A leaf member holds a value of its type, with the integer widths as one class, as a typed
//! column of the shredder does. A struct member holds an object if it names each field of the
//! object and holds the value of each. A list member holds an array if it holds each element.
//! A Variant null fits no member.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::UnionArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::UnionVariants;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::exact::Fail;
use super::exact::Leaf;
use super::exact::OrMalformed;
use super::exact::binary;
use super::exact::binary_builder;
use super::exact::boolean;
use super::exact::double;
use super::exact::float;
use super::exact::integer;
use super::exact::string;
use crate::extract::Elements;
use crate::extract::Fields;
use crate::extract::Object;

/// Whether `schema` is or holds a union.
pub(crate) fn has_union(schema: &DType) -> bool {
    match schema {
        DType::Union(..) => true,
        DType::Struct(fields, _) => fields.fields().any(|field| has_union(&field)),
        DType::List(element, _) => has_union(element),
        _ => false,
    }
}

/// The dtype of the shredded column of a node of `schema` in a union.
///
/// It is a function of the schema alone, so that every chunk of a stream has the same dtype. A
/// member is null in the rows of the other members, and a struct field is null where the object
/// does not have it, so each node is nullable.
fn member_dtype(schema: &DType) -> VortexResult<DType> {
    Ok(match schema {
        DType::Struct(fields, _) => DType::Struct(
            StructFields::new(
                fields.names().clone(),
                fields
                    .fields()
                    .map(|field| member_dtype(&field))
                    .collect::<VortexResult<Vec<_>>>()?,
            ),
            Nullability::Nullable,
        ),
        DType::List(element, _) => {
            DType::List(member_dtype(element)?.into(), Nullability::Nullable)
        }
        DType::Union(variants, _) => {
            DType::Union(member_variants(variants)?, Nullability::Nullable)
        }
        leaf => leaf.as_nullable(),
    })
}

/// The variants of the shredded column of a union of `variants`.
fn member_variants(variants: &UnionVariants) -> VortexResult<UnionVariants> {
    let dtypes = variants
        .variants()
        .map(|variant| member_dtype(&variant))
        .collect::<VortexResult<Vec<_>>>()?;
    UnionVariants::try_new(
        variants.names().clone(),
        dtypes,
        variants.type_ids().to_vec(),
    )
}

/// The values of one node of a union, each of which the node holds whole.
enum Member {
    Utf8(VarBinViewBuilder),
    Binary(VarBinViewBuilder),
    Bool(Leaf<bool>),
    I8(Leaf<i8>),
    I16(Leaf<i16>),
    I32(Leaf<i32>),
    I64(Leaf<i64>),
    F32(Leaf<f32>),
    F64(Leaf<f64>),
    Object(Box<ObjectMember>),
    List(Box<ListMember>),
    Union(Box<UnionMember>),
}

/// A struct member. Each field holds the field of the object of the same name.
struct ObjectMember {
    dtype: DType,
    /// The member of each field, with the index of its name in the [`FieldCache`].
    ///
    /// [`FieldCache`]: crate::extract::FieldCache
    fields: Vec<(usize, Member)>,
    validity: BitBufferMut,
}

/// A list member. Each element is held by the same member.
struct ListMember {
    offsets: BufferMut<u64>,
    elements: Member,
    validity: BitBufferMut,
}

/// A union. Each value goes to the first member that holds it.
pub(super) struct UnionMember {
    variants: UnionVariants,
    type_ids: BufferMut<u8>,
    validity: BitBufferMut,
    members: Vec<Member>,
}

impl Member {
    /// A member of `schema`, which adds the field names of the structs in `schema` to `names`.
    /// The strings and binaries are allocated by `allocator`.
    fn new(
        schema: &DType,
        names: &mut Vec<FieldName>,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> VortexResult<Self> {
        Ok(match schema {
            DType::Utf8(_) => Self::Utf8(VarBinViewBuilder::with_capacity_in(
                DType::Utf8(Nullability::Nullable),
                capacity,
                allocator.clone(),
            )),
            DType::Binary(_) => Self::Binary(binary_builder(capacity, allocator)),
            DType::Bool(_) => Self::Bool(Leaf::new(capacity)),
            DType::Primitive(PType::I8, _) => Self::I8(Leaf::new(capacity)),
            DType::Primitive(PType::I16, _) => Self::I16(Leaf::new(capacity)),
            DType::Primitive(PType::I32, _) => Self::I32(Leaf::new(capacity)),
            DType::Primitive(PType::I64, _) => Self::I64(Leaf::new(capacity)),
            DType::Primitive(PType::F32, _) => Self::F32(Leaf::new(capacity)),
            DType::Primitive(PType::F64, _) => Self::F64(Leaf::new(capacity)),
            DType::Struct(struct_fields, _) => {
                let mut fields = Vec::with_capacity(struct_fields.nfields());
                for (name, field) in struct_fields.names().iter().zip(struct_fields.fields()) {
                    names.push(name.clone());
                    let name = names.len() - 1;
                    fields.push((name, Self::new(&field, names, capacity, allocator)?));
                }
                Self::Object(Box::new(ObjectMember {
                    dtype: member_dtype(schema)?,
                    fields,
                    validity: BitBufferMut::with_capacity(capacity),
                }))
            }
            DType::List(element, _) => {
                let mut offsets = BufferMut::with_capacity(capacity + 1);
                offsets.push(0);
                Self::List(Box::new(ListMember {
                    offsets,
                    elements: Self::new(element, names, capacity, allocator)?,
                    validity: BitBufferMut::with_capacity(capacity),
                }))
            }
            DType::Union(..) => Self::Union(Box::new(UnionMember::new(
                schema, names, capacity, allocator,
            )?)),
            _ => vortex_bail!(
                "a Variant can only be shredded into strings, binaries, booleans, signed integers, \
                 floats, structs, lists and unions, not {schema}"
            ),
        })
    }

    /// Whether this member holds all of `value`, the bytes of a value followed by any other
    /// bytes.
    fn fits(&self, fields: &Fields<'_>, value: &[u8]) -> Result<bool, Fail> {
        Ok(match self {
            Self::Utf8(_) => string(value).is_some(),
            Self::Binary(_) => binary(value).is_some(),
            Self::Bool(_) => boolean(value).is_some(),
            Self::I8(_) => integer(value).is_some_and(|v| i8::try_from(v).is_ok()),
            Self::I16(_) => integer(value).is_some_and(|v| i16::try_from(v).is_ok()),
            Self::I32(_) => integer(value).is_some_and(|v| i32::try_from(v).is_ok()),
            Self::I64(_) => integer(value).is_some(),
            Self::F32(_) => float(value).is_some(),
            Self::F64(_) => double(value).is_some(),
            Self::Object(object) => object.fits(fields, value)?,
            Self::List(list) => list.fits(fields, value)?,
            Self::Union(union) => union.member(fields, value)?.is_some(),
        })
    }

    /// Appends `value`, which this member [fits](Self::fits).
    fn append(&mut self, fields: &Fields<'_>, value: &[u8]) -> Result<(), Fail> {
        match self {
            Self::Utf8(builder) => builder.append_value(string(value).or_malformed()?),
            Self::Binary(builder) => builder.append_value(binary(value).or_malformed()?),
            Self::Bool(leaf) => leaf.push(Some(boolean(value).or_malformed()?)),
            Self::I8(leaf) => leaf.push(Some(narrow(value)?)),
            Self::I16(leaf) => leaf.push(Some(narrow(value)?)),
            Self::I32(leaf) => leaf.push(Some(narrow(value)?)),
            Self::I64(leaf) => leaf.push(Some(integer(value).or_malformed()?)),
            Self::F32(leaf) => leaf.push(Some(float(value).or_malformed()?)),
            Self::F64(leaf) => leaf.push(Some(double(value).or_malformed()?)),
            Self::Object(object) => object.append(fields, value)?,
            Self::List(list) => list.append(fields, value)?,
            Self::Union(union) => union.append(fields, value)?,
        }
        Ok(())
    }

    /// Appends a row that this member does not hold.
    fn append_null(&mut self) {
        match self {
            Self::Utf8(builder) | Self::Binary(builder) => builder.append_null(),
            Self::Bool(leaf) => leaf.push(None),
            Self::I8(leaf) => leaf.push(None),
            Self::I16(leaf) => leaf.push(None),
            Self::I32(leaf) => leaf.push(None),
            Self::I64(leaf) => leaf.push(None),
            Self::F32(leaf) => leaf.push(None),
            Self::F64(leaf) => leaf.push(None),
            Self::Object(object) => object.append_null(),
            Self::List(list) => list.append_null(),
            Self::Union(union) => union.append_null(),
        }
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        Ok(match self {
            Self::Utf8(mut builder) | Self::Binary(mut builder) => {
                builder.finish_into_varbinview().into_array()
            }
            Self::Bool(leaf) => leaf.finish_bool(),
            Self::I8(leaf) => leaf.finish(),
            Self::I16(leaf) => leaf.finish(),
            Self::I32(leaf) => leaf.finish(),
            Self::I64(leaf) => leaf.finish(),
            Self::F32(leaf) => leaf.finish(),
            Self::F64(leaf) => leaf.finish(),
            Self::Object(object) => object.finish()?,
            Self::List(list) => list.finish()?,
            Self::Union(union) => union.finish()?,
        })
    }
}

/// The integer that `value` holds, narrowed to `T`.
fn narrow<T: TryFrom<i64>>(value: &[u8]) -> Result<T, Fail> {
    integer(value)
        .and_then(|value| T::try_from(value).ok())
        .or_malformed()
}

impl ObjectMember {
    fn fits(&self, fields: &Fields<'_>, value: &[u8]) -> Result<bool, Fail> {
        let Some(object) = Object::new(value).or_malformed()? else {
            return Ok(false);
        };

        let mut held = 0;
        for (name, field) in &self.fields {
            if let Some(position) = fields.position(&object, *name).or_malformed()? {
                if !field.fits(fields, object.value(position).or_malformed()?)? {
                    return Ok(false);
                }
                held += 1;
            }
        }
        Ok(held == object.len())
    }

    fn append(&mut self, fields: &Fields<'_>, value: &[u8]) -> Result<(), Fail> {
        let object = Object::new(value).or_malformed()?.or_malformed()?;
        for (name, field) in &mut self.fields {
            match fields.position(&object, *name).or_malformed()? {
                Some(position) => field.append(fields, object.value(position).or_malformed()?)?,
                None => field.append_null(),
            }
        }
        self.validity.append(true);
        Ok(())
    }

    fn append_null(&mut self) {
        for (_, field) in &mut self.fields {
            field.append_null();
        }
        self.validity.append(false);
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        let DType::Struct(struct_fields, _) = self.dtype else {
            vortex_bail!("a struct member has dtype {}", self.dtype);
        };
        let len = self.validity.len();
        let fields = self
            .fields
            .into_iter()
            .map(|(_, field)| field.finish())
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(StructArray::try_new(
            struct_fields.names().clone(),
            fields,
            len,
            Validity::from(self.validity.freeze()),
        )?
        .into_array())
    }
}

impl ListMember {
    fn fits(&self, fields: &Fields<'_>, value: &[u8]) -> Result<bool, Fail> {
        let Some(list) = Elements::new(value).or_malformed()? else {
            return Ok(false);
        };

        for idx in 0..list.len() {
            if !self
                .elements
                .fits(fields, list.element(idx).or_malformed()?)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn append(&mut self, fields: &Fields<'_>, value: &[u8]) -> Result<(), Fail> {
        let list = Elements::new(value).or_malformed()?.or_malformed()?;
        for idx in 0..list.len() {
            self.elements
                .append(fields, list.element(idx).or_malformed()?)?;
        }

        let end = self.offsets.last().copied().unwrap_or_default();
        self.offsets
            .push(end + u64::try_from(list.len()).ok().or_malformed()?);
        self.validity.append(true);
        Ok(())
    }

    fn append_null(&mut self) {
        let end = self.offsets.last().copied().unwrap_or_default();
        self.offsets.push(end);
        self.validity.append(false);
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        let elements = self.elements.finish()?;
        let offsets = PrimitiveArray::new(self.offsets.freeze(), Validity::NonNullable);
        // SAFETY: the offsets start at zero and grow by the number of elements each list
        // appended, so they end at the number of elements, and there is one more offset than
        // there are validity bits.
        Ok(unsafe {
            ListArray::new_unchecked(
                elements,
                offsets.into_array(),
                Validity::from(self.validity.freeze()),
            )
        }
        .into_array())
    }
}

impl UnionMember {
    /// A union of `schema`, which adds the field names of the structs in `schema` to `names`.
    pub(super) fn new(
        schema: &DType,
        names: &mut Vec<FieldName>,
        capacity: usize,
        allocator: &BufferAllocatorRef,
    ) -> VortexResult<Self> {
        let DType::Union(variants, _) = schema else {
            vortex_bail!("a union member has schema {schema}");
        };

        let members = variants
            .variants()
            .map(|variant| Member::new(&variant, names, capacity, allocator))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            variants: member_variants(variants)?,
            type_ids: BufferMut::with_capacity(capacity),
            validity: BitBufferMut::with_capacity(capacity),
            members,
        })
    }

    /// The index of the first member that holds all of `value`.
    fn member(&self, fields: &Fields<'_>, value: &[u8]) -> Result<Option<usize>, Fail> {
        for (idx, member) in self.members.iter().enumerate() {
            if member.fits(fields, value)? {
                return Ok(Some(idx));
            }
        }
        Ok(None)
    }

    /// Appends `value` to the first member that holds all of it, and returns whether there was
    /// one. If there was none, nothing is appended.
    pub(super) fn append_fitting(
        &mut self,
        fields: &Fields<'_>,
        value: &[u8],
    ) -> Result<bool, Fail> {
        let Some(held_by) = self.member(fields, value)? else {
            return Ok(false);
        };

        for (idx, member) in self.members.iter_mut().enumerate() {
            if idx == held_by {
                member.append(fields, value)?;
            } else {
                member.append_null();
            }
        }
        let type_id = self.variants.type_ids().get(held_by).or_malformed()?;
        self.type_ids.push(*type_id);
        self.validity.append(true);
        Ok(true)
    }

    fn append(&mut self, fields: &Fields<'_>, value: &[u8]) -> Result<(), Fail> {
        if self.append_fitting(fields, value)? {
            Ok(())
        } else {
            Err(Fail::Malformed)
        }
    }

    pub(super) fn append_null(&mut self) {
        for member in &mut self.members {
            member.append_null();
        }
        self.type_ids.push(0);
        self.validity.append(false);
    }

    pub(super) fn finish(self) -> VortexResult<ArrayRef> {
        let type_ids = PrimitiveArray::new(
            self.type_ids.freeze(),
            Validity::from(self.validity.freeze()),
        )
        .into_array();
        let members = self
            .members
            .into_iter()
            .map(Member::finish)
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(UnionArray::try_new(type_ids, self.variants, members)?.into_array())
    }
}

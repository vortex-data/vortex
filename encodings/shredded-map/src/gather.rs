// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Gathers `(source, position)` pairs from several arrays into one canonical array.
//!
//! Decoding a shredded map interleaves values from the residual map and from every shredded
//! column. Concatenating the sources and taking from the result copies every value twice, so
//! common leaf types are gathered directly from their canonical buffers instead.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// For each output entry, the source it comes from and its position in that source.
pub(crate) struct Positions {
    pub src: Vec<u16>,
    pub pos: Vec<u32>,
}

impl Positions {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            src: Vec::with_capacity(n),
            pos: Vec::with_capacity(n),
        }
    }

    #[inline]
    pub fn push(&mut self, src: u16, pos: usize) {
        self.src.push(src);
        #[allow(clippy::cast_possible_truncation)]
        self.pos.push(pos as u32);
    }

    pub fn len(&self) -> usize {
        self.src.len()
    }
}

/// A value source of a gather.
pub(crate) enum Source {
    /// An array whose dtype matches the output dtype up to nullability.
    Array(ArrayRef),
    /// For a union output, an array holding only the values of the variant at child `child`.
    Variant { child: usize, array: ArrayRef },
}

/// A leaf source: an array or a value repeated at every position.
enum Leaf {
    Array(ArrayRef),
    Const(Scalar),
}

/// Gathers values of `dtype` from `sources` at `positions`.
pub(crate) fn gather(
    dtype: &DType,
    sources: Vec<Source>,
    positions: &Positions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let DType::Union(variants, nullability) = dtype else {
        let leaves = sources
            .into_iter()
            .map(|s| match s {
                Source::Array(a) => Leaf::Array(a),
                Source::Variant { .. } => unreachable!("variant sources require a union dtype"),
            })
            .collect();
        return gather_leaf(dtype, leaves, positions, ctx);
    };

    let mut type_ids = Vec::with_capacity(sources.len());
    let mut children: Vec<Vec<Leaf>> = (0..variants.len())
        .map(|_| Vec::with_capacity(sources.len()))
        .collect();
    for source in sources {
        match source {
            Source::Array(array) => {
                let union = array.execute::<UnionArray>(ctx)?;
                type_ids.push(Leaf::Array(union.type_ids().clone()));
                for (c, child) in union.iter_children().enumerate() {
                    children[c].push(Leaf::Array(child.clone()));
                }
            }
            Source::Variant { child, array } => {
                let tag = variants.type_ids()[child];
                type_ids.push(Leaf::Const(Scalar::primitive(tag, *nullability)));
                for (c, variant) in variants.variants().enumerate() {
                    children[c].push(if c == child {
                        Leaf::Array(array.clone())
                    } else {
                        Leaf::Const(Scalar::default_value(&variant))
                    });
                }
            }
        }
    }
    let type_ids = gather_leaf(
        &DType::Primitive(PType::U8, *nullability),
        type_ids,
        positions,
        ctx,
    )?;
    let children = children
        .into_iter()
        .zip(variants.variants())
        .map(|(leaves, dtype)| gather_leaf(&dtype, leaves, positions, ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(UnionArray::try_new(type_ids, variants.clone(), children)?.into_array())
}

/// A canonical leaf with its validity, if the output tracks validity.
enum Resolved {
    Const(Scalar),
    Primitive(PrimitiveArray, Option<Mask>),
    Bool(BitBuffer, Option<Mask>),
    View(VarBinViewArray, Option<Mask>),
}

fn resolve(leaf: Leaf, nullable: bool, ctx: &mut ExecutionCtx) -> VortexResult<Option<Resolved>> {
    let array = match leaf {
        Leaf::Const(scalar) => return Ok(Some(Resolved::Const(scalar))),
        Leaf::Array(array) => array,
    };
    if let Some(scalar) = array.as_constant() {
        return Ok(Some(Resolved::Const(scalar)));
    }
    let mask = if nullable && array.dtype().is_nullable() {
        Some(array.validity()?.execute_mask(array.len(), ctx)?)
    } else {
        None
    };
    Ok(match array.execute::<Canonical>(ctx)? {
        Canonical::Primitive(a) => Some(Resolved::Primitive(a, mask)),
        Canonical::Bool(a) => Some(Resolved::Bool(a.to_bit_buffer(), mask)),
        Canonical::VarBinView(a) => Some(Resolved::View(a, mask)),
        _ => None,
    })
}

fn gather_validity(resolved: &[Resolved], positions: &Positions) -> BitBuffer {
    let mut bits = BitBufferMut::with_capacity(positions.len());
    for (&s, &p) in positions.src.iter().zip(&positions.pos) {
        bits.append(match &resolved[s as usize] {
            Resolved::Const(scalar) => !scalar.is_null(),
            Resolved::Primitive(_, Some(m))
            | Resolved::Bool(_, Some(m))
            | Resolved::View(_, Some(m)) => m.value(p as usize),
            _ => true,
        });
    }
    bits.freeze()
}

fn gather_leaf(
    dtype: &DType,
    leaves: Vec<Leaf>,
    positions: &Positions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullable = dtype.is_nullable();
    let mut resolved = Vec::with_capacity(leaves.len());
    let mut leaves = leaves.into_iter();
    for leaf in leaves.by_ref() {
        match resolve(leaf, nullable, ctx)? {
            Some(r) => resolved.push(r),
            None => return gather_fallback(dtype, resolved, leaves, positions, ctx),
        }
    }
    let validity = if nullable {
        Validity::from_bit_buffer(gather_validity(&resolved, positions), dtype.nullability())
    } else {
        Validity::NonNullable
    };

    match dtype {
        DType::Primitive(ptype, _) => match_each_native_ptype!(*ptype, |T| {
            enum Src<'a> {
                Slice(&'a [T]),
                Const(T),
            }
            let srcs = resolved
                .iter()
                .map(|r| match r {
                    Resolved::Primitive(a, _) if a.ptype() == *ptype => {
                        Some(Src::Slice(a.as_slice::<T>()))
                    }
                    Resolved::Const(s) => Some(Src::Const(
                        s.as_primitive_opt()
                            .and_then(|p| p.typed_value::<T>())
                            .unwrap_or_default(),
                    )),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>();
            let Some(srcs) = srcs else {
                return gather_fallback(dtype, resolved, std::iter::empty(), positions, ctx);
            };
            let mut out = BufferMut::<T>::with_capacity(positions.len());
            for (&s, &p) in positions.src.iter().zip(&positions.pos) {
                out.push(match &srcs[s as usize] {
                    Src::Slice(slice) => slice[p as usize],
                    Src::Const(value) => *value,
                });
            }
            Ok(PrimitiveArray::new(out.freeze(), validity).into_array())
        }),
        DType::Bool(_) => {
            let mut bits = BitBufferMut::with_capacity(positions.len());
            for (&s, &p) in positions.src.iter().zip(&positions.pos) {
                bits.append(match &resolved[s as usize] {
                    Resolved::Bool(b, _) => b.value(p as usize),
                    Resolved::Const(scalar) => scalar
                        .as_bool_opt()
                        .and_then(|b| b.value())
                        .unwrap_or(false),
                    _ => {
                        return gather_fallback(
                            dtype,
                            resolved,
                            std::iter::empty(),
                            positions,
                            ctx,
                        );
                    }
                });
            }
            Ok(BoolArray::new(bits.freeze(), validity).into_array())
        }
        DType::Utf8(_) | DType::Binary(_) => {
            gather_views(dtype, resolved, validity, positions, ctx)
        }
        _ => gather_fallback(dtype, resolved, std::iter::empty(), positions, ctx),
    }
}

fn gather_views(
    dtype: &DType,
    resolved: Vec<Resolved>,
    validity: Validity,
    positions: &Positions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut buffers: Vec<ByteBuffer> = Vec::new();
    enum Src<'a> {
        Views(&'a [BinaryView], u32),
        Const(BinaryView),
    }
    let mut srcs = Vec::with_capacity(resolved.len());
    for r in &resolved {
        srcs.push(match r {
            Resolved::View(a, _) => {
                #[allow(clippy::cast_possible_truncation)]
                let base = buffers.len() as u32;
                buffers.extend(a.data_buffers().iter().map(|b| b.as_host().clone()));
                Src::Views(a.views(), base)
            }
            Resolved::Const(scalar) => {
                let bytes: Vec<u8> = if let Some(s) = scalar.as_utf8_opt() {
                    s.value().map(|v| v.as_bytes().to_vec()).unwrap_or_default()
                } else if let Some(b) = scalar.as_binary_opt() {
                    b.value().map(|v| v.to_vec()).unwrap_or_default()
                } else {
                    return gather_fallback(dtype, resolved, std::iter::empty(), positions, ctx);
                };
                #[allow(clippy::cast_possible_truncation)]
                let index = buffers.len() as u32;
                let view = BinaryView::make_view(&bytes, index, 0);
                if !view.is_inlined() {
                    buffers.push(ByteBuffer::from(bytes));
                }
                Src::Const(view)
            }
            _ => return gather_fallback(dtype, resolved, std::iter::empty(), positions, ctx),
        });
    }
    let mut views = BufferMut::<BinaryView>::with_capacity(positions.len());
    for (&s, &p) in positions.src.iter().zip(&positions.pos) {
        views.push(match &srcs[s as usize] {
            Src::Views(vs, base) => {
                let view = vs[p as usize];
                if view.is_inlined() || *base == 0 {
                    view
                } else {
                    let r = view.as_view();
                    BinaryView::from(r.with_buffer_and_offset(r.buffer_index + base, r.offset))
                }
            }
            Src::Const(view) => *view,
        });
    }
    let buffers: Arc<[ByteBuffer]> = buffers.into();
    // SAFETY: every view is copied from a valid source view with its buffer index rebased onto
    // the concatenated buffer list, or built over bytes pushed into that list.
    Ok(
        unsafe { VarBinViewArray::new_unchecked(views.freeze(), buffers, dtype.clone(), validity) }
            .into_array(),
    )
}

/// Concatenates the sources and takes from them, for leaf types without a direct gather.
fn gather_fallback(
    dtype: &DType,
    resolved: Vec<Resolved>,
    rest: impl Iterator<Item = Leaf>,
    positions: &Positions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let target = dtype.as_nullable();
    let mut chunks = Vec::new();
    let mut constant = Vec::new();
    let resolved = resolved.into_iter().map(|r| match r {
        Resolved::Const(s) => Leaf::Const(s),
        Resolved::Primitive(a, _) => Leaf::Array(a.into_array()),
        Resolved::Bool(b, _) => Leaf::Array(BoolArray::new(b, Validity::NonNullable).into_array()),
        Resolved::View(a, _) => Leaf::Array(a.into_array()),
    });
    for leaf in resolved.chain(rest) {
        let (array, is_const) = match leaf {
            Leaf::Const(s) => (ConstantArray::new(s, 1).into_array(), true),
            Leaf::Array(a) => (a, false),
        };
        chunks.push(array.cast(target.clone())?);
        constant.push(is_const);
    }
    let mut bases = Vec::with_capacity(chunks.len());
    let mut next = 0u64;
    for chunk in &chunks {
        bases.push(next);
        next += chunk.len() as u64;
    }
    let indices: Buffer<u64> = positions
        .src
        .iter()
        .zip(&positions.pos)
        .map(|(&s, &p)| {
            let s = s as usize;
            bases[s] + if constant[s] { 0 } else { u64::from(p) }
        })
        .collect();
    let all = ChunkedArray::try_new(chunks, target)?
        .into_array()
        .execute::<Canonical>(ctx)?
        .into_array();
    all.take(PrimitiveArray::new(indices, Validity::NonNullable).into_array())?
        .cast(dtype.clone())?
        .execute::<Canonical>(ctx)
        .map(Canonical::into_array)
}

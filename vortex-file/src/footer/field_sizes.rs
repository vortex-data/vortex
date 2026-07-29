// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::Field;
use vortex_array::dtype::FieldPath;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutRef;
use vortex_layout::segments::SegmentId;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::footer::SegmentSpec;

/// Compressed sizes of a file's fields, keyed by field path.
///
/// Sizes are derived from the file's layout tree and segment map: walking the tree from the root,
/// each physical segment is attributed to the deepest [`FieldPath`] whose layout subtree
/// references it. A segment referenced by more than one field is attributed to their longest
/// common ancestor, so a segment is counted exactly once and the size of a field always equals
/// the sum of its child field sizes plus the bytes attributed directly to it (e.g. struct
/// validity, or auxiliary segments such as zone maps and dictionaries written above the field
/// split).
///
/// Fields nested inside non-struct types (e.g. structs within lists) are not split into separate
/// layout subtrees by the writer, so their bytes are attributed to the enclosing field.
///
/// File-level metadata, alignment padding, and the footer are not part of any segment and are
/// never attributed.
#[derive(Debug, Clone)]
pub struct CompressedFieldSizes {
    sizes: HashMap<FieldPath, u64>,
}

impl CompressedFieldSizes {
    pub(crate) fn try_new(root: &LayoutRef, segments: &[SegmentSpec]) -> VortexResult<Self> {
        // Every segment is attributed to the deepest field path referencing it, collapsing to the
        // longest common ancestor if multiple fields reference the same segment.
        let mut attribution = HashMap::<SegmentId, FieldPath>::default();
        // Seed every field path present in the layout tree so empty fields report a zero size.
        let mut sizes = HashMap::<FieldPath, u64>::default();
        sizes.insert(FieldPath::root(), 0);

        let mut stack = vec![(LayoutRef::clone(root), FieldPath::root())];
        while let Some((layout, path)) = stack.pop() {
            for segment_id in layout.segment_ids() {
                match attribution.entry(segment_id) {
                    Entry::Occupied(mut entry) => {
                        let ancestor = common_prefix(entry.get(), &path);
                        entry.insert(ancestor);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(path.clone());
                    }
                }
            }

            for (child, child_type) in layout.children()?.into_iter().zip(layout.child_types()) {
                let child_path = match child_type {
                    LayoutChildType::Field(name) => {
                        let child_path = path.clone().push(name);
                        sizes.entry(child_path.clone()).or_insert(0);
                        child_path
                    }
                    _ => path.clone(),
                };
                stack.push((child, child_path));
            }
        }

        for (segment_id, path) in attribution {
            let segment = segments.get(*segment_id as usize).ok_or_else(|| {
                vortex_err!(
                    "layout references missing segment {} (segment count: {})",
                    *segment_id,
                    segments.len()
                )
            })?;
            // A segment counts towards its attributed field and every enclosing field.
            for len in 0..=path.parts().len() {
                *sizes
                    .entry(FieldPath::from(path.parts()[..len].to_vec()))
                    .or_insert(0) += u64::from(segment.length);
            }
        }

        Ok(Self { sizes })
    }

    /// Returns the compressed size in bytes of the field at the given path, including all of its
    /// descendants, or `None` if the layout tree contains no such field.
    ///
    /// [`FieldPath::root`] returns the same value as [`total`][Self::total].
    pub fn get(&self, path: &FieldPath) -> Option<u64> {
        self.sizes.get(path).copied()
    }

    /// Returns the total compressed size in bytes of all segments referenced by the layout tree.
    pub fn total(&self) -> u64 {
        self.get(&FieldPath::root()).unwrap_or_default()
    }

    /// Iterates over all field paths in the layout tree and their compressed sizes.
    pub fn iter(&self) -> impl Iterator<Item = (&FieldPath, u64)> {
        self.sizes.iter().map(|(path, size)| (path, *size))
    }
}

fn common_prefix(left: &FieldPath, right: &FieldPath) -> FieldPath {
    left.parts()
        .iter()
        .zip(right.parts())
        .take_while(|(l, r)| l == r)
        .map(|(l, _)| l.clone())
        .collect::<Vec<Field>>()
        .into()
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::array_session;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::FieldPath;
    use vortex_array::field_path;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;

    use crate::WriteOptionsSessionExt;
    use crate::writer::WriteSummary;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        session
    });

    async fn write_nested() -> VortexResult<WriteSummary> {
        let inner = StructArray::from_fields(&[
            ("c", buffer![10i64, 20, 30, 40].into_array()),
            (
                "d",
                StructArray::from_fields(&[("e", buffer![1.0f64, 2.0, 3.0, 4.0].into_array())])?
                    .into_array(),
            ),
        ])?;
        let root = StructArray::from_fields(&[
            ("a", buffer![1u32, 2, 3, 4].into_array()),
            ("b", inner.into_array()),
        ])?;

        let mut buf = ByteBufferMut::empty();
        SESSION
            .write_options()
            .write(&mut buf, root.into_array().to_array_stream())
            .await
    }

    #[tokio::test]
    async fn nested_field_sizes() -> VortexResult<()> {
        let summary = write_nested().await?;
        let sizes = summary.footer().compressed_field_sizes()?;

        for path in [
            field_path!(a),
            field_path!(b),
            field_path!(b.c),
            field_path!(b.d),
            field_path!(b.d.e),
        ] {
            assert!(
                sizes.get(&path).is_some_and(|size| size > 0),
                "expected non-zero size for {path}, got {:?}",
                sizes.get(&path)
            );
        }
        assert_eq!(sizes.get(&field_path!(missing)), None);
        assert_eq!(sizes.get(&field_path!(b.d.e.missing)), None);

        // Attribution partitions segments: a field's size is the sum of its children plus its own
        // directly-attributed segments (none here, as the structs are non-nullable).
        let size = |path: FieldPath| sizes.get(&path).unwrap_or_default();
        assert_eq!(
            size(field_path!(b)),
            size(field_path!(b.c)) + size(field_path!(b.d))
        );
        assert_eq!(size(field_path!(b.d)), size(field_path!(b.d.e)));
        assert_eq!(sizes.total(), size(field_path!(a)) + size(field_path!(b)));

        // Every segment in the file is attributed exactly once.
        let segment_total: u64 = summary
            .footer()
            .segment_map()
            .iter()
            .map(|segment| u64::from(segment.length))
            .sum();
        assert_eq!(sizes.total(), segment_total);

        Ok(())
    }

    #[tokio::test]
    async fn column_sizes_in_schema_order() -> VortexResult<()> {
        let summary = write_nested().await?;
        let sizes = summary.footer().compressed_field_sizes()?;

        assert_eq!(
            summary.compressed_column_sizes()?,
            vec![
                sizes.get(&field_path!(a)).unwrap_or_default(),
                sizes.get(&field_path!(b)).unwrap_or_default(),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn non_struct_file_reports_single_column() -> VortexResult<()> {
        let mut buf = ByteBufferMut::empty();
        let summary = SESSION
            .write_options()
            .write(
                &mut buf,
                buffer![1i32, 2, 3, 4].into_array().to_array_stream(),
            )
            .await?;

        let sizes = summary.footer().compressed_field_sizes()?;
        assert!(sizes.total() > 0);
        assert_eq!(sizes.get(&FieldPath::root()), Some(sizes.total()));
        assert_eq!(summary.compressed_column_sizes()?, vec![sizes.total()]);
        Ok(())
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RunArray;
use arrow_array::cast::AsArray;
use arrow_array::types::BinaryViewType;
use arrow_array::types::Int32Type;
use arrow_array::types::StringViewType;
use arrow_array::types::UInt8Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_runend::RunEnd;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use super::*;
use crate::ArrowExport;
use crate::ArrowExportVTable;
use crate::ArrowSession;
use crate::ArrowSessionExt;
use crate::executor::byte_view::canonical_varbinview_to_arrow;

#[test]
fn typed_options_are_independent() {
    struct CustomOption(u32);
    let original = ArrowExportOptions::default().with(CompactBuffers(false));
    let changed = original
        .clone()
        .with(CompactBuffers(true))
        .with(CustomOption(42));
    assert_eq!(
        original.get::<CompactBuffers>(),
        Some(&CompactBuffers(false))
    );
    assert!(original.get::<CustomOption>().is_none());
    assert_eq!(changed.get::<CompactBuffers>(), Some(&CompactBuffers(true)));
    assert_eq!(changed.get::<CustomOption>().map(|value| value.0), Some(42));
    assert!(
        ArrowExportOptions::default()
            .get::<CompactBuffers>()
            .is_none()
    );
}

#[test]
fn get_or_default_falls_back_to_default() {
    assert_eq!(
        ArrowExportOptions::default().get_or_default::<CompactBuffers>(),
        CompactBuffers(true)
    );
    assert_eq!(
        ArrowExportOptions::default()
            .with(CompactBuffers(false))
            .get_or_default::<CompactBuffers>(),
        CompactBuffers(false)
    );
}

#[derive(Debug)]
struct Plugin;

struct PluginOption(Arc<AtomicU32>);

static TEST_OPTION_ID: CachedId = CachedId::new("test.options");

impl ArrowExportVTable for Plugin {
    fn arrow_ext_id(&self) -> Id {
        *TEST_OPTION_ID
    }

    fn vortex_id(&self) -> Id {
        *TEST_OPTION_ID
    }

    fn to_arrow_field(
        &self,
        _name: &str,
        _dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        Ok(None)
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        _target: &Field,
        options: &ArrowExportOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        options
            .get::<PluginOption>()
            .expect("external option")
            .0
            .store(42, Ordering::Relaxed);
        Ok(ArrowExport::Unsupported(array))
    }
}

#[test]
fn external_options_reach_nested_exporters() -> VortexResult<()> {
    let session = array_session();
    session.arrow().register_exporter(Arc::new(Plugin));
    let mut ctx = session.create_execution_ctx();
    let child = Field::new("item", DataType::Int32, false).with_metadata(
        [("ARROW:extension:name".to_owned(), "test.options".to_owned())]
            .into_iter()
            .collect(),
    );
    let field = Field::new("list", DataType::List(Arc::new(child)), false);
    let array = ListArray::try_new(
        buffer![1i32, 2, 3].into_array(),
        buffer![0i32, 3].into_array(),
        Validity::NonNullable,
    )?
    .into_array();
    let called = Arc::new(AtomicU32::new(0));
    let options = ArrowExportOptions::default().with(PluginOption(Arc::clone(&called)));
    let arrow = session
        .arrow()
        .exporter(&options)
        .execute_arrow(array, Some(&field), &mut ctx)?;
    assert_eq!(arrow.len(), 1);
    assert_eq!(called.load(Ordering::Relaxed), 42);
    Ok(())
}

/// Three rows taken from distant positions of one large view buffer, so compaction has to copy
/// rather than just trim it.
fn sparse_views(binary: bool, ctx: &mut ExecutionCtx) -> VortexResult<VarBinViewArray> {
    let value = "x".repeat(256);
    let values = (0..128).map(|i| (i != 1).then(|| format!("{i}{value}")));
    let array = if binary {
        VarBinViewArray::from_iter_nullable_bin(values)
    } else {
        VarBinViewArray::from_iter_nullable_str(values)
    };
    array
        .into_array()
        .take(buffer![0u32, 1, 12].into_array())?
        .execute::<VarBinViewArray>(ctx)
}

fn view_buffer_ptrs(array: &ArrowArrayRef) -> Vec<*const u8> {
    let buffers = match array.data_type() {
        DataType::BinaryView => array.as_binary_view().data_buffers(),
        DataType::Utf8View => array.as_string_view().data_buffers(),
        dt => unreachable!("not a view type: {dt}"),
    };
    buffers.iter().map(|buffer| buffer.as_ptr()).collect()
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Flat,
    List,
    ListView,
    FixedSizeList,
    RunEnd,
    Struct,
    ListOfStruct,
    Dictionary,
}

impl Shape {
    /// Wrap a three-row `leaf` into this shape, returning the array and its Arrow target type.
    fn wrap(
        self,
        leaf: ArrayRef,
        leaf_type: DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, DataType)> {
        let field: FieldRef = Arc::new(Field::new("item", leaf_type.clone(), true));
        Ok(match self {
            Shape::Flat => (leaf, leaf_type),
            Shape::List => (
                ListArray::try_new(leaf, buffer![0i32, 3].into_array(), Validity::NonNullable)?
                    .into_array(),
                DataType::List(field),
            ),
            Shape::ListView => (
                ListViewArray::try_new(
                    leaf,
                    buffer![0i32].into_array(),
                    buffer![3i32].into_array(),
                    Validity::NonNullable,
                )?
                .into_array(),
                DataType::ListView(field),
            ),
            Shape::FixedSizeList => (
                FixedSizeListArray::try_new(leaf, 3, Validity::NonNullable, 1)?.into_array(),
                DataType::FixedSizeList(field, 3),
            ),
            Shape::RunEnd => (
                RunEnd::try_new(buffer![1u32, 2, 3].into_array(), leaf, ctx)?.into_array(),
                DataType::RunEndEncoded(
                    Arc::new(Field::new("ends", DataType::Int32, false)),
                    field,
                ),
            ),
            Shape::Struct => (
                StructArray::try_new(
                    FieldNames::from(["item"]),
                    vec![leaf],
                    3,
                    Validity::NonNullable,
                )?
                .into_array(),
                DataType::Struct(vec![field].into()),
            ),
            Shape::ListOfStruct => {
                let (array, struct_type) = Shape::Struct.wrap(leaf, leaf_type, ctx)?;
                Shape::List.wrap(array, struct_type, ctx)?
            }
            Shape::Dictionary => (
                DictArray::try_new(buffer![0u8, 1, 2].into_array(), leaf)?.into_array(),
                DataType::Dictionary(Box::new(DataType::UInt8), Box::new(leaf_type)),
            ),
        })
    }

    /// Extract the leaf column from an exported array of this shape.
    fn leaf(self, arrow: ArrowArrayRef) -> ArrowArrayRef {
        match self {
            Shape::Flat => arrow,
            Shape::List => Arc::clone(arrow.as_list::<i32>().values()),
            Shape::ListView => Arc::clone(arrow.as_list_view::<i32>().values()),
            Shape::FixedSizeList => Arc::clone(arrow.as_fixed_size_list().values()),
            Shape::RunEnd => Arc::clone(
                arrow
                    .as_any()
                    .downcast_ref::<RunArray<Int32Type>>()
                    .expect("run array")
                    .values(),
            ),
            Shape::Struct => Arc::clone(arrow.as_struct().column(0)),
            Shape::ListOfStruct => Shape::Struct.leaf(Shape::List.leaf(arrow)),
            Shape::Dictionary => Arc::clone(arrow.as_dictionary::<UInt8Type>().values()),
        }
    }
}

#[rstest]
fn compaction_option_reaches_nested_views(
    #[values(
        Shape::Flat,
        Shape::List,
        Shape::ListView,
        Shape::FixedSizeList,
        Shape::RunEnd,
        Shape::Struct,
        Shape::ListOfStruct,
        Shape::Dictionary
    )]
    shape: Shape,
    #[values(false, true)] binary: bool,
    #[values(None, Some(false), Some(true))] compact: Option<bool>,
) -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let options = compact.map_or_else(ArrowExportOptions::default, |compact| {
        ArrowExportOptions::default().with(CompactBuffers(compact))
    });

    let view = sparse_views(binary, &mut ctx)?;
    let backing = view.data_buffers()[0].as_host().as_ptr();
    let expected = if binary {
        canonical_varbinview_to_arrow::<BinaryViewType>(&view, &mut ctx)?
    } else {
        canonical_varbinview_to_arrow::<StringViewType>(&view, &mut ctx)?
    };

    let (array, target) = shape.wrap(view.into_array(), expected.data_type().clone(), &mut ctx)?;
    let arrow = session.arrow().exporter(&options).execute_arrow(
        array,
        Some(&Field::new("", target, true)),
        &mut ctx,
    )?;

    let leaf = shape.leaf(arrow);
    assert_eq!(leaf.to_data(), expected.to_data());
    assert_eq!(
        view_buffer_ptrs(&leaf).contains(&backing),
        compact == Some(false)
    );
    Ok(())
}

#[rstest]
fn execute_arrow_compacts_by_default(#[values(false, true)] binary: bool) -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let view = sparse_views(binary, &mut ctx)?;
    let backing = view.data_buffers()[0].as_host().as_ptr();

    let arrow = session
        .arrow()
        .execute_arrow(view.into_array(), None, &mut ctx)?;

    assert!(!view_buffer_ptrs(&arrow).contains(&backing));
    Ok(())
}

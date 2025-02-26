use vortex_array::EncodingContext;

use crate::vtable::LayoutVTableRef;

pub type LayoutContext = EncodingContext<LayoutVTableRef>;
//
// impl Default for LayoutContext {
//     fn default() -> Self {
//         Self {
//             layout_refs: vec![
//                 LayoutVTableRef::new_ref(&ChunkedLayout),
//                 LayoutVTableRef::new_ref(&FlatLayout),
//                 LayoutVTableRef::new_ref(&StructLayout),
//                 LayoutVTableRef::new_ref(&StatsLayout),
//             ],
//         }
//     }
// }

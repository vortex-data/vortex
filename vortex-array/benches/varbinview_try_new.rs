// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1 << 12, 1 << 16];

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn garbage_view() -> BinaryView {
    BinaryView::new_ref(13, *b"AAAA", 0xDEAD_BEEF, 0xF000_0000)
}

fn inline_view() -> BinaryView {
    BinaryView::new_inlined(b"hello world!")
}

fn owned_nullable_views(len: usize) -> Buffer<BinaryView> {
    let garbage = garbage_view();
    let inline = inline_view();
    let views: Vec<BinaryView> = (0..len)
        .map(|i| if i % 2 == 0 { inline } else { garbage })
        .collect();
    Buffer::copy_from(views)
}

fn owned_valid_views(len: usize) -> Buffer<BinaryView> {
    let views: Vec<BinaryView> = vec![inline_view(); len];
    Buffer::copy_from(views)
}

fn nullable_validity(len: usize) -> Validity {
    let bits = BitBuffer::from_iter((0..len).map(|i| i % 2 == 0));
    Validity::from_bit_buffer(bits, Nullability::Nullable)
}

#[divan::bench(args = SIZES)]
fn nullable_exclusive(bencher: Bencher, len: usize) {
    let dtype = DType::Utf8(Nullability::Nullable);
    let buffers: Arc<[ByteBuffer]> = Arc::new([]);
    bencher
        .with_inputs(|| {
            (
                owned_nullable_views(len),
                Arc::clone(&buffers),
                dtype.clone(),
                nullable_validity(len),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(views, buffers, dtype, validity, mut ctx)| {
            VarBinViewArray::try_new(views, buffers, dtype, validity, &mut ctx)
                .vortex_expect("try_new must succeed")
        });
}

#[divan::bench(args = SIZES)]
fn all_valid_exclusive(bencher: Bencher, len: usize) {
    let dtype = DType::Utf8(Nullability::NonNullable);
    let buffers: Arc<[ByteBuffer]> = Arc::new([]);
    bencher
        .with_inputs(|| {
            (
                owned_valid_views(len),
                Arc::clone(&buffers),
                dtype.clone(),
                Validity::NonNullable,
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(views, buffers, dtype, validity, mut ctx)| {
            VarBinViewArray::try_new(views, buffers, dtype, validity, &mut ctx)
                .vortex_expect("try_new must succeed")
        });
}

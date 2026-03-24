// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WebAssembly entry points for the Vortex TUI browser.
//!
//! Provides functions callable from JavaScript to load a Vortex file from a byte array
//! and display the interactive browser in a web page using ratzilla.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use ratzilla::CanvasBackend;
use ratzilla::WebRenderer;
use ratzilla::ratatui::Terminal;
use vortex::array::MaskFuture;
use vortex::array::serde::ArrayParts;
use vortex::error::VortexExpect;
use vortex::expr::root;
use vortex::layout::layouts::flat::Flat;
use vortex::layout::segments::SegmentSource;
use vortex::session::VortexSession;
use wasm_bindgen::prelude::*;

use crate::browse::app::AppState;
use crate::browse::app::KeyMode;
use crate::browse::handle_normal_mode;
use crate::browse::handle_search_mode;
use crate::browse::input::InputEvent;
use crate::browse::ui::render_app;

/// Initialize the WASM module (sets up panic hook for better error messages).
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Spawn an async task to load the flat layout array data and cache it in AppState.
///
/// This avoids calling `block_on()` in the render loop, which would deadlock in WASM
/// since the single-threaded event loop can't process spawned tasks while busy-waiting.
fn maybe_load_flat_data(app: &Rc<RefCell<AppState>>) {
    let borrowed = app.borrow();
    if !borrowed.cursor.layout().is::<Flat>() {
        return;
    }
    if borrowed.cached_flat_array.is_some() {
        return;
    }

    // Extract everything we need before dropping the borrow.
    let layout = borrowed.cursor.layout().clone();
    let segment_source = borrowed.vxf.segment_source();
    let session = borrowed.session.clone();
    let row_count = layout.row_count();
    drop(borrowed);

    let app = app.clone();
    wasm_bindgen_futures::spawn_local(async move {
        // Load the array data.
        let array = load_flat_array(&layout, &segment_source, &session, row_count).await;

        // Load the flatbuffer size.
        let fb_size = load_flatbuffer_size(&layout, &segment_source).await;

        // Store results — the borrow is safe because spawn_local runs between
        // animation frames, so the draw_web callback won't be holding it.
        let mut app = app.borrow_mut();
        app.cached_flat_array = Some(array);
        app.cached_flatbuffer_size = Some(fb_size);
    });
}

async fn load_flat_array(
    layout: &vortex::layout::LayoutRef,
    segment_source: &Arc<dyn SegmentSource>,
    session: &VortexSession,
    row_count: u64,
) -> vortex::array::ArrayRef {
    let reader = layout
        .new_reader("".into(), segment_source.clone(), session)
        .vortex_expect("Failed to create reader");
    reader
        .projection_evaluation(
            &(0..row_count),
            &root(),
            MaskFuture::new_true(
                usize::try_from(row_count).vortex_expect("row_count overflowed usize"),
            ),
        )
        .vortex_expect("Failed to construct projection")
        .await
        .vortex_expect("Failed to read flat array")
}

async fn load_flatbuffer_size(
    layout: &vortex::layout::LayoutRef,
    segment_source: &Arc<dyn SegmentSource>,
) -> usize {
    let segment_id = layout.as_::<Flat>().segment_id();
    let segment = segment_source
        .request(segment_id)
        .await
        .vortex_expect("Failed to read segment");
    ArrayParts::try_from(segment)
        .vortex_expect("Failed to parse segment")
        .metadata()
        .len()
}

/// Open a Vortex file from raw bytes and launch the interactive browser.
///
/// Call this from JavaScript after reading a `.vtx` file (e.g. via drag-and-drop or file input).
/// The browser UI will be rendered into the DOM.
#[wasm_bindgen]
pub fn open_vortex_file(data: &[u8]) -> Result<(), JsValue> {
    use vortex::VortexSessionDefault;
    use vortex::buffer::ByteBuffer;
    use vortex::io::runtime::wasm::WasmRuntime;
    use vortex::io::session::RuntimeSessionExt;

    let session = VortexSession::default().with_handle(WasmRuntime::handle());
    let buffer = ByteBuffer::from(data.to_vec());
    let app = Rc::new(RefCell::new(
        AppState::from_buffer(session, buffer).map_err(|e| JsValue::from_str(&e.to_string()))?,
    ));

    // Size the canvas to the CSS viewport. The JS side then scales canvas.width by
    // devicePixelRatio for crisp high-DPI rendering (ctx.scale(dpr)) without changing
    // the cell count. Browser zoom changes the viewport, so zooming out gives more
    // cols/rows and the resize handler recreates the terminal.
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let vw = window
        .inner_width()
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(1200.0);
    let vh = window
        .inner_height()
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(800.0);
    let backend = CanvasBackend::new_with_size(vw as u32, vh as u32)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let terminal = Terminal::new(backend).map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Register our own keydown listener so we can call preventDefault() for keys
    // the browser would otherwise intercept (e.g. Tab switching focus, '/' opening
    // Firefox's quick-find).
    {
        let app_for_keys = app.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::KeyboardEvent| {
            event.prevent_default();

            {
                let mut app_mut = app_for_keys.borrow_mut();
                let input = InputEvent::from(event);
                match app_mut.key_mode {
                    KeyMode::Normal => {
                        handle_normal_mode(&mut app_mut, input);
                    }
                    KeyMode::Search => {
                        handle_search_mode(&mut app_mut, input);
                    }
                }
            }
            // After handling the key event, trigger async loading if we navigated
            // to a FlatLayout.
            maybe_load_flat_data(&app_for_keys);
        });
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("no document"))?;
        document
            .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())
            .map_err(|e| JsValue::from_str(&format!("addEventListener failed: {e:?}")))?;
        closure.forget();
    }

    terminal.draw_web(move |frame| {
        let mut app = app.borrow_mut();
        render_app(&mut app, frame);
    });

    Ok(())
}

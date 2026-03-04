// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WebAssembly entry points for the Vortex TUI browser.
//!
//! Provides functions callable from JavaScript to load a Vortex file from a byte array
//! and display the interactive browser in a web page using ratzilla.

use std::cell::RefCell;
use std::rc::Rc;

use ratzilla::CanvasBackend;
use ratzilla::WebRenderer;
use ratzilla::ratatui::Terminal;
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
    use vortex::session::VortexSession;

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

    terminal.on_key_event({
        let app = app.clone();
        move |key_event| {
            let mut app = app.borrow_mut();
            let input = InputEvent::from(key_event);
            match app.key_mode {
                KeyMode::Normal => {
                    handle_normal_mode(&mut app, input);
                }
                KeyMode::Search => {
                    handle_search_mode(&mut app, input);
                }
            }
        }
    });

    terminal.draw_web(move |frame| {
        let mut app = app.borrow_mut();
        render_app(&mut app, frame);
    });

    Ok(())
}

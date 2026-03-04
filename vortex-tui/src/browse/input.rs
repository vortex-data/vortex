// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Platform-agnostic key event abstraction for the TUI browser.

/// A platform-agnostic key event.
pub(crate) struct InputEvent {
    /// The key code.
    pub code: InputKeyCode,
    /// Whether Ctrl is held.
    pub ctrl: bool,
    /// Whether Alt is held.
    pub alt: bool,
    /// Whether Shift is held.
    pub shift: bool,
}

/// A platform-agnostic key code.
pub(crate) enum InputKeyCode {
    /// A character key.
    Char(char),
    /// Up arrow.
    Up,
    /// Down arrow.
    Down,
    /// Left arrow.
    Left,
    /// Right arrow.
    Right,
    /// Enter/Return.
    Enter,
    /// Escape.
    Esc,
    /// Tab.
    Tab,
    /// Page Up.
    PageUp,
    /// Page Down.
    PageDown,
    /// Home.
    Home,
    /// End.
    End,
    /// Backspace.
    Backspace,
    /// Delete.
    Delete,
    /// Any other unrecognized key.
    Other,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<crossterm::event::KeyEvent> for InputEvent {
    fn from(key: crossterm::event::KeyEvent) -> Self {
        use crossterm::event::KeyCode;
        use crossterm::event::KeyModifiers;

        let code = match key.code {
            KeyCode::Char(c) => InputKeyCode::Char(c),
            KeyCode::Up => InputKeyCode::Up,
            KeyCode::Down => InputKeyCode::Down,
            KeyCode::Left => InputKeyCode::Left,
            KeyCode::Right => InputKeyCode::Right,
            KeyCode::Enter => InputKeyCode::Enter,
            KeyCode::Esc => InputKeyCode::Esc,
            KeyCode::Tab => InputKeyCode::Tab,
            KeyCode::PageUp => InputKeyCode::PageUp,
            KeyCode::PageDown => InputKeyCode::PageDown,
            KeyCode::Home => InputKeyCode::Home,
            KeyCode::End => InputKeyCode::End,
            KeyCode::Backspace => InputKeyCode::Backspace,
            KeyCode::Delete => InputKeyCode::Delete,
            _ => InputKeyCode::Other,
        };

        InputEvent {
            code,
            ctrl: key.modifiers.contains(KeyModifiers::CONTROL),
            alt: key.modifiers.contains(KeyModifiers::ALT),
            shift: key.modifiers.contains(KeyModifiers::SHIFT),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl From<web_sys::KeyboardEvent> for InputEvent {
    fn from(event: web_sys::KeyboardEvent) -> Self {
        let key = event.key();
        let code = if key.len() == 1 {
            match key.chars().next() {
                Some(c) => InputKeyCode::Char(c),
                None => InputKeyCode::Other,
            }
        } else {
            match key.as_str() {
                "ArrowUp" => InputKeyCode::Up,
                "ArrowDown" => InputKeyCode::Down,
                "ArrowLeft" => InputKeyCode::Left,
                "ArrowRight" => InputKeyCode::Right,
                "Enter" => InputKeyCode::Enter,
                "Escape" => InputKeyCode::Esc,
                "Tab" => InputKeyCode::Tab,
                "PageUp" => InputKeyCode::PageUp,
                "PageDown" => InputKeyCode::PageDown,
                "Home" => InputKeyCode::Home,
                "End" => InputKeyCode::End,
                "Backspace" => InputKeyCode::Backspace,
                "Delete" => InputKeyCode::Delete,
                _ => InputKeyCode::Other,
            }
        };

        InputEvent {
            code,
            ctrl: event.ctrl_key(),
            alt: event.alt_key(),
            shift: event.shift_key(),
        }
    }
}

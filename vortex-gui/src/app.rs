// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};

use eframe::Frame;
use egui::{CentralPanel, Context};
use vortex::file::{VortexFile, VortexOpenOptions};

use crate::cursor::LayoutCursor;
use crate::views::start::start_view;

pub struct App {
    path: PathBuf,
    cursor: LayoutCursor,
    file: VortexFile,
    state: State,
}

#[derive(Default, Copy, Clone)]
pub enum ActiveView {
    #[default]
    Start,
}

#[derive(Default, Clone)]
pub struct State {
    active: ActiveView,
}

impl App {
    pub async fn for_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_owned();
        let file = VortexOpenOptions::file().open(&path).await?;

        let cursor = LayoutCursor::new(file.footer().clone(), file.segment_source());
        Ok(Self {
            path,
            cursor,
            file,
            state: State::default(),
        })
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &Context, frame: &mut Frame) {
        CentralPanel::default().show(ctx, move |ui| {
            match self.state.active {
                ActiveView::Start => {
                    start_view(ui, &self.path, &self.cursor, &self.file);
                }
            };
        });
    }
}

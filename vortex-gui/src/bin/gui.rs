// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::anyhow;
use rfd::FileDialog;
use vortex_gui::App;

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let file_picker = FileDialog::new();

    let file = file_picker.pick_file().unwrap_or_else(|| {
        std::process::exit(1);
    });

    let app = App::for_file(file).await?;

    eframe::run_native(
        "Vortex",
        eframe::NativeOptions::default(),
        Box::new(move |_cc| Ok(Box::new(app))),
    )
    .map_err(|_err| anyhow!("app execution error encountered"))?;

    Ok(())
}

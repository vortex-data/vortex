// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use egui::Ui;
use egui_extras::{Column, TableBuilder};
use futures::executor::block_on;
use vortex::Array;
use vortex::dtype::DType;
use vortex::expr::root;
use vortex::file::VortexFile;
use vortex::mask::Mask;
use vortex::scalar::StructScalar;

use crate::cursor::LayoutCursor;

fn data_preview(ui: &mut Ui, cursor: &LayoutCursor, file: &VortexFile) {
    let reader = cursor
        .layout()
        .new_reader("Preview".into(), file.segment_source())
        .unwrap();

    // TODO(aduffy): this is pretty dumb to be honest.
    let row_count = reader.row_count().as_exact().unwrap();
    // TODO(aduffy): spawn this, use a channel to wait for the update.
    let array = block_on(
        reader
            .projection_evaluation(&(0..row_count), &root())
            .expect("Failed to construct projection")
            .invoke(Mask::new_true(
                usize::try_from(row_count).expect("row_count overflowed usize"),
            )),
    )
    .expect("Failed to read flat array");

    let column_names = if let DType::Struct(fields, _) = array.dtype() {
        fields.names().iter().map(|n| n.to_string()).collect()
    } else {
        vec!["value".to_string()]
    };

    let column_names = ui.collapsing("Rows", |ui| {
        TableBuilder::new(ui)
            .striped(true)
            .columns(Column::auto().resizable(true), column_names.len() + 1)
            .header(10.0, |mut row| {
                // row ID column
                row.col(|mut col| {
                    col.label("Row ID");
                });

                /// the column data...
                for col_name in &column_names {
                    row.col(|mut col| {
                        col.label(col_name);
                    });
                }
            })
            .body(|mut body| {
                body.rows(10.0, array.len(), |mut row| {
                    let row_id = row.index();
                    // TODO(aduffy): make a new TableBuilder widget that is backed by a columnar
                    //  representation.
                    let scalar = array.scalar_at(row_id);
                    if let Ok(struct_scalar) = StructScalar::try_from(&scalar) {
                        for field in struct_scalar.fields().unwrap() {
                            row.col(|mut ui| {
                                ui.label(field.to_string());
                            });
                        }
                    }
                });
            });
    });
}

fn child_panel(ui: &mut Ui, layout: &LayoutCursor) {
    let child_count = layout.layout().nchildren();

    ui.collapsing(format!("Children: {child_count}"), |ui| {
        TableBuilder::new(ui)
            .striped(true)
            .column(Column::auto().resizable(true))
            .column(Column::remainder())
            .header(10.0, |mut tr| {
                tr.col(|mut ui| {
                    ui.label("Index");
                });
                tr.col(|mut ui| {
                    ui.label("Encoding");
                });
            })
            .body(|mut tb| {
                tb.rows(10.0, child_count, |mut row| {
                    // name
                    let index = row.index();
                    let encoding = layout.child(index).layout().encoding_id();

                    row.col(|mut ui| {
                        ui.label(format!("{index}"));
                    });

                    // layout type
                    row.col(|mut ui| {
                        ui.label(encoding.to_string());
                    });
                });
            });
    });
}

pub fn start_view(ui: &mut Ui, path: &Path, cursor: &LayoutCursor, file: &VortexFile) {
    ui.heading(format!("{}", path.display()));

    ui.vertical(|ui| {
        data_preview(ui, cursor, file);

        child_panel(ui, cursor);
    });
}

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
# "sphobjinv",
# ]
# ///
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# https://github.com/pola-rs/polars/issues/7027#issuecomment-2661198001

import sphobjinv as soi

inv = soi.Inventory()
inv.project = "Polars"
inv.version = ""

data_obj_strs = [
    soi.DataObjStr(
        name="polars.DataFrame",
        domain="py",
        role="class",
        priority="1",
        uri="reference/dataframe/index.html",
        dispname="DataFrame",
    ),
    soi.DataObjStr(
        name="polars.LazyFrame",
        domain="py",
        role="class",
        priority="1",
        uri="reference/lazyframe/index.html",
        dispname="LazyFrame",
    ),
    soi.DataObjStr(
        name="Expr",
        domain="py",
        role="class",
        priority="1",
        uri="expressions/index.html",
        dispname="Expr",
    ),
    soi.DataObjStr(
        name="Series",
        domain="py",
        role="class",
        priority="1",
        uri="series/index.html",
        dispname="Series",
    ),
]

for data_obj_str in data_obj_strs:
    inv.objects.append(data_obj_str)

text = inv.data_file()
ztext = soi.compress(text)

soi.writebytes("polars.objects.inv", ztext)

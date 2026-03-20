// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow schemas for TPC-H tables.
//!
//! Adapted from the SQL definitions in <https://github.com/dimitri/tpch-citus/blob/master/schema/tpch-schema.sql>

use std::sync::LazyLock;

use arrow_schema::Schema;

use crate::schema_from_ddl;

pub static NATION: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        n_nationkey BIGINT NOT NULL,
        n_name VARCHAR NOT NULL,
        n_regionkey BIGINT NOT NULL,
        n_comment VARCHAR NOT NULL,
    ",
    )
});

pub static REGION: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        r_regionkey BIGINT NOT NULL,
        r_name VARCHAR NOT NULL,
        r_comment VARCHAR NOT NULL,
    ",
    )
});

pub static PART: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        p_partkey BIGINT NOT NULL,
        p_name VARCHAR NOT NULL,
        p_mfgr VARCHAR NOT NULL,
        p_brand VARCHAR NOT NULL,
        p_type VARCHAR NOT NULL,
        p_size INTEGER NOT NULL,
        p_container VARCHAR NOT NULL,
        p_retailprice DECIMAL(15,2) NOT NULL,
        p_comment VARCHAR NOT NULL,
    ",
    )
});

pub static SUPPLIER: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        s_suppkey BIGINT NOT NULL,
        s_name VARCHAR NOT NULL,
        s_address VARCHAR NOT NULL,
        s_nationkey BIGINT NOT NULL,
        s_phone VARCHAR NOT NULL,
        s_acctbal DECIMAL(15,2) NOT NULL,
        s_comment VARCHAR NOT NULL,
    ",
    )
});

pub static PARTSUPP: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        ps_partkey BIGINT NOT NULL,
        ps_suppkey BIGINT NOT NULL,
        ps_availqty INTEGER NOT NULL,
        ps_supplycost DECIMAL(15,2) NOT NULL,
        ps_comment VARCHAR NOT NULL,
    ",
    )
});

pub static CUSTOMER: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        c_custkey BIGINT NOT NULL,
        c_name VARCHAR NOT NULL,
        c_address VARCHAR NOT NULL,
        c_nationkey BIGINT NOT NULL,
        c_phone VARCHAR NOT NULL,
        c_acctbal DECIMAL(15,2) NOT NULL,
        c_mktsegment VARCHAR NOT NULL,
        c_comment VARCHAR NOT NULL,
    ",
    )
});

pub static ORDERS: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        o_orderkey BIGINT NOT NULL,
        o_custkey BIGINT NOT NULL,
        o_orderstatus VARCHAR NOT NULL,
        o_totalprice DECIMAL(15,2) NOT NULL,
        o_orderdate DATE NOT NULL,
        o_orderpriority VARCHAR NOT NULL,
        o_clerk VARCHAR NOT NULL,
        o_shippriority INTEGER NOT NULL,
        o_comment VARCHAR NOT NULL,
    ",
    )
});

pub static LINEITEM: LazyLock<Schema> = LazyLock::new(|| {
    schema_from_ddl(
        "
        l_orderkey BIGINT NOT NULL,
        l_partkey BIGINT NOT NULL,
        l_suppkey BIGINT NOT NULL,
        l_linenumber INTEGER NOT NULL,
        l_quantity DECIMAL(15,2) NOT NULL,
        l_extendedprice DECIMAL(15,2) NOT NULL,
        l_discount DECIMAL(15,2) NOT NULL,
        l_tax DECIMAL(15,2) NOT NULL,
        l_returnflag VARCHAR NOT NULL,
        l_linestatus VARCHAR NOT NULL,
        l_shipdate DATE NOT NULL,
        l_commitdate DATE NOT NULL,
        l_receiptdate DATE NOT NULL,
        l_shipinstruct VARCHAR NOT NULL,
        l_shipmode VARCHAR NOT NULL,
        l_comment VARCHAR NOT NULL,
    ",
    )
});

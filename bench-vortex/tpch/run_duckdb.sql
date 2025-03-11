
LOAD 'build/debug/extension/duckdb_vortex_rs/duckdb_vortex_rs.duckdb_extension';

SET VARIABLE vortex_directory='PATH_TO_VORTEX';
SET VARIABLE data_directory=getvariable('vortex_directory') || '/bench-vortex/data/tpch/1/vortex_compressed/';

create view customer as
select *
from duckdb_vortex_rs( getvariable('data_directory') || 'customer.vortex');

create view lineitem as
select *
from duckdb_vortex_rs( getvariable('data_directory') || 'lineitem.vortex');

create view nation as
select *
from duckdb_vortex_rs( getvariable('data_directory') || 'nation.vortex');

create view orders as
select *
from duckdb_vortex_rs( getvariable('data_directory') || 'orders.vortex');

create view part as
select *
from duckdb_vortex_rs( getvariable('data_directory') || 'part.vortex');

create view partsupp as
select *
from duckdb_vortex_rs( getvariable('data_directory') || 'partsupp.vortex');

create view region as
select *
from duckdb_vortex_rs( getvariable('data_directory') || 'region.vortex');

create view supplier as
select *
from duckdb_vortex_rs(getvariable('data_directory') || 'supplier.vortex');


SET VARIABLE q1='/Users/joeisaacs/git/spiraldb/vortex/bench-vortex/tpch/q1.sql';
.set temp_query getvariable('q1')


-- Sadly the read command doesn't work with variable interpolation.
-- Replace the below <PATH_TO_VORTEX> with you path to the vortex repo
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q1.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q2.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q3.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q4.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q5.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q6.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q7.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q8.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q9.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q10.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q11.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q12.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q13.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q14.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q15.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q16.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q17.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q18.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q19.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q20.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q21.sql'
.read 'PATH_TO_VORTEX/vortex/bench-vortex/tpch/q22.sql'

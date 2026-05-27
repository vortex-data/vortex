WITH supplier_part_summary AS (
    SELECT 
        s.s_suppkey, 
        s.s_name, 
        p.p_partkey, 
        SUM(ps.ps_availqty) AS total_available_qty,
        AVG(ps.ps_supplycost) AS avg_supply_cost,
        SUM(l.l_quantity) AS total_ordered_qty
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    LEFT JOIN 
        lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY 
        s.s_suppkey, s.s_name, p.p_partkey
),
ranked_suppliers AS (
    SELECT 
        sps.s_suppkey, 
        sps.s_name, 
        sps.p_partkey, 
        sps.total_available_qty, 
        sps.avg_supply_cost, 
        sps.total_ordered_qty,
        RANK() OVER (PARTITION BY sps.p_partkey ORDER BY sps.total_ordered_qty DESC) AS rank_ordered_qty
    FROM 
        supplier_part_summary sps
)
SELECT 
    rs.s_suppkey, 
    rs.s_name, 
    p.p_name, 
    rs.total_available_qty, 
    rs.avg_supply_cost, 
    rs.total_ordered_qty
FROM 
    ranked_suppliers rs
JOIN 
    part p ON rs.p_partkey = p.p_partkey
WHERE 
    rs.rank_ordered_qty = 1
ORDER BY 
    rs.total_ordered_qty DESC, rs.s_suppkey;

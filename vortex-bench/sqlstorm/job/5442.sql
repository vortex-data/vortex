SELECT 
    a.name AS aka_name,
    t.title AS movie_title,
    t.production_year,
    c.role_id,
    rc.role AS role_name,
    cm.name AS company_name,
    it.info AS movie_info,
    k.keyword AS movie_keyword
FROM 
    aka_name a
JOIN 
    cast_info c ON a.person_id = c.person_id
JOIN 
    title t ON c.movie_id = t.id
JOIN 
    role_type rc ON c.role_id = rc.id
JOIN 
    movie_companies mc ON t.id = mc.movie_id
JOIN 
    company_name cm ON mc.company_id = cm.id
LEFT JOIN 
    movie_info mi ON t.id = mi.movie_id
LEFT JOIN 
    info_type it ON mi.info_type_id = it.id
LEFT JOIN 
    movie_keyword mk ON t.id = mk.movie_id
LEFT JOIN 
    keyword k ON mk.keyword_id = k.id
WHERE 
    t.production_year BETWEEN 2000 AND 2020
    AND a.name IS NOT NULL
    AND mc.company_type_id = (
        SELECT id FROM company_type WHERE kind = 'Distributor' LIMIT 1
    )
ORDER BY 
    t.production_year DESC, 
    a.name, 
    t.title;

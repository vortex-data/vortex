SELECT 
    a.name AS aka_name,
    t.title AS movie_title,
    c.note AS cast_note,
    c.nr_order AS cast_order,
    n.name AS person_name,
    rt.role AS role,
    m.info AS movie_info,
    k.keyword AS movie_keyword
FROM 
    aka_name a
JOIN 
    cast_info c ON a.person_id = c.person_id
JOIN 
    title t ON c.movie_id = t.id
JOIN 
    name n ON a.person_id = n.imdb_id
JOIN 
    role_type rt ON c.role_id = rt.id
JOIN 
    movie_info m ON t.id = m.movie_id
JOIN 
    movie_keyword mk ON t.id = mk.movie_id
JOIN 
    keyword k ON mk.keyword_id = k.id
WHERE 
    t.production_year = 2020
ORDER BY 
    t.title, c.nr_order;

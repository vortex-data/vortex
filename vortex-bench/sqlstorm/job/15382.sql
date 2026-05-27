SELECT 
    a.id AS aka_id, 
    a.name AS aka_name, 
    t.id AS title_id, 
    t.title AS movie_title, 
    c.person_role_id, 
    p.info AS person_info
FROM 
    aka_name a
JOIN 
    cast_info c ON a.person_id = c.person_id
JOIN 
    title t ON c.movie_id = t.id
JOIN 
    person_info p ON a.person_id = p.person_id
WHERE 
    t.production_year = 2023;

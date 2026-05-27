SELECT 
    t.title, 
    a.name AS actor_name, 
    r.role 
FROM 
    title t 
JOIN 
    cast_info c ON t.id = c.movie_id 
JOIN 
    aka_name a ON c.person_id = a.person_id 
JOIN 
    role_type r ON c.role_id = r.id 
WHERE 
    t.production_year = 2020 
ORDER BY 
    t.title;

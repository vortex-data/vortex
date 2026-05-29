SELECT 
    a.name AS actor_name,
    t.title AS movie_title,
    ct.kind AS company_type,
    ki.keyword AS movie_keyword,
    pi.info AS person_info,
    COUNT(DISTINCT c.id) AS cast_count
FROM 
    aka_name a
JOIN 
    cast_info c ON a.person_id = c.person_id
JOIN 
    aka_title t ON c.movie_id = t.id
JOIN 
    movie_companies mc ON t.id = mc.movie_id
JOIN 
    company_type ct ON mc.company_type_id = ct.id
JOIN 
    movie_keyword mk ON t.id = mk.movie_id
JOIN 
    keyword ki ON mk.keyword_id = ki.id
JOIN 
    person_info pi ON a.person_id = pi.person_id
WHERE 
    t.production_year BETWEEN 2000 AND 2020
    AND ct.kind LIKE 'Production%'
GROUP BY 
    a.name, t.title, ct.kind, ki.keyword, pi.info
ORDER BY 
    cast_count DESC, a.name;

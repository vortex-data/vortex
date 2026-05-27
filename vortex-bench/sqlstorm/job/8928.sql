SELECT
    a.name AS actor_name,
    t.title AS movie_title,
    t.production_year,
    ckt.kind AS cast_type,
    co.name AS company_name,
    mi.info AS movie_info,
    kv.keyword AS movie_keyword
FROM
    aka_name a
JOIN
    cast_info ci ON a.person_id = ci.person_id
JOIN
    title t ON ci.movie_id = t.id
JOIN
    complete_cast cc ON t.id = cc.movie_id
JOIN
    comp_cast_type ckt ON ci.person_role_id = ckt.id
JOIN
    movie_companies mc ON t.id = mc.movie_id
JOIN
    company_name co ON mc.company_id = co.id
JOIN
    movie_info mi ON t.id = mi.movie_id
JOIN
    movie_keyword mk ON t.id = mk.movie_id
JOIN
    keyword kv ON mk.keyword_id = kv.id
WHERE
    t.production_year BETWEEN 1990 AND 2020
    AND a.name ILIKE '%Smith%'
    AND ckt.kind = 'Actor'
ORDER BY
    t.production_year DESC, a.name;

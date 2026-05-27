WITH RankedPosts AS (
    SELECT 
        p.Id,
        p.Title,
        p.CreationDate,
        p.Score,
        p.ViewCount,
        u.DisplayName AS OwnerDisplayName,
        DENSE_RANK() OVER (PARTITION BY p.OwnerUserId ORDER BY p.Score DESC) AS PostRank
    FROM 
        Posts p
    JOIN 
        Users u ON p.OwnerUserId = u.Id
    WHERE 
        p.PostTypeId = 1 
        AND p.CreationDate >= '2022-01-01'
),
MaxRank AS (
    SELECT 
        OwnerDisplayName,
        MAX(PostRank) AS MaxPostRank
    FROM 
        RankedPosts
    GROUP BY 
        OwnerDisplayName
)
SELECT 
    rp.Title,
    rp.CreationDate,
    rp.Score,
    rp.ViewCount,
    rp.OwnerDisplayName
FROM 
    RankedPosts rp
JOIN 
    MaxRank mr ON rp.OwnerDisplayName = mr.OwnerDisplayName
WHERE 
    rp.PostRank = mr.MaxPostRank
ORDER BY 
    rp.Score DESC, 
    rp.ViewCount DESC;

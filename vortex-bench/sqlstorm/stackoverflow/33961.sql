
WITH RecentPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.CreationDate,
        p.Score,
        p.ViewCount,
        u.DisplayName AS OwnerDisplayName,
        COUNT(DISTINCT c.Id) AS CommentCount,
        ROW_NUMBER() OVER (PARTITION BY u.Id ORDER BY p.CreationDate DESC) AS UserPostRank
    FROM 
        Posts p
    JOIN 
        Users u ON p.OwnerUserId = u.Id
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    WHERE 
        p.CreationDate >= TIMESTAMP '2024-10-01 12:34:56' - INTERVAL '30 days' AND
        p.PostTypeId = 1 
    GROUP BY 
        p.Id, p.Title, p.CreationDate, p.Score, p.ViewCount, u.DisplayName, u.Id
),
TopUsers AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        SUM(p.Score) AS TotalScore,
        DENSE_RANK() OVER (ORDER BY SUM(p.Score) DESC) AS UserRank
    FROM 
        Users u
    JOIN 
        Posts p ON u.Id = p.OwnerUserId
    WHERE 
        p.PostTypeId = 1 
    GROUP BY 
        u.Id, u.DisplayName
    HAVING 
        COUNT(p.Id) > 5 
),
PostHistoryAggregated AS (
    SELECT 
        ph.PostId,
        STRING_AGG(ph.Comment, ', ') AS EditComments,
        MAX(ph.CreationDate) AS LastEditDate,
        MAX(ph.UserDisplayName) AS LastEditedBy
    FROM 
        PostHistory ph
    WHERE 
        ph.PostHistoryTypeId IN (4, 5, 6) 
    GROUP BY 
        ph.PostId
)
SELECT 
    rp.PostId,
    rp.Title,
    rp.CreationDate,
    rp.Score,
    rp.ViewCount,
    rp.OwnerDisplayName,
    rp.CommentCount,
    tu.TotalScore AS OwnerTotalScore,
    tu.UserRank AS OwnerRank,
    pha.EditComments,
    pha.LastEditDate,
    pha.LastEditedBy,
    CASE 
        WHEN rp.CommentCount >= 10 THEN 'Hot'
        WHEN rp.Score > 100 THEN 'Popular'
        ELSE 'Normal' 
    END AS PostStatus
FROM 
    RecentPosts rp
LEFT JOIN 
    TopUsers tu ON rp.OwnerDisplayName = tu.DisplayName
LEFT JOIN 
    PostHistoryAggregated pha ON rp.PostId = pha.PostId
WHERE 
    rp.UserPostRank = 1 
ORDER BY 
    rp.Score DESC
LIMIT 50;

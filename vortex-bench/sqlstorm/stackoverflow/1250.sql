
WITH RankedPosts AS (
    SELECT 
        p.Id, 
        p.Title, 
        p.CreationDate,
        p.Score,
        COUNT(c.Id) AS CommentCount,
        RANK() OVER (PARTITION BY p.PostTypeId ORDER BY p.Score DESC) AS ScoreRank
    FROM 
        Posts p
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    WHERE 
        p.CreationDate >= '2023-01-01' 
        AND p.Score IS NOT NULL
    GROUP BY 
        p.Id, p.Title, p.CreationDate, p.Score, p.PostTypeId
),

TopUsers AS (
    SELECT 
        u.Id AS UserId, 
        u.DisplayName, 
        SUM(v.BountyAmount) AS TotalBounty
    FROM 
        Users u
    JOIN 
        Votes v ON u.Id = v.UserId
    WHERE 
        v.VoteTypeId IN (8, 9)  
    GROUP BY 
        u.Id, u.DisplayName
    HAVING 
        SUM(v.BountyAmount) > 0
),

ClosedPosts AS (
    SELECT 
        ph.PostId, 
        STRING_AGG(DISTINCT ctr.Name, ', ') AS ClosedReasons
    FROM 
        PostHistory ph
    JOIN 
        CloseReasonTypes ctr ON CAST(ph.Comment AS INTEGER) = ctr.Id
    WHERE 
        ph.PostHistoryTypeId = 10  
    GROUP BY 
        ph.PostId
)

SELECT 
    rp.Title, 
    rp.Score, 
    rp.CommentCount,
    tu.DisplayName AS TopUser,
    tu.TotalBounty,
    cp.ClosedReasons
FROM 
    RankedPosts rp
LEFT JOIN 
    TopUsers tu ON rp.ScoreRank = 1 AND tu.TotalBounty IS NOT NULL
LEFT JOIN 
    ClosedPosts cp ON rp.Id = cp.PostId
WHERE 
    rp.Score > 0 
    AND COALESCE(cp.ClosedReasons, '') <> ''
ORDER BY 
    rp.Score DESC, rp.CreationDate DESC;

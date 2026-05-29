WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId, 
        p.Title, 
        p.CreationDate, 
        p.OwnerUserId,
        COUNT(c.Id) AS CommentCount,
        ROW_NUMBER() OVER (PARTITION BY p.OwnerUserId ORDER BY p.CreationDate DESC) AS RN
    FROM 
        Posts p
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    WHERE 
        p.CreationDate >= cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '1 year'
    GROUP BY 
        p.Id, p.Title, p.CreationDate, p.OwnerUserId
),
UserStatistics AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        SUM(CASE WHEN b.Class = 1 THEN 1 ELSE 0 END) AS GoldBadges,
        SUM(CASE WHEN b.Class = 2 THEN 1 ELSE 0 END) AS SilverBadges,
        SUM(CASE WHEN b.Class = 3 THEN 1 ELSE 0 END) AS BronzeBadges,
        COALESCE(SUM(v.BountyAmount), 0) AS TotalBounty
    FROM 
        Users u
    LEFT JOIN 
        Badges b ON u.Id = b.UserId
    LEFT JOIN 
        Votes v ON u.Id = v.UserId AND v.VoteTypeId = 9
    WHERE 
        u.Reputation > 1000
    GROUP BY 
        u.Id, u.DisplayName
),
ClosedPosts AS (
    SELECT 
        ph.PostId,
        ph.CreationDate,
        MAX(ph.CreationDate) AS LastClosedDate
    FROM 
        PostHistory ph
    WHERE 
        ph.PostHistoryTypeId = 10
    GROUP BY 
        ph.PostId, ph.CreationDate
)
SELECT 
    p.Title,
    p.CommentCount,
    u.DisplayName AS Owner,
    us.GoldBadges,
    us.SilverBadges,
    us.BronzeBadges,
    us.TotalBounty,
    cp.LastClosedDate,
    COALESCE(CASE WHEN cp.LastClosedDate IS NOT NULL THEN 'Closed' ELSE 'Open' END, 'Unknown') AS PostStatus
FROM 
    RankedPosts p
JOIN 
    Users u ON p.OwnerUserId = u.Id
JOIN 
    UserStatistics us ON u.Id = us.UserId
LEFT JOIN 
    ClosedPosts cp ON p.PostId = cp.PostId
WHERE 
    p.RN = 1
ORDER BY 
    p.CreationDate DESC
LIMIT 100;
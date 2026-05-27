WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.OwnerUserId,
        p.CreationDate,
        p.Score,
        ROW_NUMBER() OVER (PARTITION BY p.OwnerUserId ORDER BY p.Score DESC) AS RowNum,
        COUNT(*) OVER (PARTITION BY p.OwnerUserId) AS TotalPosts
    FROM Posts p
    WHERE p.CreationDate >= cast('2024-10-01' as date) - INTERVAL '1 year'
),
PostStatistics AS (
    SELECT 
        rp.OwnerUserId,
        COUNT(rp.PostId) AS PostCount,
        AVG(rp.Score) AS AvgScore,
        MAX(rp.CreationDate) AS LastPostDate
    FROM RankedPosts rp
    WHERE rp.RowNum = 1  
    GROUP BY rp.OwnerUserId
),
UserBadges AS (
    SELECT 
        u.Id AS UserId,
        STRING_AGG(b.Name, ', ') AS BadgeNames
    FROM Users u
    LEFT JOIN Badges b ON u.Id = b.UserId
    GROUP BY u.Id
),
PostHistorySummary AS (
    SELECT 
        ph.PostId,
        COUNT(ph.Id) AS EditCount,
        MAX(ph.CreationDate) AS LastEditTime
    FROM PostHistory ph
    WHERE ph.PostHistoryTypeId IN (4, 5)  
    GROUP BY ph.PostId
)
SELECT 
    u.DisplayName,
    u.Reputation,
    ps.PostCount,
    ps.AvgScore,
    ps.LastPostDate,
    ub.BadgeNames,
    COALESCE(phs.EditCount, 0) AS EditCount,
    phs.LastEditTime,
    CASE 
        WHEN ps.PostCount > 5 THEN 'Active User' 
        WHEN ps.LastPostDate >= cast('2024-10-01' as date) - INTERVAL '6 months' THEN 'Recent Contributor'
        ELSE 'Inactive User' 
    END AS UserStatus
FROM Users u
JOIN PostStatistics ps ON u.Id = ps.OwnerUserId
LEFT JOIN UserBadges ub ON u.Id = ub.UserId
LEFT JOIN PostHistorySummary phs ON ps.OwnerUserId = phs.PostId
WHERE u.Reputation > 1000
ORDER BY u.Reputation DESC, ps.AvgScore DESC
LIMIT 50;
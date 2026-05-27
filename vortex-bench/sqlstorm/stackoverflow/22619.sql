
WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.Tags,
        p.CreationDate,
        p.Score,
        ROW_NUMBER() OVER (PARTITION BY p.OwnerUserId ORDER BY p.CreationDate DESC) AS UserPostRank,
        p.OwnerUserId
    FROM 
        Posts p
    WHERE 
        p.CreationDate >= DATE_TRUNC('month', CAST('2024-10-01' AS DATE)) 
),
UserBadges AS (
    SELECT 
        u.Id AS UserId,
        COUNT(b.Id) AS BadgeCount,
        SUM(CASE WHEN b.Class = 1 THEN 1 ELSE 0 END) AS GoldBadges,
        SUM(CASE WHEN b.Class = 2 THEN 1 ELSE 0 END) AS SilverBadges,
        SUM(CASE WHEN b.Class = 3 THEN 1 ELSE 0 END) AS BronzeBadges
    FROM 
        Users u
    LEFT JOIN 
        Badges b ON u.Id = b.UserId
    GROUP BY 
        u.Id
),
PostActivity AS (
    SELECT 
        ph.PostId,
        ph.CreationDate,
        ph.PostHistoryTypeId,
        CASE 
            WHEN ph.PostHistoryTypeId IN (10, 11) THEN 'Closure Activity'
            WHEN ph.PostHistoryTypeId IN (6, 4) THEN 'Tag or Title Activity'
            ELSE 'Other Activity'
        END AS ActivityType,
        COUNT(DISTINCT ph.UserId) AS UniqueUserCount
    FROM 
        PostHistory ph
    GROUP BY 
        ph.PostId, ph.CreationDate, ph.PostHistoryTypeId
)
SELECT 
    rp.PostId,
    rp.Title,
    rp.Tags,
    ub.BadgeCount,
    ub.GoldBadges,
    ub.SilverBadges,
    ub.BronzeBadges,
    pa.ActivityType,
    pa.UniqueUserCount,
    CASE 
        WHEN pa.UniqueUserCount > 5 THEN 'Highly Active'
        WHEN pa.UniqueUserCount BETWEEN 3 AND 5 THEN 'Moderately Active'
        ELSE 'Low Activity'
    END AS ActivityLevel
FROM 
    RankedPosts rp
LEFT JOIN 
    UserBadges ub ON rp.OwnerUserId = ub.UserId
LEFT JOIN 
    PostActivity pa ON rp.PostId = pa.PostId
WHERE 
    rp.UserPostRank <= 3 
    AND (pa.ActivityType IS NOT NULL OR ub.BadgeCount > 0) 
ORDER BY 
    rp.Score DESC, rp.CreationDate ASC;

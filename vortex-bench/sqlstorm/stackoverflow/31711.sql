
WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId, 
        p.Title, 
        p.PostTypeId, 
        p.OwnerUserId, 
        p.CreationDate,
        RANK() OVER (PARTITION BY p.OwnerUserId ORDER BY p.Score DESC) AS Rank,
        COUNT(DISTINCT c.Id) AS CommentCount,
        p.Score
    FROM 
        Posts p
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    WHERE 
        p.CreationDate >= CURRENT_DATE - INTERVAL '1 year' 
    GROUP BY 
        p.Id, p.Title, p.PostTypeId, p.OwnerUserId, p.CreationDate, p.Score
),
UserBadges AS (
    SELECT 
        b.UserId,
        COUNT(CASE WHEN b.Class = 1 THEN 1 END) AS GoldBadges,
        COUNT(CASE WHEN b.Class = 2 THEN 1 END) AS SilverBadges,
        COUNT(CASE WHEN b.Class = 3 THEN 1 END) AS BronzeBadges
    FROM 
        Badges b
    GROUP BY 
        b.UserId
),
TopPosts AS (
    SELECT 
        rp.PostId, 
        rp.Title, 
        rp.OwnerUserId,
        ub.GoldBadges,
        ub.SilverBadges,
        ub.BronzeBadges,
        rp.CommentCount,
        ROW_NUMBER() OVER (ORDER BY rp.Score DESC) AS TopRank
    FROM 
        RankedPosts rp
    JOIN 
        UserBadges ub ON rp.OwnerUserId = ub.UserId
    WHERE 
        rp.Rank = 1 
)
SELECT 
    p.Title, 
    u.DisplayName, 
    u.Reputation,
    COALESCE(tp.GoldBadges, 0) AS GoldBadges,
    COALESCE(tp.SilverBadges, 0) AS SilverBadges,
    COALESCE(tp.BronzeBadges, 0) AS BronzeBadges,
    tp.CommentCount
FROM 
    TopPosts tp 
JOIN 
    Users u ON tp.OwnerUserId = u.Id
JOIN 
    Posts p ON tp.PostId = p.Id
WHERE 
    u.Reputation > 500 
ORDER BY 
    tp.CommentCount DESC, 
    u.Reputation DESC;

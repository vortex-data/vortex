
WITH UserActivity AS (
    SELECT 
        U.Id AS UserId,
        U.DisplayName,
        COALESCE(SUM(CASE WHEN V.VoteTypeId = 2 THEN 1 ELSE 0 END), 0) AS UpVotes,
        COALESCE(SUM(CASE WHEN V.VoteTypeId = 3 THEN 1 ELSE 0 END), 0) AS DownVotes,
        COUNT(DISTINCT P.Id) AS PostCount,
        COUNT(DISTINCT C.Id) AS CommentCount,
        COUNT(DISTINCT B.Id) AS BadgeCount
    FROM Users U
    LEFT JOIN Posts P ON U.Id = P.OwnerUserId
    LEFT JOIN Comments C ON P.Id = C.PostId
    LEFT JOIN Votes V ON P.Id = V.PostId AND V.UserId = U.Id
    LEFT JOIN Badges B ON U.Id = B.UserId
    WHERE U.CreationDate >= '2023-01-01'
    GROUP BY U.Id, U.DisplayName
),
PostStatistics AS (
    SELECT 
        P.Id AS PostId,
        P.Title,
        P.CreationDate,
        P.ViewCount,
        P.Score,
        P.Tags,
        COUNT(DISTINCT C.Id) AS CommentCount,
        COUNT(DISTINCT V.Id) AS VoteCount
    FROM Posts P
    LEFT JOIN Comments C ON P.Id = C.PostId
    LEFT JOIN Votes V ON P.Id = V.PostId
    GROUP BY P.Id, P.Title, P.CreationDate, P.ViewCount, P.Score, P.Tags
),
TopPosts AS (
    SELECT 
        PS.PostId,
        PS.Title,
        PS.CreationDate,
        PS.ViewCount,
        PS.Score,
        PS.Tags,
        PS.CommentCount,
        PS.VoteCount,
        ROW_NUMBER() OVER (ORDER BY PS.Score DESC, PS.ViewCount DESC) AS Rank
    FROM PostStatistics PS
)
SELECT 
    UA.DisplayName,
    UA.UpVotes,
    UA.DownVotes,
    UA.PostCount,
    UA.CommentCount,
    UA.BadgeCount,
    TP.Title AS TopPostTitle,
    TP.ViewCount AS TopPostViewCount,
    TP.Score AS TopPostScore,
    TP.Rank
FROM UserActivity UA
LEFT JOIN TopPosts TP ON UA.UserId = TP.PostId
WHERE TP.Rank <= 10
ORDER BY UA.BadgeCount DESC, UA.UpVotes DESC;

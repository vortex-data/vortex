
WITH UserReputation AS (
    SELECT Id AS UserId, Reputation, LastAccessDate,
           ROW_NUMBER() OVER (ORDER BY Reputation DESC) AS Rank
    FROM Users
    WHERE Reputation IS NOT NULL
),
PostStatistics AS (
    SELECT P.Id AS PostId, P.OwnerUserId, P.PostTypeId,
           COUNT(CASE WHEN C.Id IS NOT NULL THEN 1 END) AS CommentCount,
           SUM(CASE WHEN V.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpVotes,
           SUM(CASE WHEN V.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownVotes,
           AVG(LENGTH(P.Body)) AS AvgBodyLength
    FROM Posts P
    LEFT JOIN Comments C ON P.Id = C.PostId
    LEFT JOIN Votes V ON P.Id = V.PostId
    GROUP BY P.Id, P.OwnerUserId, P.PostTypeId
),
TopPosts AS (
    SELECT PS.PostId, PS.OwnerUserId, PS.CommentCount, PS.UpVotes, PS.DownVotes, 
           PS.AvgBodyLength, 
           COALESCE(UR.Reputation, 0) AS OwnerReputation,
           CASE 
               WHEN PS.CommentCount > 10 THEN 'Highly Discussed'
               WHEN PS.CommentCount BETWEEN 5 AND 10 THEN 'Moderately Discussed'
               ELSE 'Less Discussed'
           END AS DiscussionCategory
    FROM PostStatistics PS
    LEFT JOIN UserReputation UR ON PS.OwnerUserId = UR.UserId
),
AggregateResults AS (
    SELECT DiscussionCategory, 
           COUNT(*) AS TotalPosts, 
           AVG(OwnerReputation) AS AvgOwnerReputation, 
           SUM(UpVotes) AS TotalUpVotes, 
           SUM(DownVotes) AS TotalDownVotes
    FROM TopPosts
    GROUP BY DiscussionCategory
)
SELECT AR.DiscussionCategory, AR.TotalPosts, 
       AR.AvgOwnerReputation, AR.TotalUpVotes, AR.TotalDownVotes,
       CASE 
           WHEN AR.TotalPosts > 50 THEN 'Very Active'
           WHEN AR.TotalPosts BETWEEN 20 AND 50 THEN 'Active'
           ELSE 'Less Active'
       END AS ActivityLevel
FROM AggregateResults AR
ORDER BY AR.TotalPosts DESC;

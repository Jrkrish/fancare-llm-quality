-- The same tool call logged twice for one trace.
SELECT 'tool_calls' AS tbl, trace_id || '#' || seq AS row_id, 'duplicate (trace_id, seq)' AS problem
FROM tool_calls GROUP BY trace_id, seq HAVING count(*) > 1
UNION ALL
-- The same request logged twice (double write): same session, timestamp and message.
SELECT 'traces', min(trace_id), 'duplicate trace (session, created_at, message)'
FROM traces GROUP BY session_id, created_at, user_message HAVING count(*) > 1
UNION ALL
-- The same guardrail verdict recorded twice for one trace.
SELECT 'guardrail_events', trace_id, 'duplicate guardrail event'
FROM guardrail_events GROUP BY trace_id, stage, action, category, detail HAVING count(*) > 1;

-- Values outside their allowed domain.
SELECT 'traces' AS tbl, trace_id AS row_id, 'invalid status: ' || status AS problem
FROM traces WHERE status NOT IN ('ok', 'blocked', 'error')
UNION ALL
SELECT 'traces', trace_id, 'invalid input_verdict: ' || input_verdict
FROM traces WHERE input_verdict NOT IN ('allow', 'redact', 'block')
UNION ALL
SELECT 'traces', trace_id, 'invalid output_verdict: ' || output_verdict
FROM traces WHERE output_verdict IS NOT NULL AND output_verdict NOT IN ('allow', 'redact', 'block')
UNION ALL
SELECT 'traces', trace_id, 'blocked input but status is not blocked'
FROM traces WHERE input_verdict = 'block' AND status != 'blocked'
UNION ALL
SELECT 'traces', trace_id, 'negative latency, tokens or cost'
FROM traces WHERE latency_ms < 0 OR prompt_tokens < 0 OR completion_tokens < 0 OR cost_usd < 0
UNION ALL
SELECT 'traces', trace_id, 'created_at is not an ISO timestamp'
FROM traces WHERE datetime(created_at) IS NULL
UNION ALL
SELECT 'traces', trace_id, 'retrieved/cited chunk ids are not a JSON array'
FROM traces WHERE json_valid(retrieved_chunk_ids) = 0 OR json_type(retrieved_chunk_ids) != 'array'
   OR json_valid(cited_chunk_ids) = 0 OR json_type(cited_chunk_ids) != 'array'
UNION ALL
SELECT 'traces', trace_id, 'unknown prompt_version: ' || prompt_version
FROM traces WHERE prompt_version NOT IN ('v1', 'v2', 'v3')
UNION ALL
SELECT 'tool_calls', CAST(id AS TEXT), 'invalid tool status: ' || status
FROM tool_calls WHERE status NOT IN ('ok', 'invalid_arguments', 'unknown_tool', 'timeout', 'malformed', 'empty', 'error')
UNION ALL
SELECT 'tool_calls', CAST(id AS TEXT), 'arguments are not a JSON object'
FROM tool_calls WHERE json_valid(arguments) = 0 OR json_type(arguments) != 'object'
UNION ALL
SELECT 'guardrail_events', CAST(id AS TEXT), 'invalid stage/action'
FROM guardrail_events WHERE stage NOT IN ('input', 'retrieval', 'output') OR action NOT IN ('allow', 'redact', 'block');

-- Rows whose parent is missing (foreign keys are declared but not enforced by SQLite).
SELECT 'tool_calls' AS tbl, CAST(c.id AS TEXT) AS row_id, 'tool call without a trace' AS problem
FROM tool_calls c LEFT JOIN traces t ON t.trace_id = c.trace_id WHERE t.trace_id IS NULL
UNION ALL
SELECT 'guardrail_events', CAST(g.id AS TEXT), 'guardrail event without a trace'
FROM guardrail_events g LEFT JOIN traces t ON t.trace_id = g.trace_id WHERE t.trace_id IS NULL
UNION ALL
SELECT 'traces', t.trace_id, 'trace without a session'
FROM traces t LEFT JOIN sessions s ON s.session_id = t.session_id WHERE s.session_id IS NULL
UNION ALL
SELECT 'traces', t.trace_id, 'trace without any guardrail verdict'
FROM traces t WHERE NOT EXISTS (SELECT 1 FROM guardrail_events g WHERE g.trace_id = t.trace_id)
UNION ALL
SELECT 'sessions', s.session_id, 'session without any trace'
FROM sessions s WHERE NOT EXISTS (SELECT 1 FROM traces t WHERE t.session_id = s.session_id);

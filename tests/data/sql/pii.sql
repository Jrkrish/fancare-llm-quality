-- Personal data that reached storage. has_pii() is registered from app.guardrails.redact_pii,
-- so the check uses exactly the same patterns as the live redaction.
SELECT 'traces' AS tbl, trace_id AS row_id, 'PII in user_message' AS problem FROM traces WHERE has_pii(user_message)
UNION ALL
SELECT 'traces', trace_id, 'PII in answer' FROM traces WHERE has_pii(answer)
UNION ALL
SELECT 'traces', trace_id, 'PII in error text' FROM traces WHERE has_pii(error)
UNION ALL
SELECT 'tool_calls', CAST(id AS TEXT), 'PII in tool arguments' FROM tool_calls WHERE has_pii(arguments)
UNION ALL
SELECT 'guardrail_events', CAST(id AS TEXT), 'PII in guardrail detail' FROM guardrail_events WHERE has_pii(detail);

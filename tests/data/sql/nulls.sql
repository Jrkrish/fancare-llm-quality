-- Required values that are missing.
SELECT 'traces' AS tbl, trace_id AS row_id, 'required column is NULL' AS problem
FROM traces
WHERE session_id IS NULL OR created_at IS NULL OR status IS NULL OR prompt_version IS NULL
   OR retriever_mode IS NULL OR user_message IS NULL OR latency_ms IS NULL OR input_verdict IS NULL
UNION ALL
SELECT 'traces', trace_id, 'successful trace has no answer'
FROM traces WHERE status = 'ok' AND (answer IS NULL OR trim(answer) = '')
UNION ALL
SELECT 'traces', trace_id, 'successful trace has no model recorded'
FROM traces WHERE status = 'ok' AND (model IS NULL OR provider IS NULL)
UNION ALL
SELECT 'traces', trace_id, 'error trace has no error message'
FROM traces WHERE status = 'error' AND (error IS NULL OR error = '')
UNION ALL
SELECT 'tool_calls', CAST(id AS TEXT), 'required column is NULL'
FROM tool_calls WHERE trace_id IS NULL OR name IS NULL OR arguments IS NULL OR status IS NULL;

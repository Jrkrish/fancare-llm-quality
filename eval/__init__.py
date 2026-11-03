import os

# DeepEval sends anonymous usage telemetry by default; keep eval runs local.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

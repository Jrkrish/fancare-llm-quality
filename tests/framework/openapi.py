"""Validates live responses against the app's own OpenAPI document."""
from jsonschema import Draft202012Validator


class OpenApiValidator:
    def __init__(self, spec: dict):
        self.spec = spec

    def response_schema(self, path: str, method: str, status: int) -> dict:
        op = self.spec["paths"][path][method.lower()]
        responses = op["responses"]
        if str(status) not in responses:
            raise AssertionError(f"{method.upper()} {path} does not document status {status} (documents {sorted(responses)})")
        content = responses[str(status)].get("content", {})
        return content.get("application/json", {}).get("schema", {})

    def validate(self, path: str, method: str, response) -> None:
        """Raise AssertionError if the status is undocumented or the JSON body breaks the schema."""
        schema = self.response_schema(path, method, response.status_code)
        if not schema:
            return
        root = {**schema, "components": self.spec.get("components", {})}
        errors = sorted(Draft202012Validator(root).iter_errors(response.json()), key=lambda e: list(e.path))
        if errors:
            details = "; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5])
            raise AssertionError(f"{method.upper()} {path} {response.status_code} violates OpenAPI schema: {details}")

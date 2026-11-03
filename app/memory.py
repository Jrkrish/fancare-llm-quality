"""Per-session conversation memory (in-process)."""
import threading
from collections import defaultdict


class SessionMemory:
    """Keeps the last N user/assistant turns per session_id, plus session notes.

    Only the (already PII-redacted) user text and final assistant answer are
    stored; tool-call messages are not replayed across turns. Notes hold
    validated facts from uploaded documents.
    """

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self._turns: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._notes: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def history(self, session_id: str) -> list[dict]:
        messages = []
        for user, assistant in self.turns(session_id):
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        return messages

    def turns(self, session_id: str) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._turns.get(session_id, []))

    def append(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            turns = self._turns[session_id]
            turns.append((user, assistant))
            del turns[: -self.max_turns]

    def add_note(self, session_id: str, note: str) -> None:
        with self._lock:
            self._notes[session_id].append(note)

    def notes(self, session_id: str) -> list[str]:
        with self._lock:
            return list(self._notes.get(session_id, []))

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._turns.clear()
                self._notes.clear()
            else:
                self._turns.pop(session_id, None)
                self._notes.pop(session_id, None)

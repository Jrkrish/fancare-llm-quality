"""Locust load test for /chat. Keep concurrency low: free LLM tiers allow ~30 requests/minute.

    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 2 -r 1 -t 3m --csv reports/load/run --html reports/load/report.html
    python -m tests.load.summarize reports/load/run_stats.csv --since <ISO start time printed by locust>

Each simulated user holds a 3-turn conversation in its own session, then starts
a new one, so "conversations" are countable for the cost-per-100 estimate.
"""
import os
import random
import uuid

from locust import HttpUser, between, task

API_KEY = os.getenv("APP_API_KEY", "dev-local-key")
CONVERSATIONS = [
    ["My fan blinks red four times.", "What should I try first?", "Is the receiver covered by warranty?"],
    ["Where is my order ORD-10001?", "Which carrier?", "Can I still cancel it?"],
    ["What size fan for a 200 sq ft room?", "Which one is quiet?", "Can it go outdoors?"],
    ["How do I pair the remote?", "And the app over Bluetooth?", "How many phones can pair?"],
]


class ChatUser(HttpUser):
    # Think time between messages, like a real user reading the answer.
    wait_time = between(5, 10)

    def on_start(self):
        self.new_conversation()

    def new_conversation(self):
        self.session_id = f"load-{uuid.uuid4().hex[:10]}"
        self.script = list(random.choice(CONVERSATIONS))

    @task
    def chat_turn(self):
        if not self.script:
            self.new_conversation()
        message = self.script.pop(0)
        with self.client.post("/chat", json={"session_id": self.session_id, "message": message},
                              headers={"X-API-Key": API_KEY}, name="/chat", catch_response=True) as resp:
            if resp.status_code == 200 and resp.json().get("answer"):
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

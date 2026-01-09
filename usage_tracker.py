from datetime import datetime

# ---- Model pricing (USD per 1K tokens) ----
MODEL_PRICING = {
    "gpt-4.1-mini": {
        "input": 0.0003,
        "output": 0.0006
    },
    "gpt-4.1-nano": {
        "input": 0.00005,
        "output": 0.0001
    }
}

class UsageTracker:
    def __init__(self):
        self.calls = []
        self.started_at = datetime.utcnow()

    def record(self, response, model: str, stage: str):
        """
        Records the token usage from an OpenAI response object.
        """
        usage = getattr(response, "usage", None)
        if not usage:
            return

        # FIXED: OpenAI uses 'prompt_tokens' and 'completion_tokens'
        # We map them to 'input_tokens' and 'output_tokens' for our internal consistency
        self.calls.append({
            "stage": stage,
            "model": model,
            "input_tokens": usage.prompt_tokens,      # Was usage.input_tokens (Error source)
            "output_tokens": usage.completion_tokens, # Was usage.output_tokens
            "total_tokens": usage.total_tokens
        })

    def calculate_cost(self, call):
        pricing = MODEL_PRICING.get(call["model"])
        if not pricing:
            return 0.0
        
        input_cost = (call["input_tokens"] / 1000) * pricing["input"]
        output_cost = (call["output_tokens"] / 1000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)

    def summary(self):
        total_input = sum(c["input_tokens"] for c in self.calls)
        total_output = sum(c["output_tokens"] for c in self.calls)
        total_tokens = sum(c["total_tokens"] for c in self.calls)
        total_cost = sum(self.calculate_cost(c) for c in self.calls)

        return {
            "started_at": self.started_at.isoformat(),
            "total_calls": len(self.calls),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "calls": [
                {**c, "cost_usd": self.calculate_cost(c)}
                for c in self.calls
            ]
        }

    def print_detailed_report(self):
        summary = self.summary()
        print("\n========== OpenAI API Usage Report ==========")
        print(f"Execution started at (UTC): {summary['started_at']}")
        print(f"Total API calls: {summary['total_calls']}")
        print(f"Total input tokens: {summary['total_input_tokens']}")
        print(f"Total output tokens: {summary['total_output_tokens']}")
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"Total estimated cost (USD): ${summary['total_cost_usd']}")
        
        print("\n--- Per Call Breakdown ---")
        for i, c in enumerate(summary["calls"], start=1):
            print(
                f"{i}. Stage: {c['stage']} | "
                f"Model: {c['model']} | "
                f"In: {c['input_tokens']} | "
                f"Out: {c['output_tokens']} | "
                f"Total: {c['total_tokens']} | "
                f"Cost: ${c['cost_usd']}"
            )
        print("============================================\n")

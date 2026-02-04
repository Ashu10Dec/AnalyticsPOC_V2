from datetime import datetime

# ---- Model pricing (USD per 1M tokens) ----
# Pricing updated as of February 2026

MODEL_PRICING = {
    # === OpenAI Models ===
    
    # GPT-5 Series
    "gpt-5.2": {
        "input": 3.50,
        "output": 28.00
    },
    "gpt-5.2-pro": {
        "input": 21.00,
        "output": 168.00
    },
    "gpt-5.1": {
        "input": 1.25,
        "output": 10.00
    },
    "gpt-5-mini": {
        "input": 0.25,
        "output": 2.00
    },
    "gpt-5-nano": {
        "input": 0.05,
        "output": 0.40
    },
    
    # GPT-4.1 Series
    "gpt-4.1": {
        "input": 3.00,
        "output": 12.00
    },
    "gpt-4.1-mini": {
        "input": 0.80,
        "output": 3.20
    },
    "gpt-4.1-nano": {
        "input": 0.20,
        "output": 0.80
    },
    
    # GPT-4o Series
    "gpt-4o": {
        "input": 3.75,
        "output": 15.00
    },
    "gpt-4o-mini": {
        "input": 0.30,
        "output": 1.20
    },
    
    # GPT-4 Realtime Series
    "gpt-realtime": {
        "input": 32.00,
        "output": 64.00
    },
    "gpt-realtime-mini": {
        "input": 10.00,
        "output": 20.00
    },
    "gpt-4o-realtime-preview": {
        "input": 40.00,
        "output": 80.00
    },
    
    # GPT-3.5 Series
    "gpt-3.5-turbo": {
        "input": 3.00,
        "output": 6.00
    },
    
    # Legacy Models
    "davinci-002": {
        "input": 12.00,
        "output": 12.00
    },
    "babbage-002": {
        "input": 1.60,
        "output": 1.60
    },
    
    # === Claude Models (Anthropic) ===
    
    # Claude 4.5 Series (Latest)
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00
    },
    "claude-opus-4-5": {
        "input": 5.00,
        "output": 25.00
    },
    "claude-opus-4-5-20251101": {
        "input": 5.00,
        "output": 25.00
    },
    
    # Claude 4 Series
    "claude-sonnet-4": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-opus-4": {
        "input": 15.00,
        "output": 75.00
    },
    "claude-opus-4.1": {
        "input": 15.00,
        "output": 75.00
    },
    
    # Claude 3.5 Series
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-5-sonnet-20240620": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.00
    },
    
    # Claude 3 Series
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00
    },
    "claude-3-sonnet-20240229": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25
    },
    
    # Claude Legacy
    "claude-2.1": {
        "input": 11.02,
        "output": 32.68
    },
    "claude-instant-1.2": {
        "input": 1.63,
        "output": 5.51
    }
}


class UsageTracker:
    def __init__(self):
        self.calls = []
        self.started_at = datetime.utcnow()

    def record(self, response, model: str, stage: str, provider: str = "openai"):
        """
        Records the token usage from an API response object.
        Supports both OpenAI and Claude response formats.
        
        Args:
            response: API response object
            model: Model name/identifier
            stage: Stage of processing (e.g., "query", "filter")
            provider: API provider ("openai" or "claude")
        """
        usage = getattr(response, "usage", None)
        if not usage:
            return

        if provider == "openai":
            # OpenAI uses 'prompt_tokens' and 'completion_tokens'
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
        elif provider == "claude":
            # Claude uses 'input_tokens' and 'output_tokens'
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            total_tokens = input_tokens + output_tokens
        else:
            return

        self.calls.append({
            "stage": stage,
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        })

    def calculate_cost(self, call):
        """
        Calculate cost for a single API call.
        
        Args:
            call: Dictionary containing call details
            
        Returns:
            float: Cost in USD
        """
        pricing = MODEL_PRICING.get(call["model"])
        if not pricing:
            # Return 0 if model not found in pricing
            return 0.0
        
        # Pricing is per 1M tokens, so divide by 1,000,000
        input_cost = (call["input_tokens"] / 1_000_000) * pricing["input"]
        output_cost = (call["output_tokens"] / 1_000_000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def summary(self):
        """
        Generate a summary of all API calls.
        
        Returns:
            dict: Summary statistics including total tokens and costs
        """
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
        """
        Print a detailed report of all API usage and costs.
        """
        summary = self.summary()
        print("\n========== AI API Usage Report ==========")
        print(f"Execution started at (UTC): {summary['started_at']}")
        print(f"Total API calls: {summary['total_calls']}")
        print(f"Total input tokens: {summary['total_input_tokens']}")
        print(f"Total output tokens: {summary['total_output_tokens']}")
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"Total estimated cost (USD): ${summary['total_cost_usd']}")
        print("\n--- Per Call Breakdown ---")
        for i, c in enumerate(summary["calls"], start=1):
            print(
                f"{i}. Provider: {c['provider']} | "
                f"Stage: {c['stage']} | "
                f"Model: {c['model']} | "
                f"In: {c['input_tokens']} | "
                f"Out: {c['output_tokens']} | "
                f"Total: {c['total_tokens']} | "
                f"Cost: ${c['cost_usd']}"
            )
        print("============================================\n")

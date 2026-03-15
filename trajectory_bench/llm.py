import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import ModelConfig, JudgeConfig

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
)
async def generate(config: ModelConfig, messages: list[dict]) -> tuple[str, dict]:
    """Send messages to the model and return (completion_text, usage_dict)."""
    client = _get_client()

    # Anthropic API needs first message to be user role — our data already satisfies this
    response = await client.messages.create(
        model=config.model_name,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, usage


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
)
async def judge_call(config: JudgeConfig, prompt: str) -> tuple[str, dict]:
    """Send a single user message to the judge model."""
    client = _get_client()

    response = await client.messages.create(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, usage

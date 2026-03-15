import anthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import ModelConfig, JudgeConfig

_anthropic_client: anthropic.AsyncAnthropic | None = None
_openai_clients: dict[str, AsyncOpenAI] = {}


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def _get_openai_client(
    base_url: str | None = None,
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> AsyncOpenAI:
    key = f"{base_url}:{api_key}:{sorted(extra_headers.items()) if extra_headers else ''}"
    if key not in _openai_clients:
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        elif base_url:
            # vLLM doesn't need a real key, but the SDK requires one
            kwargs["api_key"] = "EMPTY"
        if extra_headers:
            kwargs["default_headers"] = extra_headers
        _openai_clients[key] = AsyncOpenAI(**kwargs)
    return _openai_clients[key]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
async def generate(config: ModelConfig, messages: list[dict]) -> tuple[str, dict]:
    """Send messages to the model and return (completion_text, usage_dict)."""
    if config.provider == "anthropic":
        return await _generate_anthropic(config, messages)
    elif config.provider == "openai":
        return await _generate_openai(config, messages)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


async def _generate_anthropic(config: ModelConfig, messages: list[dict]) -> tuple[str, dict]:
    client = _get_anthropic_client()
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


async def _generate_openai(config: ModelConfig, messages: list[dict]) -> tuple[str, dict]:
    client = _get_openai_client(config.base_url, config.api_key, config.extra_headers)
    response = await client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": getattr(response.usage, "prompt_tokens", 0),
        "output_tokens": getattr(response.usage, "completion_tokens", 0),
    }
    return text, usage


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
)
async def judge_call(config: JudgeConfig, prompt: str) -> tuple[str, dict]:
    """Send a single user message to the judge model."""
    messages = [{"role": "user", "content": prompt}]

    if config.provider == "anthropic":
        client = _get_anthropic_client()
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
    elif config.provider == "openai":
        client = _get_openai_client(config.base_url, config.api_key, config.extra_headers)
        response = await client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0),
        }
    else:
        raise ValueError(f"Unknown provider: {config.provider}")

    return text, usage

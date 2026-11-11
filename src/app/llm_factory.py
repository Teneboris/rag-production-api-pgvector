"""Chat-LLM factory shared by the RAG graph and the plain agent.

Provider is selected via MODEL_PROVIDER ("ollama" | "bedrock")
"""

from src.app.config import get_settings

settings = get_settings()


def _ollama_kwargs() -> dict:
    kwargs = {"base_url": settings.ollama_base_url}
    if settings.ollama_api_key:
        kwargs["client_kwargs"] = {
            "headers": {
                "Authorization": f"Bearer {settings.ollama_api_key}",
            },
        }
    return kwargs


def build_chat_llm(model: str | None = None, **overrides):
    """Return a LangChain chat model for the configured provider.

    `model` defaults to settings.primary_model. Extra keyword arguments
    are forwarded to the provider class.
    """
    model_name = model or settings.primary_model
    provider = settings.model_provider.lower()

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        # Ollama-only option (JSON mode); Bedrock gets structure via tool calling.
        overrides.pop("format", None)

        params = {
            "model": model_name,
            "temperature": 0,
            "region_name": settings.aws_region,
        }
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            params["aws_access_key_id"] = settings.aws_access_key_id
            params["aws_secret_access_key"] = settings.aws_secret_access_key

        params.update(overrides)
        return ChatBedrockConverse(**params)

    if provider != "ollama":
        raise ValueError(
            f"Unsupported MODEL_PROVIDER '{settings.model_provider}'. "
            "Use 'ollama' or 'bedrock'."
        )

    from langchain_ollama import ChatOllama

    params = {
        "model": model_name,
        "temperature": 0,
        **_ollama_kwargs(),
    }
    params.update(overrides)
    return ChatOllama(**params)

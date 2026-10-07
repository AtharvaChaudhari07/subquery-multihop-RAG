from app.core.settings import get_settings

settings = get_settings()

LANGFUSE_ENABLED = (
    settings.langfuse_enabled
    and bool(settings.langfuse_public_key)
    and bool(settings.langfuse_secret_key)
)


if LANGFUSE_ENABLED:

    from langfuse import Langfuse, observe

    # ---------------------------------------------------------
    # Initialize the Langfuse singleton using values loaded
    # from our Pydantic Settings object.
    # ---------------------------------------------------------

    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )

    print("[startup] Langfuse client initialized.")

else:

    langfuse = None

    def observe(*_args, **_kwargs):
        """
        No-op decorator when Langfuse is disabled
        or credentials are missing.
        """
        def decorator(function):
            return function

        return decorator
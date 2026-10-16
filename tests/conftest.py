from __future__ import annotations

import pytest

from wikize_memory.config import Settings

USER_AGENT = "WikiMemoryHarvester/1.0 (+https://example.org/contact)"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        user_agent=USER_AGENT,
        trust_env=False,
        max_retries=1,
        retry_base_seconds=0,
        retry_max_seconds=0,
        max_continuations=10,
    )

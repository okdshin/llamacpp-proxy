import pytest
from llamacpp_proxy.config.rate_limit import RateLimitSettings

def test_validate_no_api_keys():
    settings = RateLimitSettings(
        unlimited_api_key="",
        limited_api_key=""
    )
    with pytest.raises(ValueError, match="At least one API key must be configured"):
        settings.validate()

def test_validate_unlimited_api_key():
    settings = RateLimitSettings(
        unlimited_api_key="test_key",
        limited_api_key=""
    )
    settings.validate()  # should not raise

def test_validate_limited_api_key():
    settings = RateLimitSettings(
        unlimited_api_key="",
        limited_api_key="test_key"
    )
    settings.validate()  # should not raise

def test_validate_both_api_keys():
    settings = RateLimitSettings(
        unlimited_api_key="unlimited_key",
        limited_api_key="limited_key"
    )
    settings.validate()  # should not raise
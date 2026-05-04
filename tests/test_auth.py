"""
Tests for authentication utilities.
"""
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from src.utils.auth import verify_api_key, get_optional_api_key
from src.config import settings


class TestVerifyAPIKey:
    """Test verify_api_key function."""

    def test_valid_api_key(self):
        """Test with valid API key."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=settings.api_key
        )
        result = verify_api_key(credentials)
        assert result == settings.api_key

    def test_invalid_api_key(self):
        """Test with invalid API key."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_key"
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(credentials)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail

    def test_empty_api_key(self):
        """Test with empty API key."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(credentials)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in exc_info.value.detail


class TestGetOptionalAPIKey:
    """Test get_optional_api_key function."""

    def test_valid_api_key(self):
        """Test with valid API key."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=settings.api_key
        )
        result = get_optional_api_key(credentials)
        assert result == settings.api_key

    def test_invalid_api_key(self):
        """Test with invalid API key."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_key"
        )
        result = get_optional_api_key(credentials)
        assert result is None

    def test_empty_api_key(self):
        """Test with empty API key."""
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
        result = get_optional_api_key(credentials)
        assert result is None

    def test_no_credentials(self):
        """Test with no credentials."""
        result = get_optional_api_key(None)
        assert result is None

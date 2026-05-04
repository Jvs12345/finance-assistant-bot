"""
Authentication utilities for API key validation.
"""
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# HTTP Bearer security scheme
security = HTTPBearer()


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Verify API key from Authorization header.

    Args:
        credentials: HTTP authorization credentials with Bearer token

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not credentials:
        logger.warning("Missing authorization credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = credentials.credentials

    if not api_key:
        logger.warning("Empty API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate against configured API key
    if api_key != settings.api_key:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.debug("API key validated successfully")
    return api_key


def get_optional_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[str]:
    """
    Get API key from Authorization header without requiring it.

    This is useful for endpoints that are optionally authenticated.

    Args:
        credentials: HTTP authorization credentials with Bearer token

    Returns:
        The API key if provided and valid, None otherwise
    """
    if not credentials or not credentials.credentials:
        return None

    api_key = credentials.credentials

    # Validate against configured API key
    if api_key != settings.api_key:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        return None

    logger.debug("API key validated successfully")
    return api_key

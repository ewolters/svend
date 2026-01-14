"""
Simple API key authentication for Svend.

For alpha/beta, this is just a list of valid keys.
For production, you'd use a proper auth system.
"""

import os
import secrets
import json
from pathlib import Path
from typing import Optional, Dict, Set
from datetime import datetime

from fastapi import HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class APIKeyManager:
    """
    Simple API key manager.

    Stores keys in a JSON file for simplicity.
    For production, use a database.
    """

    def __init__(self, keys_file: str = "config/api_keys.json"):
        self.keys_file = Path(keys_file)
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_keys()

    def _load_keys(self):
        """Load keys from file."""
        if self.keys_file.exists():
            with open(self.keys_file) as f:
                self.keys = json.load(f)
        else:
            self.keys = {}
            self._save_keys()

    def _save_keys(self):
        """Save keys to file."""
        with open(self.keys_file, "w") as f:
            json.dump(self.keys, f, indent=2)

    def generate_key(self, name: str, tier: str = "alpha") -> str:
        """
        Generate a new API key.

        Args:
            name: User/tester name
            tier: "alpha", "beta", "paid"

        Returns:
            The generated API key
        """
        key = f"svend_{secrets.token_hex(16)}"

        self.keys[key] = {
            "name": name,
            "tier": tier,
            "created": datetime.utcnow().isoformat(),
            "active": True,
            "requests_today": 0,
            "last_request": None,
        }

        self._save_keys()
        return key

    def validate_key(self, key: str) -> Optional[Dict]:
        """
        Validate an API key.

        Returns key info if valid, None if invalid.
        """
        if key not in self.keys:
            return None

        key_info = self.keys[key]

        if not key_info.get("active", True):
            return None

        return key_info

    def get_tier(self, key: str) -> Optional[str]:
        """Get the tier for a key."""
        info = self.validate_key(key)
        return info.get("tier") if info else None

    def record_request(self, key: str):
        """Record a request for rate limiting."""
        if key in self.keys:
            self.keys[key]["requests_today"] += 1
            self.keys[key]["last_request"] = datetime.utcnow().isoformat()
            # Don't save on every request - too slow
            # self._save_keys()

    def revoke_key(self, key: str):
        """Revoke an API key."""
        if key in self.keys:
            self.keys[key]["active"] = False
            self._save_keys()

    def list_keys(self) -> Dict:
        """List all keys (redacted)."""
        return {
            k[:12] + "...": v
            for k, v in self.keys.items()
        }

    def reset_daily_counts(self):
        """Reset daily request counts."""
        for key in self.keys:
            self.keys[key]["requests_today"] = 0
        self._save_keys()


# Rate limits by tier
RATE_LIMITS = {
    "alpha": 100,    # 100 requests/day
    "beta": 200,     # 200 requests/day
    "paid": 1000,    # 1000 requests/day
    "unlimited": float("inf"),
}


# Global instance
_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Get or create the key manager."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


# FastAPI security scheme
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict:
    """
    FastAPI dependency to verify API key.

    Usage:
        @app.post("/v1/chat/completions")
        async def chat(request: Request, auth: Dict = Depends(verify_api_key)):
            # auth contains key info
            pass
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Use 'Authorization: Bearer <key>'",
        )

    key = credentials.credentials
    manager = get_key_manager()

    key_info = manager.validate_key(key)
    if key_info is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    # Check rate limit
    tier = key_info.get("tier", "alpha")
    limit = RATE_LIMITS.get(tier, 100)
    requests = key_info.get("requests_today", 0)

    if requests >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. {tier} tier allows {limit} requests/day.",
        )

    # Record request
    manager.record_request(key)

    return {
        "key": key[:8] + "...",
        **key_info,
    }


async def optional_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict]:
    """
    Optional API key verification.

    Returns None if no key provided (for public endpoints).
    """
    if credentials is None:
        return None

    try:
        return await verify_api_key(credentials)
    except HTTPException:
        return None


# CLI for managing keys
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage Svend API keys")
    subparsers = parser.add_subparsers(dest="command")

    # Generate key
    gen_parser = subparsers.add_parser("generate", help="Generate a new key")
    gen_parser.add_argument("name", help="User/tester name")
    gen_parser.add_argument("--tier", default="alpha", choices=["alpha", "beta", "paid", "unlimited"])

    # List keys
    subparsers.add_parser("list", help="List all keys")

    # Revoke key
    revoke_parser = subparsers.add_parser("revoke", help="Revoke a key")
    revoke_parser.add_argument("key", help="API key to revoke")

    # Reset counts
    subparsers.add_parser("reset", help="Reset daily request counts")

    args = parser.parse_args()

    manager = APIKeyManager()

    if args.command == "generate":
        key = manager.generate_key(args.name, args.tier)
        print(f"Generated key for {args.name} ({args.tier}):")
        print(f"  {key}")
        print(f"\nRate limit: {RATE_LIMITS.get(args.tier, 100)} requests/day")

    elif args.command == "list":
        keys = manager.list_keys()
        print("API Keys:")
        for k, v in keys.items():
            status = "active" if v.get("active") else "revoked"
            print(f"  {k} | {v['name']} | {v['tier']} | {status}")

    elif args.command == "revoke":
        manager.revoke_key(args.key)
        print(f"Revoked key: {args.key[:12]}...")

    elif args.command == "reset":
        manager.reset_daily_counts()
        print("Reset all daily request counts")

    else:
        parser.print_help()

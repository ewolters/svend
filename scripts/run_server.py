#!/usr/bin/env python3
"""
Run Svend API server with logging and authentication.

Usage:
    # Development (no auth)
    py -3 scripts/run_server.py --model-path checkpoints/svend-1.8b.pt --no-auth

    # Alpha (with auth)
    py -3 scripts/run_server.py --model-path checkpoints/svend-1.8b.pt

    # Generate API key
    py -3 scripts/run_server.py --generate-key "Alice" --tier alpha

    # List keys
    py -3 scripts/run_server.py --list-keys
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run Svend API server")

    # Server options
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "vllm"])

    # Auth options
    parser.add_argument("--no-auth", action="store_true", help="Disable API key authentication")
    parser.add_argument("--generate-key", type=str, metavar="NAME", help="Generate API key for user")
    parser.add_argument("--tier", type=str, default="alpha", choices=["alpha", "beta", "paid", "unlimited"])
    parser.add_argument("--list-keys", action="store_true", help="List all API keys")
    parser.add_argument("--revoke-key", type=str, help="Revoke an API key")

    # Logging options
    parser.add_argument("--no-logging", action="store_true", help="Disable request logging")
    parser.add_argument("--log-bodies", action="store_true", help="Log request/response bodies")

    args = parser.parse_args()

    # Handle key management commands
    if args.generate_key or args.list_keys or args.revoke_key:
        from src.server.auth import APIKeyManager, RATE_LIMITS

        manager = APIKeyManager()

        if args.generate_key:
            key = manager.generate_key(args.generate_key, args.tier)
            print(f"\nGenerated API key for '{args.generate_key}' ({args.tier}):")
            print(f"\n  {key}\n")
            print(f"Rate limit: {RATE_LIMITS.get(args.tier, 100)} requests/day")
            print("\nUsage:")
            print(f"  curl -H 'Authorization: Bearer {key}' http://localhost:8000/v1/models")

        elif args.list_keys:
            keys = manager.list_keys()
            print("\nAPI Keys:")
            print("-" * 60)
            for k, v in keys.items():
                status = "active" if v.get("active") else "REVOKED"
                requests = v.get("requests_today", 0)
                print(f"  {k} | {v['name']:<15} | {v['tier']:<8} | {status:<8} | {requests} reqs today")
            print("-" * 60)

        elif args.revoke_key:
            manager.revoke_key(args.revoke_key)
            print(f"Revoked key: {args.revoke_key[:12]}...")

        return

    # Require model path for server
    if not args.model_path:
        parser.error("--model-path is required to run the server")

    # Import server components
    from fastapi import FastAPI, Depends
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    from src.server.api import SvendServer
    from src.server.logging_middleware import RequestLogger, LoggingMiddleware
    from src.server.auth import verify_api_key, optional_api_key

    # Create server
    server = SvendServer(
        model_path=args.model_path,
        backend=args.backend,
    )

    app = server.create_app()

    # Add logging middleware
    if not args.no_logging:
        logger = RequestLogger()
        app.add_middleware(LoggingMiddleware, logger=logger, log_bodies=args.log_bodies)
        print(f"Request logging enabled: logs/requests.db")

        # Add stats endpoint
        @app.get("/admin/stats")
        async def get_stats():
            return logger.get_stats()

        @app.get("/admin/logs")
        async def get_logs(limit: int = 100):
            return logger.get_recent(limit)

        @app.get("/admin/errors")
        async def get_errors(limit: int = 100):
            return logger.get_errors(limit)

    # Add auth if enabled
    if not args.no_auth:
        print("API key authentication enabled")
        print("Generate keys with: py -3 scripts/run_server.py --generate-key NAME")

        # Override endpoints to require auth
        # This is a bit hacky - in production you'd structure this better
        original_routes = app.routes.copy()

        for route in original_routes:
            if hasattr(route, "path") and route.path.startswith("/v1/"):
                # Add auth dependency
                if hasattr(route, "dependant"):
                    route.dependant.dependencies.append(Depends(verify_api_key))

    print(f"\n{'='*50}")
    print("SVEND API SERVER")
    print(f"{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Backend: {args.backend}")
    print(f"Auth: {'disabled' if args.no_auth else 'enabled'}")
    print(f"Logging: {'disabled' if args.no_logging else 'enabled'}")
    print(f"{'='*50}")
    print(f"\nServer starting at http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/health\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

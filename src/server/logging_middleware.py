"""
Request/Response logging for Svend API.

Logs all requests for:
- Debugging
- Safety review
- Usage analytics
"""

import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLogger:
    """
    SQLite-based request logger.

    Stores all API requests for review and debugging.
    """

    def __init__(self, db_path: str = "logs/requests.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    api_key TEXT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    request_body TEXT,
                    response_body TEXT,
                    status_code INTEGER,
                    latency_ms REAL,
                    error TEXT,
                    ip_address TEXT,
                    user_agent TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_key ON requests(api_key)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_endpoint ON requests(endpoint)
            """)

    @contextmanager
    def _get_conn(self):
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_request(
        self,
        endpoint: str,
        method: str,
        request_body: Optional[str] = None,
        response_body: Optional[str] = None,
        status_code: int = 200,
        latency_ms: float = 0,
        api_key: Optional[str] = None,
        error: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log a request."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO requests (
                    timestamp, api_key, endpoint, method,
                    request_body, response_body, status_code,
                    latency_ms, error, ip_address, user_agent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                api_key,
                endpoint,
                method,
                request_body,
                response_body,
                status_code,
                latency_ms,
                error,
                ip_address,
                user_agent,
            ))

    def get_recent(self, limit: int = 100) -> list:
        """Get recent requests."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM requests
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_by_api_key(self, api_key: str, limit: int = 100) -> list:
        """Get requests by API key."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM requests
                WHERE api_key = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (api_key, limit))
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_errors(self, limit: int = 100) -> list:
        """Get requests with errors."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM requests
                WHERE error IS NOT NULL OR status_code >= 400
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics."""
        with self._get_conn() as conn:
            # Total requests
            cursor = conn.execute("""
                SELECT COUNT(*) FROM requests
                WHERE timestamp > datetime('now', ?)
            """, (f"-{hours} hours",))
            total = cursor.fetchone()[0]

            # By endpoint
            cursor = conn.execute("""
                SELECT endpoint, COUNT(*) as count
                FROM requests
                WHERE timestamp > datetime('now', ?)
                GROUP BY endpoint
                ORDER BY count DESC
            """, (f"-{hours} hours",))
            by_endpoint = dict(cursor.fetchall())

            # Errors
            cursor = conn.execute("""
                SELECT COUNT(*) FROM requests
                WHERE timestamp > datetime('now', ?)
                AND (error IS NOT NULL OR status_code >= 400)
            """, (f"-{hours} hours",))
            errors = cursor.fetchone()[0]

            # Avg latency
            cursor = conn.execute("""
                SELECT AVG(latency_ms) FROM requests
                WHERE timestamp > datetime('now', ?)
            """, (f"-{hours} hours",))
            avg_latency = cursor.fetchone()[0] or 0

            return {
                "total_requests": total,
                "by_endpoint": by_endpoint,
                "errors": errors,
                "error_rate": errors / total if total > 0 else 0,
                "avg_latency_ms": avg_latency,
                "period_hours": hours,
            }


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request logging.
    """

    def __init__(self, app, logger: RequestLogger, log_bodies: bool = True):
        super().__init__(app)
        self.logger = logger
        self.log_bodies = log_bodies

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()

        # Extract info
        endpoint = request.url.path
        method = request.method
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")

        # Get request body (if applicable)
        request_body = None
        if self.log_bodies and method in ("POST", "PUT", "PATCH"):
            try:
                body_bytes = await request.body()
                request_body = body_bytes.decode("utf-8")[:10000]  # Limit size

                # Reconstruct request for downstream handlers
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                request = Request(request.scope, receive)
            except Exception:
                pass

        # Call endpoint
        error = None
        response_body = None
        try:
            response = await call_next(request)
            status_code = response.status_code

            # Capture response body for logging (careful - streaming!)
            if self.log_bodies and status_code < 400:
                # Only log non-streaming responses
                if not response.headers.get("content-type", "").startswith("text/event-stream"):
                    body_parts = []
                    async for chunk in response.body_iterator:
                        body_parts.append(chunk)
                    response_body = b"".join(body_parts).decode("utf-8")[:10000]

                    # Reconstruct response
                    async def body_iterator():
                        for part in body_parts:
                            yield part
                    response.body_iterator = body_iterator()

        except Exception as e:
            error = str(e)
            status_code = 500
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000

            # Log (skip health checks to reduce noise)
            if endpoint != "/health":
                self.logger.log_request(
                    endpoint=endpoint,
                    method=method,
                    request_body=request_body,
                    response_body=response_body,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    api_key=api_key[:8] + "..." if api_key else None,  # Truncate key
                    error=error,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )

        return response

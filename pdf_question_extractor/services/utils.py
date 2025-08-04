"""Shared utilities for services"""

import asyncio
from typing import List

class RateLimiter:
    """Rate limiter for API calls to prevent quota exhaustion"""
    
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter
        
        Args:
            calls_per_minute: Maximum number of calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.calls: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """
        Acquire permission to make an API call.
        Blocks if rate limit would be exceeded.
        """
        async with self.lock:
            now = asyncio.get_event_loop().time()
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            if len(self.calls) >= self.calls_per_minute:
                # Wait until we can make another call
                sleep_time = 60 - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
                self.calls = self.calls[1:]
            
            self.calls.append(now)
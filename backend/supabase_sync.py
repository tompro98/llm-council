"""Sync council runs to Supabase for persistent storage."""

import os
import httpx
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


async def save_run_to_supabase(
    conversation_id: str,
    source: str,
    question: str,
    title: Optional[str],
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any],
    metadata: Dict[str, Any],
):
    """
    Upsert a council run to Supabase council_runs table.
    Silently skips if Supabase is not configured.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/council_runs",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates",
                },
                json={
                    "conversation_id": conversation_id,
                    "source": source,
                    "question": question,
                    "title": title,
                    "stage1": stage1,
                    "stage2": stage2,
                    "stage3": stage3,
                    "metadata": metadata,
                },
            )
            if resp.status_code not in (200, 201):
                logger.warning(f"Supabase sync failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.warning(f"Supabase sync error (non-fatal): {e}")

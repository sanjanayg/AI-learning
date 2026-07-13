import asyncio
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db
from cache.cache_config import cache_settings
from cache.cache_metrics import metrics_service
from cache.kb_version_tracker import KBVersionTracker
from cache.semantic_cache import SemanticCacheService
from cache.cache_schemas import CacheEntryPayload

logger = logging.getLogger(__name__)

class CacheCleanupService:
    def __init__(self, cache_service: SemanticCacheService):
        self.cache_service = cache_service
        self.cleanup_task = None
        self._running = False

    def start(self):
        """Starts the cleanup worker task in the background."""
        self._running = True
        self.cleanup_task = asyncio.create_task(self._run_loop())
        logger.info("CacheCleanupService started background cleanup loop.")

    def stop(self):
        """Stops the background cleanup loop."""
        self._running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            logger.info("CacheCleanupService stopped.")

    async def _run_loop(self):
        # Run cleanup immediately on startup, then periodically
        while self._running:
            try:
                await self.perform_cleanup()
            except Exception as e:
                logger.error("Error in cache cleanup loop: %s", e)
            
            # Wait for the next interval
            interval_seconds = cache_settings.CACHE_CLEANUP_INTERVAL_HOURS * 3600
            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break

    async def perform_cleanup(self):
        """
        Scans all cache entries in Qdrant and deletes entries that are:
        - Expired (TTL reached)
        - Unused for CACHE_UNUSED_EXPIRY_DAYS
        - Obsolete knowledge version (different from current DB version of tenant)
        """
        logger.info("Starting semantic cache cleanup scan...")
        client = self.cache_service.client
        collection_name = self.cache_service.collection_name
        
        now = datetime.now(timezone.utc)
        unused_cutoff = now - timedelta(days=cache_settings.CACHE_UNUSED_EXPIRY_DAYS)
        
        # We fetch active tenants' current KB versions
        # Open a DB session to query DB
        db_generator = get_db()
        db: AsyncSession = await anext(db_generator)
        
        tenant_versions = {}
        points_to_delete = []
        
        offset = None
        scanned_count = 0
        deleted_expired = 0
        deleted_unused = 0
        deleted_obsolete = 0
        
        try:
            while True:
                # Scroll all points
                batch, offset = await asyncio.to_thread(
                    client.scroll,
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not batch:
                    break
                
                for pt in batch:
                    scanned_count += 1
                    payload = pt.payload
                    if not payload:
                        points_to_delete.append(pt.id)
                        continue
                        
                    try:
                        entry = CacheEntryPayload(**payload)
                    except Exception as e:
                        logger.warning("Failed to deserialize payload during cleanup, scheduling delete: %s", e)
                        points_to_delete.append(pt.id)
                        continue
                        
                    # Check 1: Expired (TTL)
                    expires_at = datetime.fromisoformat(entry.expires_at)
                    if expires_at < now:
                        points_to_delete.append(pt.id)
                        deleted_expired += 1
                        continue
                        
                    # Check 2: Unused for long time
                    last_accessed = datetime.fromisoformat(entry.last_accessed)
                    if last_accessed < unused_cutoff:
                        points_to_delete.append(pt.id)
                        deleted_unused += 1
                        continue
                        
                    # Check 3: Obsolete knowledge version
                    tenant_id = entry.tenant_id
                    if tenant_id not in tenant_versions:
                        # Fetch and cache active version for this tenant
                        tenant_versions[tenant_id] = await KBVersionTracker.get_version(db, tenant_id)
                        
                    current_version = tenant_versions[tenant_id]
                    if entry.kb_version != current_version:
                        points_to_delete.append(pt.id)
                        deleted_obsolete += 1
                        continue
                
                if offset is None:
                    break
                    
            # Perform batch deletion if any points identified
            if points_to_delete:
                logger.info("Deleting %d obsolete/expired cache points from Qdrant...", len(points_to_delete))
                # Delete in chunks of 100 to avoid long request size
                for i in range(0, len(points_to_delete), 100):
                    chunk = points_to_delete[i:i+100]
                    await asyncio.to_thread(
                        client.delete,
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(points=chunk)
                    )
                
                metrics_service.record_expired_removed(deleted_expired + deleted_unused + deleted_obsolete)
                metrics_service.record_deletion(len(points_to_delete) - (deleted_expired + deleted_unused + deleted_obsolete))
                
            logger.info(
                "Cleanup complete. Scanned: %d, Deleted: %d (Expired: %d, Unused: %d, Obsolete KB: %d)",
                scanned_count, len(points_to_delete), deleted_expired, deleted_unused, deleted_obsolete
            )
            
        except Exception as e:
            logger.error("Error during cache cleanup: %s", e)
        finally:
            # Safely close DB generator
            try:
                await db_generator.aclose()
            except Exception:
                pass

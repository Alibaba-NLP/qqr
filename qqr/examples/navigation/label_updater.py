"""
POI label updater for navigation task.

Calls amap POI detail API to refresh coordinates in sample.label
based on POI IDs in label["ref"]. Updates both:
  - label["ref"][i]["location"]
  - label["groundtruth"]["arguments"] (origin/destination/waypoints)

Should be called at the start of rollout (before inspector/OPD)
so all downstream components see accurate coordinates.
"""

import asyncio
import logging
from typing import Any

import httpx

from qqr.schemas import Sample
from qqr.utils.envs import AMAP_MAPS_API_KEY

logger = logging.getLogger(__name__)

_AMAP_DETAIL_URL = "https://restapi.amap.com/v5/place/detail"


async def _fetch_poi_location_by_id(poi_id: str) -> str | None:
    """Call amap POI detail API to get current location for a POI id."""
    params = {
        "key": AMAP_MAPS_API_KEY,
        "id": poi_id,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(_AMAP_DETAIL_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        if data.get("status") != "1":
            return None
        pois = data.get("pois")
        if not pois or not isinstance(pois, list):
            return None
        location = pois[0].get("location")
        return location if isinstance(location, str) and "," in location else None
    except Exception as e:
        logger.debug(f"[label_updater] POI detail failed for id={poi_id}: {e}")
        return None


async def update_label_locations(sample: Sample) -> dict:
    """
    Fetch latest POI locations by ID and update sample.label in-place.

    Updates:
      1) label["ref"][i]["location"] — refreshed from amap API
      2) label["groundtruth"]["arguments"] — origin/destination/waypoints coordinates

    No-op if label structure is missing or all locations are already current.
    """
    label = sample.label
    if not isinstance(label, dict):
        return label

    ref_list = label.get("ref")
    gt = label.get("groundtruth")

    if not isinstance(ref_list, list) or not isinstance(gt, dict):
        return label
    gt_args = gt.get("arguments")
    if not isinstance(gt_args, dict):
        return label

    # Collect POI IDs to query
    ids_to_query = []
    for item in ref_list:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids_to_query.append(item["id"])

    if not ids_to_query:
        return label

    # Concurrent API calls for all POI IDs
    results = await asyncio.gather(
        *[_fetch_poi_location_by_id(pid) for pid in ids_to_query]
    )
    id2loc = {pid: loc for pid, loc in zip(ids_to_query, results) if loc is not None}

    if not id2loc:
        return label

    # Update ref locations, build old→new mapping
    old2new_loc: dict[str, str] = {}
    for item in ref_list:
        if not isinstance(item, dict):
            continue
        poi_id = item.get("id")
        if not isinstance(poi_id, str):
            continue
        new_loc = id2loc.get(poi_id)
        if not isinstance(new_loc, str):
            continue
        old_loc = item.get("location")
        if isinstance(old_loc, str) and old_loc and old_loc != new_loc:
            old2new_loc[old_loc] = new_loc
            item["location"] = new_loc

    if not old2new_loc:
        return label

    # Update groundtruth arguments
    def _replace(s: str | None) -> str | None:
        if not isinstance(s, str):
            return s
        return old2new_loc.get(s, s)

    gt_args["origin"] = _replace(gt_args.get("origin"))
    gt_args["destination"] = _replace(gt_args.get("destination"))

    wps = gt_args.get("waypoints")
    if isinstance(wps, str) and wps.strip():
        parts = [old2new_loc.get(seg.strip(), seg.strip()) for seg in wps.split(";") if seg.strip()]
        gt_args["waypoints"] = ";".join(parts)

    return label

import pickle

# Handles two different timelines:
# 1. Active memory (memory that arrays are actually using right now): alloc/free_completed.
_ALLOC_ACTIONS = {"AllocNew", "AllocReuse", "AllocMakeBuffer"}
_FREE_ACTIONS = {"FreeActiveToCache", "FreeActiveToOS", "Release"}
# 2. Reserved memory (memory that's taken from OS and not released yet, either for active
#    memory or retained in cache).
_SEGMENT_ALLOC = {"AllocNew", "AllocMakeBuffer"}
_SEGMENT_FREE = {"FreeActiveToOS", "Release", "FreeCacheToOS"}


def _frames(traceback):
    if not traceback:
        return []
    return [
        {"filename": filename, "line": line, "name": name}
        for filename, name, line in reversed(traceback)
    ]


def to_snapshot(events):
    """Builds a snapshot dict from mx.get_memory_events()."""
    trace = []
    for event in events:
        entry = {
            "addr": event["buffer_ptr"],
            "size": event["size"],
            "requested_size": event["requested_size"],
            "stream": 0,  # TODO: add properly.
            "version": 0,
            "time_us": event["timestamp_us"],
            "frames": _frames(event.get("traceback")),
            "category": event.get("primitive_name") or "unknown",
        }
        # A buffer can appear on both timelines.
        if event["action"] in _SEGMENT_ALLOC:
            trace.append({**entry, "action": "segment_alloc"})
        if event["action"] in _ALLOC_ACTIONS:
            trace.append({**entry, "action": "alloc"})
        if event["action"] in _FREE_ACTIONS:
            trace.append({**entry, "action": "free_completed"})
        if event["action"] in _SEGMENT_FREE:
            trace.append({**entry, "action": "segment_free"})

    return {"device_traces": [trace], "segments": []}


def dump_snapshot(events, path="mlx_snapshot.pickle"):
    with open(path, "wb") as f:
        pickle.dump(to_snapshot(events), f)
    return path

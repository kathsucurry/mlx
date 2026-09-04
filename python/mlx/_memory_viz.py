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
    alloc_addrs_infos = dict()
    for event in events:
        primitive_name = event.get("primitive_name") or "unknown"
        frames = _frames(event.get("traceback"))
        stream = event["stream"]

        # Replace some of the free's fields with alloc's fields.
        # TODO: add proper tests.
        if event["action"] in _ALLOC_ACTIONS:
            alloc_addrs_infos[event["addr"]] = (
                event["stream"],
                primitive_name,
                frames,
            )
        elif event["action"] in _FREE_ACTIONS and event["addr"] in alloc_addrs_infos:
            stream, primitive_name, frames = alloc_addrs_infos.pop(event["addr"])

        entry = {
            "addr": event["addr"],
            "size": event["size"],
            "requested_size": event["requested_size"],
            "stream": stream,
            "version": 0,
            "time_us": event["timestamp_us"],
            "frames": frames,
            "category": primitive_name,
            "user_metadata": {"primitive": primitive_name},
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

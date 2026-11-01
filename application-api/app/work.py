import hashlib


def chained_sha256(iterations: int) -> bytes:
    """Fixed CPU work: the cost is set in iterations, never in time."""
    digest = bytes(32)
    for _ in range(iterations):
        digest = hashlib.sha256(digest).digest()
    return digest


def request_work(request) -> None:
    # Runs synchronously in the route, like application code processing data.
    iterations = request.app.state.request_work_iterations
    if iterations:
        chained_sha256(iterations)

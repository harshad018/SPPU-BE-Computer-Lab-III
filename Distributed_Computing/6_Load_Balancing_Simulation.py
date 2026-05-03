"""
Practical 5: Load Balancing Simulation
============================================================
Simulates distribution of incoming client requests across multiple
servers using three classic load-balancing strategies:

    1. Round-Robin       – requests cycle through servers in fixed order.
    2. Least Connections – each request goes to the server currently
                           handling the fewest active connections.
    3. Random            – each request is assigned to a randomly chosen server.

Metrics reported per strategy:
    • Requests handled per server
    • Total simulated processing time per server
    • Load imbalance factor (std-dev of request distribution)
    • Average request processing time

Usage:
    python 6_Load_Balancing_Simulation.py

Dependencies: None (Python stdlib only)
"""

import random
import statistics
from typing import List

# ---------------------------------------------------------------------------
# Server model
# ---------------------------------------------------------------------------

class Server:
    """Represents a single server node in the cluster."""

    def __init__(self, server_id: int, capacity: int = 10):
        """Initialise a server.

        Args:
            server_id: Unique identifier for the server.
            capacity:  Nominal max requests per time unit (informational).
        """
        self.server_id = server_id
        self.capacity = capacity
        self.active_connections: int = 0
        self.total_requests: int = 0
        self.total_process_time: float = 0.0

    def handle_request(self, processing_time: float):
        """Simulate handling one client request.

        Args:
            processing_time: Simulated processing duration (arbitrary units).
        """
        self.active_connections += 1
        self.total_requests += 1
        self.total_process_time += processing_time
        # Request completes instantly in simulation
        self.active_connections = max(0, self.active_connections - 1)

    def reset(self):
        """Reset all counters (used between strategy runs)."""
        self.active_connections = 0
        self.total_requests = 0
        self.total_process_time = 0.0

    def __repr__(self):
        return (f"Server(id={self.server_id}, requests={self.total_requests}, "
                f"total_time={self.total_process_time:.3f})")


# ---------------------------------------------------------------------------
# Load-balancer strategies
# ---------------------------------------------------------------------------

class RoundRobinBalancer:
    """Distribute requests in a cyclic order across all servers."""

    def __init__(self, servers: List[Server]):
        self.servers = servers
        self._index = 0

    def get_server(self) -> Server:
        """Return the next server in rotation."""
        server = self.servers[self._index % len(self.servers)]
        self._index += 1
        return server

    @property
    def name(self) -> str:
        return "Round-Robin"


class LeastConnectionsBalancer:
    """Send each request to the server with the fewest active connections.

    In a synchronous simulation requests complete instantly, so active
    connections are always 0.  Total accumulated processing time is used
    as a secondary key to reflect the server's real load — the server
    that has done the least work so far receives the next request.
    """

    def __init__(self, servers: List[Server]):
        self.servers = servers

    def get_server(self) -> Server:
        """Return the least-loaded server (fewest active connections;
        ties broken by lowest total processing time accumulated)."""
        return min(
            self.servers,
            key=lambda s: (s.active_connections, s.total_process_time),
        )

    @property
    def name(self) -> str:
        return "Least Connections"


class RandomBalancer:
    """Assign each request to a randomly chosen server."""

    def __init__(self, servers: List[Server]):
        self.servers = servers

    def get_server(self) -> Server:
        """Return a randomly selected server."""
        return random.choice(self.servers)

    @property
    def name(self) -> str:
        return "Random"


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def simulate(balancer, num_requests: int, seed: int = 42) -> dict:
    """Run the load-balancing simulation.

    Args:
        balancer:     A balancer instance with a .get_server() method.
        num_requests: Total number of client requests to simulate.
        seed:         Random seed for reproducibility.

    Returns:
        Dictionary containing per-server and aggregate metrics.
    """
    rng = random.Random(seed)

    for _ in range(num_requests):
        server = balancer.get_server()
        # Simulate random processing time (0.01 – 0.50 arbitrary time units)
        proc_time = round(rng.uniform(0.01, 0.50), 3)
        server.handle_request(proc_time)

    requests_per_server = {s.server_id: s.total_requests for s in balancer.servers}
    time_per_server = {
        s.server_id: round(s.total_process_time, 3) for s in balancer.servers
    }
    counts = list(requests_per_server.values())
    imbalance = round(statistics.stdev(counts), 3) if len(counts) > 1 else 0.0
    avg_proc = round(
        sum(s.total_process_time for s in balancer.servers) / num_requests, 3
    )

    return {
        "requests_per_server": requests_per_server,
        "time_per_server": time_per_server,
        "imbalance_factor": imbalance,
        "avg_processing_time": avg_proc,
        "total_requests": num_requests,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_metrics(strategy_name: str, metrics: dict, servers: List[Server]):
    """Pretty-print simulation metrics for one strategy."""
    print(f"\n{'=' * 55}")
    print(f"  Strategy : {strategy_name}")
    print(f"{'=' * 55}")
    print(f"  Total requests    : {metrics['total_requests']}")
    print(f"  Avg process time  : {metrics['avg_processing_time']:.3f} units/request")
    print(f"  Imbalance factor  : {metrics['imbalance_factor']:.3f}  "
          f"(std-dev of load; lower = better)")
    print()
    print(f"  {'Server':<12} {'Requests':>10} {'Total Time':>12}")
    print(f"  {'-' * 36}")
    for s in servers:
        print(f"  Server {s.server_id:<5}  "
              f"{metrics['requests_per_server'][s.server_id]:>10}  "
              f"{metrics['time_per_server'][s.server_id]:>12.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run simulation with all three strategies and compare results."""
    random.seed(42)

    NUM_SERVERS = 5
    NUM_REQUESTS = 50

    print("--- Load Balancing Simulation ---")
    print(f"Servers: {NUM_SERVERS}  |  Client requests: {NUM_REQUESTS}\n")

    servers = [Server(server_id=i, capacity=10) for i in range(NUM_SERVERS)]

    strategies = [
        RoundRobinBalancer(servers),
        LeastConnectionsBalancer(servers),
        RandomBalancer(servers),
    ]

    all_metrics = {}
    for balancer in strategies:
        # Reset server counters between runs
        for s in servers:
            s.reset()
        balancer._index = 0 if hasattr(balancer, "_index") else None

        metrics = simulate(balancer, NUM_REQUESTS)
        all_metrics[balancer.name] = metrics
        print_metrics(balancer.name, metrics, servers)

    # Summary comparison
    print(f"\n{'=' * 55}")
    print("  SUMMARY COMPARISON")
    print(f"{'=' * 55}")
    print(f"  {'Strategy':<22} {'Imbalance':>12} {'Avg ProcTime':>14}")
    print(f"  {'-' * 50}")
    for name, m in all_metrics.items():
        print(f"  {name:<22} {m['imbalance_factor']:>12.3f} "
              f"{m['avg_processing_time']:>14.3f}")

    best = min(all_metrics.items(), key=lambda x: x[1]["imbalance_factor"])
    print(f"\n  Best strategy (lowest imbalance): {best[0]}")


if __name__ == "__main__":
    main()

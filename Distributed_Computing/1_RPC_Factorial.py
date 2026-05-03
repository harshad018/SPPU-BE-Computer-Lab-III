"""
Practical 1: RPC Calculation
============================================================
Distributed application where a client sends an integer to a server,
which calculates and returns the factorial using Python's XML-RPC.

This simulates a Remote Procedure Call (RPC) pattern — the client
calls the remote 'factorial' function as if it were local.

Usage:
    # Terminal 1 – Start the server:
    python 1_RPC_Factorial.py server

    # Terminal 2 – Calculate factorial of 7:
    python 1_RPC_Factorial.py client 7

    # Demo mode (server + client run in same process via threads):
    python 1_RPC_Factorial.py

Dependencies: None (uses Python stdlib xmlrpc)
"""

import math
import sys
import threading
import time
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client

HOST = "localhost"
PORT = 8001


# ---------------------------------------------------------------------------
# Server-side logic
# ---------------------------------------------------------------------------

def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial n! as an integer.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    return math.factorial(n)


def start_server():
    """Start the XML-RPC server and register the factorial function."""
    server = SimpleXMLRPCServer((HOST, PORT), logRequests=False, allow_none=True)
    server.register_function(factorial, "factorial")
    print(f"[Server] RPC Factorial Server started on {HOST}:{PORT}")
    print("[Server] Waiting for client requests... (Press Ctrl+C to stop)")
    server.serve_forever()


# ---------------------------------------------------------------------------
# Client-side logic
# ---------------------------------------------------------------------------

def start_client(n=None):
    """Connect to the RPC server and compute factorial(n).

    Args:
        n: The integer whose factorial to compute. If None, runs a demo.
    """
    time.sleep(1.0)  # Wait briefly for server to start
    proxy = xmlrpc.client.ServerProxy(f"http://{HOST}:{PORT}/")
    print("\n--- RPC Client ---")
    if n is not None:
        result = proxy.factorial(int(n))
        print(f"factorial({n}) = {result}")
    else:
        # Demo: compute factorial for several values
        for num in [0, 1, 5, 7, 10, 12]:
            result = proxy.factorial(num)
            print(f"  factorial({num:>2}) = {result}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse CLI arguments and run server, client, or demo mode."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "server":
            start_server()
        elif mode == "client":
            n = sys.argv[2] if len(sys.argv) > 2 else 5
            start_client(n)
        else:
            print(f"Usage: python {sys.argv[0]} [server | client <n>]")
            sys.exit(1)
    else:
        # Demo mode: server in background thread, client in main thread
        print("--- RPC Factorial Demo (Server + Client in same process) ---")
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        start_client()


if __name__ == "__main__":
    main()

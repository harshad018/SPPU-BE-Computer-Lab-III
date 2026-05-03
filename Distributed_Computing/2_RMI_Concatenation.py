"""
Practical 2: RMI String Concatenation
============================================================
Distributed application that simulates Java RMI-style remote method
invocation using Python's XML-RPC framework.  A client submits two
strings to a remote service object; the server concatenates them and
returns the result.

Key concept: Unlike raw RPC (which exposes bare functions), this example
registers an *object instance* on the server — mirroring the OO nature
of Java RMI where you invoke methods on a remote object.

Usage:
    # Terminal 1 – Start the server:
    python 2_RMI_Concatenation.py server

    # Terminal 2 – Concatenate two strings:
    python 2_RMI_Concatenation.py client "Hello" "World"

    # Demo mode (server + client run in same process via threads):
    python 2_RMI_Concatenation.py

Dependencies: None (uses Python stdlib xmlrpc)
"""

import sys
import threading
import time
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client

HOST = "localhost"
PORT = 8002


# ---------------------------------------------------------------------------
# Remote service object (analogous to a Java Remote interface + implementation)
# ---------------------------------------------------------------------------

class StringService:
    """Remote string-manipulation service (simulates a Java RMI remote object)."""

    def concatenate(self, str1: str, str2: str) -> str:
        """Concatenate two strings directly.

        Args:
            str1: First string.
            str2: Second string.

        Returns:
            str1 + str2
        """
        return str1 + str2

    def concatenate_with_separator(self, str1: str, str2: str, sep: str = " ") -> str:
        """Concatenate two strings with an optional separator.

        Args:
            str1: First string.
            str2: Second string.
            sep:  Separator character (default: single space).

        Returns:
            str1 + sep + str2
        """
        return str1 + sep + str2

    def reverse_concat(self, str1: str, str2: str) -> str:
        """Concatenate strings in reverse order (str2 first, then str1).

        Args:
            str1: First string.
            str2: Second string.

        Returns:
            str2 + str1
        """
        return str2 + str1


# ---------------------------------------------------------------------------
# Server-side logic
# ---------------------------------------------------------------------------

def start_server():
    """Start the XML-RPC server with the StringService object registered."""
    server = SimpleXMLRPCServer((HOST, PORT), logRequests=False, allow_none=True)
    server.register_instance(StringService())
    print(f"[Server] RMI String Service started on {HOST}:{PORT}")
    print("[Server] Waiting for client requests... (Press Ctrl+C to stop)")
    server.serve_forever()


# ---------------------------------------------------------------------------
# Client-side logic
# ---------------------------------------------------------------------------

def start_client(str1: str = None, str2: str = None):
    """Connect to the RMI server and invoke string operations.

    Args:
        str1: First string (None triggers demo mode).
        str2: Second string.
    """
    time.sleep(1.0)  # Wait for server to start
    proxy = xmlrpc.client.ServerProxy(f"http://{HOST}:{PORT}/")

    print("\n--- RMI Client (String Service) ---")
    if str1 is not None and str2 is not None:
        result = proxy.concatenate(str1, str2)
        print(f'  concatenate("{str1}", "{str2}") = "{result}"')
        sep_result = proxy.concatenate_with_separator(str1, str2, "-")
        print(f'  concatenate_with_separator("{str1}", "{str2}", "-") = "{sep_result}"')
        rev = proxy.reverse_concat(str1, str2)
        print(f'  reverse_concat("{str1}", "{str2}") = "{rev}"')
    else:
        # Demo mode
        pairs = [("Hello", "World"), ("SPPU", "Lab"), ("Distributed", "Computing")]
        for s1, s2 in pairs:
            print(f'  concatenate("{s1}", "{s2}") = "{proxy.concatenate(s1, s2)}"')
            print(f'  concatenate_with_separator("{s1}", "{s2}", "-") = '
                  f'"{proxy.concatenate_with_separator(s1, s2, "-")}"')
            print(f'  reverse_concat("{s1}", "{s2}") = "{proxy.reverse_concat(s1, s2)}"')
            print()


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
            s1 = sys.argv[2] if len(sys.argv) > 2 else "Hello"
            s2 = sys.argv[3] if len(sys.argv) > 3 else "World"
            start_client(s1, s2)
        else:
            print(f"Usage: python {sys.argv[0]} [server | client <str1> <str2>]")
            sys.exit(1)
    else:
        print("--- RMI String Concatenation Demo (Server + Client in same process) ---")
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        start_client()


if __name__ == "__main__":
    main()

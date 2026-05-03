# Note: Requires Pyro4 library (pip install Pyro4)
# Run the server first, then run the client in a separate terminal.

import Pyro4
import threading
import time
import sys

# --- SERVER CODE ---
@Pyro4.expose
class Calculator(object):
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            return "Error: Division by zero"
        return a / b

def start_server():
    daemon = Pyro4.Daemon()                # make a Pyro daemon
    uri = daemon.register(Calculator)      # register the calculator as a Pyro object
    print(f"Ready. Object uri = {uri}")    # print the uri so we can use it in the client later
    
    # Save URI to a file so client can read it
    with open("calculator_uri.txt", "w") as f:
        f.write(str(uri))
        
    daemon.requestLoop()                   # start the event loop of the server to wait for calls

# --- CLIENT CODE ---
def start_client():
    time.sleep(2) # Wait for server to start
    try:
        with open("calculator_uri.txt", "r") as f:
            uri = f.read().strip()
    except FileNotFoundError:
        print("URI file not found. Ensure server is running.")
        return

    calc = Pyro4.Proxy(uri)         # get a Pyro proxy to the calculator object
    
    print("\n--- RMI Client ---")
    print("5 + 3 =", calc.add(5, 3))
    print("10 - 4 =", calc.subtract(10, 4))
    print("6 * 7 =", calc.multiply(6, 7))
    print("8 / 2 =", calc.divide(8, 2))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        start_client()
    elif len(sys.argv) > 1 and sys.argv[1] == "server":
        start_server()
    else:
        print("Running both server and client in threads for demonstration...")
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()
        
        start_client()

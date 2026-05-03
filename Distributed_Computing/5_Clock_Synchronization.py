import threading
import time
import random

# Berkeley Algorithm for Clock Synchronization Simulation

class Node:
    def __init__(self, node_id, time_offset):
        self.node_id = node_id
        # Simulate local clock with an offset
        self.local_time = time.time() + time_offset

    def get_time(self):
        # Update local time to current real time + offset
        return self.local_time + (time.time() - self.local_time)

    def set_time(self, new_time):
        self.local_time = new_time
        print(f"Node {self.node_id} clock adjusted to: {time.ctime(self.local_time)}")

class TimeServer:
    def __init__(self, nodes):
        self.nodes = nodes

    def synchronize_clocks(self):
        print("--- Berkeley Algorithm Clock Synchronization ---")
        
        # 1. Master requests time from all nodes
        node_times = []
        for node in self.nodes:
            node_times.append(node.get_time())
            print(f"Node {node.node_id} reported time: {time.ctime(node_times[-1])}")

        # 2. Master calculates the average time
        average_time = sum(node_times) / len(node_times)
        print(f"\nCalculated Average Time: {time.ctime(average_time)}\n")

        # 3. Master sends the adjustment to all nodes
        for i, node in enumerate(self.nodes):
            offset = average_time - node_times[i]
            print(f"Sending offset {offset:.4f} seconds to Node {node.node_id}")
            node.set_time(node.get_time() + offset)

def main():
    # Create 3 nodes with random time offsets (between -50 and 50 seconds)
    nodes = [Node(i, random.uniform(-50, 50)) for i in range(3)]
    
    server = TimeServer(nodes)
    server.synchronize_clocks()

if __name__ == "__main__":
    main()

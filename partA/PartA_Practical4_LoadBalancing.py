# ============================================================
# FILE: PartA_Practical4_LoadBalancing.py
# STANDALONE FILE — No other files needed.
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# STEP 1 — Install required libraries (only once):
#   pip install matplotlib numpy
#   (random and time are Python built-ins — no install needed)
#
# OPTION A — Jupyter Notebook / Google Colab (RECOMMENDED):
#   1. Open Jupyter or Colab, create a new notebook
#   2. Paste the entire code into one cell
#   3. Press Shift+Enter
#   4. Output: Step-by-step request assignment logs for BOTH algorithms
#   5. Two bar charts appear inline comparing Round Robin vs Least Connections
#   6. 'load_balancing.png' is saved in current directory
#
# OPTION B — PyCharm:
#   1. Open this file in PyCharm
#   2. Install libraries if prompted (pip install matplotlib numpy)
#   3. Click ▶ Run
#   4. Terminal shows all request distribution logs
#   5. Bar chart opens as separate popup window
#
# OPTION C — Terminal:
#   python PartA_Practical4_LoadBalancing.py
#
# EXPECTED OUTPUT:
#   For each of 15 requests, shows which server got the request and active connections
#   Final table: Server name | Total Requests | Active Connections
#   Two-panel bar chart: Round Robin distribution vs Least Connections distribution
#   Round Robin: each server should get ~5 requests (15 requests / 3 servers)
#   Least Connections: distribution varies based on simulated release probability
#
# TO CHANGE NUMBER OF REQUESTS: Edit NUM_REQUESTS = 15 near the bottom
# ============================================================

# Import random to simulate random request processing times
import random
# Import time for displaying timestamps
import time
# Import matplotlib for visualization of load distribution
import matplotlib.pyplot as plt
# Import numpy for numerical operations
import numpy as np

# ============================================================
# SERVER CLASS — Represents a single server in the system
# ============================================================
class Server:
    """Represents a server node in the distributed system"""

    def __init__(self, server_id, capacity=100):
        # server_id: unique identifier for this server (e.g., "Server-1")
        self.server_id = server_id
        # active_connections: number of currently active client connections
        self.active_connections = 0
        # total_requests: total requests this server has handled
        self.total_requests = 0
        # capacity: maximum number of connections server can handle
        self.capacity = capacity

    def handle_request(self, request_id):
        """Simulates accepting and processing a client request"""
        # Increment active connections when request arrives
        self.active_connections += 1
        # Increment total request counter
        self.total_requests += 1
        # Print which server is handling which request
        print(f"  → {self.server_id} | Active Connections: {self.active_connections} | "
              f"Handling Request #{request_id}")

    def release_request(self):
        """Simulates completing a request (freeing up a connection slot)"""
        # Decrement active connections when request is completed
        if self.active_connections > 0:
            self.active_connections -= 1  # Connection freed after processing

    def __str__(self):
        # String representation of server state for easy printing
        return f"{self.server_id}(Active={self.active_connections}, Total={self.total_requests})"


# ============================================================
# ALGORITHM 1: ROUND ROBIN LOAD BALANCER
# ============================================================
class RoundRobinLoadBalancer:
    """
    Round Robin distributes requests to servers in cyclic order.
    Server 1 → Server 2 → Server 3 → Server 1 → Server 2 → ...
    Each server gets equal number of requests regardless of current load.
    """

    def __init__(self, servers):
        # Store the list of server objects
        self.servers = servers
        # current_index: tracks which server is next in rotation
        self.current_index = 0

    def get_next_server(self):
        """Returns the next server in the round-robin cycle"""
        # Select the server at current index position
        server = self.servers[self.current_index]
        # Move index to next server — wrap around using modulo
        self.current_index = (self.current_index + 1) % len(self.servers)
        # Return the selected server
        return server

    def distribute_request(self, request_id):
        """Distributes a single request using round robin selection"""
        # Get the next server in the cycle
        server = self.get_next_server()
        # Assign request to selected server
        server.handle_request(request_id)
        # Simulate some requests completing randomly (to vary active connections)
        if random.random() < 0.4:  # 40% chance that a request completes
            server.release_request()  # Free up one connection slot


# ============================================================
# ALGORITHM 2: LEAST CONNECTIONS LOAD BALANCER
# ============================================================
class LeastConnectionsLoadBalancer:
    """
    Least Connections assigns each request to the server
    with the fewest currently active connections.
    More intelligent than Round Robin — adapts to server load.
    """

    def __init__(self, servers):
        # Store the list of server objects
        self.servers = servers

    def get_least_loaded_server(self):
        """Returns the server with the minimum active connections"""
        # Use min() with key=lambda to find server with fewest active connections
        return min(self.servers, key=lambda s: s.active_connections)

    def distribute_request(self, request_id):
        """Distributes a single request to the least loaded server"""
        # Find the server with fewest active connections
        server = self.get_least_loaded_server()
        # Assign request to that server
        server.handle_request(request_id)
        # Simulate random request completions
        if random.random() < 0.4:  # 40% chance a connection frees up
            server.release_request()


# ============================================================
# SIMULATION FUNCTION
# ============================================================
def simulate_load_balancing(algorithm_name, load_balancer, num_requests=10):
    """
    Simulates distributing a number of client requests using the given algorithm.
    Returns statistics about the distribution.
    """
    print(f"\n{'='*60}")
    print(f"  ALGORITHM: {algorithm_name}")
    print(f"  Distributing {num_requests} requests across {len(load_balancer.servers)} servers")
    print(f"{'='*60}")

    # Loop through each request and distribute it
    for request_id in range(1, num_requests + 1):
        print(f"\nRequest #{request_id} arriving...")
        # Distribute this request using the load balancer
        load_balancer.distribute_request(request_id)

    # Print final server statistics after all requests
    print(f"\n{'='*60}")
    print(f"FINAL SERVER LOAD DISTRIBUTION ({algorithm_name}):")
    print(f"{'='*60}")
    print(f"{'Server':<15} {'Total Requests':>15} {'Active Connections':>20}")
    print(f"{'-'*52}")

    # Collect data for plotting
    server_names = []
    total_requests_list = []

    for server in load_balancer.servers:
        # Print each server's statistics
        print(f"{server.server_id:<15} {server.total_requests:>15} {server.active_connections:>20}")
        server_names.append(server.server_id)
        total_requests_list.append(server.total_requests)

    return server_names, total_requests_list


# ============================================================
# MAIN PROGRAM — Create servers and run both algorithms
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("  LOAD BALANCING SIMULATION")
    print("  Distributed Computing - Practical 4")
    print("="*60)

    # Define number of requests to simulate
    NUM_REQUESTS = 15

    # --- ROUND ROBIN SIMULATION ---
    # Create 3 fresh server objects for Round Robin
    rr_servers = [
        Server("Server-1"),  # Server 1 with default capacity
        Server("Server-2"),  # Server 2
        Server("Server-3"),  # Server 3
    ]
    # Create Round Robin load balancer with these servers
    rr_balancer = RoundRobinLoadBalancer(rr_servers)
    # Run the simulation and get statistics
    rr_names, rr_totals = simulate_load_balancing("ROUND ROBIN", rr_balancer, NUM_REQUESTS)

    # --- LEAST CONNECTIONS SIMULATION ---
    # Create 3 fresh server objects for Least Connections
    lc_servers = [
        Server("Server-A"),  # Server A
        Server("Server-B"),  # Server B
        Server("Server-C"),  # Server C
    ]
    # Create Least Connections load balancer with these servers
    lc_balancer = LeastConnectionsLoadBalancer(lc_servers)
    # Run the simulation and get statistics
    lc_names, lc_totals = simulate_load_balancing("LEAST CONNECTIONS", lc_balancer, NUM_REQUESTS)

    # ============================================================
    # VISUALIZATION — Bar charts comparing both algorithms
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Two side-by-side plots

    # Round Robin Bar Chart
    bars1 = ax1.bar(rr_names, rr_totals, color=['#4CAF50', '#2196F3', '#FF9800'])
    ax1.set_title('Round Robin - Request Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Servers')
    ax1.set_ylabel('Total Requests Handled')
    ax1.set_ylim(0, NUM_REQUESTS + 2)
    # Add value labels on top of each bar
    for bar, val in zip(bars1, rr_totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha='center', va='bottom', fontweight='bold')

    # Add ideal distribution line for Round Robin
    ideal = NUM_REQUESTS / len(rr_servers)  # Equal distribution would be this value
    ax1.axhline(y=ideal, color='red', linestyle='--', label=f'Ideal = {ideal:.1f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Least Connections Bar Chart
    bars2 = ax2.bar(lc_names, lc_totals, color=['#9C27B0', '#F44336', '#00BCD4'])
    ax2.set_title('Least Connections - Request Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Servers')
    ax2.set_ylabel('Total Requests Handled')
    ax2.set_ylim(0, NUM_REQUESTS + 2)
    # Add value labels on top of each bar
    for bar, val in zip(bars2, lc_totals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 str(val), ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add main title and save
    plt.suptitle('Load Balancing Algorithm Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('load_balancing.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved as 'load_balancing.png'")


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Server class models each server with connection counters
# 2. RoundRobinLoadBalancer cycles through servers in fixed order (1→2→3→1→...)
#    using current_index and modulo arithmetic
# 3. LeastConnectionsLoadBalancer dynamically picks server with min active connections
#    using Python's min() with a lambda key function
# 4. simulate_load_balancing() sends NUM_REQUESTS requests through the chosen algorithm
# 5. With 40% random release probability, active_connections vary — showing the
#    difference between the two algorithms in dynamic scenarios
# 6. Bar charts compare total requests per server for both algorithms
# Key insight: Round Robin gives equal distribution regardless of actual load.
# Least Connections adapts to actual server load — better for varying request durations.
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Load Balancing in Distributed Systems
# Load balancing distributes incoming requests across multiple servers to:
#   - Prevent server overload
#   - Reduce response time
#   - Improve overall system throughput
# Round Robin: Simple, cyclic, good for homogeneous servers
# Least Connections: Dynamic, considers actual load, better for varying request times
# Other algorithms: Weighted Round Robin, IP Hash, Random, Least Response Time
# Used in: Web servers (NGINX, HAProxy), cloud platforms (AWS ELB), CDNs
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is load balancing and why is it important?
# A1. Load balancing distributes incoming network traffic/requests across multiple
#     servers to ensure no single server is overwhelmed. Importance:
#     - Prevents server overload and single point of failure
#     - Improves response time and availability
#     - Enables horizontal scaling (add more servers)
#     - Increases fault tolerance — if one server fails, others continue

# Q2. Explain the working principle of the Round Robin algorithm.
# A2. Round Robin assigns requests to servers in a fixed cyclic order:
#     Request 1 → Server A, Request 2 → Server B, Request 3 → Server C,
#     Request 4 → Server A (cycle repeats)
#     Uses a pointer/index that increments with each request and wraps around.
#     Equal distribution regardless of server load or request processing time.
#
# Q3. How does the Least Connections algorithm dynamically allocate requests?
# A3. Least Connections tracks active (in-progress) connections for each server.
#     When a new request arrives:
#     1. Load balancer checks active connection count of ALL servers
#     2. Assigns request to server with MINIMUM active connections
#     3. When a request completes, server's count decreases
#     Adapts to varying request durations — busy servers get fewer new requests.
#
# Q4. Which load balancing algorithm is best suited for real-time applications?
# A4. Least Connections is generally better for real-time applications because:
#     - Requests may have varying processing times in real-time systems
#     - It considers actual server load, not just request count
#     - Prevents overloading slower servers
#     However, for very short-lived requests with equal processing time,
#     Round Robin is simpler and equally effective.
#
# Q5. What are the limitations of Round Robin?
# A5. (i) Ignores server load — a slow server gets the same requests as a fast one
#     (ii) No fault tolerance — sends requests to failed servers until updated
#     (iii) Ignores server capacity — different-spec servers treated equally
#     (iv) Long-running connections can cause imbalance over time
#
# Q6. What is Weighted Round Robin?
# A6. Weighted Round Robin assigns a weight to each server based on capacity.
#     Higher weight = more requests. Example: if Server A has weight 3 and
#     Server B has weight 1, then A gets 3 requests for every 1 request B gets.
#     Handles heterogeneous server environments better than standard Round Robin.
#
# Q7. What is Weighted Least Connections?
# A7. Weighted Least Connections combines weights with active connection counts.
#     Effective connections = actual_connections / weight
#     Server with lowest effective connection count receives the next request.
#     Balances both server capacity and current load simultaneously.
#
# Q8. What other load balancing algorithms exist in modern systems?
# A8. (i) IP Hash: Routes based on client IP — same client always goes to same server
#     (ii) Least Response Time: Considers both connections AND response time
#     (iii) Resource Based: Routes based on CPU/memory availability
#     (iv) Random: Assigns to random server (simple, effective for stateless services)
#     (v) Consistent Hashing: Used in distributed caches (e.g., Redis)
#
# Q9. What is the difference between hardware and software load balancing?
# A9. Hardware LB: Dedicated physical devices (F5 BIG-IP), very fast, expensive,
#     limited flexibility, handles millions of requests per second.
#     Software LB: Running as software on commodity hardware (NGINX, HAProxy),
#     flexible, cheaper, easier to configure, scales through additional instances.
#
# Q10. What is sticky session / session persistence in load balancing?
# A10. Sticky sessions ensure that all requests from a specific client go to the
#      same server for the duration of their session. Used when server stores
#      session state (like shopping cart). Implemented via: cookie-based routing,
#      IP hash, or session table. Conflicts with optimal load distribution.
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# Load balancing = distributing traffic across multiple servers to maximize:
# availability, throughput, and low latency.
#
# Key performance terms:
#   - Throughput: requests handled per second.
#   - Latency: response delay per request.
#   - Utilization: resource use (CPU/RAM/network).
#   - Tail latency (p95/p99): worst-case user experience metrics.
#
# Layer types:
#   - L4 LB (TCP/UDP): faster, packet/connection-level.
#   - L7 LB (HTTP): content-aware routing (URL, headers, cookies).
#
# Health checks:
#   - Passive: infer failure from request errors.
#   - Active: periodic probes (/health endpoint).
#
# Practical tradeoff:
#   - Round Robin: simple + fair count.
#   - Least Connections: adapts to long-running workloads.
#   - Weighted variants: support heterogeneous servers.
#
# Cloud relevance:
#   AWS ALB/NLB, GCP LB, Azure LB all implement these ideas with auto-scaling.
#
# Exam-ready line:
# "Modern distributed systems require load balancers not only for scale but for
# resilience, failover, zero-downtime maintenance, and traffic shaping."
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. What is failover in load balancing?
# A11. Automatic rerouting of traffic to healthy servers when one server becomes unavailable.
#
# Q12. Why does least-connections work better for variable request times?
# A12. It considers current active load, so busy nodes receive fewer new requests.
#
# Q13. What is auto-scaling relation with load balancing?
# A13. Auto-scaling adds/removes servers based on load metrics, and load balancer
#      immediately distributes traffic across the updated server pool.
#
# Q14. What is north-south vs east-west traffic?
# A14. North-south: client-to-service traffic entering/exiting data center.
#      East-west: service-to-service internal traffic in microservices.
#
# Q15. What is rate limiting at load balancer?
# A15. Restricting request rate per client/API key/IP to prevent abuse, overload,
#      and denial-of-service patterns.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) LOAD BALANCER POSITION
#    - Sits between clients and backend servers.
#    - Receives all incoming requests and forwards to selected server.
#
# 2) ROUND ROBIN (RR)
#    - Deterministic cyclic assignment.
#    - Best for similar servers and similar request durations.
#    - May become unfair under long-running uneven workloads.
#
# 3) LEAST CONNECTIONS (LC)
#    - Dynamic: checks active connections before assignment.
#    - Better when request durations vary.
#    - Requires accurate real-time connection tracking.
#
# 4) REAL-WORLD PRODUCTION CONCERNS
#    - Health checks, failover, sticky sessions, weighted routing, auto-scaling.
#    - SSL termination and WAF integration often done at load balancer layer.
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "The simulation compares static cyclic routing (RR) with dynamic load-aware
#     routing (LC), then visualizes request distribution to show why adaptive
#     algorithms perform better in variable-load environments."
# ============================================================

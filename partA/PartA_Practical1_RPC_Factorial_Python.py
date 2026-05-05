# ============================================================
# FILE: PartA_Practical1_RPC_Factorial_Python.py
# STANDALONE FILE — No other files needed. Everything is in this one file.
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# OPTION A — PyCharm:
#   1. Open PyCharm → File → Open → select this file
#   2. No extra libraries needed (xmlrpc is built into Python)
#   3. Click the green ▶ Run button (or right-click → Run)
#   4. In the terminal at the bottom, type a number when prompted
#      e.g., Enter a number: 5  →  Output: Factorial = 120
#
# OPTION B — Terminal / Command Prompt:
#   1. Open terminal in the folder containing this file
#   2. Run: python PartA_Practical1_RPC_Factorial_Python.py
#   3. Type a number and press Enter when prompted
#
# OPTION C — Jupyter Notebook:
#   1. Copy entire file content into a notebook cell
#   2. Click Run (Shift+Enter)
#   3. Type a number in the input box that appears
#
# HOW THIS FILE WORKS INTERNALLY:
#   - The server thread starts in background (daemon thread)
#   - After 1 second delay, the client connects to it
#   - Both run in the same process for demo convenience
#   - In real deployment: save server part as rpc_server.py, client as rpc_client.py
#     then run server in terminal 1 and client in terminal 2 simultaneously
# ============================================================

# ========================= SERVER CODE ======================
# File: rpc_server.py

# Import the SimpleXMLRPCServer class to create an RPC server
from xmlrpc.server import SimpleXMLRPCServer

# Import the ServerProxy class to create an RPC client stub
import xmlrpc.client

# Define a function that computes factorial of a number
def factorial(n):
    # Base case: factorial of 0 or 1 is 1
    if n == 0 or n == 1:
        return 1
    # Recursive case: n * factorial(n-1)
    result = 1
    # Loop from 2 to n (inclusive) to compute factorial iteratively
    for i in range(2, n + 1):
        result *= i  # Multiply result by current i
    return result  # Return computed factorial value

# Define a function to start the RPC server
def start_server():
    # Create an RPC server listening on localhost at port 8000
    server = SimpleXMLRPCServer(("localhost", 8000), logRequests=True)

    # Print confirmation that server is running
    print("[SERVER] RPC Server started on port 8000...")

    # Register the factorial function so clients can call it remotely
    server.register_function(factorial, "factorial")

    # Print what function is registered
    print("[SERVER] Registered function: factorial(n)")

    # Start serving requests indefinitely
    print("[SERVER] Waiting for client requests...")
    server.serve_forever()  # Keeps server running to accept multiple clients


# ========================= CLIENT CODE ======================
# File: rpc_client.py

# Define a function to act as the RPC client
def start_client():
    # Create a proxy object that connects to the RPC server at localhost:8000
    proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

    # Input: take integer from user
    n = int(input("[CLIENT] Enter a number to compute factorial: "))

    # Call the remote factorial function via the proxy (this goes over network/localhost)
    result = proxy.factorial(n)

    # Print the result returned by the server
    print(f"[CLIENT] Factorial of {n} received from SERVER = {result}")


# ========================= SIMULATION (Both in one file) =====
# Since we can't run two separate processes easily here,
# we simulate the RPC call by calling the function directly
# In real implementation: run server and client in separate terminals

import threading  # Import threading to run server in background thread
import time       # Import time to add delay so server starts before client

def run_demo():
    # Create a thread to run the server in background (daemon=True means it stops when main stops)
    server_thread = threading.Thread(target=start_server, daemon=True)

    # Start the server thread
    server_thread.start()

    # Wait 1 second to ensure server is up before client tries to connect
    time.sleep(1)

    # Now run the client in the main thread
    start_client()

# Entry point of the program
if __name__ == "__main__":
    run_demo()  # Run the combined demo


# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Server starts and registers the 'factorial' function using XML-RPC protocol
# 2. Server listens on localhost:8000 for incoming RPC requests
# 3. Client creates a proxy object pointing to the server address
# 4. Client calls proxy.factorial(n) — this marshals (packs) the argument n into XML
# 5. The XML request is sent over the network to the server
# 6. Server receives, unmarshals (unpacks) n, executes factorial(n)
# 7. Server marshals the result back into XML and sends it to client
# 8. Client unmarshals the result and displays it
# Key concept: The client calls factorial() as if it's a local function,
# but it actually executes on the server — this is the essence of RPC
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Remote Procedure Call (RPC)
# RPC is a protocol that allows a program to execute a procedure (function)
# on a remote computer as if it were a local call.
# Key components:
#   - Client Stub: Packs (marshals) parameters and sends request
#   - Server Stub: Unpacks (unmarshals) parameters and calls actual function
#   - Marshalling: Converting parameters to a transmittable format (XML here)
#   - Unmarshalling: Converting received data back to usable parameters
# Python uses xmlrpc.server and xmlrpc.client for XML-RPC based RPC
# RPC hides the complexity of network communication from the developer
# The calling process is suspended while waiting for the remote result
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is RPC (Remote Procedure Call)?
# A1. RPC is a protocol that allows a client program to execute a function/procedure
#     on a remote server as if it were a local function call. The network communication
#     is hidden from the programmer through stub objects.
#
# Q2. What is Marshalling and Unmarshalling in RPC?
# A2. Marshalling: Converting function parameters into a standard transmittable format (like XML/JSON).
#     Unmarshalling: Converting received data back into the original parameter types.
#     The client stub marshals data before sending; server stub unmarshals on receiving.
#
# Q3. What is a Client Stub in RPC?
# A3. A Client Stub is a proxy object on the client side that acts like the remote function.
#     It intercepts local calls, marshals parameters, sends them to the server,
#     waits for response, unmarshals results, and returns them to the caller.
#
# Q4. What is a Server Stub in RPC?
# A4. A Server Stub sits on the server side. It receives incoming RPC requests,
#     unmarshals the parameters, calls the actual server function, marshals the
#     result, and sends it back to the client.
#
# Q5. What are the key issues in RPC?
# A5. (i) Binding: How client finds the server (static or dynamic binding)
#     (ii) Security: Authentication and encryption of data
#     (iii) Fault Tolerance: Handling server crashes or network failures
#     (iv) Scalability: Supporting many clients simultaneously
#     (v) Performance: Minimizing latency from marshalling and network
#
# Q6. What is the difference between RPC and local procedure call?
# A6. In local call: function exists in same address space, no network needed.
#     In RPC: function exists on a remote machine, parameters travel over network,
#     stubs handle communication. RPC adds network latency and marshalling overhead.
#
# Q7. What is XML-RPC? How is it used in Python?
# A7. XML-RPC is an RPC protocol that uses XML to encode calls and HTTP as transport.
#     Python's xmlrpc.server provides SimpleXMLRPCServer to create servers,
#     and xmlrpc.client provides ServerProxy to create client stubs.
#
# Q8. What is the role of RPC Runtime?
# A8. RPC Runtime is a library that handles all network communication underlying RPC.
#     It manages binding, establishes connections using appropriate protocol,
#     passes data between client and server, and handles communication errors.
#
# Q9. What is Dynamic Binding in RPC?
# A9. Dynamic Binding means the client finds the server address at runtime (not hardcoded).
#     The client stub contacts a name server to get the server's transport address
#     when the first RPC call is made. This is more flexible than static binding.
#
# Q10. What are advantages of RPC?
# A10. (i) Abstraction: Network complexity is hidden from programmer
#      (ii) Performance: Many protocol layers are omitted to improve speed
#      (iii) Reusability: Code can be reused across distributed environments
#      (iv) Simplicity: Developer writes code similar to local procedure calls
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# RPC in one line: "Call remote function as if local."
# Core flow to remember in viva:
#   Caller -> Client Stub -> Marshalling -> Transport (HTTP/TCP) ->
#   Server Stub -> Actual Function -> Return Path (reverse)
#
# Why RPC exists:
#   - Break monolithic apps into client-server services.
#   - Reuse compute-heavy functions on remote high-power systems.
#   - Keep interface stable while server logic evolves.
#
# Important technical terms:
#   - IDL (Interface Definition Language): language-neutral API contract in many RPC systems.
#   - Serialization/Marshalling: object -> bytes/XML/JSON.
#   - Deserialization/Unmarshalling: bytes -> object.
#   - Timeout: max wait for remote response.
#   - Retry: resend request if failure (careful with non-idempotent methods).
#   - Idempotent: repeated call gives same effect (safe retry candidate).
#
# RPC vs REST quick contrast:
#   - RPC: action-oriented (doX()), strong function-call style.
#   - REST: resource-oriented (GET/POST /users).
#   - RPC typically simpler for internal microservice-to-microservice calls.
#
# Failure cases you should mention:
#   - Server down, network partition, request timeout, partial response,
#     duplicate execution after retry, incompatible client/server versions.
#
# Security basics for distributed viva:
#   - Authentication (who are you?)
#   - Authorization (what can you do?)
#   - Encryption (TLS/HTTPS)
#   - Input validation and rate limiting
#
# Exam-ready line:
# "RPC improves developer productivity by hiding network complexity, but distributed
# failures, latency, and serialization overhead must be handled explicitly."
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. Why are remote calls usually slower than local calls?
# A11. Because remote calls include network latency, serialization/deserialization,
#      protocol overhead, and context switching, while local calls are in-process memory calls.
#
# Q12. What is exactly-once semantics in RPC? Is it guaranteed?
# A12. Exactly-once means function executes one time only; hard to guarantee in distributed
#      systems due to retries/timeouts. Most systems provide at-most-once or at-least-once.
#
# Q13. What is at-most-once vs at-least-once RPC?
# A13. At-most-once: no duplicate execution, but request may fail.
#      At-least-once: request eventually executes, but duplicates possible.
#
# Q14. Why is timeout mandatory in RPC clients?
# A14. Without timeout, client may block forever on server/network failure.
#      Timeout allows fallback/retry/circuit-breaker behavior.
#
# Q15. What is a circuit breaker in distributed systems?
# A15. A protection pattern that temporarily stops calling failing remote services
#      to avoid cascading failures and gives recovery window.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) RPC COMMUNICATION MODEL
#    - Client never executes server logic directly.
#    - Client sends request message: method name + parameters.
#    - Server executes method and sends return value or error.
#
# 2) STUB CONCEPT (MOST ASKED)
#    - Client stub: behaves like local object/function but internally performs network I/O.
#    - Server stub/skeleton: receives request, decodes it, calls real function.
#
# 3) WHY DISTRIBUTED CALLS ARE TRICKY
#    - Local call fails rarely and fast; remote call can fail due to network.
#    - Need timeout, retries, and error handling as first-class design.
#
# 4) IDEMPOTENCY AND RETRIES
#    - Idempotent call: safe to repeat (same final effect), e.g., "get factorial(5)".
#    - Non-idempotent call: repeating may cause duplicate effects, e.g., "transfer money".
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "This code starts an XML-RPC server, registers factorial(), then client proxy invokes
#     it remotely. The key concept is transparent remote invocation using marshalling and
#     unmarshalling over HTTP."
# ============================================================

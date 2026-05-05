# ============================================================
# FILE: PartA_Practical2_RMI_Python.py
# STANDALONE FILE — No other files needed. Everything is in this one file.
#
# ── HOW TO RUN (Detailed) ──────────────────────────────────
# OPTION A — PyCharm:
#   1. Open PyCharm → Open this file
#   2. No extra libraries needed (xmlrpc is built into Python 3)
#   3. Click the green ▶ Run button
#   4. When prompted:
#      Enter first string:  Hello
#      Enter second string: World
#      Output: Result received from SERVER (via RMI): 'HelloWorld'
#
# OPTION B — Terminal:
#   1. Open terminal in folder containing this file
#   2. Run: python PartA_Practical2_RMI_Python.py
#   3. Type both strings when prompted
#
# OPTION C — Jupyter Notebook:
#   1. Paste entire content into a cell
#   2. Run cell (Shift+Enter)
#   3. Enter strings in the input boxes that appear
#
# HOW THIS FILE WORKS INTERNALLY:
#   - Server thread starts in background (daemon) on port 9000
#   - 1-second delay ensures server is ready before client connects
#   - Client proxy calls concatenate() remotely via XML-RPC protocol
#   - In real RMI (Java): 3 separate files are needed (see Java version)
#   - This Python version demonstrates the exact same RMI concept
# ============================================================

# Import SimpleXMLRPCServer — Python's built-in RPC server (simulates RMI)
from xmlrpc.server import SimpleXMLRPCServer
# Import xmlrpc.client to create client proxy (simulates RMI stub)
import xmlrpc.client
# Import threading to run server in background during demo
import threading
# Import time for delay between server start and client call
import time

# ========================= REMOTE OBJECT (SERVER SIDE) =====
# This class represents the "Remote Object" in RMI terminology
class StringService:
    """
    This class acts as the Remote Object on the server side.
    In Java RMI: this would implement a Remote Interface.
    The method concatenate() is the Remote Method being invoked.
    """

    # Define the remote method that clients can invoke
    def concatenate(self, str1, str2):
        # str1: first string sent by client
        # str2: second string sent by client
        print(f"[SERVER] Received: str1='{str1}', str2='{str2}'")

        # Perform the actual string concatenation on the server side
        result = str1 + str2

        # Print the result being sent back
        print(f"[SERVER] Concatenated result: '{result}'")

        # Return the result to the client (this travels over network in RMI)
        return result

# ========================= SERVER CODE ======================
def start_rmi_server():
    """
    Starts the RMI-like server (XML-RPC server simulating Java RMI registry)
    In Java RMI: LocateRegistry.createRegistry(1099) + Naming.rebind()
    """

    # Create an XML-RPC server on localhost port 9000
    server = SimpleXMLRPCServer(("localhost", 9000), logRequests=False)

    # Print server start message
    print("[SERVER] RMI-like Server started on port 9000")

    # Create an instance of the remote object (StringService)
    remote_obj = StringService()

    # Register the remote object's instance — all its methods become remotely callable
    # In Java RMI: Naming.rebind("//localhost/StringService", remote_obj)
    server.register_instance(remote_obj)

    # Print what service is registered (like RMI Registry)
    print("[SERVER] Registered Remote Object: StringService")
    print("[SERVER] Available Remote Method: concatenate(str1, str2)")
    print("[SERVER] Waiting for remote method invocations...")

    # Start serving requests in a loop
    server.serve_forever()

# ========================= CLIENT CODE ======================
def start_rmi_client():
    """
    RMI Client that looks up the remote object and invokes its method.
    In Java RMI: Naming.lookup("//localhost/StringService") returns stub.
    """

    # Create a proxy (stub) that connects to the RMI-like server
    # In Java RMI: proxy = (StringRemote) Naming.lookup("rmi://localhost/StringService")
    proxy = xmlrpc.client.ServerProxy("http://localhost:9000/")

    # Take two strings from user
    str1 = input("[CLIENT] Enter first string: ")
    str2 = input("[CLIENT] Enter second string: ")

    # Invoke the remote method on the proxy — looks like a local method call
    # Internally: stub marshals arguments, sends to server, waits for result
    print(f"[CLIENT] Invoking remote method: concatenate('{str1}', '{str2}')")
    result = proxy.concatenate(str1, str2)

    # Display the result received from the server (unmarshalled by stub)
    print(f"[CLIENT] Result received from SERVER (via RMI): '{result}'")

# ========================= DEMO RUNNER ======================
def run_demo():
    # Create a background thread for the server (daemon so it stops when main ends)
    server_thread = threading.Thread(target=start_rmi_server, daemon=True)

    # Start the server thread
    server_thread.start()

    # Wait 1 second for server to initialize before client connects
    time.sleep(1)

    # Run the client in the main thread
    start_rmi_client()

# Entry point of the program
if __name__ == "__main__":
    run_demo()  # Start both server and client

# ============================================================
# HOW THE ENTIRE CODE WORKS:
# 1. Server creates a StringService remote object and registers it with XML-RPC server
# 2. Server starts listening on port 9000 for remote method invocations
# 3. Client creates a ServerProxy (stub) pointing to the server address
# 4. Client calls proxy.concatenate(str1, str2) — looks like a local method call
# 5. Stub marshals str1 and str2 into XML format and sends HTTP request to server
# 6. Server receives request, unmarshals parameters, calls concatenate() method
# 7. Method concatenates the strings and returns the result
# 8. Server marshals result back into XML and sends HTTP response to client
# 9. Client stub unmarshals the result and returns it to the calling code
# 10. Client displays the result
# This demonstrates the core RMI concept: method invocation across JVMs/processes
# ============================================================

# ============================================================
# ABOUT THIS PRACTICAL:
# Topic: Remote Method Invocation (RMI)
# RMI is Java's mechanism for invoking methods on objects in another JVM.
# Key components:
#   - Remote Interface: Defines which methods can be called remotely
#   - Remote Object: Server-side class implementing the Remote Interface
#   - Stub: Client-side proxy representing the remote object
#   - Skeleton: Server-side handler that receives calls and dispatches to object
#   - RMI Registry: Name service for looking up remote objects (like DNS for objects)
# In Java 2 SDK+: Skeleton is no longer needed (stub protocol updated)
# Python equivalent uses XML-RPC to demonstrate the same concept
# The key idea: client calls methods as if object were local, but execution is remote
# ============================================================

# ============================================================
# VIVA QUESTIONS AND ANSWERS:
#
# Q1. What is RMI (Remote Method Invocation)?
# A1. RMI is a Java API that allows a program to invoke methods on an object
#     running in a different JVM (possibly on a different machine).
#     It uses stub and skeleton objects for communication.
#     Key package: java.rmi
#
# Q2. What is the role of Stub in RMI?
# A2. Stub is the client-side proxy object that represents the remote object.
#     When client calls a method on the stub, the stub:
#     (i) Initiates connection to remote JVM
#     (ii) Marshals (writes/transmits) parameters
#     (iii) Waits for result
#     (iv) Unmarshals the return value or exception
#     (v) Returns value to caller

# Q3. What is the role of Skeleton in RMI?
# A3. Skeleton is the server-side handler that:
#     (i) Reads parameters for the remote method
#     (ii) Invokes the actual method on the remote object
#     (iii) Marshals the result to send back to client
#     Note: In Java 2 SDK and later, skeletons are no longer needed.
#
# Q4. What is RMI Registry?
# A4. RMI Registry is a naming service (like a phone book) for remote objects.
#     Server registers remote objects with a name: Naming.rebind("//host/name", obj)
#     Client looks up objects by name: Naming.lookup("rmi://host/name")
#     Default port: 1099. Started using: rmiregistry command.
#
# Q5. What are the 6 steps to write an RMI program in Java?
# A5. 1. Create the Remote Interface (extends java.rmi.Remote)
#     2. Implement the Remote Interface on the server
#     3. Compile with rmic tool to generate stub and skeleton
#     4. Start RMI Registry: rmiregistry
#     5. Create and start the Remote (Server) application
#     6. Create and start the Client application
#
# Q6. How is RMI different from RPC?
# A6. RMI: Java-specific, object-oriented, full object serialization support
#     RPC: Language-neutral, procedure-oriented, simpler data types
#     RMI supports passing objects by value (serialization) and by reference
#     RPC typically only passes primitive types and simple structures
#
# Q7. What is marshalling in RMI?
# A7. Marshalling is the process of converting Java objects into a byte stream
#     for transmission over the network. In RMI, Java's object serialization
#     is used for marshalling. On the receiving end, unmarshalling reconstructs
#     the original Java objects from the byte stream.
#
# Q8. What requirements must a class meet to be a Remote Object in RMI?
# A8. (i) Must implement an interface that extends java.rmi.Remote
#     (ii) Remote interface methods must throw java.rmi.RemoteException
#     (iii) Server class must extend UnicastRemoteObject OR call exportObject()
#     (iv) Class must be serializable if passed by value between JVMs
#
# Q9. What is the difference between passing by value and by reference in RMI?
# A9. Pass by value: Object is serialized, copied to other JVM — changes don't affect original
#     Pass by reference: Remote stub is passed — method calls go back to original object
#     Primitive types and non-remote objects are passed by value in RMI
#     Remote objects are passed by reference (stub is passed)
#
# Q10. What is distributed application? How does RMI enable it?
# A10. A distributed application runs across multiple networked machines.
#      RMI enables it by: (i) Locating remote methods via Registry
#      (ii) Providing communication with remote objects via stubs
#      (iii) Loading class definitions dynamically from remote locations
#      This allows Java programs to collaborate across a network transparently.
# ============================================================

# ============================================================
# RAPID REVISION BOOSTER (DETAILED TOPIC NOTES):
# RMI concept:
#   Remote Method Invocation allows Java objects in different JVMs to communicate
#   as if method calls are local object calls.
#
# Architecture memory trick:
#   Remote Interface -> Remote Object Implementation -> Registry Binding ->
#   Client Lookup -> Stub Call -> Network -> Server Execution -> Return.
#
# Key classes in Java RMI:
#   - java.rmi.Remote
#   - java.rmi.RemoteException
#   - java.rmi.registry.LocateRegistry
#   - java.rmi.Naming
#   - java.rmi.server.UnicastRemoteObject
#
# Best viva explanation:
# "RMI is object-oriented distributed computing, while basic RPC is procedure-oriented."
#
# Practical deployment note:
#   - Server and client may run on different hosts.
#   - Registry typically runs on port 1099.
#   - Firewalls and network policies must allow registry + service ports.
#
# Common interview/viva pitfalls:
#   - Forgetting RemoteException in interface methods.
#   - Not exporting remote object.
#   - Version mismatch in serialized classes.
#   - DNS/host resolution failures in lookup URL.
#
# Security concern:
#   RMI should run over secure channels and controlled networks.
#   Never expose sensitive remote methods without auth checks.
# ============================================================

# ============================================================
# ADDITIONAL VIVA Q&A (HIGH-VALUE):
# Q11. Why must remote methods throw RemoteException?
# A11. Because network calls can fail unpredictably (timeout, disconnect, routing issues),
#      and RemoteException standardizes communication failure handling.
#
# Q12. What is the role of rmiregistry?
# A12. It is a naming service that maps logical service names to remote object references.
#      Clients use it to discover service endpoints dynamically.
#
# Q13. What is pass-by-value in RMI serialization?
# A13. Object state is copied and sent to another JVM; receiver gets a separate copy.
#      Modifying copied object does not affect original object.
#
# Q14. What is pass-by-reference in RMI?
# A14. For remote objects, a stub reference is sent so method invocations still execute
#      on the original remote JVM object.
#
# Q15. How does RMI support scalability?
# A15. Through distributed object deployment across multiple servers with registry-based
#      discovery, load balancing, and horizontal scaling strategies.
# ============================================================

# ============================================================
# SUBTOPIC DEEP DIVE (READ BEFORE VIVA):
# 1) RMI = OBJECT-ORIENTED REMOTE CALLING
#    - RPC calls functions.
#    - RMI calls methods on remote objects.
#    - This is why remote interface + object identity matter.
#
# 2) RMI LIFECYCLE
#    - Define remote interface.
#    - Implement interface on server object.
#    - Export/bind object in registry.
#    - Client lookup to obtain stub.
#    - Client invokes method on stub.
#
# 3) SERIALIZATION ROLE
#    - Method arguments/return values are serialized across JVM/process boundaries.
#    - Class compatibility/versioning issues can break remote calls.
#
# 4) COMMON FAILURE QUESTIONS
#    - Registry not running.
#    - Wrong host/port.
#    - RemoteException due to timeout/disconnect.
#    - ClassNotFound for serialized payload types.
#
# 5) PRACTICAL EXAM EXPLANATION TEMPLATE
#    "The server hosts a remote string concatenation service; client obtains proxy and
#     calls concatenate() as if local. Under the hood request data is marshalled, sent,
#     executed remotely, and response is unmarshalled."
# ============================================================

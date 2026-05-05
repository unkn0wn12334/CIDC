// ============================================================
// CLUSTER: Practical 1 Java RPC — FILES IN THIS CLUSTER:
//   1. RPCServer.java  ← Start this FIRST (see full run steps there)
//   2. RPCClient.java  ← YOU ARE HERE — run AFTER server is running
//
// SHORT RUN STEPS:
//   Terminal 1: javac RPCServer.java RPCClient.java  (compile once)
//               java RPCServer                       (start server)
//   Terminal 2: java RPCClient                       (run client)
//               Enter a number when prompted
// ============================================================

// File: RPCClient.java

// Import Socket to connect to the RPC server
import java.net.Socket;
// Import DataInputStream to receive data from server
import java.io.DataInputStream;
// Import DataOutputStream to send data to server
import java.io.DataOutputStream;
// Import IOException to handle connection errors
import java.io.IOException;
// Import Scanner to take user input
import java.util.Scanner;

// Define the RPCClient class
public class RPCClient {

    // Main method — entry point of the client program
    public static void main(String[] args) {

        // Define server address and port to connect to
        String serverAddress = "localhost"; // Server is on same machine
        int port = 8000;                   // Same port as server

        // Create Scanner to read user input from keyboard
        Scanner scanner = new Scanner(System.in);

        // Ask user to enter a number whose factorial is needed
        System.out.print("[CLIENT] Enter a number to compute factorial: ");
        int n = scanner.nextInt(); // Read the integer input

        try {
            // Create a Socket to connect to the RPC server
            // This is equivalent to the "client stub initiating connection"
            Socket socket = new Socket(serverAddress, port);
            System.out.println("[CLIENT] Connected to RPC Server at " + serverAddress + ":" + port);

            // Create DataOutputStream to send the integer n to the server
            DataOutputStream dos = new DataOutputStream(socket.getOutputStream());

            // Create DataInputStream to receive the result from the server
            DataInputStream dis = new DataInputStream(socket.getInputStream());

            // Send the integer n to the server — this is "marshalling" the argument
            dos.writeInt(n);
            System.out.println("[CLIENT] Sent n = " + n + " to server (marshalled)");

            // Wait and read the result from server — this is "unmarshalling" the return value
            long result = dis.readLong();

            // Display the result returned by the server
            System.out.println("[CLIENT] Factorial of " + n + " received from SERVER = " + result);

            // Close the streams and socket after communication is done
            dos.close();
            dis.close();
            socket.close();

        } catch (IOException e) {
            // Print error if connection to server fails
            System.out.println("[CLIENT] Error connecting to server: " + e.getMessage());
        }

        // Close the scanner
        scanner.close();
    }
}

// ============================================================
// HOW THE ENTIRE CLIENT CODE WORKS:
// 1. Scanner reads integer n from user input
// 2. Socket connects to RPCServer at localhost:8000 (client stub connecting)
// 3. DataOutputStream sends integer n to server (marshalling the argument)
// 4. Client waits for server to compute and respond (suspended during execution)
// 5. DataInputStream reads the long result from server (unmarshalling)
// 6. Result is displayed to user
// 7. All connections are closed
// This simulates the "client stub" part of RPC — client thinks it's calling factorial locally,
// but the execution actually happens on the server
// ============================================================

// ============================================================
// ABOUT THIS PRACTICAL:
// Topic: Remote Procedure Call (RPC) using Java Socket Programming
// RPC allows a client to invoke a function on a remote server transparently.
// In this Java implementation:
//   - Socket programming replaces the XML transport (used in Python version)
//   - DataInputStream/DataOutputStream handle marshalling/unmarshalling of data
//   - Server computes factorial and returns result over the socket
//   - Client sends integer, receives result — simulating remote invocation
// In production RPC systems (like Java RMI, gRPC), stubs are auto-generated
// from interface definition files, and marshalling is handled automatically
// ============================================================

// ============================================================
// VIVA QUESTIONS AND ANSWERS:
//
// Q1. What is RPC (Remote Procedure Call)?
// A1. RPC is a protocol that allows a client program to execute a function/procedure
//     on a remote server as if it were a local function call. The network communication
//     is hidden from the programmer through stub objects.
//
// Q2. What is Marshalling and Unmarshalling in RPC?
// A2. Marshalling: Converting function parameters into a standard transmittable format.
//     Unmarshalling: Converting received data back into the original parameter types.
//     In Java: DataOutputStream.writeInt() = marshalling, DataInputStream.readLong() = unmarshalling.
//
// Q3. What is a Client Stub in RPC?
// A3. The client stub is a proxy that looks like the actual remote function.
//     In Java: The Socket + DataOutputStream combination acts as the client stub.
//     It packs data, sends to server, waits for result, and returns it.
//
// Q4. How does the server know which procedure to execute in RPC?
// A4. The client sends a procedure identifier (method name or ID) along with parameters.
//     The server stub reads this identifier and calls the appropriate server function.
//     In our simplified version, there's only one function (factorial), so it's implicit.
//
// Q5. What is the difference between RPC and RMI?
// A5. RPC: Language-neutral, procedure-oriented, used across different languages.
//     RMI: Java-specific, object-oriented, allows calling methods on remote Java objects.
//     RMI uses interfaces and stubs generated by rmic tool; RPC uses simpler protocol.
//
// Q6. What are call semantics in RPC?
// A6. Call semantics define what happens if a message is lost or server crashes:
//     - Maybe: Request sent once, no guarantee of execution
//     - At least once: Request retried until acknowledged (may execute multiple times)
//     - At most once: Duplicates filtered, executes 0 or 1 times
//     - Exactly once: Guaranteed to execute exactly once (hardest to implement)
//
// Q7. What is Binding in RPC?
// A7. Binding is the process of connecting the client to the server.
//     Static Binding: Server address is hardcoded (like localhost:8000 in our code).
//     Dynamic Binding: Client contacts a Name Server/Registry at runtime to find server.
//
// Q8. What are the security considerations in RPC?
// A8. (i) Authentication: Verify identity of client before allowing execution
//     (ii) Encryption: Encrypt data transmitted between client and server
//     (iii) Authorization: Check if client has permission to call the procedure
//     (iv) Input Validation: Validate parameters to prevent injection attacks
//
// Q9. What is the role of ServerSocket in Java RPC implementation?
// A9. ServerSocket listens on a specific port for incoming client connections.
//     server.accept() blocks until a client connects, then returns a Socket object
//     representing that connection, through which data can be exchanged.
//
// Q10. How does RPC differ from message passing?
// A10. RPC: Synchronous, client blocks waiting for result, looks like a function call.
//      Message Passing: Asynchronous, client sends message and continues,
//      result received later. RPC is simpler for request-response patterns.
// ============================================================

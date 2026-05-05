// ============================================================
// CLUSTER: Practical 1 Java RPC — FILES IN THIS CLUSTER:
//   1. RPCServer.java  ← YOU ARE HERE (PRIMARY — read full run steps here)
//   2. RPCClient.java  ← Run after server is running
//
// ── HOW TO RUN (Detailed) ─────────────────────────────────
// PRE-REQUISITE: Java JDK installed. Check: java -version in terminal.
//
// STEP 1 — Compile both files (do this once):
//   Open terminal in the folder containing RPCServer.java and RPCClient.java
//   Run: javac RPCServer.java RPCClient.java
//   This creates RPCServer.class and RPCClient.class in the same folder
//
// STEP 2 — Start the Server (Terminal Window 1):
//   Run: java RPCServer
//   You will see: [SERVER] RPC Server started on port 8000
//                 [SERVER] Waiting for client connection...
//   Leave this terminal open and running — server waits for client
//
// STEP 3 — Run the Client (Terminal Window 2 — open a NEW terminal):
//   Navigate to the same folder
//   Run: java RPCClient
//   You will see: [CLIENT] Enter a number to compute factorial:
//   Type a number (e.g., 5) and press Enter
//   Output: [CLIENT] Factorial of 5 received from SERVER = 120
//
// STEP 4 — Check Server Terminal:
//   Server terminal will show: [SERVER] Received n = 5, Result = 120
//
// NOTE: Server handles ONE client then closes. Run java RPCServer again for next client.
// ============================================================

// ========================= SERVER CODE ======================
// File: RPCServer.java

// Import ServerSocket to create a server that listens for connections
import java.net.ServerSocket;
// Import Socket to handle individual client connections
import java.net.Socket;
// Import DataInputStream to read data sent by client
import java.io.DataInputStream;
// Import DataOutputStream to send data back to client
import java.io.DataOutputStream;
// Import IOException for handling input/output errors
import java.io.IOException;

// Define the RPCServer class
public class RPCServer {

    // Define the static factorial method that server will execute remotely
    public static long factorial(int n) {
        // Base case: if n is 0 or 1, return 1
        if (n <= 1) return 1;
        long result = 1; // Initialize result variable
        // Loop from 2 to n to compute factorial iteratively
        for (int i = 2; i <= n; i++) {
            result *= i; // Multiply result by current i
        }
        return result; // Return the computed factorial
    }

    // Main method — entry point of the server program
    public static void main(String[] args) {
        // Define port number on which server will listen
        int port = 8000;

        try {
            // Create a ServerSocket that listens on port 8000
            ServerSocket serverSocket = new ServerSocket(port);
            System.out.println("[SERVER] RPC Server started on port " + port);
            System.out.println("[SERVER] Waiting for client connection...");

            // Accept a client connection — this blocks until client connects
            Socket clientSocket = serverSocket.accept();
            System.out.println("[SERVER] Client connected: " + clientSocket.getInetAddress());

            // Create DataInputStream to read the integer sent by client
            DataInputStream dis = new DataInputStream(clientSocket.getInputStream());

            // Create DataOutputStream to send the result back to client
            DataOutputStream dos = new DataOutputStream(clientSocket.getOutputStream());

            // Read the integer value sent by the client (this is the RPC argument)
            int n = dis.readInt();
            System.out.println("[SERVER] Received n = " + n + " from client");

            // Call the factorial function — this is the actual remote procedure execution
            long result = factorial(n);
            System.out.println("[SERVER] Computed factorial(" + n + ") = " + result);

            // Send the result back to the client (this is the RPC return value)
            dos.writeLong(result);
            System.out.println("[SERVER] Result sent to client");

            // Close all connections after communication
            dis.close();
            dos.close();
            clientSocket.close();
            serverSocket.close();

        } catch (IOException e) {
            // Print error if connection fails
            System.out.println("[SERVER] Error: " + e.getMessage());
        }
    }
}

// ============================================================
// HOW THE ENTIRE SERVER CODE WORKS:
// 1. ServerSocket is created on port 8000 to listen for incoming client connections
// 2. server.accept() blocks and waits until a client connects
// 3. Once connected, DataInputStream reads the integer n from the client
// 4. The factorial(n) method is executed on the server side
// 5. DataOutputStream sends the computed result back to the client
// 6. All streams and sockets are closed after communication
// This simulates the "server stub" and "server procedure" parts of RPC
// ============================================================

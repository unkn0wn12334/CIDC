// ============================================================
// CLUSTER: Practical 2 Java RMI — FILES IN THIS CLUSTER:
//   1. StringRemote.java ← Remote Interface (compile first, no run needed)
//   2. StringServer.java ← YOU ARE HERE (PRIMARY — full run steps here)
//   3. StringClient.java ← Run after server is ready
//
// ── HOW TO RUN (Detailed) ─────────────────────────────────
// PRE-REQUISITE: Java JDK installed. All 3 .java files in the SAME folder.
//
// STEP 1 — Compile all three files (open terminal in the folder, run once):
//   javac StringRemote.java StringServer.java StringClient.java
//   → Creates StringRemote.class, StringServer.class, StringClient.class
//   → If "error: package does not exist" — check all 3 .java files are present
//
// STEP 2 — Start the RMI Server (Terminal Window 1):
//   java StringServer
//   Expected output:
//     [SERVER] RMI Registry created on port 1099
//     [SERVER] Remote Object (StringServer) created
//     [SERVER] Object bound to RMI Registry as 'StringService'
//     [SERVER] Server is ready and waiting for client calls...
//   Leave this terminal running — server waits for clients indefinitely
//
// STEP 3 — Run the Client (open a NEW Terminal Window 2):
//   cd /same/folder/as/server
//   java StringClient
//   Expected:
//     [CLIENT] Successfully connected to RMI Server
//     [CLIENT] Enter first string:   → type e.g. "Hello "
//     [CLIENT] Enter second string:  → type e.g. "World"
//     [CLIENT] Result from server: 'Hello World'
//
// STEP 4 — Check Server Terminal to see server-side logs:
//     [SERVER] Remote method invoked!
//     [SERVER] Received str1: 'Hello ', str2: 'World'
//     [SERVER] Returning result: 'Hello World'
//
// TROUBLESHOOTING:
//   "Connection refused" → Server not running yet, complete Step 2 first
//   "Port 1099 in use"  → Kill existing rmiregistry:
//     Linux/Mac: kill -9 $(lsof -ti:1099)
//     Windows:   netstat -ano | findstr 1099  then  taskkill /PID <id> /F
// ============================================================

// File 2: StringServer.java (Remote Object Implementation + Server)

// Import Remote interface
import java.rmi.Remote;
// Import RemoteException for handling network errors
import java.rmi.RemoteException;
// Import UnicastRemoteObject — base class for exportable remote objects
import java.rmi.server.UnicastRemoteObject;
// Import Naming to register the remote object with RMI Registry
import java.rmi.Naming;
// Import LocateRegistry to create the RMI Registry programmatically
import java.rmi.registry.LocateRegistry;

// StringServer implements the StringRemote interface (the Remote Interface)
// Extends UnicastRemoteObject to make it exportable over the network
public class StringServer extends UnicastRemoteObject implements StringRemote {

    // Constructor must throw RemoteException because UnicastRemoteObject requires it
    public StringServer() throws RemoteException {
        super(); // Call parent class constructor to export this remote object
    }

    // Implement the remote method defined in StringRemote interface
    // This method runs on the SERVER when client invokes it remotely
    @Override
    public String concatenate(String str1, String str2) throws RemoteException {
        // Print the received strings on the server side
        System.out.println("[SERVER] Remote method invoked!");
        System.out.println("[SERVER] Received str1: '" + str1 + "'");
        System.out.println("[SERVER] Received str2: '" + str2 + "'");

        // Perform concatenation — the actual remote computation
        String result = str1 + str2;

        // Print what result is being sent back
        System.out.println("[SERVER] Returning result: '" + result + "'");

        // Return the concatenated result to the client (marshalled by RMI)
        return result;
    }

    // Main method — starts the RMI server
    public static void main(String[] args) {
        try {
            // Create the RMI Registry on port 1099 (default RMI port)
            // Equivalent to running "rmiregistry" command in terminal
            LocateRegistry.createRegistry(1099);
            System.out.println("[SERVER] RMI Registry created on port 1099");

            // Create an instance of the Remote Object (StringServer)
            StringServer serverObj = new StringServer();
            System.out.println("[SERVER] Remote Object (StringServer) created");

            // Bind the remote object to a name in the RMI Registry
            // Clients will use this name to look up the remote object
            // Format: "rmi://hostname/ServiceName"
            Naming.rebind("rmi://localhost/StringService", serverObj);
            System.out.println("[SERVER] Object bound to RMI Registry as 'StringService'");
            System.out.println("[SERVER] Server is ready and waiting for client calls...");

        } catch (Exception e) {
            // Print error details if server startup fails
            System.out.println("[SERVER] Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

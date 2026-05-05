// ============================================================
// CLUSTER: Practical 2 Java RMI — FILES IN THIS CLUSTER:
//   1. StringRemote.java ← Remote Interface
//   2. StringServer.java ← See this for FULL detailed run steps
//   3. StringClient.java ← YOU ARE HERE — run AFTER server is running
//
// SHORT RUN STEPS:
//   Compile (once): javac StringRemote.java StringServer.java StringClient.java
//   Terminal 1:     java StringServer    (keep running)
//   Terminal 2:     java StringClient    (enter two strings when prompted)
// ============================================================

// File 3: StringClient.java (RMI Client)

// Import Naming to look up the remote object from RMI Registry
import java.rmi.Naming;
// Import Scanner for reading user input
import java.util.Scanner;

// Define the RMI Client class
public class StringClient {

    // Main method — entry point of the client program
    public static void main(String[] args) {

        // Create Scanner to read user input from keyboard
        Scanner scanner = new Scanner(System.in);

        try {
            // Look up the remote object from RMI Registry using its registered name
            // This returns a stub (proxy) object that represents the remote object
            // The stub implements StringRemote interface
            // In Java RMI: Naming.lookup() contacts the RMI Registry, downloads the stub
            StringRemote stub = (StringRemote) Naming.lookup("rmi://localhost/StringService");
            System.out.println("[CLIENT] Successfully connected to RMI Server");
            System.out.println("[CLIENT] Remote stub obtained for 'StringService'");

            // Take two strings from user
            System.out.print("[CLIENT] Enter first string: ");
            String str1 = scanner.nextLine(); // Read first string

            System.out.print("[CLIENT] Enter second string: ");
            String str2 = scanner.nextLine(); // Read second string

            // Invoke the remote method on the stub
            // This LOOKS like a local method call, but:
            //   - Stub marshals str1 and str2
            //   - Stub sends request to server
            //   - Server's skeleton receives it, calls actual concatenate()
            //   - Server returns result, skeleton marshals it
            //   - Stub unmarshals result and returns it here
            System.out.println("[CLIENT] Invoking remote method: concatenate()");
            String result = stub.concatenate(str1, str2);

            // Display the result returned by the remote server
            System.out.println("[CLIENT] Result from server: '" + result + "'");

        } catch (Exception e) {
            // Handle errors like server not running or network failure
            System.out.println("[CLIENT] Error: " + e.getMessage());
            e.printStackTrace();
        }

        // Close scanner
        scanner.close();
    }
}

// ============================================================
// HOW THE ENTIRE JAVA RMI CODE WORKS:
// 1. StringRemote interface defines the contract for remote methods
// 2. StringServer implements the interface and extends UnicastRemoteObject
// 3. Server creates RMI Registry on port 1099 and binds the object as "StringService"
// 4. Client calls Naming.lookup() to get the stub from RMI Registry
// 5. Client calls stub.concatenate(str1, str2) — looks like local call
// 6. RMI stub marshals str1 and str2 into byte stream, sends to server
// 7. Server skeleton (internal to RMI) receives bytes, unmarshals strings
// 8. Actual concatenate() method runs on the SERVER JVM
// 9. Result is marshalled and sent back to client
// 10. Client stub unmarshals result and returns it to client code
// 11. Client displays the result
//
// THREE FILES NEEDED:
// - StringRemote.java: Remote Interface (shared between client and server)
// - StringServer.java: Server implementation + registry setup
// - StringClient.java: Client that looks up and calls remote methods
// ============================================================

// ============================================================
// ABOUT THIS PRACTICAL:
// Topic: Java RMI (Remote Method Invocation)
// RMI allows Java objects to invoke methods on objects in other JVMs.
// Key difference from RPC: RMI is object-oriented, supports full Java objects.
// Uses Java serialization for marshalling (automatic for Java types).
// RMI Registry acts as a naming service (like DNS for objects).
// In Java 2+ SDK: Stub protocol was updated, skeleton class is no longer needed.
// Modern Java uses dynamic stubs generated at runtime (no rmic tool needed).
// Real-world use: Early Java enterprise systems, EJB (Enterprise Java Beans).
// ============================================================

// ============================================================
// VIVA QUESTIONS AND ANSWERS:
//
// Q1. What is RMI? How does it work?
// A1. RMI (Remote Method Invocation) is a Java API to invoke methods on objects
//     in another JVM. Works by: Client gets a stub from RMI Registry → calls method
//     on stub → stub serializes args → sends to server → server deserializes →
//     executes method → serializes result → sends back → client deserializes → returns.
//
// Q2. What is the role of stub and skeleton in RMI?
// A2. Stub (client-side): Acts as local proxy for remote object. Marshals arguments,
//     sends to server, waits, unmarshals result.
//     Skeleton (server-side, pre-Java 2): Receives calls, unmarshals args, invokes
//     actual object method, marshals result. Now handled automatically by RMI runtime.
//
// Q3. What is RMI Registry?
// A3. RMI Registry is a name-to-object mapping service. Server binds objects with names
//     using Naming.rebind(). Clients find objects using Naming.lookup(). Default port: 1099.
//     Acts like a phone book: "I want StringService" → Registry returns the stub.
//
// Q4. Why must remote methods throw RemoteException?
// A4. RemoteException handles network-related failures like server crash, network timeout,
//     or serialization errors that can occur during remote communication. It's mandatory
//     in the Remote interface so callers are forced to handle distributed system failures.
//
// Q5. What is UnicastRemoteObject?
// A5. UnicastRemoteObject is a base class for remote objects. When a server class extends it,
//     RMI automatically exports the object (makes it available for remote calls) when
//     constructed. It handles socket creation and communication infrastructure.
//
// Q6. Describe Java RMI Example with all 6 steps.
// A6. 1. Create Remote Interface extending java.rmi.Remote
//     2. Implement Remote Interface (extend UnicastRemoteObject)
//     3. Compile and use rmic tool: rmic StringServer (generates stubs)
//     4. Start RMI Registry: rmiregistry (in same directory as .class files)
//     5. Start server: java StringServer (binds object to registry)
//     6. Start client: java StringClient (looks up stub, calls method)
//
// Q7. What is Java serialization and why is it important for RMI?
// A7. Java serialization converts objects into a byte stream for transmission.
//     RMI uses serialization to marshal (pack) method arguments and return values.
//     Classes passed as arguments must implement java.io.Serializable.
//     String, Integer and other Java built-in types are already serializable.
//
// Q8. What are requirements for a distributed application?
// A8. Three key requirements (met by RMI):
//     (i) Locate the remote method (via RMI Registry / Naming.lookup())
//     (ii) Communicate with remote objects (via stubs and serialization)
//     (iii) Load class definitions for objects (via class loading mechanism in RMI)
//
// Q9. How does RMI handle network failures?
// A9. RMI throws RemoteException when network failures occur. Programmers must catch
//     this exception and implement retry logic or failover. RMI does not automatically
//     retry failed calls. Advanced systems use replicated servers or fault-tolerant middleware.
//
// Q10. What is the Naming class in RMI?
// A10. java.rmi.Naming provides static methods to interact with the RMI Registry:
//      - rebind(name, obj): Register/update an object with a name
//      - bind(name, obj): Register object (fails if name already taken)
//      - lookup(name): Get stub for a registered object
//      - unbind(name): Remove a binding from the registry
//      - list(host): List all registered names on a host
// ============================================================

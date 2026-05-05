// ============================================================
// CLUSTER: Practical 2 Java RMI — FILES IN THIS CLUSTER:
//   1. StringRemote.java ← YOU ARE HERE (Remote Interface — compile first)
//   2. StringServer.java ← See this file for FULL detailed run steps
//   3. StringClient.java ← Run last after server is running
//
// SHORT RUN STEPS:
//   Compile: javac StringRemote.java StringServer.java StringClient.java
//   Run Server: java StringServer   (in Terminal 1)
//   Run Client: java StringClient   (in Terminal 2)
// ============================================================

// File 1: StringRemote.java (Remote Interface)

// Import Remote interface — all remote interfaces must extend this
import java.rmi.Remote;
// Import RemoteException — all remote methods must declare this exception
import java.rmi.RemoteException;

// Define the Remote Interface — specifies which methods can be invoked remotely
// Any class that implements this interface becomes a Remote Object
public interface StringRemote extends Remote {

    // Declare the remote method that clients can call
    // Must throw RemoteException to handle network/communication failures
    // str1 and str2 are strings that the client will send from another JVM
    String concatenate(String str1, String str2) throws RemoteException;
}

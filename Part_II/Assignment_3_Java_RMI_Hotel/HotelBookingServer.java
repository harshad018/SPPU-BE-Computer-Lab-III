import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * HotelBookingServer – implementation of HotelBookingService.
 *
 * Stores bookings in-memory (HashMap).  Run this class first, then run
 * HotelBookingClient in one or more separate terminals.
 *
 * Compile & run:
 *   javac *.java
 *   rmiregistry &          (or use createRegistry below – already embedded)
 *   java HotelBookingServer
 */
public class HotelBookingServer extends UnicastRemoteObject
        implements HotelBookingService {

    // room number → guest name
    private final Map<Integer, String> bookings = new HashMap<>();
    private static final int TOTAL_ROOMS = 10;

    public HotelBookingServer() throws RemoteException {
        super();
    }

    // -----------------------------------------------------------------------
    // Remote method implementations
    // -----------------------------------------------------------------------

    @Override
    public synchronized String bookRoom(String guestName, int roomNumber)
            throws RemoteException {
        if (roomNumber < 1 || roomNumber > TOTAL_ROOMS) {
            return "[ERROR] Room " + roomNumber
                    + " does not exist. Valid rooms: 1-" + TOTAL_ROOMS;
        }
        if (bookings.containsKey(roomNumber)) {
            String existing = bookings.get(roomNumber);
            if (existing.equalsIgnoreCase(guestName)) {
                return "[INFO] Room " + roomNumber
                        + " is already booked by " + guestName + ".";
            }
            return "[ERROR] Room " + roomNumber
                    + " is already booked by " + existing + ".";
        }
        // Check if this guest already has a booking
        for (Map.Entry<Integer, String> entry : bookings.entrySet()) {
            if (entry.getValue().equalsIgnoreCase(guestName)) {
                return "[ERROR] Guest \"" + guestName
                        + "\" already has a booking for room "
                        + entry.getKey() + ". Cancel it first.";
            }
        }
        bookings.put(roomNumber, guestName);
        return "[SUCCESS] Room " + roomNumber
                + " booked successfully for guest \"" + guestName + "\".";
    }

    @Override
    public synchronized String cancelBooking(String guestName)
            throws RemoteException {
        Integer bookedRoom = null;
        for (Map.Entry<Integer, String> entry : bookings.entrySet()) {
            if (entry.getValue().equalsIgnoreCase(guestName)) {
                bookedRoom = entry.getKey();
                break;
            }
        }
        if (bookedRoom == null) {
            return "[ERROR] No booking found for guest \"" + guestName + "\".";
        }
        bookings.remove(bookedRoom);
        return "[SUCCESS] Booking for guest \"" + guestName
                + "\" (Room " + bookedRoom + ") has been cancelled.";
    }

    @Override
    public synchronized List<String> listBookings() throws RemoteException {
        List<String> result = new ArrayList<>();
        if (bookings.isEmpty()) {
            result.add("  (no bookings at the moment)");
        } else {
            List<Integer> rooms = new ArrayList<>(bookings.keySet());
            java.util.Collections.sort(rooms);
            for (int room : rooms) {
                result.add("  Room " + room + "  →  " + bookings.get(room));
            }
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // main
    // -----------------------------------------------------------------------

    public static void main(String[] args) {
        int port = 1099;
        if (args.length > 0) {
            try {
                port = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.err.println("[WARN] Invalid port argument; using default 1099.");
            }
        }
        try {
            HotelBookingServer server = new HotelBookingServer();
            // Create the RMI registry programmatically (no need to run rmiregistry manually)
            Registry registry = LocateRegistry.createRegistry(port);
            registry.rebind("HotelBookingService", server);
            System.out.println("=================================================");
            System.out.println("  Hotel Booking RMI Server started on port " + port);
            System.out.println("  Registered service: HotelBookingService");
            System.out.println("  Available rooms: 1–" + TOTAL_ROOMS);
            System.out.println("  Waiting for client connections …");
            System.out.println("=================================================");
        } catch (RemoteException e) {
            System.err.println("[ERROR] Server failed to start: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

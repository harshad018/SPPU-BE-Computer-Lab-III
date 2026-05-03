import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.List;

/**
 * HotelBookingClient – demonstrates booking and cancellation via RMI.
 *
 * Usage (after server is running):
 *   java HotelBookingClient [host] [port]
 *
 * Examples:
 *   java HotelBookingClient              # connects to localhost:1099
 *   java HotelBookingClient 192.168.1.5  # remote host, default port
 */
public class HotelBookingClient {

    public static void main(String[] args) {
        String host = "localhost";
        int port = 1099;

        if (args.length >= 1) host = args[0];
        if (args.length >= 2) {
            try { port = Integer.parseInt(args[1]); }
            catch (NumberFormatException e) {
                System.err.println("[WARN] Invalid port; using 1099.");
            }
        }

        try {
            Registry registry = LocateRegistry.getRegistry(host, port);
            HotelBookingService service =
                    (HotelBookingService) registry.lookup("HotelBookingService");

            System.out.println("=================================================");
            System.out.println("  Hotel Booking RMI Client");
            System.out.println("  Connected to " + host + ":" + port);
            System.out.println("=================================================\n");

            // ---- Demo operations ----------------------------------------

            // 1. Book rooms for several guests
            System.out.println("--- Booking Rooms ---");
            System.out.println(service.bookRoom("Alice", 3));
            System.out.println(service.bookRoom("Bob",   7));
            System.out.println(service.bookRoom("Carol", 1));

            // 2. Attempt duplicate bookings
            System.out.println("\n--- Duplicate / Error Cases ---");
            System.out.println(service.bookRoom("Dave",  3));  // room already taken
            System.out.println(service.bookRoom("Alice", 5));  // guest already booked
            System.out.println(service.bookRoom("Eve",  15));  // invalid room

            // 3. List current bookings
            System.out.println("\n--- Current Bookings ---");
            List<String> bookings = service.listBookings();
            bookings.forEach(System.out::println);

            // 4. Cancel a booking
            System.out.println("\n--- Cancellation ---");
            System.out.println(service.cancelBooking("Bob"));
            System.out.println(service.cancelBooking("Unknown")); // not booked

            // 5. List bookings after cancellation
            System.out.println("\n--- Bookings After Cancellation ---");
            bookings = service.listBookings();
            bookings.forEach(System.out::println);

            System.out.println("\nClient demo complete.");

        } catch (Exception e) {
            System.err.println("[ERROR] Client exception: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

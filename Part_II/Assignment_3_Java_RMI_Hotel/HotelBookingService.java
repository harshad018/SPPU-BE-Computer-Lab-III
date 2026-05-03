import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;

/**
 * HotelBookingService – RMI remote interface.
 * Clients can book or cancel a room for a guest.
 */
public interface HotelBookingService extends Remote {

    /**
     * Book a room for a guest.
     *
     * @param guestName  Name of the guest.
     * @param roomNumber Room number to book.
     * @return Confirmation message.
     * @throws RemoteException if a network/RMI error occurs.
     */
    String bookRoom(String guestName, int roomNumber) throws RemoteException;

    /**
     * Cancel the booking for a guest.
     *
     * @param guestName Name of the guest whose booking should be cancelled.
     * @return Cancellation message.
     * @throws RemoteException if a network/RMI error occurs.
     */
    String cancelBooking(String guestName) throws RemoteException;

    /**
     * Return all current bookings (for display purposes).
     *
     * @return List of booking strings "Room X → Guest".
     * @throws RemoteException if a network/RMI error occurs.
     */
    List<String> listBookings() throws RemoteException;
}

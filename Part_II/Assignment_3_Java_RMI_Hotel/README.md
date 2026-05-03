# Assignment 3 – Distributed Hotel Booking Application (Java RMI)

## Overview
A client–server distributed system built with **Java RMI** where:
- The **server** maintains in-memory room-booking state (HashMap).
- **Clients** invoke two remote methods:
  - `bookRoom(guestName, roomNumber)` – reserve a room for a guest.
  - `cancelBooking(guestName)` – release the guest's current room.
  - `listBookings()` – view all current bookings (utility/display method).

## Files
| File | Description |
|---|---|
| `HotelBookingService.java` | RMI remote interface |
| `HotelBookingServer.java` | Server implementation + `main()` |
| `HotelBookingClient.java` | Demo client with sample operations |

## Requirements
- Java Development Kit (JDK) 8 or later
- No external libraries needed (uses standard `java.rmi.*`)

## Compile
```bash
javac *.java
```

## Run

### Terminal 1 – Start the server
```bash
java HotelBookingServer
```
The server creates its own RMI registry on port **1099** (default).
You should see:
```
=================================================
  Hotel Booking RMI Server started on port 1099
  Registered service: HotelBookingService
  Available rooms: 1–10
  Waiting for client connections …
=================================================
```

### Terminal 2 – Run a client
```bash
java HotelBookingClient
```
Expected output:
```
--- Booking Rooms ---
[SUCCESS] Room 3 booked successfully for guest "Alice".
[SUCCESS] Room 7 booked successfully for guest "Bob".
[SUCCESS] Room 1 booked successfully for guest "Carol".

--- Duplicate / Error Cases ---
[ERROR] Room 3 is already booked by Alice.
[ERROR] Guest "Alice" already has a booking for room 3. Cancel it first.
[ERROR] Room 15 does not exist. Valid rooms: 1-10

--- Current Bookings ---
  Room 1  →  Carol
  Room 3  →  Alice
  Room 7  →  Bob

--- Cancellation ---
[SUCCESS] Booking for guest "Bob" (Room 7) has been cancelled.
[ERROR] No booking found for guest "Unknown".

--- Bookings After Cancellation ---
  Room 1  →  Carol
  Room 3  →  Alice
```

### Multiple clients
You can open as many client terminals as you like — all share the same server state.
Methods are `synchronized` so concurrent access is safe.

### Remote host
```bash
java HotelBookingClient 192.168.1.100        # default port 1099
java HotelBookingClient 192.168.1.100 2000   # custom port
```

## Notes
- The server uses `LocateRegistry.createRegistry()` so **no separate `rmiregistry`
  process** is needed.
- Bookings are stored in memory; they are lost when the server is restarted.
- Allowed room numbers: 1–10 (configurable via `TOTAL_ROOMS` in the server).

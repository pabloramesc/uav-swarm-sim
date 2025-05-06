import logging
import socket
import time
from dataclasses import dataclass

logging.basicConfig(
    level=logging.DEBUG,
    format="[SwarmSim:IpcSocket] %(levelname)s: %(message)s",
)


@dataclass
class IpcMessage:
    time: float
    addr: str
    port: int
    data: bytes


class IpcSocket:

    def __init__(
        self, addr: str = "127.0.0.1", port: int = 9001, ns3_port: int = 9000
    ) -> None:
        self.addr = addr
        self.port = port
        self.ns3_port = ns3_port
        self.sock = self._setup_socket()

    def _setup_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((self.addr, self.port))
        sock.setblocking(False)
        logging.debug(f"Socket listening in {self.addr}:{self.port}")
        return sock

    def _close_socket(self) -> None:
        self.sock.close()

    def read_socket(self) -> IpcMessage | None:
        try:
            data, addr = self.sock.recvfrom(1024)
            logging.debug(f"Message received from {addr}: {data}")

            msg = IpcMessage(time=time.time(), addr=addr[0], port=addr[1], data=data)

            if msg.port != self.ns3_port:
                logging.debug(f"Message received from foreign port. Ignoring.")
                return None

            return msg

        except BlockingIOError:
            logging.debug("No message received.")
            return None

    def send_to_ns3(self, data: bytes) -> None:
        self.sock.sendto(data, (self.addr, self.ns3_port))
        logging.debug(f"Message sent to {(self.addr, self.ns3_port)}: {data}")
        
    def close(self) -> None:
        self.sock.close()
        logging.debug(f"Socket closed.")

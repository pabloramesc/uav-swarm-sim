"""
Copyright (c) 2025 Pablo Ramirez Escudero

This software is released under the MIT License.
https://opensource.org/licenses/MIT
"""

import logging
import socket
import time
from dataclasses import dataclass
import threading
from collections import deque

# Create a logger for this module
logger = logging.getLogger("SIM:IpcSocket")
logger.setLevel(logging.NOTSET)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class IpcMessage:
    time: float
    addr: str
    port: int
    data: bytes


class IpcSocket:

    def __init__(
        self,
        addr: str = "127.0.0.1",
        port: int = 9001,
        ns3_port: int = 9000,
        buffer_size: int = 100,
    ) -> None:
        self.addr = addr
        self.port = port
        self.ns3_port = ns3_port
        self.buffer_size = buffer_size
        
        self.sock = self._setup_socket()
        self.message_buffer = deque(maxlen=buffer_size)
        
        self.running = False
        self.thread = None

    def _setup_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((self.addr, self.port))
        sock.setblocking(False)
        logger.debug(f"Socket listening in {self.addr}:{self.port}")
        return sock

    def send_to_ns3(self, data: bytes) -> None:
        self.sock.sendto(data, (self.addr, self.ns3_port))
        logger.debug(f"Message sent to {(self.addr, self.ns3_port)}: {data}")

    def start_reading(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.debug("Started reading messages in a separate thread.")

    def _read_loop(self) -> None:
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                logger.debug(f"Message received from {addr}: {data}")

                msg = IpcMessage(
                    time=time.time(), addr=addr[0], port=addr[1], data=data
                )

                if msg.port == self.ns3_port:
                    self.message_buffer.append(msg)  # Add message to buffer
                    logger.debug(
                        f"Message added to buffer. Buffer size: {len(self.message_buffer)}"
                    )
                else:
                    logger.debug(f"Message received from foreign port. Ignoring.")

            except BlockingIOError:
                # logger.debug("No message received.")
                continue

    def close(self) -> None:
        self.stop_reading()
        self.sock.close()
        logger.debug(f"Socket closed.")

    def stop_reading(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def clear_buffer(self) -> None:
        self.message_buffer.clear()

    def get_buffer_size(self) -> int:
        """Returns the number of messages currently in the buffer."""
        return len(self.message_buffer)

    def get_first_message(self) -> IpcMessage | None:
        """Returns and removes the first message in the buffer, or None if the buffer is empty."""
        if self.message_buffer:
            return self.message_buffer.popleft()
        return None

    def get_last_message(self) -> IpcMessage | None:
        """Returns and removes the last message in the buffer, or None if the buffer is empty."""
        if self.message_buffer:
            return self.message_buffer.pop()
        return None

    def get_all_messages(self) -> list[IpcMessage]:
        """Returns all messages in the buffer and clears it."""
        messages = list(self.message_buffer)
        self.message_buffer.clear()
        return messages

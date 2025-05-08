#include "ipc-socket.h"

#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

#define DEBUG false

using namespace std;

IpcSocket::IpcSocket(const string &addr, uint16_t port)
    : m_addr(addr), m_port(port), m_sock(-1) {
    SetupSocket();
}

IpcSocket::~IpcSocket() {
    Close();
}

void IpcSocket::Close() {
    if (m_sock != -1) {
        close(m_sock);
        m_sock = -1;
#if DEBUG
        cout << "[NS3:IpcSocket] DEBUG: Socket closed." << endl;
#endif
    }
}

void IpcSocket::SetupSocket() {
    m_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_sock < 0) {
        cerr << "[NS3:IpcSocket] ERROR: Failed to create socket. Error: "
             << strerror(errno) << endl;
        std::exit(EXIT_FAILURE); // Critical error, terminate the program
    }

#if DEBUG
    cout << "[NS3:IpcSocket] DEBUG: Socket created successfully. FD: " << m_sock << endl;
#endif

    int opt = 1;
    if (setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        cerr << "[NS3:IpcSocket] ERROR: Failed to set SO_REUSEADDR option. Error: "
             << strerror(errno) << endl;
        std::exit(EXIT_FAILURE); // Critical error, terminate the program
    }

#if DEBUG
    cout << "[NS3:IpcSocket] DEBUG: Reuse address option set successfully." << endl;
#endif

    if (setsockopt(m_sock, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        cerr << "[NS3:IpcSocket] ERROR: Failed to set SO_REUSEPORT option. Error: "
             << strerror(errno) << endl;
        std::exit(EXIT_FAILURE); // Critical error, terminate the program
    }

#if DEBUG
    cout << "[NS3:IpcSocket] DEBUG: Reuse port option set successfully." << endl;
#endif

    int flags = fcntl(m_sock, F_GETFL, 0);
    if (flags == -1) {
        cerr << "[NS3:IpcSocket] ERROR: Failed to get socket flags. Error: "
             << strerror(errno) << endl;
        std::exit(EXIT_FAILURE); // Critical error, terminate the program
    }

    if (fcntl(m_sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        cerr << "[NS3:IpcSocket] ERROR: Failed to set socket to non-blocking mode. Error: "
             << strerror(errno) << endl;
        std::exit(EXIT_FAILURE); // Critical error, terminate the program
    }

#if DEBUG
    cout << "[NS3:IpcSocket] DEBUG: Socket set to non-blocking mode successfully." << endl;
#endif

    memset(&m_listenAddr, 0, sizeof(m_listenAddr));
    m_listenAddr.sin_family = AF_INET;
    m_listenAddr.sin_addr.s_addr = inet_addr(m_addr.c_str());
    m_listenAddr.sin_port = htons(m_port);

    if (bind(m_sock, (struct sockaddr *)&m_listenAddr, sizeof(m_listenAddr)) < 0) {
        cerr << "[NS3:IpcSocket] ERROR: Failed to bind socket to " << m_addr << ":" << m_port
             << ". Error: " << strerror(errno) << endl;
        std::exit(EXIT_FAILURE); // Critical error, terminate the program
    }

#if DEBUG
    cout << "[NS3:IpcSocket] DEBUG: Socket bound successfully to " << m_addr << ":" << m_port << endl;
#endif
}

int IpcSocket::ReadSocket(uint8_t *buffer, size_t size) {
    int numBytes = recvfrom(m_sock, buffer, size, 0,
                            (struct sockaddr *)&m_remoteAddr, &m_remoteLen);

    if (numBytes < 0) {
        return numBytes;
    }

#if DEBUG
    string msg((char *)buffer, numBytes);
    const char *clientIp = inet_ntoa(m_remoteAddr.sin_addr);
    uint16_t clientPort = ntohs(m_remoteAddr.sin_port);
    cout << "[NS3:IpcSocket] DEBUG: Received " << numBytes << " bytes from "
         << "(" << clientIp << ":" << clientPort << "): " << msg << endl;
#endif

    return numBytes;
}

void IpcSocket::SendToRemote(const uint8_t *data, size_t size) {
    int result = sendto(m_sock, data, size, 0,
                        (sockaddr *)&m_remoteAddr, m_remoteLen);

    if (result < 0) {
        cerr << "[NS3:IpcSocket] ERROR: Failed to send data. Error: " << strerror(errno) << endl;
        return;
    }
#if DEBUG
    string msg(reinterpret_cast<const char *>(data), size);
    const char *clientIp = inet_ntoa(m_remoteAddr.sin_addr);
    uint16_t clientPort = ntohs(m_remoteAddr.sin_port);
    cout << "[NS3:IpcSocket] DEBUG: Sent " << size << " bytes to "
         << "(" << clientIp << ":" << clientPort << "): " << msg << endl;
#endif
}
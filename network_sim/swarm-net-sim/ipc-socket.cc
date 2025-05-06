#include "ipc-socket.h"

#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

#define DEBUG true

using namespace std;

IpcSocket::IpcSocket(const string &addr, uint16_t port)
    : m_addr(addr), m_port(port), m_sock(-1) {
    SetupSocket();
}

IpcSocket::~IpcSocket() {
    if (m_sock != -1)
        close(m_sock);
}

void IpcSocket::SetupSocket() {
    m_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_sock < 0) {
        cerr << "[IpcSocket] Failed to create socket" << endl;
    }

    int flags = fcntl(m_sock, F_GETFL, 0);
    if (flags == -1 || fcntl(m_sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        cerr << "[IpcSocket] Failed to set socket in non-blocking mode" << endl;
    }

    memset(&m_listenAddr, 0, sizeof(m_listenAddr));
    m_listenAddr.sin_family = AF_INET;
    m_listenAddr.sin_addr.s_addr = inet_addr(m_addr.c_str());
    m_listenAddr.sin_port = htons(m_port);

    if (bind(m_sock, (struct sockaddr *)&m_listenAddr, sizeof(m_listenAddr)) < 0) {
        cerr << "[IpcSocket] Failed to bind socket to " << m_addr << ":" << m_port << endl;
    }

    cout << "[IpcSocket] Socket listening on " << m_addr << ":" << m_port << endl;
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

    cout << "[IpcSocket] Received from " << "(" << clientIp << ":" << clientPort << "): " << msg << endl;
#endif

    return numBytes;
}

void IpcSocket::SendToRemote(const uint8_t *data, size_t size) {
    sendto(m_sock, data, size, 0,
           (sockaddr *)&m_remoteAddr, m_remoteLen);

#if DEBUG
    string msg(reinterpret_cast<const char *>(data), size);
    const char *clientIp = inet_ntoa(m_remoteAddr.sin_addr);
    uint16_t clientPort = ntohs(m_remoteAddr.sin_port);

    cout << "[IpcSocket] Sent to " << "(" << clientIp << ":" << clientPort << "): " << msg << endl;
#endif
}
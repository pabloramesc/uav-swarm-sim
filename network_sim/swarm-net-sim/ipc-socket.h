#ifndef IPC_SOCKET_H
#define IPC_SOCKET_H

#include <netinet/in.h>
#include <string>
using namespace std;

class IpcSocket {
public:
    IpcSocket(const string &addr, uint16_t port);
    ~IpcSocket();
    
    int ReadSocket(uint8_t *buffer, size_t size);
    void SendToRemote(const uint8_t *data, size_t size);
    void Close();

private:
    void SetupSocket();

    string m_addr;
    uint16_t m_port;

    int m_sock;
    struct sockaddr_in m_listenAddr, m_remoteAddr;
    socklen_t m_remoteLen;
};

#endif // IPC_SOCKET_H
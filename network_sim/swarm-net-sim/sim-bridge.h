#ifndef SIM_BRIDGE_H
#define SIM_BRIDGE_H

#include "ipc-socket.h"
#include "nodes-manager.h"

#include "ns3/core-module.h"
#include "ns3/network-module.h"

#define BUFFER_SIZE 1024

using namespace std;
using namespace ns3;

enum SimCommandCode {
    CMD_DO_NOTHING = 0x00,
    CMD_SET_POSITIONS = 0x01,
    CMD_REQUEST_POSITIONS = 0x02,
    CMD_INGRESS_PACKET = 0x03,
    CMD_REQUEST_ADDRESSES = 0x04,
    CMD_STOP_SIMULATION = 0xFF,
    REPLY_ALL_POSITIONS = 0xA1,
    REPLY_EGRESS_PACKET = 0xA2,
    REPLY_ALL_ADDRESSES = 0xA3,
};

class SimBridge {
public:
    SimBridge(float pollingInterval = 0.1);
    ~SimBridge();

    void RegisterNode(int nodeId, Ptr<Node> node);
    void StartPolling();
    void StopSimulation();

private:
    void PollSocket();

    void RxCallback(Ptr<Socket> socket);

    void ProcessCommand(int numBytes);
    void HandleDoNothing(int numBytes);
    void HandleSetPositions(int numBytes);
    void HandleRequestPositions(int numBytes);
    void HandleRequestAddresses(int numBytes);
    void HandleIngressPacket(int numBytes);
    void HandleStopSimulation(int numBytes);

    void ReplyDoNothing();
    void ReplyAllPositions();
    void ReplyAllAddresses();
    void ReplyEgressPacket(int nodeId, Ipv4Address srcAddr, Ipv4Address destAddr, const uint8_t *data, size_t size);

    float m_pollingInterval;
    uint8_t m_buffer[BUFFER_SIZE];

    IpcSocket m_ipcSocket;
    NodesManager m_nodesManager;

    bool m_running;
};

#endif // SIM_BRIDGE_H
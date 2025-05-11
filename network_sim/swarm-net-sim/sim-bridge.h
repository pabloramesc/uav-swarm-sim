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
    // Command codes for simulation control
    DO_NOTHING = 0x00,
    STOP_SIMULATION = 0xFF,
    // Command codes to control nodes
    SET_POSITIONS = 0x01,
    INGRESS_PACKET = 0x02,
    EGRESS_PACKET = 0x03,
    // Command codes for simulation status
    REQUEST_POSITIONS = 0xA1,
    REQUEST_ADDRESSES = 0xA2,
    REQUEST_SIM_TIME = 0xA3,
    // Reply codes for simulation status
    REPLY_ALL_POSITIONS = 0xB1,
    REPLY_ALL_ADDRESSES = 0xB2,
    REPLY_SIM_TIME = 0xB3,
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
    void HandleStopSimulation(int numBytes);
    
    void HandleSetPositions(int numBytes);
    void HandleIngressPacket(int numBytes);
    void SendEgressPacket(int nodeId, Ipv4Address srcAddr, Ipv4Address destAddr, const uint8_t *data, size_t size);

    void HandleRequestPositions(int numBytes);
    void HandleRequestAddresses(int numBytes);
    void HandleRequestSimTime(int numBytes);

    void ReplyDoNothing();
    void ReplyAllPositions();
    void ReplyAllAddresses();
    void ReplySimTime();

    float m_pollingInterval;
    uint8_t m_buffer[BUFFER_SIZE];

    IpcSocket m_ipcSocket;
    NodesManager m_nodesManager;

    bool m_running;
};

#endif // SIM_BRIDGE_H
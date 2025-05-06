#include "sim-bridge.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"

using namespace std;
using namespace ns3;

#define DEBUG true

SimBridge::SimBridge(float pollingInterval)
    : m_pollingInterval(pollingInterval),
      m_ipcSocket("127.0.0.1", 9000),
      m_nodesManager() {
    PollSocket();
}

SimBridge::~SimBridge() {
    StopSimulation();
}

void SimBridge::PollSocket() {
    for (int i = 0; i < 100; i++) {
        int numBytes = m_ipcSocket.ReadSocket(m_buffer, sizeof(m_buffer));

        if (numBytes < 0) {
            break;
        }

#if DEBUG
        string msg((char *)m_buffer, numBytes);
        cout << "[SimBridge] At " << Simulator::Now().GetSeconds() << " s received: " << msg << endl;
#endif

        ProcessCommand(numBytes);
    }

    Simulator::Schedule(Seconds(m_pollingInterval), &SimBridge::PollSocket, this);
}

void SimBridge::StartSimulation() {
    Simulator::Run();
}

void SimBridge::StopSimulation() {
    Simulator::Stop();
    Simulator::Destroy();
}

void SimBridge::RegisterNode(int nodeId, Ptr<Node> node) {
    m_nodesManager.RegisterNode(nodeId, node, 12345, true);
    m_nodesManager.SetNodeRxCallback(nodeId, MakeCallback(&SimBridge::RxCallback, this));
}

void SimBridge::RxCallback(Ptr<Socket> socket) {
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);
    uint8_t buffer[BUFFER_SIZE];
    packet->CopyData(buffer, packet->GetSize());
    string msg((char *)buffer);

    Ptr<Node> node = socket->GetNode();
    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    Ipv4Address ipv4Addr = ipv4->GetAddress(1, 0).GetLocal();

    InetSocketAddress inetFrom = InetSocketAddress::ConvertFrom(from);
    Ipv4Address ipv4From = inetFrom.GetIpv4();

#if DEBUG
    cout << "[RxCallback] At " << Simulator::Now().GetSeconds() << "s "
         << "Node " << node->GetId() << " (" << ipv4Addr << ") "
         << "received: '" << msg << "' from " << ipv4From << endl;
#endif

    ReplyEgressPacket(node->GetId(), ipv4From, ipv4Addr, buffer, packet->GetSize());
}

void SimBridge::ProcessCommand(int numBytes) {
    if (numBytes <= 0) {
        cerr << "[SimBridge] Empty package received. Ignoring." << endl;
        return;
    }

    uint8_t commandCode = m_buffer[0]; // First byte is the command code

    switch (commandCode) {
    case CMD_DO_NOTHING:
        HandleDoNothing(numBytes);
        break;

    case CMD_SET_POSITIONS:
        HandleSetPositions(numBytes);
        break;

    case CMD_REQUEST_POSITIONS:
        HandleRequestPositions(numBytes);
        break;

    case CMD_INGRESS_PACKET:
        HandleIngressPacket(numBytes);
        break;

    case CMD_STOP_SIMULATION:
        HandleStopSimulation(numBytes);
        break;

    default:
        cerr << "[SimBridge] Unknown command code received: " << (int)commandCode << ". Ignoring." << endl;
        break;
    }
}

void SimBridge::HandleDoNothing(int numBytes) {
    if (numBytes > 1) {
        cout << "[SimBridge] CMD_DO_NOTHING received with more than 1 byte. No action taken." << endl;
        return;
    }
    cout << "[SimBridge] CMD_DO_NOTHING received. No action taken." << endl;
}

void SimBridge::HandleSetPositions(int numBytes) {
    if (numBytes < 1 + 13) { // At least 1 byte for command + 1 node entry (1 byte for id + 3x4 bytes for position)
        cout << "[SimBridge] CMD_SET_POSITIONS received with insufficient data." << endl;
        return;
    }

    int offset = 1;                   // Start after the command byte
    while (offset + 13 <= numBytes) { // Ensure there is enough data for [id, px, py, pz]
        int nodeId = (int)m_buffer[offset];
        double x = (double)m_buffer[offset + 1];
        double y = (double)m_buffer[offset + 5];
        double z = (double)m_buffer[offset + 9];

        m_nodesManager.SetNodePosition(nodeId, Vector(x, y, z));

#if DEBUG
        cout << "[SimBridge] CMD_SET_POSITIONS: Node " << nodeId
             << " position set to (" << x << ", " << y << ", " << z << ")" << endl;
#endif

        offset += 13; // Move to the next node entry
    }

    if (offset != numBytes) {
        NS_FATAL_ERROR("[SimBridge] CMD_SET_POSITIONS received with extra or malformed data.");
    }
}

void SimBridge::HandleRequestPositions(int numBytes) {
    if (numBytes > 1) {
        cout << "[SimBridge] CMD_REQUEST_POSITIONS received with more than 1 byte. No action taken." << endl;
        return;
    }

#if DEBUG
    cout << "[SimBridge] CMD_REQUEST_POSITIONS received. Replying with all positions." << endl;
#endif

    ReplyAllPositions();
    return;
}

void SimBridge::HandleStopSimulation(int numBytes) {
    if (numBytes > 1) {
        cout << "[SimBridge] CMD_STOP_SIMULATION received with more than 1 byte. No action taken." << endl;
        return;
    }
    cout << "[SimBridge] CMD_STOP_SIMULATION received. Stopping simulation." << endl;
    StopSimulation();
}

void SimBridge::HandleIngressPacket(int numBytes) {
    if (numBytes < 13) { // 1 byte for command + 4 bytes for nodeId + 4 bytes for srcAddr + 4 bytes for destAddr
        NS_FATAL_ERROR("[SimBridge] CMD_INGRESS_PACKET received with insufficient data.");
        return;
    }

    int nodeId = (int)m_buffer[1];
    // uint32_t srcAddr = (uint32_t)m_buffer[5];
    uint32_t destAddr = (uint32_t)m_buffer[9];

    size_t payloadSize = numBytes - 13; // Remaining bytes are the payload
    if (payloadSize == 0) {
        NS_FATAL_ERROR("[SimBridge] CMD_INGRESS_PACKET received with no payload.");
        return;
    }

    m_nodesManager.SendPacket(nodeId, Ipv4Address(destAddr), &m_buffer[13], payloadSize);
}

void SimBridge::ReplyDoNothing() {
    uint8_t response[BUFFER_SIZE];
    size_t size = 1;
    response[0] = 0x00;
    m_ipcSocket.SendToRemote(response, size);
}

void SimBridge::ReplyAllPositions() {
    uint8_t response[BUFFER_SIZE];
    size_t offset = 0;

    response[offset++] = REPLY_ALL_POSITIONS; // Add the reply command code

    int numNodes = m_nodesManager.GetNumNodes();
    for (int nodeId = 0; nodeId < numNodes; ++nodeId) {

        if (offset + 13 > BUFFER_SIZE) {
            NS_FATAL_ERROR("[SimBridge] Buffer overflow while preparing REPLY_ALL_POSITIONS.");
            return;
        }

        Vector pos = m_nodesManager.GetNodePosition(nodeId);

        response[offset++] = (uint8_t)nodeId;
        memcpy(&response[offset], &pos.x, sizeof(double));
        offset += sizeof(double);
        memcpy(&response[offset], &pos.y, sizeof(double));
        offset += sizeof(double);
        memcpy(&response[offset], &pos.z, sizeof(double));
        offset += sizeof(double);

#if DEBUG
        cout << "[SimBridge] REPLY_ALL_POSITIONS: Node " << nodeId
             << " position is (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << endl;
#endif
    }

    m_ipcSocket.SendToRemote(response, offset);

#if DEBUG
    cout << "[SimBridge] REPLY_ALL_POSITIONS sent with " << offset << " bytes." << endl;
#endif
}

void SimBridge::ReplyEgressPacket(int nodeId, Ipv4Address srcAddr, Ipv4Address destAddr, const uint8_t *data, size_t size) {
    uint8_t response[1024];
    size_t responseSize = 13 + size; // 1 byte for command + 4 bytes for nodeId + 4 bytes for srcAddr + 4 bytes for destAddr + payload

    if (responseSize > sizeof(response)) {
        NS_FATAL_ERROR("[SimBridge] Response size exceeds buffer limit.");
        return;
    }

    response[0] = REPLY_EGRESS_PACKET; // Command code for egress packet
    *reinterpret_cast<int *>(&response[1]) = nodeId;
    *reinterpret_cast<uint32_t *>(&response[5]) = srcAddr.Get();
    *reinterpret_cast<uint32_t *>(&response[9]) = destAddr.Get();
    memcpy(&response[13], data, size);

    m_ipcSocket.SendToRemote(response, responseSize);

    cout << "[SimBridge] REPLY_EGRESS_PACKET: Node " << nodeId
         << " sent a response from " << srcAddr
         << " to " << destAddr
         << " with payload size " << size << " bytes." << endl;
}
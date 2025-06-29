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

#define DEBUG false
#define TIMESTAMP true

SimBridge::SimBridge(float pollingInterval)
    : m_pollingInterval(pollingInterval),
      m_ipcSocket("127.0.0.1", 9000),
      m_nodesManager(),
      m_running(false) {
}

SimBridge::~SimBridge() {
    if (m_running) StopSimulation();
}

void SimBridge::ScheduleScoketPolling() {
    for (int i = 0; i < 100; i++) {
        int numBytes = m_ipcSocket.ReadSocket(m_buffer, sizeof(m_buffer));

        if (numBytes < 0) {
            break;
        }

#if DEBUG
        cout << "[NS3:SimBridge] DEBUG: At " << Simulator::Now().GetSeconds()
             << " s received " << numBytes << " bytes. Processing..." << endl;
#endif

        ProcessCommand(numBytes);
        if (!m_running) break;
    }

    if (m_running) {
        Simulator::Schedule(Seconds(m_pollingInterval), &SimBridge::ScheduleScoketPolling, this);
    }
}

void SimBridge::ScheduleStatusReport() {
    if (m_ipcSocket.is_remote) {
        ReplySimTime();
        ReplyAllPositions();
    }

    if (m_running) {
        Simulator::Schedule(Seconds(1.0), &SimBridge::ScheduleStatusReport, this);
    }
}

void SimBridge::StartPolling() {
    Simulator::ScheduleNow(&SimBridge::ScheduleScoketPolling, this);
    // Simulator::ScheduleNow(&SimBridge::ScheduleStatusReport, this);
    m_running = true;
}

void SimBridge::StopSimulation() {
    m_running = false;
    m_ipcSocket.Close();
    m_nodesManager.Clear();
    Simulator::Stop();
#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Simulation stopped successfully." << endl;
#endif
}

void SimBridge::RegisterNode(int nodeId, Ptr<Node> node) {
    m_nodesManager.RegisterNode(nodeId, node);
    m_nodesManager.SetNodeRxCallback(nodeId, MakeCallback(&SimBridge::RxCallback, this));
}

void SimBridge::RxCallback(Ptr<Socket> socket) {
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);
    uint8_t buffer[BUFFER_SIZE];
    uint32_t size = packet->GetSize();
    packet->CopyData(buffer, size);

    Ptr<Node> node = socket->GetNode();

    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    Ipv4Address ipv4Addr = ipv4->GetAddress(1, 0).GetLocal();

    InetSocketAddress inetFrom = InetSocketAddress::ConvertFrom(from);
    Ipv4Address ipv4From = inetFrom.GetIpv4();

#if DEBUG
    string msg((char *)buffer, size);
    cout << "[NS3:RxCallback] DEBUG: At " << Simulator::Now().GetSeconds() << "s "
         << "Node " << node->GetId() << " (" << ipv4Addr << ") "
         << "received " << size << " bytes from " << ipv4From << endl;
#endif

    SendEgressPacket(node->GetId(), ipv4From, ipv4Addr, buffer, size);
}

void SimBridge::ProcessCommand(int numBytes) {
    if (numBytes <= 0) {
        cerr << "[NS3:SimBridge] ERROR: Empty package received. Ignoring." << endl;
        return;
    }

    uint8_t commandCode = m_buffer[0]; // First byte is the command code

    switch (commandCode) {
    case DO_NOTHING:
        HandleDoNothing(numBytes);
        break;

    case SET_POSITIONS:
        HandleSetPositions(numBytes);
        break;

    case REQUEST_POSITIONS:
        HandleRequestPositions(numBytes);
        break;

    case REQUEST_ADDRESSES:
        HandleRequestAddresses(numBytes);
        break;

    case REQUEST_SIM_TIME:
        HandleRequestSimTime(numBytes);
        break;

    case INGRESS_PACKET:
        HandleIngressPacket(numBytes);
        break;

    case STOP_SIMULATION:
        HandleStopSimulation(numBytes);
        break;

    default:
        cerr << "[NS3:SimBridge] ERROR: Unknown command code received: " << (int)commandCode << ". Ignoring." << endl;
        break;
    }
}

void SimBridge::HandleDoNothing(int numBytes) {
    if (numBytes > 1) {
        cerr << "[NS3:SimBridge] ERROR: Received DO_NOTHING with more than 1 byte. Ignoring." << endl;
        return;
    }
#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Executing DO_NOTHING. Sending heartbeat response." << endl;
#endif
    ReplyDoNothing();
}

void SimBridge::HandleSetPositions(int numBytes) {
    if (numBytes < 1 + 13) { // At least 1 byte for command + 1 node entry (1 byte for id + 3x4 bytes for position)
        cout << "[NS3:SimBridge] ERROR: Received SET_POSITIONS with insufficient data." << endl;
        return;
    }

    int offset = 1;                   // Start after the command byte
    while (offset + 13 <= numBytes) { // Ensure there is enough data for [id, px, py, pz]
        int nodeId = (int)m_buffer[offset];
        offset += 1;

        float x = 0.0f, y = 0.0f, z = 0.0f;
        memcpy(&x, &m_buffer[offset], sizeof(float));
        offset += sizeof(float);
        memcpy(&y, &m_buffer[offset], sizeof(float));
        offset += sizeof(float);
        memcpy(&z, &m_buffer[offset], sizeof(float));
        offset += sizeof(float);

#if DEBUG
        cout << "[NS3:SimBridge] DEBUG: Node " << nodeId << " "
             << "new position is (" << x << ", " << y << ", " << z << ")" << endl;
#endif

        m_nodesManager.SetNodePosition(nodeId, Vector(x, y, z));
    }

    if (offset != numBytes) {
        cerr << "[NS3:SimBridge] ERROR: Set positions received with extra or malformed data. Expected "
             << numBytes << " bytes but got " << offset << endl;
    }
}

void SimBridge::HandleRequestPositions(int numBytes) {
    if (numBytes > 1) {
        cout << "[NS3:SimBridge] ERROR: Positions request received with more than 1 byte. Ignoring." << endl;
        return;
    }

#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Positions request received. Replying with all positions." << endl;
#endif

    ReplyAllPositions();
    return;
}

void SimBridge::HandleStopSimulation(int numBytes) {
    if (numBytes > 1) {
        cout << "[NS3:SimBridge] ERROR: Stop simulation received with more than 1 byte. Ignoring." << endl;
        return;
    }
    cout << "[NS3:SimBridge] Stopping simulation." << endl;
    StopSimulation();
}

void SimBridge::HandleIngressPacket(int numBytes) {
    if (numBytes < 13) { // 1 byte for command + 1 byte for nodeId + 4 bytes for srcAddr + 4 bytes for dstAddr
        NS_FATAL_ERROR("[NS3:SimBridge] ERROR: Ingress packet received with insufficient data.");
        return;
    }

    int nodeId = (int)m_buffer[1];
    uint32_t srcAddr, dstAddr;

    memcpy(&srcAddr, &m_buffer[2], sizeof(uint32_t));
    srcAddr = ntohl(srcAddr);
    memcpy(&dstAddr, &m_buffer[6], sizeof(uint32_t));
    dstAddr = ntohl(dstAddr);

    size_t payloadSize = numBytes - 10; // Remaining bytes are the payload
    if (payloadSize == 0) {
        NS_FATAL_ERROR("[NS3:SimBridge] ERROR: Ingress packet received with no payload.");
        return;
    }

    uint8_t data[BUFFER_SIZE];
    memcpy(&data[0], &m_buffer[10], payloadSize);

#if TIMESTAMP
    double ingressTimestamp = Simulator::Now().GetSeconds();
    if (10 + 8 + payloadSize > BUFFER_SIZE) {
        NS_FATAL_ERROR("[NS3:SimBridge] ERROR: Payload with timestamp too large.");
        return;
    }
    memcpy(&data[payloadSize], &ingressTimestamp, sizeof(double));
    payloadSize += sizeof(double);
#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Ingress packet marked with timestamp: "
         << ingressTimestamp << endl;
#endif
#endif

#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Ingress packet received. Node " << nodeId
         << ", Source Address: " << Ipv4Address(srcAddr)
         << ", Destination Address: " << Ipv4Address(dstAddr)
         << ", Payload Size: " << payloadSize << " bytes." << endl;
#endif

    m_nodesManager.SendPacket(nodeId, Ipv4Address(dstAddr), data, payloadSize);
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
            NS_FATAL_ERROR("[NS3:SimBridge] ERROR: Buffer overflow while preparing REPLY_ALL_POSITIONS.");
            return;
        }

        Vector pos = m_nodesManager.GetNodePosition(nodeId);
        float px = static_cast<float>(pos.x);
        float py = static_cast<float>(pos.y);
        float pz = static_cast<float>(pos.z);

        response[offset++] = (uint8_t)nodeId;
        memcpy(&response[offset], &px, sizeof(float));
        offset += sizeof(float);
        memcpy(&response[offset], &py, sizeof(float));
        offset += sizeof(float);
        memcpy(&response[offset], &pz, sizeof(float));
        offset += sizeof(float);

#if DEBUG
        cout << "[NS3:SimBridge] DEBUG: Node " << nodeId << " position is "
             << pos.x << ", " << pos.y << ", " << pos.z << endl;
#endif
    }

    m_ipcSocket.SendToRemote(response, offset);
}

void SimBridge::SendEgressPacket(int nodeId, Ipv4Address srcAddr, Ipv4Address dstAddr, const uint8_t *data, size_t size) {
    uint8_t response[1024];
    size_t responseSize = 10 + size; // 1 byte for command + 1 byte for nodeId + 4 bytes for srcAddr + 4 bytes for dstAddr + payload

    if (responseSize > sizeof(response)) {
        NS_FATAL_ERROR("[NS3:SimBridge] ERROR: Response size exceeds buffer limit.");
        return;
    }

    uint32_t srcAddrUint32 = htonl(srcAddr.Get());
    uint32_t dstAddrUint32 = htonl(dstAddr.Get());

    response[0] = EGRESS_PACKET; // Command code for egress packet
    response[1] = (uint8_t)nodeId;
    memcpy(&response[2], &srcAddrUint32, sizeof(uint32_t)); // Source address
    memcpy(&response[6], &dstAddrUint32, sizeof(uint32_t)); // Destination address
    memcpy(&response[10], data, size);

#if TIMESTAMP
    double egressTimestamp = Simulator::Now().GetSeconds();
    memcpy(&response[10 + size], &egressTimestamp, sizeof(double));
    responseSize += sizeof(double);
#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Egress packet marked with timestamp: "
         << egressTimestamp << endl;
#endif
#endif

    m_ipcSocket.SendToRemote(response, responseSize);

#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Egress packet sent to remote. Node " << nodeId
         << ", Source Address: " << srcAddr
         << ", Destination Address: " << dstAddr
         << ", Payload Size: " << size << " bytes." << endl;
#endif
}

void SimBridge::HandleRequestAddresses(int numBytes) {
    if (numBytes > 1) {
        cerr << "[NS3:SimBridge] ERROR: REQUEST_ADDRESSES received with extra data. Ignoring." << endl;
        return;
    }

#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Handling REQUEST_ADDRESSES. Replying with all node addresses." << endl;
#endif

    ReplyAllAddresses();
}

void SimBridge::ReplyAllAddresses() {
    uint8_t response[BUFFER_SIZE];
    size_t offset = 0;

    response[offset++] = REPLY_ALL_ADDRESSES; // Add the reply command code

    int numNodes = m_nodesManager.GetNumNodes();
    for (int nodeId = 0; nodeId < numNodes; nodeId++) {
        Ipv4Address ipAddr = m_nodesManager.GetNodeIpAddress(nodeId);

        if (offset + 5 > BUFFER_SIZE) { // Ensure there is enough space for 1 byte (nodeId) + 4 bytes (ipAddr)
            NS_FATAL_ERROR("[NS3:SimBridge] ERROR: Buffer overflow while preparing REPLY_ALL_ADDRESSES.");
            return;
        }

        response[offset++] = static_cast<uint8_t>(nodeId); // Add nodeId (1 byte)
        uint32_t ipAddrRaw = htonl(ipAddr.Get());
        memcpy(&response[offset], &ipAddrRaw, sizeof(uint32_t)); // Add ipAddr (4 bytes)
        offset += sizeof(uint32_t);
    }

    m_ipcSocket.SendToRemote(response, offset);

#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Sent REPLY_ALL_ADDRESSES with " << (offset - 1) / 5
         << " entries (" << offset << " bytes)." << endl;
#endif
}

void SimBridge::HandleRequestSimTime(int numBytes) {
    if (numBytes > 1) {
        cerr << "[NS3:SimBridge] ERROR: REQUEST_SIM_TIME received with extra data. Ignoring." << endl;
        return;
    }
#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Handling REQUEST_SIM_TIME. Replying with simulation time." << endl;
#endif
    ReplySimTime();
}

void SimBridge::ReplySimTime() {
    uint8_t response[BUFFER_SIZE];
    size_t size = 1;

    response[0] = REPLY_SIM_TIME; // Add the reply command code

    double simTime = Simulator::Now().GetSeconds();
    memcpy(&response[size], &simTime, sizeof(double));
    size += sizeof(double);

#if DEBUG
    cout << "[NS3:SimBridge] DEBUG: Sending simulation time: " << simTime << " seconds." << endl;
#endif
    m_ipcSocket.SendToRemote(response, size);
}
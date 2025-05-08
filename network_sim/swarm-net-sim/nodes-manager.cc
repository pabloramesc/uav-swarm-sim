#include "nodes-manager.h"

#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"

#define DEBUG false

#define PORT 12345
#define BCAST true

NodesManager::NodesManager() {}

NodesManager::~NodesManager() {}

void NodesManager::Clear() {
    m_sockets.clear();
    m_nodes.clear();
#if DEBUG
    cout << "[NS3:NodesManager] DEBUG: All nodes and sockets registry cleared." << endl;
#endif
}

void NodesManager::RegisterNode(int nodeId, Ptr<Node> node) {
    if (m_nodes.find(nodeId) != m_nodes.end()) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Node ID " << nodeId << " is already registered.");
    }

    Ptr<Socket> socket = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
    InetSocketAddress localAddress(Ipv4Address::GetAny(), PORT);
    socket->Bind(localAddress);
    socket->SetAllowBroadcast(BCAST);

    m_nodes[nodeId] = node;
    m_sockets[nodeId] = socket;

#if DEBUG
    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    Ipv4Address addr = ipv4->GetAddress(1, 0).GetLocal();
    cout << "[NS3:NodesManager] DEBUG: Node " << nodeId
         << " (" << addr << ") and socket registered." << endl;
#endif
}

void NodesManager::SetNodeRxCallback(int nodeId, Callback<void, Ptr<Socket>> callback) {
    auto it = m_sockets.find(nodeId);
    if (it != m_sockets.end()) {
        Ptr<Socket> socket = it->second;
        socket->SetRecvCallback(callback);
    } else {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Node ID " << nodeId << " not found when setting callback.");
    }
}

void NodesManager::SetNodePosition(int nodeId, const Vector &pos) {
    auto it = m_nodes.find(nodeId);
    if (it == m_nodes.end()) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Node " << nodeId << " not found in registry.");
        return;
    }

    Ptr<Node> node = it->second;
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    if (mobility == nullptr) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: MobilityModel not found for Node " << nodeId);
        return;
    }

    mobility->SetPosition(pos);

#if DEBUG
    cout << "[NS3:NodesManager] DEBUG: Node " << nodeId << " position set to ("
         << pos.x << ", " << pos.y << ", " << pos.z << ")" << endl;
#endif
}

Vector NodesManager::GetNodePosition(int nodeId) const {
    auto it = m_nodes.find(nodeId);
    if (it == m_nodes.end()) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Node " << nodeId << " not found in registry.");
        return Vector(0.0, 0.0, 0.0); // Return a default Vector
    }

    Ptr<Node> node = it->second;
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    if (mobility == nullptr) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Mobility model not found for Node " << nodeId);
        return Vector(0.0, 0.0, 0.0); // Return a default Vector
    }

    Vector pos = mobility->GetPosition();

#if DEBUG
    cout << "[NS3:NodesManager] DEBUG: Node " << nodeId << " position is ("
         << pos.x << ", " << pos.y << ", " << pos.z << ")" << endl;
#endif

    return pos;
}

void NodesManager::SendPacket(int nodeId, Ipv4Address destAddr, const uint8_t *data, size_t size) {
    auto it = m_sockets.find(nodeId);
    if (it == m_sockets.end()) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Node ID " << nodeId << " not found in socket registry.");
        return;
    }

    Ptr<Socket> socket = it->second;
    Ptr<Packet> packet = Create<Packet>(data, size);
    InetSocketAddress remoteAddr = InetSocketAddress(destAddr, PORT);
    socket->Connect(remoteAddr);
    socket->Send(packet);

#if DEBUG
    Ptr<Ipv4> ipv4 = socket->GetNode()->GetObject<Ipv4>();
    Ipv4Address srcAddr = ipv4->GetAddress(1, 0).GetLocal();
    string msg((char *)data, size);
    cout << "[NS3:NodesManager] DEBUG: At " << Simulator::Now().GetSeconds() << "s "
         << "Node " << socket->GetNode()->GetId() << " (" << srcAddr << ") "
         << "sent to " << destAddr << " msg: " << msg << endl;
#endif
}

int NodesManager::GetNumNodes() const {
    return m_nodes.size();
}

Ptr<Node> NodesManager::GetNode(int nodeId) const {
    auto it = m_nodes.find(nodeId);
    if (it == m_nodes.end()) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Node ID " << nodeId << " not found.");
    }
    return it->second;
}

Ipv4Address NodesManager::GetNodeIpAddress(int nodeId) const {
    auto it = m_nodes.find(nodeId);
    if (it == m_nodes.end()) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Node ID " << nodeId << " not found in registry.");
    }

    Ptr<Node> node = it->second;
    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    if (ipv4 == nullptr) {
        NS_FATAL_ERROR("[NS3:NodesManager] ERROR: Ipv4 object not found for Node " << nodeId);
    }

    // Assuming interface 1 is used for the primary IP address
    Ipv4Address ipAddr = ipv4->GetAddress(1, 0).GetLocal();

#if DEBUG
    cout << "[NS3:NodesManager] DEBUG: Node " << nodeId << " has IP address " << ipAddr << endl;
#endif

    return ipAddr;
}
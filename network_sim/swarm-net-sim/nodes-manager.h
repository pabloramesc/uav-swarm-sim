#ifndef NODES_MANAGER_H
#define NODES_MANAGER_H

#include <map>

#include "ns3/core-module.h"
#include "ns3/network-module.h"

using namespace std;
using namespace ns3;

class NodesManager {
public:
    NodesManager();
    ~NodesManager();

    void RegisterNode(int nodeId, Ptr<Node> node);
    void SetNodeRxCallback(int nodeId, Callback<void, Ptr<Socket>>);

    void SetNodePosition(int nodeId, const Vector &pos);
    Vector GetNodePosition(int nodeId) const;
    void SendPacket(int nodeId, Ipv4Address destAddr, const uint8_t *data, size_t size);

    int GetNumNodes() const;
    Ptr<Node> GetNode(int nodeId) const;

    void Clear();

private:
    map<int, Ptr<Node>> m_nodes;     // Node registry
    map<int, Ptr<Socket>> m_sockets; // Socket registry
};

#endif // NODES_MANAGER_H
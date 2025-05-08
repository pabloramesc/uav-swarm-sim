#include "ns3/aodv-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include "sim-bridge.h" // Asegúrate de que este header esté en tu include path

using namespace ns3;

// void ReceivePacket(Ptr<Socket> socket) {
//     Address from;
//     Ptr<Packet> packet = socket->RecvFrom(from);
//     uint8_t buffer[1024];
//     packet->CopyData(buffer, packet->GetSize());
//     std::string msg((char *)buffer);

//     Ptr<Node> node = socket->GetNode();
//     Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
//     Ipv4Address ipv4Addr = ipv4->GetAddress(1, 0).GetLocal();

//     InetSocketAddress inetFrom = InetSocketAddress::ConvertFrom(from);
//     Ipv4Address ipv4From = inetFrom.GetIpv4();

//     std::cout << "Nodo " << node->GetId() << " (" << ipv4Addr << ") recibió: '"
//               << msg << "' desde " << ipv4From << std::endl;
// }

// void SendPacket(Ptr<Socket> socket, Ipv4Address destAddr) {
//     InetSocketAddress remoteAddr = InetSocketAddress(destAddr, 12345);
//     // source->SetAllowBroadcast(true);
//     socket->Connect(remoteAddr);

//     Ptr<Node> node = socket->GetNode();
//     std::string msg = "Hello from " + std::to_string(node->GetId());
//     Ptr<Packet> packet = Create<Packet>((uint8_t *)msg.c_str(), msg.length() + 1);
//     socket->Send(packet);

//     Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
//     Ipv4Address ipv4Addr = ipv4->GetAddress(1, 0).GetLocal();

//     std::cout << "Nodo " << node->GetId() << " (" << ipv4Addr << ") envió: '"
//               << msg << "' a " << destAddr << std::endl;
// }

int main(int argc, char *argv[]) {
    uint32_t nGCS = 1;
    uint32_t nUAV = 5;
    uint32_t nUser = 3;

    CommandLine cmd;
    cmd.AddValue("nGCS", "Number of GCS nodes", nGCS);
    cmd.AddValue("nUAV", "Number of UAV nodes", nUAV);
    cmd.AddValue("nUser", "Number of User nodes", nUser);
    cmd.Parse(argc, argv);

    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));
    Config::SetDefault("ns3::RealtimeSimulatorImpl::SynchronizationMode", StringValue("BestEffort"));
    Time::SetResolution(Time::NS);

    NodeContainer gcsNodes;
    NodeContainer uavNodes;
    NodeContainer userNodes;
    gcsNodes.Create(nGCS);
    uavNodes.Create(nUAV);
    userNodes.Create(nUser);

    NodeContainer allNodes;
    allNodes.Add(gcsNodes);
    allNodes.Add(uavNodes);
    allNodes.Add(userNodes);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    YansWifiPhyHelper phy;
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    phy.SetChannel(channel.Create());

    NetDeviceContainer gcsDevices = wifi.Install(phy, mac, gcsNodes);
    NetDeviceContainer uavDevices = wifi.Install(phy, mac, uavNodes);
    NetDeviceContainer userDevices = wifi.Install(phy, mac, userNodes);

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);

    AodvHelper aodv;
    Ipv4ListRoutingHelper list;
    list.Add(aodv, 100);
    InternetStackHelper stack;
    stack.SetRoutingHelper(aodv);
    stack.Install(allNodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.0.0", "0.0.1.1");
    Ipv4InterfaceContainer gcsIfaces = ipv4.Assign(gcsDevices);
    ipv4.SetBase("10.0.0.0", "255.255.0.0", "0.0.2.1");
    Ipv4InterfaceContainer uavIfaces = ipv4.Assign(uavDevices);
    ipv4.SetBase("10.0.0.0", "255.255.0.0", "0.0.3.1");
    Ipv4InterfaceContainer userIfaces = ipv4.Assign(userDevices);

    // Ipv4InterfaceContainer allIfaces;
    // allIfaces.Add(gcsIfaces);
    // allIfaces.Add(uavIfaces);
    // allIfaces.Add(userIfaces);

    SimBridge bridge(0.01);
    for (uint32_t i = 0; i < allNodes.GetN(); ++i) {
        bridge.RegisterNode(i, allNodes.Get(i));
    }
    bridge.StartPolling();

    // Ptr<Node> node = allNodes.Get(0);
    // Ptr<Socket> socket = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
    // InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), 12345);
    // socket->Bind(local);
    // socket->SetRecvCallback(MakeCallback(&ReceivePacket));
    // socket->SetAllowBroadcast(true);
    // Simulator::Schedule(Seconds(1.0), &SendPacket, socket, Ipv4Address("10.0.2.1"));

    Simulator::Run();
    Simulator::Destroy();
    return 0;
}

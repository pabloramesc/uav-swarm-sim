#include "ns3/aodv-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include "ns3/olsr-module.h"


using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiAdhocMultihop");

const uint16_t port = 54321;

void ReceivePacket(Ptr<Socket> socket) {
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);
    Ipv4Address ip = socket->GetNode()->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    std::cout << "At " << Simulator::Now().GetSeconds() << "s, Node "
              << socket->GetNode()->GetId() << " (" << ip << ") received "
              << packet->GetSize() << " bytes" << std::endl;
}

void SendPacket(Ptr<Socket> socket, Ipv4Address destAddr) {
    InetSocketAddress remoteAddr = InetSocketAddress(destAddr, port);
    socket->Connect(remoteAddr);

    std::string msg = "Hello from User1!";
    Ptr<Packet> packet = Create<Packet>((uint8_t *)msg.c_str(), msg.length() + 1);
    socket->Send(packet);

    Ipv4Address ip = socket->GetNode()->GetObject<Ipv4>()->GetAddress(1, 0).GetLocal();

    std::cout << "At " << Simulator::Now().GetSeconds() << "s, Node "
              << socket->GetNode()->GetId() << " (" << ip << ") sent "
              << packet->GetSize() << " bytes to " << destAddr << std::endl;
}

int main(int argc, char *argv[]) {
    uint32_t nDrones = 5;
    double spacing = 100.0;
    double droneHeight = 50.0;
    uint32_t nPackets = 10;
    double txInterval = 1.0;
    double txPower = 20.0;
    double pathExponent = 2.4;

    CommandLine cmd;
    cmd.AddValue("nDrones", "Number of drone nodes", nDrones);
    cmd.AddValue("spacing", "Spacing between drones", spacing);
    cmd.AddValue("droneHeight", "Height of drones above ground", droneHeight);
    cmd.AddValue("nPackets", "Number of packets to send", nPackets);
    cmd.AddValue("txInterval", "Interval between packets (s)", txInterval);
    cmd.AddValue("txPower", "Transmission power in dBm", txPower);
    cmd.AddValue("pathExponent", "Path loss exponent", pathExponent);
    cmd.Parse(argc, argv);

    std::cout << "Simulation parameters:\n";
    std::cout << "  nDrones       = " << nDrones << "\n";
    std::cout << "  spacing       = " << spacing << " m\n";
    std::cout << "  droneHeight   = " << droneHeight << " m\n";
    std::cout << "  nPackets      = " << nPackets << "\n";
    std::cout << "  txInterval    = " << txInterval << " s\n";
    std::cout << "  txPower       = " << txPower << " dBm\n";
    std::cout << "  pathExponent  = " << pathExponent << "\n";

    // Total nodes: 2 users + N drones
    NodeContainer allNodes;
    allNodes.Create(nDrones + 2);
    NodeContainer user1 = NodeContainer(allNodes.Get(0));
    NodeContainer user2 = NodeContainer(allNodes.Get(nDrones + 1));
    NodeContainer drones;
    for (uint32_t i = 1; i <= nDrones; ++i) {
        drones.Add(allNodes.Get(i));
    }

    // Set up Wi-Fi
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    Ptr<YansWifiChannel> wifiChannel = channel.Create();
    Ptr<LogDistancePropagationLossModel> lossModel = CreateObject<LogDistancePropagationLossModel>();
    Ptr<PropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel>();
    lossModel->SetPathLossExponent(pathExponent);
    lossModel->SetReference(1.0, 40.05);
    wifiChannel->SetPropagationLossModel(lossModel);
    wifiChannel->SetPropagationDelayModel(delayModel);

    YansWifiPhyHelper phy;
    phy.SetChannel(wifiChannel);
    phy.Set("TxPowerStart", DoubleValue(txPower));
    phy.Set("TxPowerEnd", DoubleValue(txPower));

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(phy, mac, allNodes);

    // Mobility
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);

    // Set positions
    allNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, 0.0)); // User 1
    for (uint32_t i = 0; i < nDrones; ++i) {
        double x = (i + 1) * spacing;
        double y = (i + 1) * spacing;
        allNodes.Get(i + 1)->GetObject<MobilityModel>()->SetPosition(Vector(x, y, droneHeight));
    }
    allNodes.Get(nDrones + 1)->GetObject<MobilityModel>()->SetPosition(Vector((nDrones + 1) * spacing, (nDrones + 1) * spacing, 0.0)); // User 2

    // Internet stack
    InternetStackHelper stack;

    AodvHelper aodv;
    stack.SetRoutingHelper(aodv);

    // OlsrHelper olsr;
    // olsr.Set("HelloInterval", TimeValue(Seconds(1.0)));
    // olsr.Set("TcInterval", TimeValue(Seconds(2.0)));
    // stack.SetRoutingHelper(olsr);

    stack.Install(allNodes);

    Ipv4AddressHelper address;
    address.SetBase("10.0.0.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // Create sockets
    Ptr<Socket> sendSocket = Socket::CreateSocket(user1.Get(0), UdpSocketFactory::GetTypeId());
    Ptr<Socket> recvSocket = Socket::CreateSocket(user2.Get(0), UdpSocketFactory::GetTypeId());

    recvSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    recvSocket->SetRecvCallback(MakeCallback(&ReceivePacket));

    // Schedule packet send
    for (uint32_t i = 0; i < nPackets; ++i) {
        Simulator::Schedule(Seconds(1.0 + i * txInterval), &SendPacket, sendSocket, interfaces.GetAddress(nDrones + 1));
    }

    double stopTime = 1.0 + nPackets * txInterval + 1.0;
    Simulator::Stop(Seconds(stopTime));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}

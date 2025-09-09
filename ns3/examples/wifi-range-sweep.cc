#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiRangeSweep");

const uint16_t port = 54321;

void ReceivePacket(Ptr<Socket> socket) {
    Address from;
    Ptr<Packet> packet = socket->RecvFrom(from);

    uint8_t buffer[1024];
    size_t size = packet->GetSize();
    packet->CopyData(buffer, size);

    Ptr<Node> node = socket->GetNode();
    Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
    Ipv4Address ipv4Addr = ipv4->GetAddress(1, 0).GetLocal();

    InetSocketAddress inetFrom = InetSocketAddress::ConvertFrom(from);
    Ipv4Address ipv4From = inetFrom.GetIpv4();

    std::cout << "At " << Simulator::Now().GetSeconds() << "s "
              << "Node " << node->GetId() << " (" << ipv4Addr << ") "
              << "received " << size << " bytes from " << ipv4From << std::endl;
}

void SendPacket(Ptr<Socket> socket, Ipv4Address destAddr) {
    InetSocketAddress remoteAddr = InetSocketAddress(destAddr, port);
    socket->Connect(remoteAddr);

    Ptr<Node> node = socket->GetNode();
    std::string msg = "This is a test message.";
    Ptr<Packet> packet = Create<Packet>((uint8_t *)msg.c_str(), msg.length() + 1);
    socket->Send(packet);
}

void MoveNode(Ptr<Node> node, double distance, double rxPowerDbm) {
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    mobility->SetPosition(Vector(distance, 0.0, 0.0));

    std::cout << "At " << Simulator::Now().GetSeconds() << "s, "
              << "Node " << node->GetId() << " moved to " << distance << " m, "
              << "Estimated RSSI: " << rxPowerDbm << " dBm" << std::endl;
}

int main(int argc, char *argv[]) {
    double txPower = 20.0;
    double pathExponent = 2.4;
    double spacing = 10.0;
    uint32_t steps = 100;

    CommandLine cmd;
    cmd.AddValue("txPower", "Transmission power in dBm", txPower);
    cmd.AddValue("pathExponent", "Path loss exponent (2.0 for free space)", pathExponent);
    cmd.AddValue("spacing", "Distance increment per second", spacing);
    cmd.AddValue("steps", "Number of sweep steps", steps);
    cmd.Parse(argc, argv);

    NodeContainer nodes;
    nodes.Create(2);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    Ptr<YansWifiChannel> wifiChannel = channel.Create();

    Ptr<LogDistancePropagationLossModel> lossModel = CreateObject<LogDistancePropagationLossModel>();
    lossModel->SetPathLossExponent(pathExponent);
    lossModel->SetReference(1.0, 40.05);

    Ptr<PropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel>();

    wifiChannel->SetPropagationLossModel(lossModel);
    wifiChannel->SetPropagationDelayModel(delayModel);

    YansWifiPhyHelper phy;
    phy.SetChannel(wifiChannel);
    phy.Set("TxPowerStart", DoubleValue(txPower));
    phy.Set("TxPowerEnd", DoubleValue(txPower));

    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    MobilityHelper mobility;
    mobility.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    Ptr<Socket> sendSocket = Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());

    Ptr<Socket> recvSocket = Socket::CreateSocket(nodes.Get(1), UdpSocketFactory::GetTypeId());
    recvSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    recvSocket->SetRecvCallback(MakeCallback(&ReceivePacket));

    for (uint32_t step = 0; step <= steps; ++step) {
        double t = step;
        double distance = spacing * step;

        Ptr<MobilityModel> txMobility = nodes.Get(0)->GetObject<MobilityModel>();
        Ptr<MobilityModel> rxMobility = nodes.Get(1)->GetObject<MobilityModel>();
        rxMobility->SetPosition(Vector(distance, 0.0, 0.0));

        double rxPowerDbm = lossModel->CalcRxPower(txPower, txMobility, rxMobility);

        Simulator::Schedule(Seconds(t), MoveNode, nodes.Get(1), distance, rxPowerDbm);
        Simulator::Schedule(Seconds(t + 0.1), SendPacket, sendSocket, interfaces.GetAddress(1));
    }

    Simulator::Stop(Seconds(steps + 1.0));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}

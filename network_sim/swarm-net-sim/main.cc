#include "ns3/aodv-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include "sim-bridge.h" // Asegúrate de que este header esté en tu include path

using namespace ns3;

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

    SimBridge bridge(0.1);
    for (uint32_t i = 0; i < allNodes.GetN(); ++i) {
        bridge.RegisterNode(i, allNodes.Get(i));
    }
    bridge.StartPolling();

    Simulator::Run();
    Simulator::Destroy();
    return 0;
}

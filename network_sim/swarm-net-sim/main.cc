#include "ns3/aodv-module.h"
#include "ns3/olsr-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include "sim-bridge.h"

using namespace ns3;

int main(int argc, char *argv[]) {
    // --- Command-line arguments ---
    uint32_t nGCS = 1; // Number of Ground Control Station nodes
    uint32_t nUAV = 5; // Number of UAV nodes
    uint32_t nUser = 3; // Number of User nodes

    CommandLine cmd;
    cmd.AddValue("nGCS", "Number of GCS nodes", nGCS);
    cmd.AddValue("nUAV", "Number of UAV nodes", nUAV);
    cmd.AddValue("nUser", "Number of User nodes", nUser);
    cmd.Parse(argc, argv);

    // --- Real time configuration ---
    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));
    Config::SetDefault("ns3::RealtimeSimulatorImpl::SynchronizationMode", StringValue("BestEffort"));
    Time::SetResolution(Time::NS);

    // --- Node creation ---
    NodeContainer gcsNodes, uavNodes, userNodes, allNodes;
    gcsNodes.Create(nGCS);
    uavNodes.Create(nUAV);
    userNodes.Create(nUser);
    allNodes.Add(gcsNodes);
    allNodes.Add(uavNodes);
    allNodes.Add(userNodes);

    // --- WiFi configuration ---
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("DsssRate1Mbps"),
                                 "ControlMode", StringValue("DsssRate1Mbps"));

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    // --- WiFi channel configuration ---
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    Ptr<YansWifiChannel> wifiChannel = channel.Create();

    Ptr<LogDistancePropagationLossModel> lossModel = CreateObject<LogDistancePropagationLossModel>();
    lossModel->SetPathLossExponent(2.4); // Free space: 2.0, urban: ~3.0–4.0
    lossModel->SetReference(1.0, 40.05); // Reference distance (m) and loss (dB)

    Ptr<PropagationDelayModel> delayModel = CreateObject<ConstantSpeedPropagationDelayModel>();
    wifiChannel->SetPropagationLossModel(lossModel);
    wifiChannel->SetPropagationDelayModel(delayModel);

    YansWifiPhyHelper phy;
    phy.SetChannel(wifiChannel);
    // phy.Set("TxPowerStart", DoubleValue(30.0));
    // phy.Set("TxPowerEnd", DoubleValue(30.0));
    // phy.Set("RxGain", DoubleValue(0));
    // phy.Set("TxGain", DoubleValue(0));

    // --- Device installation ---
    NetDeviceContainer gcsDevices = wifi.Install(phy, mac, gcsNodes);
    NetDeviceContainer uavDevices = wifi.Install(phy, mac, uavNodes);
    NetDeviceContainer userDevices = wifi.Install(phy, mac, userNodes);

    // --- Mobility configuration ---
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);

    // --- Ad-hoc routing configuration ---
    // Config::SetDefault("ns3::aodv::RoutingProtocol::HelloInterval", TimeValue(Seconds(2)));
    // Config::SetDefault("ns3::aodv::RoutingProtocol::ActiveRouteTimeout", TimeValue(Seconds(20)));
    // Config::SetDefault("ns3::aodv::RoutingProtocol::RequestRateLimit", UintegerValue(3));
    // Config::SetDefault("ns3::aodv::RoutingProtocol::ErrorRateLimit", UintegerValue(1));
    // Config::SetDefault("ns3::aodv::RoutingProtocol::EnableHello", BooleanValue(true));

    Ipv4ListRoutingHelper list;
    AodvHelper aodv;
    list.Add(aodv, 100);
    // OlsrHelper olsr;
    // list.Add(olsr, 100);
    
    InternetStackHelper stack;
    stack.SetRoutingHelper(list);
    stack.Install(allNodes);

    // --- IP address assignment ---
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.0.0.0", "255.255.0.0", "0.0.1.1");
    Ipv4InterfaceContainer gcsIfaces = ipv4.Assign(gcsDevices);
    ipv4.SetBase("10.0.0.0", "255.255.0.0", "0.0.2.1");
    Ipv4InterfaceContainer uavIfaces = ipv4.Assign(uavDevices);
    ipv4.SetBase("10.0.0.0", "255.255.0.0", "0.0.3.1");
    Ipv4InterfaceContainer userIfaces = ipv4.Assign(userDevices);

    // --- SimBridge setup ---
    SimBridge bridge(0.01); // Polling interval
    for (uint32_t i = 0; i < allNodes.GetN(); ++i) {
        bridge.RegisterNode(i, allNodes.Get(i));
    }
    bridge.StartPolling();

    // --- Simulation execution ---
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}

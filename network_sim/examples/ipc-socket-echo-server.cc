#include "ns3/core-module.h"

#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

using namespace ns3;

#define SERVER_ADDR "127.0.0.1"
#define SERVER_PORT 9000
#define BUFFER_SIZE 1024
#define POLLING_INTERVAL 0.1 // seconds

int sock;
struct sockaddr_in server_addr, client_addr;
char buffer[BUFFER_SIZE];
socklen_t client_len = sizeof(client_addr);

// Polling function scheduled by NS-3
void PollSocket() {
    int n = recvfrom(sock, buffer, sizeof(buffer), 0,
                     (struct sockaddr *)&client_addr, &client_len);

    if (n > 0) {
        buffer[n] = '\0'; // Null-terminate the string
        std::string msg(buffer);

        const char *client_ip = inet_ntoa(client_addr.sin_addr);
        unsigned short client_port = ntohs(client_addr.sin_port);

        std::cout << "[NS-3] At " << Simulator::Now().GetSeconds() << " s "
                  << "received from " << client_ip << ":" << client_port
                  << ": " << msg << std::endl;

        if (msg == "exit") {
            std::cout << "[NS-3] Received 'exit' command. Shutting down..." << std::endl;
            Simulator::Stop(); // Ends the NS-3 simulation
            return;
        }

        // Echo the message back
        sendto(sock, buffer, n, 0, (struct sockaddr *)&client_addr, client_len);
        std::cout << "[NS-3] Sent back" << std::endl;
    }

    // Re-schedule the poll
    Simulator::Schedule(Seconds(POLLING_INTERVAL), &PollSocket);
}

int main(int argc, char *argv[]) {
    CommandLine cmd;
    cmd.Parse(argc, argv);

    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));
    Config::SetDefault("ns3::RealtimeSimulatorImpl::SynchronizationMode", StringValue("BestEffort"));
    Time::SetResolution(Time::NS);

    // Create UDP socket
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "[NS-3] Error creating UDP socket" << std::endl;
        return 1;
    }

    // Set to non-blocking
    int flags = fcntl(sock, F_GETFL, 0);
    if (flags == -1 || fcntl(sock, F_SETFL, flags | O_NONBLOCK) == -1) {
        std::cerr << "[NS-3] Error setting non-blocking socket" << std::endl;
        return 1;
    }

    // Bind
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_ADDR);
    server_addr.sin_port = htons(SERVER_PORT);

    if (bind(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "[NS-3] Error binding UDP socket to " << SERVER_ADDR << ":" << SERVER_PORT << std::endl;
        return 1;
    }

    std::cout << "[NS-3] UDP Server listening on " << SERVER_ADDR << ":" << SERVER_PORT << std::endl;

    // Schedule the first polling call
    Simulator::ScheduleNow(&PollSocket);

    // Run simulation
    Simulator::Stop(Seconds(100)); // Example stop time
    Simulator::Run();
    Simulator::Destroy();

    close(sock);
    return 0;
}

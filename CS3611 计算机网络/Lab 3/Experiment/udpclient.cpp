#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <netdb.h>
#include <ifaddrs.h>
#include <sys/socket.h>

#define BUFFER_SIZE 1024 // buffer size
#define CLIENT_PORT 3366 // client port

std::map<std::string, std::string> lookup_table, reverse_table; // map host name to IP address
char receive_buffer[BUFFER_SIZE], send_buffer[BUFFER_SIZE];     // buffer for receiving and sending messages
char local_hostname[32], local_address[INET_ADDRSTRLEN];        // local hostname and address
char port_name[32];                                             // client port in string
const int terminal_handler = 0;                                 // terminal socket file descriptor
int server_handler, client_handler;                             // server and client socket file descriptor
struct sockaddr_in broadcast_descriptor;                        // broadcast descriptor
const char broadcast_address[] = "10.255.255.255";              // broadcast address

void setup() // initialize host list and socket server
{
    int return_value;

    printf("building domain name system...\n");

    sprintf(port_name, "%d", CLIENT_PORT); // convert port to string
    lookup_table["h1"] = "10.0.0.1";
    lookup_table["h2"] = "10.0.0.2";
    lookup_table["h3"] = "10.0.0.3";
    lookup_table["h4"] = "10.0.0.4";
    lookup_table["h5"] = "10.0.0.5";
    reverse_table["10.0.0.1"] = "h1";
    reverse_table["10.0.0.2"] = "h2";
    reverse_table["10.0.0.3"] = "h3";
    reverse_table["10.0.0.4"] = "h4";
    reverse_table["10.0.0.5"] = "h5";

    struct ifaddrs *config_list, *config_pointer;
    return_value = getifaddrs(&config_list); // fetch network interface list
    if (return_value == -1)                  // fail to fetch network interface list
        throw "setup.getifaddrs";

    for (config_pointer = config_list; config_pointer; config_pointer = config_pointer->ifa_next) // traverse interface list
    {
        if (strcmp(config_pointer->ifa_name, "lo") == 0) // loopback interface
            continue;

        sockaddr *current_address = config_pointer->ifa_addr; // extract address
        if (current_address->sa_family != AF_INET)            // ipv4 only
            continue;

        getnameinfo(current_address, sizeof(sockaddr_in), local_address, INET_ADDRSTRLEN, nullptr, 0, NI_NUMERICHOST); // fetch local address
        strcpy(local_hostname, reverse_table[local_address].c_str());                                                  // convert address to hostname

        break; // successfully find some required interface
    }

    freeifaddrs(config_list);      // free interface list
    if (config_pointer == nullptr) // fail to find any required interface
        throw "setup.getifaddrs";

    printf("starting local socket server...\n");

    struct addrinfo hints, *address_list, *address;
    memset(&hints, 0, sizeof(hints)); // clear hints
    hints.ai_family = AF_INET;        // ipv4 only
    hints.ai_socktype = SOCK_DGRAM;   // udp connection
    hints.ai_flags = AI_PASSIVE;      // allow bind and listen

    server_handler = socket(AF_INET, SOCK_DGRAM, 0); // create socket
    if (server_handler == -1)                        // fail to create socket
        throw "setup.socket";

    int broadcast_enabled = 1;                                                                                          // enable broadcast
    return_value = setsockopt(server_handler, SOL_SOCKET, SO_BROADCAST, &broadcast_enabled, sizeof(broadcast_enabled)); // modify socket option
    if (return_value == -1)                                                                                             // fail to modify socket option
        throw "setup.setsockopt";

    struct hostent *entry = gethostbyname(broadcast_address); // fetch broadcast
    if (entry == nullptr)                                     // fail to fetch broadcast
        throw "setup.gethostbyname";

    memset(&broadcast_descriptor, 0, sizeof(broadcast_descriptor));     // clear broadcast descriptor
    broadcast_descriptor.sin_family = AF_INET;                          // ipv4 only
    broadcast_descriptor.sin_port = htons(CLIENT_PORT);                 // client port
    broadcast_descriptor.sin_addr = *((struct in_addr *)entry->h_addr); // broadcast address

    return_value = getaddrinfo(nullptr, port_name, &hints, &address_list); // fetch local address list
    if (return_value)                                                      // fail to fetch local address list
        throw "setup.getaddrinfo";

    for (address = address_list; address != nullptr; address = address->ai_next) // traverse address list
    {
        client_handler = socket(address->ai_family, address->ai_socktype, address->ai_protocol); // create socket
        if (client_handler == -1)                                                                // fail to create socket
            continue;                                                                            // try next address

        int enabled = 1;                                                                                // enable reuse address
        return_value = setsockopt(client_handler, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)); // modify socket option
        if (return_value == -1)                                                                         // fail to modify socket option
            throw "setup.setsockopt";

        return_value = bind(client_handler, address->ai_addr, address->ai_addrlen); // bind local address to socket
        if (return_value == -1)                                                     // fail to bind local address to socket
        {
            shutdown(client_handler, 2); // close created socket
            continue;                    // try next address
        }

        break; // successfully bind to some address
    }

    freeaddrinfo(address_list); // free address list
    if (address == nullptr)     // fail to bind to any address
        throw "setup.bind";
}

void execute()
{
    int return_value;

    printf("waiting for client connection...\n");

    struct sockaddr_storage source_storage;
    socklen_t storage_length = sizeof(source_storage);
    char source_hostname[32], source_address[INET_ADDRSTRLEN];
    fd_set handler_list;
    int handler_limit = 0;

    while (true)
    {
        FD_ZERO(&handler_list);                  // clear handler list
        FD_SET(terminal_handler, &handler_list); // add terminal handler to list
        FD_SET(client_handler, &handler_list);   // add local handler to list
        if (handler_limit < client_handler)      // maximum handler increases
            handler_limit = client_handler;      // update maximum handler

        return_value = select(handler_limit + 1, &handler_list, nullptr, nullptr, nullptr); // select event
        if (return_value == -1)                                                             // fail to select event
            throw "execute.select";
        else if (return_value == 0) // no event
            continue;               // select again

        if (FD_ISSET(terminal_handler, &handler_list)) // input from local terminal
        {
            bzero(send_buffer, BUFFER_SIZE);                                                                                                                      // clear send buffer
            scanf("%[^\n]", send_buffer);                                                                                                                         // read message from terminal
            char deprecated = getchar();                                                                                                                          // absorb redundant character
            return_value = sendto(server_handler, send_buffer, strlen(send_buffer), 0, (struct sockaddr *)(&broadcast_descriptor), sizeof(broadcast_descriptor)); // send message
            if (return_value == -1)                                                                                                                               // fail to send message
                throw "execute.sendto";
        }

        if (FD_ISSET(client_handler, &handler_list)) // message from broadcast server
        {
            bzero(receive_buffer, BUFFER_SIZE);                                                                                             // clear receive buffer
            return_value = recvfrom(client_handler, receive_buffer, BUFFER_SIZE, 0, (struct sockaddr *)(&source_storage), &storage_length); // receive message
            if (return_value <= 0)                                                                                                          // fail to receive message
                throw "execute.recvfrom";

            getnameinfo((struct sockaddr *)(&source_storage), storage_length, source_address, INET_ADDRSTRLEN, nullptr, 0, NI_NUMERICHOST); // fetch target address
            strcpy(source_hostname, reverse_table[source_address].c_str());                                                                 // convert address to hostname
            if (strcmp(source_hostname, local_hostname) != 0)                                                                               // ignore local message
                printf("%s From %s\n", receive_buffer, source_hostname);                                                                    // display message
        }
    }
}

int main()
{
    try
    {
        setup();   // initialize host list and socket server
        execute(); // main loop for client
    }
    catch (const char *message)
    {
        printf("[exception] %s\n", message); // print exception message
        return 1;                            // exit with error code
    }
    return 0;
}
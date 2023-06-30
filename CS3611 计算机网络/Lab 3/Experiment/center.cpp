#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <netdb.h>
#include <sys/socket.h>

#define BACKLOG 10       // maximum pending connections allowed
#define BUFFER_SIZE 1024 // buffer size
#define CLIENT_PORT 3366 // client port

std::map<std::string, std::string> lookup_table, reverse_table; // map host name to IP address
char receive_buffer[BUFFER_SIZE], send_buffer[BUFFER_SIZE];     // buffer for receiving and sending messages
char port_name[32];                                             // client port in string
int server_handler;                                             // server socket file descriptor

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

    printf("starting local socket server...\n");

    struct addrinfo hints, *address_list, *address;
    memset(&hints, 0, sizeof(hints)); // clear hints
    hints.ai_family = AF_INET;        // ipv4 only
    hints.ai_socktype = SOCK_STREAM;  // tcp connection
    hints.ai_flags = AI_PASSIVE;      // allow bind and listen

    return_value = getaddrinfo(nullptr, port_name, &hints, &address_list); // fetch local address list
    if (return_value)                                                      // fail to fetch local address list
        throw "setup.getaddrinfo";

    for (address = address_list; address != nullptr; address = address->ai_next) // traverse address list
    {
        server_handler = socket(address->ai_family, address->ai_socktype, address->ai_protocol); // create socket
        if (server_handler == -1)                                                                // fail to create socket
            continue;                                                                            // try next address

        int enabled = 1;                                                                                // enable reuse address
        return_value = setsockopt(server_handler, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)); // modify socket option
        if (return_value == -1)                                                                         // fail to modify socket option
            throw "setup.setsockopt";

        return_value = bind(server_handler, address->ai_addr, address->ai_addrlen); // bind local address to socket
        if (return_value == -1)                                                     // fail to bind local address to socket
        {
            shutdown(server_handler, 2); // close created socket
            continue;                    // try next address
        }

        break; // successfully bind to some address
    }

    freeaddrinfo(address_list); // free address list
    if (address == nullptr)     // fail to bind to any address
        throw "setup.bind";

    return_value = listen(server_handler, BACKLOG); // start listening
    if (return_value == -1)                         // fail to start listening
        throw "setup.listen";
}

void execute()
{
    printf("waiting for client connection...\n");

    struct sockaddr_storage source_storage;
    socklen_t storage_length = sizeof(source_storage);
    struct addrinfo hints, *address_list, *address;
    char source_hostname[32], source_address[INET_ADDRSTRLEN];
    char target_hostname[32], target_address[INET_ADDRSTRLEN];
    int return_value;

    while (true)
    {
        int source_handler = accept(server_handler, (struct sockaddr *)(&source_storage), &storage_length); // accept connection
        if (source_handler == -1)                                                                           // fail to accept connection
            throw "execute.accept";

        bzero(receive_buffer, BUFFER_SIZE);                                  // clear receive buffer
        return_value = recv(source_handler, receive_buffer, BUFFER_SIZE, 0); // receive message
        if (return_value <= 0)                                               // fail to receive message
            throw "execute.recv";

        getnameinfo((struct sockaddr *)(&source_storage), storage_length, source_address, INET_ADDRSTRLEN, nullptr, 0, NI_NUMERICHOST); // fetch source address
        strcpy(source_hostname, reverse_table[source_address].c_str());                                                                 // convert address to hostname
        shutdown(source_handler, 2);                                                                                                    // close source socket

        bzero(send_buffer, BUFFER_SIZE);                                                                                       // clear send buffer
        sscanf(receive_buffer, "To %[^:]: %n", target_hostname, &return_value);                                                // parse message
        char *message_content = receive_buffer + return_value;                                                                 // parse message
        strcpy(target_address, lookup_table[target_hostname].c_str());                                                         // convert hostname to address
        sprintf(send_buffer, "%s From %s", message_content, source_hostname);                                                  // construct message
        printf("%s (%s) -> %s (%s): %s\n", source_hostname, source_address, target_hostname, target_address, message_content); // print message

        memset(&hints, 0, sizeof(hints)); // clear hints
        hints.ai_family = AF_INET;        // ipv4 only
        hints.ai_socktype = SOCK_STREAM;  // tcp connection

        return_value = getaddrinfo(target_address, port_name, &hints, &address_list); // fetch target address list
        if (return_value)                                                             // fail to fetch target address
            throw "execute.getaddrinfo";

        int target_handler;                                                          // target socket file descriptor
        for (address = address_list; address != nullptr; address = address->ai_next) // traverse address list
        {
            target_handler = socket(address->ai_family, address->ai_socktype, address->ai_protocol); // create socket
            if (target_handler == -1)                                                                // fail to create socket
                continue;                                                                            // try next addresss

            return_value = connect(target_handler, address->ai_addr, address->ai_addrlen); // establish connection
            if (return_value == -1)                                                        // fail to establish connection
            {
                shutdown(target_handler, 2); // close created socket
                continue;                    // try next address
            }

            break; // sucessfully connect to some address
        }

        freeaddrinfo(address_list); // free address list
        if (address == nullptr)     // fail to connect to any address
            throw "execute.connect";

        return_value = send(target_handler, send_buffer, BUFFER_SIZE, 0); // send message
        if (return_value == -1)                                           // fail to send message
            throw "execute.send";

        shutdown(target_handler, 2); // close target socket
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
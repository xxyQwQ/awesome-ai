#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <netdb.h>
#include <sys/socket.h>

#define BACKLOG 10       // maximum pending connections allowed
#define BUFFER_SIZE 1024 // buffer size
#define CLIENT_PORT 3366 // client port

char receive_buffer[BUFFER_SIZE], send_buffer[BUFFER_SIZE];                          // buffer for receiving and sending messages
const char server_hostname[32] = "h5", server_address[INET_ADDRSTRLEN] = "10.0.0.5"; // server hostname and address
char port_name[32];                                                                  // client port in string
const int terminal_handler = 0;                                                      // standard input file descriptor
int client_handler;                                                                  // server socket file descriptor

void setup() // initialize host list and socket server
{
    int return_value;

    sprintf(port_name, "%d", CLIENT_PORT); // convert port to string

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

    return_value = listen(client_handler, BACKLOG); // start listening
    if (return_value == -1)                         // fail to start listening
        throw "setup.listen";
}

void execute()
{
    int return_value;

    printf("waiting for server message...\n");

    struct sockaddr_storage source_storage;
    socklen_t storage_length = sizeof(source_storage);
    struct addrinfo hints, *address_list, *address;
    fd_set handler_list;
    int handler_limit = 0;

    while (true)
    {
        FD_ZERO(&handler_list);                  // clear handler list
        FD_SET(terminal_handler, &handler_list); // add terminal handler to list
        FD_SET(client_handler, &handler_list);   // add socket handler to list
        if (handler_limit < client_handler)      // maximum handler increases
            handler_limit = client_handler;      // update maximum handler

        return_value = select(handler_limit + 1, &handler_list, nullptr, nullptr, nullptr); // select event
        if (return_value == -1)                                                             // fail to select event
            throw "execute.select";
        else if (return_value == 0) // no event
            continue;               // select again

        if (FD_ISSET(terminal_handler, &handler_list)) // input from terminal
        {
            bzero(send_buffer, BUFFER_SIZE); // clear send buffer
            scanf("%[^\n]", send_buffer);    // read message from terminal
            char deprecated = getchar();     // absorb redundant character

            struct addrinfo hints, *address_list, *address; // save address information
            memset(&hints, 0, sizeof(hints));               // clear hints
            hints.ai_family = AF_INET;                      // ipv4 only
            hints.ai_socktype = SOCK_STREAM;                // tcp connection

            return_value = getaddrinfo(server_address, port_name, &hints, &address_list); // fetch target address list
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

        if (FD_ISSET(client_handler, &handler_list)) // connection from server
        {
            int source_handler = accept(client_handler, (struct sockaddr *)(&source_storage), &storage_length); // accept connection
            if (source_handler == -1)                                                                           // fail to accept connection
                throw "execute.accept";

            bzero(receive_buffer, BUFFER_SIZE);                                  // clear receive buffer
            return_value = recv(source_handler, receive_buffer, BUFFER_SIZE, 0); // receive message
            if (return_value <= 0)                                               // fail to receive message
                throw "execute.recv";

            printf("%s\n", receive_buffer); // display message
            shutdown(source_handler, 2);    // close target socket
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
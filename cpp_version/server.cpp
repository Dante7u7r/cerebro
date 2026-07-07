#include "server.hpp"
#ifndef NO_SERVER
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
typedef int SOCKET;
const SOCKET INVALID_SOCKET = -1;
const int SOCKET_ERROR = -1;
#define closesocket(s) close(s)
#endif

// Variables globales para el servidor
static std::mutex json_mutex;
static std::string json_state_str = "{}";
static std::atomic<bool> server_running(false);
static std::thread server_thread;
static SOCKET listen_socket = INVALID_SOCKET;

// Helper para comprobar si un archivo existe en disco
static bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

// Resuelve la ruta del archivo buscando en local y en directorio padre
static std::string resolve_filepath(const std::string& filename) {
    // Sanitizar el nombre del archivo para prevenir path traversal
    std::string clean_name = filename;
    if (clean_name[0] == '/') {
        clean_name = clean_name.substr(1);
    }
    if (clean_name.empty()) {
        clean_name = "index.html";
    }

    std::string path1 = "./web_visualizer/" + clean_name;
    std::string path2 = "../web_visualizer/" + clean_name;
    
    if (file_exists(path1)) return path1;
    if (file_exists(path2)) return path2;
    return "";
}

// Determinar el MIME type adecuado
static std::string get_mime_type(const std::string& filename) {
    if (filename.find(".html") != std::string::npos) return "text/html; charset=utf-8";
    if (filename.find(".css") != std::string::npos) return "text/css";
    if (filename.find(".js") != std::string::npos) return "application/javascript";
    if (filename.find(".json") != std::string::npos) return "application/json";
    return "text/plain";
}

// Procesador de cada cliente individual
static void handle_client(SOCKET client_socket) {
    std::vector<char> buffer(4096);
    int bytes_received = recv(client_socket, buffer.data(), (int)buffer.size() - 1, 0);
    if (bytes_received <= 0) {
        closesocket(client_socket);
        return;
    }
    buffer[bytes_received] = '\0';

    std::string request(buffer.data());
    std::istringstream request_stream(request);
    std::string method, path, protocol;
    request_stream >> method >> path >> protocol;

    // Solo soportamos GET por simplicidad
    if (method != "GET") {
        std::string response = "HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
        send(client_socket, response.c_str(), (int)response.length(), 0);
        closesocket(client_socket);
        return;
    }

    // Remover parámetros query si los hay (ej. ?t=123)
    size_t query_pos = path.find('?');
    if (query_pos != std::string::npos) {
        path = path.substr(0, query_pos);
    }

    // Ruta especial: sim_state.json dinámico directo desde RAM
    if (path == "/sim_state.json") {
        std::string current_json;
        {
            std::lock_guard<std::mutex> lock(json_mutex);
            current_json = json_state_str;
        }

        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << current_json.length() << "\r\n"
            << "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Connection: close\r\n\r\n"
            << current_json;

        std::string resp_str = oss.str();
        send(client_socket, resp_str.c_str(), (int)resp_str.length(), 0);
    }
    // Servir archivos estáticos
    else {
        std::string resolved_path = resolve_filepath(path);
        if (resolved_path.empty()) {
            std::string not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_socket, not_found.c_str(), (int)not_found.length(), 0);
        } else {
            std::ifstream file(resolved_path, std::ios::binary);
            if (!file.is_open()) {
                std::string forbidden = "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                send(client_socket, forbidden.c_str(), (int)forbidden.length(), 0);
            } else {
                // Leer todo el archivo
                std::stringstream file_buffer;
                file_buffer << file.rdbuf();
                std::string file_content = file_buffer.str();

                std::ostringstream oss;
                oss << "HTTP/1.1 200 OK\r\n"
                    << "Content-Type: " << get_mime_type(resolved_path) << "\r\n"
                    << "Content-Length: " << file_content.length() << "\r\n"
                    << "Connection: close\r\n\r\n";
                
                std::string headers = oss.str();
                send(client_socket, headers.c_str(), (int)headers.length(), 0);
                send(client_socket, file_content.c_str(), (int)file_content.length(), 0);
            }
        }
    }

    closesocket(client_socket);
}

// Bucle principal de escucha en el puerto
static void server_listen_loop(int port) {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "[HTTP ERROR] Fallo al iniciar Winsock.\n";
        return;
    }
#endif

    listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_socket == INVALID_SOCKET) {
        std::cerr << "[HTTP ERROR] Fallo al crear socket de escucha.\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    // Configurar reutilización de dirección
    int optval = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, (const char*)&optval, sizeof(optval));

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(listen_socket, (sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
        std::cerr << "[HTTP ERROR] Fallo en el bind del puerto " << port << ".\n";
        closesocket(listen_socket);
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    if (listen(listen_socket, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "[HTTP ERROR] Fallo en listen.\n";
        closesocket(listen_socket);
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    std::cout << "[HTTP] Servidor en ejecucion en http://localhost:" << port << " (Directorio: web_visualizer)\n";

    while (server_running) {
        sockaddr_in client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        SOCKET client_socket = accept(listen_socket, (sockaddr*)&client_addr, &client_addr_len);
        
        if (client_socket == INVALID_SOCKET) {
            // Si el servidor se detuvo, salimos pacíficamente
            if (!server_running) break;
            continue;
        }

        // Procesar cliente en un hilo separado
        std::thread(handle_client, client_socket).detach();
    }

    closesocket(listen_socket);
    listen_socket = INVALID_SOCKET;
#ifdef _WIN32
    WSACleanup();
#endif
}

void start_server(int port) {
    if (server_running) return;
    server_running = true;
    server_thread = std::thread(server_listen_loop, port);
}

void stop_server() {
    if (!server_running) return;
    server_running = false;
    
    // Cerrar socket para desbloquear accept()
    if (listen_socket != INVALID_SOCKET) {
        closesocket(listen_socket);
    }
    
    if (server_thread.joinable()) {
        server_thread.join();
    }
    std::cout << "[HTTP] Servidor detenido.\n";
}

void update_json_data(const std::string& new_json) {
    std::lock_guard<std::mutex> lock(json_mutex);
    json_state_str = new_json;
}
#else
void start_server(int port) {}
void stop_server() {}
void update_json_data(const std::string& new_json) {}
#endif

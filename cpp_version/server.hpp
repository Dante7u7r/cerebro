#ifndef SERVER_HPP
#define SERVER_HPP

#include <string>

// Inicializa e inicia el servidor web en el puerto especificado en un hilo de fondo.
void start_server(int port = 8000);

// Detiene el servidor web y libera recursos.
void stop_server();

// Actualiza el string de estado JSON en memoria de forma segura (con exclusión mutua).
void update_json_data(const std::string& new_json);

#endif // SERVER_HPP

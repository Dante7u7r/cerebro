cmake_minimum_required(VERSION 3.12)
project(CerebroSim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cerebro_sim
    main.cpp
    cerebro.cpp
    entorno.cpp
    server.cpp
    virtual_bridge.cpp
)

if(WIN32)
    target_link_libraries(cerebro_sim ws2_32)
endif()

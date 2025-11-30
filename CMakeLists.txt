cmake_minimum_required(VERSION 3.31)
project(Neural-Network LANGUAGES CXX) # cxx = c++

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Ensure the standard is strictly enforced
set(CMAKE_CXX_EXTENSIONS OFF)  # Usually good practice to turn off GNU extensions if not needed

set(APP_SOURCES
        src/main.cpp
        src/CsvReader.cpp
        src/Numpty.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/DataStorage.cpp
        src/AdamOptimiser.cpp
        src/AdamLayer.cpp
)

add_executable(Neural-Network ${APP_SOURCES})

target_include_directories(Neural-Network PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include) # setting header files location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
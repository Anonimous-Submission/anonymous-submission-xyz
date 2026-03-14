cmake_minimum_required(VERSION 3.15)
project(MAS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MAS_BUILD_LLAMA "Build llama-based agents" ON)

if(MAS_BUILD_LLAMA)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE) 
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    
    add_subdirectory(llama.cpp)
    
    message(STATUS "[MAS] Llama integration enabled")
else()
    message(STATUS "[MAS] Llama integration disabled")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(Python3_Interpreter_FOUND)
    message(STATUS "[MAS] Python found: ${Python3_EXECUTABLE}")
else()
    message(FATAL_ERROR "[MAS] Python required for TextAnalyzer")
endif()

# Core library
add_library(mas_core
    src/core/UniversalAgent.cpp
)

# Agents library
add_library(mas_agents
    src/agents/FeedbackAgent.cpp
    src/agents/InterfaceGenerator.cpp
    src/agents/TextAnalyzer.cpp
)

target_include_directories(mas_core
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(mas_agents
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR}
        ${Python3_INCLUDE_DIRS}
)

# Link core to agents
target_link_libraries(mas_agents
    PUBLIC
        mas_core
)

find_package(nlohmann_json 3.2.0 QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(mas_agents PRIVATE nlohmann_json::nlohmann_json)
    target_link_libraries(mas_core PRIVATE nlohmann_json::nlohmann_json)
    message(STATUS "[MAS] Using system nlohmann/json")
else()
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(nlohmann_json)
    target_link_libraries(mas_agents PRIVATE nlohmann_json)
    target_link_libraries(mas_core PRIVATE nlohmann_json)
    message(STATUS "[MAS] Using downloaded nlohmann/json")
endif()

if(MAS_BUILD_LLAMA)
    target_include_directories(mas_agents PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    )
    target_include_directories(mas_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    )

    target_link_libraries(mas_agents PUBLIC
        llama
        ggml
        pthread
    )
    target_link_libraries(mas_core PUBLIC
        llama
        ggml
        pthread
    )
    message(STATUS "[MAS] Linked llama.cpp libraries")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(mas_agents PRIVATE stdc++fs)
    target_link_libraries(mas_core PRIVATE stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(mas_agents PRIVATE c++fs)
    target_link_libraries(mas_core PRIVATE c++fs)
endif()

target_link_libraries(mas_agents PRIVATE ${Python3_LIBRARIES})
target_link_libraries(mas_core PRIVATE ${Python3_LIBRARIES})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/integrations)
add_custom_command(
    TARGET mas_agents POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/integrations/
        ${CMAKE_CURRENT_BINARY_DIR}/integrations/
    COMMENT "[MAS] Copying integration scripts to binary directory"
)

# Main executable (only test_orchestration now)
add_executable(test_orchestration
    tests/test_orchestration.cpp
)
target_link_libraries(test_orchestration
    PRIVATE
        mas_agents
        mas_core
)

# Optional: Keep this if you want to run tests
if(BUILD_TESTS)
    enable_testing()
    add_test(NAME OrchestrationTest COMMAND test_orchestration)
    message(STATUS "[MAS] Tests enabled")
endif()

install(TARGETS mas_core mas_agents test_orchestration
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/
    DESTINATION include/mas
    FILES_MATCHING PATTERN "*.hpp"
)

message(STATUS "[MAS] Configuration complete")

file(GLOB_RECURSE SOURCE_LIST LIST_DIRECTORIES false CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADER_LIST LIST_DIRECTORIES false CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

add_library(Source ${SOURCE_LIST} ${HEADER_LIST})

target_include_directories(Source PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(Source PRIVATE OpenMP::OpenMP_CXX)
endif ()
target_link_libraries(Source PRIVATE METAL_CPP)
file(GLOB TEST_SOURCES LIST_DIRECTORIES false CONFIGURE_DEPENDENTS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_executable(testapp ${TEST_SOURCES})

target_link_libraries(testapp PRIVATE Catch2::Catch2)
target_link_libraries(testapp PRIVATE Source)
target_link_libraries(testapp PRIVATE METAL_CPP)
# Make sure the C++ target depends on the Metal library
add_dependencies(testapp MetalLibrary)

include(CTest)
include(Catch)
catch_discover_tests(
        testapp
        TEST_PREFIX "functional:"
        EXTRA_ARGS --ProjDir ${PROJECT_SOURCE_DIR}
)
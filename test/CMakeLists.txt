file(GLOB TEST_SOURCES LIST_DIRECTORIES false CONFIGURE_DEPENDENTS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_executable(testapp ${TEST_SOURCES})

target_link_libraries(testapp PRIVATE Catch2::Catch2)
target_link_libraries(testapp PRIVATE Source)

include(CTest)
include(Catch)
catch_discover_tests(
        testapp
        TEST_PREFIX "functional:"
        EXTRA_ARGS --ProjDir ${PROJECT_SOURCE_DIR}
)
cmake_minimum_required(VERSION 3.27)
project(llm_inference)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Werror)
add_compile_options(-Wall)
# disable this warning because of size in c++ is unsigned and everything else is signed.
add_compile_options(-Wno-sign-compare)

# sanitizer flags
if (ASAN)
    add_compile_options(-fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer)
    # gcc12 build does not like regex in catch2 lib giving maybe-uninitialized error. disabling for asan build.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-Wno-error=maybe-uninitialized)
    endif ()
    add_link_options(-fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer)
endif ()

if (TSAN)
    add_compile_options(-fsanitize=thread -fPIE -pie)
    add_link_options(-fsanitize=thread -fPIE -pie)
endif ()

if (UBSAN)
    add_compile_options(-fsanitize=undefined -fno-sanitize-recover)
    add_link_options(-fsanitize=undefined -fno-sanitize-recover)
endif ()

if (ISAN)
    add_compile_options(-fsanitize=integer -fno-sanitize-recover)
    add_link_options(-fsanitize=integer -fno-sanitize-recover)
endif ()

if (NSAN)
    add_compile_options(-fsanitize=nullability -fno-sanitize-recover)
    add_link_options(-fsanitize=nullability -fno-sanitize-recover)
endif ()

if (MSAN)
    add_compile_options(-fsanitize=memory -fPIE -pie -fno-omit-frame-pointer -fno-optimize-sibling-calls)
    add_link_options(-fsanitize=memory -fPIE -pie -fno-omit-frame-pointer -fno-optimize-sibling-calls)
endif ()

# code coverage flags
if (CODE_COVERAGE)
    add_compile_options(-fprofile-arcs)
    add_compile_options(-ftest-coverage)
    add_link_options(--coverage)
endif ()

enable_testing()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/external/catch2/extras")

add_subdirectory(external/catch2)
add_subdirectory(external/metal-cmake)
add_subdirectory(src)
add_subdirectory(test)

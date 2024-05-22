
# Set the path to the metal compiler
find_program(METAL_COMPILER NAMES xcrun REQUIRED)
if (NOT METAL_COMPILER)
    message(FATAL_ERROR "Metal compiler not found. Make sure Xcode Command Line Tools are installed.")
endif ()

# Function to compile .metal files
function(compile_metal_shader INPUT_FILE OUTPUT_DIR)
    get_filename_component(SHADER_NAME ${INPUT_FILE} NAME_WE)
    set(OUTPUT_FILE "${OUTPUT_DIR}/${SHADER_NAME}.air")

    # Create command to compile shader
    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${METAL_COMPILER} -sdk macosx metal -c ${INPUT_FILE} -o ${OUTPUT_FILE}
            DEPENDS ${INPUT_FILE}
            COMMENT "Compiling Metal shader: ${INPUT_FILE}"
    )

    add_custom_target(${SHADER_NAME}_shader DEPENDS ${OUTPUT_FILE})

    set(SHADER_OUTPUT_FILE ${OUTPUT_FILE} PARENT_SCOPE)
endfunction()

# Compile shaders and collect output files
set(SHADERS
        "${CMAKE_SOURCE_DIR}/shader/Compute.metal"
        "${CMAKE_SOURCE_DIR}/shader/MatMulShader.metal"
)
set(OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/shader")

file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY})

foreach (SHADER_FILE IN LISTS SHADERS)
    compile_metal_shader(${SHADER_FILE} ${OUTPUT_DIRECTORY})
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    list(APPEND COMPILED_SHADERS "${OUTPUT_DIRECTORY}/${SHADER_NAME}.air")
endforeach ()

# Link the compiled shaders into a Metal library
add_custom_command(
        OUTPUT "${OUTPUT_DIRECTORY}/default.metallib"
        COMMAND xcrun -sdk macosx metallib ${COMPILED_SHADERS} -o "${OUTPUT_DIRECTORY}/default.metallib"
        DEPENDS ${COMPILED_SHADERS}
        COMMENT "Linking Metal shaders into metallib"
)

add_custom_target(MetalLibrary ALL DEPENDS "${OUTPUT_DIRECTORY}/default.metallib")
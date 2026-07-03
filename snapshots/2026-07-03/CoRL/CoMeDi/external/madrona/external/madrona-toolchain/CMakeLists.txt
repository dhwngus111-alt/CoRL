set(TOOLCHAIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundled-toolchain")

find_library(MADRONA_BUNDLED_LIBCXX libc++-noexcept.so.1 libc++-noexcept.1.dylib libc++-noexcept.dll
    PATHS "${TOOLCHAIN_ROOT}/libcxx-noexcept/lib"
    REQUIRED
    NO_DEFAULT_PATH
)

find_library(MADRONA_BUNDLED_LIBCXX_EXCEPT libc++-except.so.1 libc++-except.1.dylib libc++-except.dll
    PATHS "${TOOLCHAIN_ROOT}/libcxx-except/lib"
    REQUIRED
    NO_DEFAULT_PATH
)

cmake_path(GET MADRONA_BUNDLED_LIBCXX FILENAME MADRONA_LIBCXX_NAME)
cmake_path(GET MADRONA_BUNDLED_LIBCXX_EXCEPT FILENAME MADRONA_LIBCXX_EXCEPT_NAME)
set(MADRONA_LIBCXX_BUILDLOC "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${MADRONA_LIBCXX_NAME}")
set(MADRONA_LIBCXX_EXCEPT_BUILDLOC "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${MADRONA_LIBCXX_EXCEPT_NAME}")

add_custom_command(
    OUTPUT ${MADRONA_LIBCXX_BUILDLOC}
    DEPENDS ${MADRONA_BUNDLED_LIBCXX}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MADRONA_BUNDLED_LIBCXX}  ${MADRONA_LIBCXX_BUILDLOC}
)

add_custom_command(
    OUTPUT ${MADRONA_LIBCXX_EXCEPT_BUILDLOC}
    DEPENDS ${MADRONA_BUNDLED_LIBCXX_EXCEPT}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MADRONA_BUNDLED_LIBCXX_EXCEPT}  ${MADRONA_LIBCXX_EXCEPT_BUILDLOC}
)

add_custom_target(madrona_libcxx_shlib
    DEPENDS ${MADRONA_LIBCXX_BUILDLOC}
)

add_custom_target(madrona_libcxx_except_shlib
    DEPENDS ${MADRONA_LIBCXX_EXCEPT_BUILDLOC}
)

add_library(madrona_libcxx SHARED IMPORTED GLOBAL)
add_library(madrona_libcxx_except SHARED IMPORTED GLOBAL)
add_dependencies(madrona_libcxx madrona_libcxx_shlib)
add_dependencies(madrona_libcxx_except madrona_libcxx_except_shlib)
set_target_properties(madrona_libcxx PROPERTIES IMPORTED_LOCATION ${MADRONA_LIBCXX_BUILDLOC})
set_target_properties(madrona_libcxx_except PROPERTIES IMPORTED_LOCATION ${MADRONA_LIBCXX_EXCEPT_BUILDLOC})

target_compile_options(madrona_libcxx INTERFACE
    -nostdinc++ -fno-exceptions -fno-rtti
)
target_link_options(madrona_libcxx INTERFACE
    -fno-exceptions -fno-rtti
)
target_include_directories(madrona_libcxx SYSTEM INTERFACE
    $<BUILD_INTERFACE:${TOOLCHAIN_ROOT}/libcxx-noexcept/include/c++/v1>
)

target_compile_options(madrona_libcxx_except INTERFACE
    -nostdinc++
)

target_include_directories(madrona_libcxx_except SYSTEM INTERFACE
    $<BUILD_INTERFACE:${TOOLCHAIN_ROOT}/libcxx-except/include/c++/v1>
)

add_library(madrona_libcxx_static INTERFACE)

target_compile_options(madrona_libcxx_static INTERFACE
    -nostdinc++
)

target_include_directories(madrona_libcxx_static SYSTEM INTERFACE
    $<BUILD_INTERFACE:${TOOLCHAIN_ROOT}/libcxx-noexcept/include/c++/v1>
)

target_link_libraries(madrona_libcxx_static INTERFACE
    "${TOOLCHAIN_ROOT}/libcxx-noexcept/lib/libc++-noexcept.a"
)

target_link_options(madrona_libcxx_static INTERFACE
    -fno-exceptions -fno-rtti
)

add_library(madrona_libcxx_except_static INTERFACE)

target_compile_options(madrona_libcxx_except_static INTERFACE
    -nostdinc++
)

target_include_directories(madrona_libcxx_except_static SYSTEM INTERFACE
    $<BUILD_INTERFACE:${TOOLCHAIN_ROOT}/libcxx-except/include/c++/v1>
)

target_link_libraries(madrona_libcxx_except_static INTERFACE
    "${TOOLCHAIN_ROOT}/libcxx-except/lib/libc++-except.a"
)

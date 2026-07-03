cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)

if (APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
endif()

set(NAME madrona-deps-compile)
project(${NAME})

include(FetchContent)
include(ExternalProject)

get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)
include(${PARENT_DIR}/cmake/sys-detect.cmake)

find_package(Python 3.6 COMPONENTS Interpreter)

include(versions.cmake)

if (MADRONA_LINUX)
    set (SDL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/SDL")
    set (SDL_INSTALL "${SDL_PREFIX}/install")
    
    list(APPEND SDL_ARGS
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_INSTALL_PREFIX=${SDL_INSTALL}"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DSDL_SHARED=ON"
        "-DSDL_STATIC=OFF"
        "-DSDL_ASSERTIONS=release"
        "-DSDL_WAYLAND=OFF"
        "-DSDL_KMSDRM=OFF"
        "-DSDL_OPENGLES=OFF"
        "-DSDL_DISKAUDIO=OFF"
        "-DSDL_X11=ON"
        "-DSDL_OPENGL=ON"
        "-DSDL_VULKAN=ON"
    )

    ExternalProject_Add(SDL
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG "${SDL_VERSION}"
        GIT_SHALLOW ON
        PREFIX "${SDL_PREFIX}"
        INSTALL_DIR "${SDL_INSTALL}"
        CMAKE_ARGS ${SDL_ARGS}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    
    install(DIRECTORY "${SDL_INSTALL}/include/SDL2"
        DESTINATION SDL/include
    )
    
    install(DIRECTORY ${SDL_INSTALL}/lib/
        DESTINATION SDL/lib
        FILES_MATCHING
            PATTERN *
            PATTERN *pkgconfig* EXCLUDE
    )
endif()

# Vulkan SDK fetch. All platforms use the Vulkan SDK at least for the pre-compiled copy of DXC
if (MADRONA_LINUX)
    set(VULKAN_URL "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz")
    set(VULKAN_HASH "92a44158004a05fd3fba18963eebe3698442c93488e94c126361851a3ea002cf")
    set(VULKAN_NO_EXTRACT OFF)
    set(VULKAN_EXTRACT_ARG DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
elseif (MADRONA_MACOS)
    set(VULKAN_DMG_NAME "vulkansdk-macos-${VULKAN_SDK_VERSION}.dmg")
    set(VULKAN_URL "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/mac/${VULKAN_DMG_NAME}")
    set(VULKAN_HASH "18b138c3fd91d5d457ec60bf264296b04332715bb379babc90c6e9c4fa45086c")
    set(VULKAN_NO_EXTRACT ON)
    set(VULKAN_EXTRACT_ARG)
endif()

FetchContent_Declare(VulkanSDK
    URL ${VULKAN_URL}
    URL_HASH SHA256=${VULKAN_HASH}
    DOWNLOAD_NO_EXTRACT ${VULKAN_NO_EXTRACT}
    ${VULKAN_EXTRACT_ARG}
)

FetchContent_MakeAvailable(VulkanSDK)

if (MADRONA_LINUX)
    set(VULKAN_SDK_ROOT "${vulkansdk_SOURCE_DIR}/x86_64")
elseif (MADRONA_MACOS)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" "${PARENT_DIR}/scripts/extract_mac_vulkan_sdk.py" "${vulkansdk_SOURCE_DIR}/${VULKAN_DMG_NAME}" "${VULKAN_HASH}" "${vulkansdk_SOURCE_DIR}" "${sevenzip_SOURCE_DIR}/7zz"
        COMMAND_ERROR_IS_FATAL ANY
    )

    set(VULKAN_SDK_ROOT "${vulkansdk_SOURCE_DIR}/macOS")
endif()

if (MADRONA_LINUX OR MADRONA_WINDOWS)
    install(DIRECTORY ${VULKAN_SDK_ROOT}/include/vulkan
        DESTINATION vk_sdk/include/
    )

    file(GLOB vk_loader_libs ${VULKAN_SDK_ROOT}/lib/libvulkan*)

    install(FILES ${vk_loader_libs}
        DESTINATION vk_sdk/lib
    )

    file(GLOB vk_layer_libs ${VULKAN_SDK_ROOT}/lib/libVkLayer*)

    install(FILES ${vk_layer_libs}
        DESTINATION vk_sdk/layers/lib/
    )

    foreach (layer_lib ${vk_layer_libs})
        get_filename_component(layer_file "${layer_lib}" NAME)
        install(CODE "execute_process(COMMAND strip ${CMAKE_INSTALL_PREFIX}/vk_sdk/layers/lib/${layer_file})")
    endforeach()

    install(DIRECTORY ${VULKAN_SDK_ROOT}/etc/vulkan/explicit_layer.d
        DESTINATION vk_sdk/layers/
    )

    install(DIRECTORY ${VULKAN_SDK_ROOT}/include/spirv-tools
        DESTINATION spirv_tools/include/
    )

    file(GLOB spirv_tools_libs ${VULKAN_SDK_ROOT}/lib/*SPIRV-Tools-shared*)
    
    install(FILES ${spirv_tools_libs}
        DESTINATION spirv_tools/lib
    )
endif()

install(DIRECTORY ${VULKAN_SDK_ROOT}/include/dxc
    DESTINATION dxc/include/
)

file(GLOB dxc_libs ${VULKAN_SDK_ROOT}/lib/*dxcompiler*)

install(FILES ${dxc_libs}
    DESTINATION dxc/lib
)

if (MADRONA_MACOS)
    install(DIRECTORY ${VULKAN_SDK_ROOT}/include/spirv_cross
        DESTINATION spirv_cross/include/
    )

    file(GLOB spirv_cross_libs ${VULKAN_SDK_ROOT}/lib/*spirv-cross-c-shared*)
    
    install(FILES ${spirv_cross_libs}
        DESTINATION spirv_cross/lib
    )

    FetchContent_Declare(metal-cpp
        URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS13_iOS16.zip"
        URL_HASH SHA256=6f741894229e9c750add1afc3797274fc008c7507e2ae726370c17c34b7c6a68
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(metal-cpp)

    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/Foundation
        DESTINATION metal-cpp/include
    )
    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/Metal
        DESTINATION metal-cpp/include
    )
    install(DIRECTORY ${metal-cpp_SOURCE_DIR}/QuartzCore
        DESTINATION metal-cpp/include
    )
    install(FILES ${metal-cpp_SOURCE_DIR}/LICENSE.txt
        DESTINATION metal-cpp
    )

    # Want metal-cpp-extensions from this example
    # (why is it not distributed separately??)
    FetchContent_Declare(metal-cpp-extensions
        URL "https://developer.apple.com/metal/LearnMetalCPP.zip"
        URL_HASH SHA256=a709f3c0b532d5b9e3a5db3da3a35f3b783af27eb50f23a711115c02f86a256d
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
    )
    FetchContent_MakeAvailable(metal-cpp-extensions)

    set(metal_cpp_ext_dir "${metal-cpp-extensions_SOURCE_DIR}/metal-cpp-extensions")
    install(DIRECTORY "${metal_cpp_ext_dir}/AppKit"
        DESTINATION metal-cpp-extensions/include
    )
    install(DIRECTORY "${metal_cpp_ext_dir}/MetalKit"
        DESTINATION metal-cpp-extensions/include
    )

endif ()

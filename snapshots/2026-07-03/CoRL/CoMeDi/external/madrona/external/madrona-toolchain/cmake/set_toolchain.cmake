include(FetchContent)

option(MADRONA_USE_TOOLCHAIN "Use prebuilt toolchain binaries" ON)
if (NOT MADRONA_USE_TOOLCHAIN)
    return()
endif()

function(madrona_setup_toolchain)
    cmake_path(GET CMAKE_CURRENT_FUNCTION_LIST_DIR PARENT_PATH TOOLCHAIN_REPO)

    include("${TOOLCHAIN_REPO}/cmake/current-hashes.cmake")
    include("${TOOLCHAIN_REPO}/cmake/sys-detect.cmake")

    find_package(Git QUIET)
    if (NOT DEFINED MADRONA_TOOLCHAIN_VERSION)
        if (NOT Git_FOUND)
            message(FATAL_ERROR "git not found, you must set MADRONA_TOOLCHAIN_VERSION to the short hash of the toolchain commit")
        endif()
    
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
            WORKING_DIRECTORY "${TOOLCHAIN_REPO}"
            OUTPUT_VARIABLE MADRONA_TOOLCHAIN_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()

    if (MADRONA_LINUX)
        set(TOOLCHAIN_OS_NAME "linux")
        if (NOT DEFINED MADRONA_TOOLCHAIN_HASH)
            set(MADRONA_TOOLCHAIN_HASH "${MADRONA_TOOLCHAIN_LINUX_HASH}")
        endif()
    elseif (MADRONA_MACOS)
        set(TOOLCHAIN_OS_NAME "mac")
        if (NOT DEFINED MADRONA_TOOLCHAIN_HASH)
            set(MADRONA_TOOLCHAIN_HASH "${MADRONA_TOOLCHAIN_MACOS_HASH}")
        endif()
    endif()
    
    set(DEPS_URL "https://github.com/shacklettbp/madrona-toolchain/releases/download/${MADRONA_TOOLCHAIN_VERSION}/madrona-toolchain-${MADRONA_TOOLCHAIN_VERSION}-${TOOLCHAIN_OS_NAME}.tar.xz")
    
    FetchContent_Declare(MadronaBundledToolchain
        URL ${DEPS_URL}
        URL_HASH SHA256=${MADRONA_TOOLCHAIN_HASH}
        DOWNLOAD_DIR "${TOOLCHAIN_REPO}/download"
        DOWNLOAD_NAME cur.tar # Can't name it .tar.xz or CMake will ignore
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_DIR "${TOOLCHAIN_REPO}/bundled-toolchain"
        STAMP_DIR "${TOOLCHAIN_REPO}/download/timestamps"
    )
    
    FetchContent_MakeAvailable(MadronaBundledToolchain)
    
    set(TOOLCHAIN_ROOT "${madronabundledtoolchain_SOURCE_DIR}")
    set(TOOLCHAIN_SYSROOT "${madronabundledtoolchain_SOURCE_DIR}/toolchain")
    
    if (MADRONA_MACOS)
        file(GLOB TOOLCHAIN_SYSROOT "${TOOLCHAIN_SYSROOT}/Toolchains/LLVM*.xctoolchain/usr")
    endif()

    set(CMAKE_C_COMPILER "${TOOLCHAIN_SYSROOT}/bin/clang" CACHE STRING "")
    set(CMAKE_CXX_COMPILER "${TOOLCHAIN_SYSROOT}/bin/clang++" -nostdlib++ CACHE STRING "")
endfunction()

madrona_setup_toolchain()
unset(madrona_setup_toolchain)

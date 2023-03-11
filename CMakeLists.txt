cmake_minimum_required(VERSION 3.11)
project(PyCoriander)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.c src/*.h src/*.hpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu src/*.cuh)
list(REMOVE_ITEM CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/MutualInformationKraskov.cu)

include_directories(src)

if(MSVC)
    set(CMAKE_CXX_FLAGS "-W3 /EHsc")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wall")
endif()

if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    # https://cmake.org/cmake/help/git-stage/module/FindCUDA.html
    find_package(CUDA QUIET)
    if(${CUDA_FOUND})
        add_library(pycoriander SHARED ${SOURCES} ${CUDA_SOURCES})
    else()
        add_library(pycoriander SHARED ${SOURCES})
    endif()
else()
    # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
    find_package(CUDAToolkit QUIET)
    if(${CUDAToolkit_FOUND})
        add_library(pycoriander SHARED ${SOURCES} ${CUDA_SOURCES})
    else()
        add_library(pycoriander SHARED ${SOURCES})
    endif()
endif()

cmake_policy(SET CMP0012 NEW)
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)

find_package(Python3 COMPONENTS Development REQUIRED)
target_link_libraries(pycoriander PRIVATE Python3::Python)

target_link_libraries(pycoriander PRIVATE ${TORCH_LIBRARIES})
target_include_directories(pycoriander PUBLIC ${TORCH_INCLUDE_DIR})

if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    # https://cmake.org/cmake/help/git-stage/module/FindCUDA.html
    if(${CUDA_FOUND})
        enable_language(CUDA)
        set_target_properties(pycoriander PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_compile_definitions(USE_CUDA)
    endif()
else()
    # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
    if(${CUDAToolkit_FOUND})
        enable_language(CUDA)
        # https://en.wikipedia.org/wiki/CUDA
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
        #set_property(TARGET pycoriander PROPERTY CUDA_ARCHITECTURES 61 75 86)
        set_property(TARGET pycoriander PROPERTY CUDA_ARCHITECTURES 61 75)
        set_target_properties(pycoriander PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_compile_definitions(USE_CUDA)
    endif()
endif()

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


find_program(XXD xxd)
IF(XXD)
    list(APPEND CUDA_MODULES "${CMAKE_CURRENT_SOURCE_DIR}/src/PearsonCorrelation.cu")
    list(APPEND CUDA_MODULES "${CMAKE_CURRENT_SOURCE_DIR}/src/SpearmanRankCorrelation.cu")
    list(APPEND CUDA_MODULES "${CMAKE_CURRENT_SOURCE_DIR}/src/KendallRankCorrelation.cu")
    list(APPEND CUDA_MODULES "${CMAKE_CURRENT_SOURCE_DIR}/src/MutualInformationBinned.cu")
    list(APPEND CUDA_MODULES "${CMAKE_CURRENT_SOURCE_DIR}/src/MutualInformationKraskov.cu")
    foreach(CUDA_MODULE IN LISTS CUDA_MODULES)
        get_filename_component(CUDA_MODULE_NAME "${CUDA_MODULE}" NAME_WE)
        set(OUTPUT "${CMAKE_SOURCE_DIR}/src/${CUDA_MODULE_NAME}Header.hpp")
        add_custom_command(
                OUTPUT "${OUTPUT}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
                COMMAND ${XXD} -i "${CUDA_MODULE_NAME}.cu" > "${OUTPUT}"
                MAIN_DEPENDENCY "${CUDA_MODULE}"
                VERBATIM
        )
        list(APPEND CUDA_HEADER_FILES "${OUTPUT}")
    endforeach()
    add_custom_target(PREPARE_CUDA_MODULES ALL DEPENDS ${CUDA_HEADER_FILES})
    add_dependencies(pycoriander PREPARE_CUDA_MODULES)
endif()


install(
        TARGETS pycoriander
        ARCHIVE DESTINATION modules
        LIBRARY DESTINATION modules
        RUNTIME DESTINATION bin
)

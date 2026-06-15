cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(Theseus)

option(BUILD_TESTING "Build unit tests" ON)
# option(PARABOLIC "Enable Compressible Navier-Stokes Solver" OFF)
option(SUBCELL_FV_BLENDING "Enable subcell blending" ON)
option(AXISYMMETRIC "Enable Axisymmetric domain solver" OFF)
option(SUTHERLAND "Enable Sutherland transport model" OFF)
option(ENABLE_TIMERS "Enable performance timers." OFF)
option(ENABLE_CUDA "Compile CUDA kernels" OFF)

find_package(MFEM REQUIRED CONFIG) 
if (TARGET MFEM::mfem)
  set(THESEUS_MFEM_TARGET MFEM::mfem)
elseif (TARGET mfem)
  set(THESEUS_MFEM_TARGET mfem)
else()
  message(FATAL_ERROR "MFEM package found, but no MFEM target (MFEM::mfem or mfem) was provided.")
endif()
get_target_property(_mfem_inc ${THESEUS_MFEM_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_mfem_defs ${THESEUS_MFEM_TARGET} INTERFACE_COMPILE_DEFINITIONS)
get_target_property(_mfem_libs ${THESEUS_MFEM_TARGET} INTERFACE_LINK_LIBRARIES)
message(STATUS "THESEUS_MFEM_TARGET = ${THESEUS_MFEM_TARGET}")
message(STATUS "MFEM include dirs = ${_mfem_inc}")
message(STATUS "MFEM compile defs = ${_mfem_defs}")
message(STATUS "MFEM link libs = ${_mfem_libs}")

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(theseus src/main.cpp)

if (ENABLE_CUDA)
   message(STATUS "Building ${PROJECT_NAME} for CUDA platform.")
   set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
   enable_language(CUDA)
   set_source_files_properties(
     src/Operators/DGSEMOperator.cpp
     src/Operators/NonlinearForm/DGSEMNonlinearForm.cpp
     PROPERTIES LANGUAGE CUDA
   )
   set_property(TARGET theseus PROPERTY CUDA_ARCHITECTURES 80)
   target_include_directories(theseus PRIVATE
     $<$<COMPILE_LANGUAGE:CUDA>:${MPI_CXX_INCLUDE_DIRS}>
   )
   target_compile_options(theseus PRIVATE
     $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>
   )
endif()

# --- Python Script for Compile Definitions ---
set(CONFIG_FILE "${CONFIG_FILE}" CACHE STRING "Path to config file")
if(IS_ABSOLUTE "${CONFIG_FILE}")
    set(FULL_CONFIG_PATH "${CONFIG_FILE}")
else()
    set(FULL_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_FILE}")
endif()
message(STATUS "CONFIG_FILE = ${CONFIG_FILE}")
if(NOT "${CONFIG_FILE}" STREQUAL "")
    message(STATUS "Parsing configuration file: ${FULL_CONFIG_PATH}")
    execute_process(
        COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/parse_compile_config.py" "${FULL_CONFIG_PATH}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE COMPILE_DEFINES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )       
    string(REPLACE "\n" ";" COMPILE_DEFINES_LIST "${COMPILE_DEFINES}")
    message(STATUS "COMPILE_DEFINES_LIST = ${COMPILE_DEFINES_LIST}")

    target_compile_definitions(theseus
        PRIVATE ${COMPILE_DEFINES_LIST}
    )
else()
    message(STATUS "No configuration file given.")
endif()

if(ENABLE_TIMERS)
  message(STATUS "Enabling performance timers.")
  target_compile_definitions(theseus PRIVATE ENABLE_TIMERS)
endif()

if(SUBCELL_FV_BLENDING)
  message(STATUS "Building ${PROJECT_NAME} with: Subcell FV Blending enabled")
  target_compile_definitions(theseus PRIVATE SUBCELL_FV_BLENDING)
else()
  message(STATUS "Building ${PROJECT_NAME} with: Subcell FV Blending OFF")
endif()

if(SUTHERLAND)
  message(STATUS "Building ${PROJECT_NAME} with: Sutherland transport model enabled")
  target_compile_definitions(theseus PRIVATE SUTHERLAND)
endif()

if(AXISYMMETRIC)
  message(STATUS "Building ${PROJECT_NAME} with: Axisymmetric domain enabled")
  target_compile_definitions(theseus PRIVATE AXISYMMETRIC)
endif()

# -------------------------------------------------------------------------------
# Target-local sanitizers for a single executable (Clang/AppleClang/GNU)
# -------------------------------------------------------------------------------

option(ENABLE_ASAN_UBSAN "Enable ASan+UBSan for the main executable" OFF)
option(NO_OPT "Disable optimizations for testing" OFF)


target_compile_definitions(theseus PRIVATE
  "$<$<OR:$<CONFIG:Debug>,$<BOOL:${NO_OPT}>,$<BOOL:${ENABLE_ASAN_UBSAN}>>:MFEM_DEBUG>"
)

if (ENABLE_ASAN_UBSAN AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  message(STATUS "Enabling ASan+UBSan on target: ${PROJECT_NAME}")

  # Compile flags
  target_compile_options(theseus PRIVATE
    -O1
    -g
    -fno-fast-math
    -ffp-contract=off
    -fno-unsafe-math-optimizations
    -fno-associative-math
    -fno-reciprocal-math
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
    -fsanitize=address,undefined
  )

  # Link flags
  target_link_options(theseus PRIVATE
    -fsanitize=address,undefined
  )

  # Prefer hard-fail on first UB to get the cleanest stack trace
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    target_compile_options(theseus PRIVATE -fno-sanitize-recover=all)
  endif()
endif()

if (NO_OPT AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  message(STATUS "Disabling all optimizations on target: theseus")

  # Compile flags
  target_compile_options(theseus PRIVATE
    -O0
    -g
    -fno-fast-math
    -ffp-contract=off
    -fno-unsafe-math-optimizations
    -fno-associative-math
    -fno-reciprocal-math
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
  )

endif()

target_sources(theseus
    PRIVATE src/ModalBasis.cpp
    PRIVATE src/Indicator.cpp
    PRIVATE src/PerssonPeraireIndicator.cpp
    PRIVATE src/Simulation.cpp
    PRIVATE src/DGSEMNonlinearForm.cpp
)

message(STATUS "Adding ${CMAKE_PREFIX_PATH}/include/mfem to include path")
target_include_directories(theseus
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_PREFIX_PATH}/include/mfem
)

target_link_libraries(theseus
    PRIVATE
    ${THESEUS_MFEM_TARGET}
    MPI::MPI_CXX
)

if (BUILD_TESTING)
  include(CTest)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
  endif()
endif()

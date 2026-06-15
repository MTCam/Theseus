# This file is part of the unit testing infrastructure for Theseus
#
# Example on how to add unit tests
# add_executable(suite_test_exe
#     unit_test_main.cpp
#     suite_tests.cpp
# )
#
# Run the whole suite of tests like this:
# add_test(NAME test_suite_name COMMAND suite_test_exe)
# Select a single test from the suite like this:
# add_test(NAME suite::test1 COMMAND suite_test_exe test1)
#

# Unit testing self-test suite
add_executable(unit_testing_stub
    unit_test_main.cpp
    test_stub.cpp
)

add_test(NAME UnitTestingInfrastructure COMMAND unit_testing_stub)
add_test(NAME SingleUnitTestInfrastructure COMMAND unit_testing_stub unit_testing_funcs)

# Gas Physics testing suite
add_executable(gas_tests
   unit_test_main.cpp
   eos_tests.cpp
   transport_tests.cpp
   gas_model_tests.cpp
)

target_include_directories(gas_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_PREFIX_PATH}/include/mfem
)

# Propagate SUTHERLAND compile definition to gas_tests if enabled
if(SUTHERLAND)
    target_compile_definitions(gas_tests PRIVATE SUTHERLAND)
endif()

target_link_libraries(gas_tests PRIVATE ${THESEUS_MFEM_TARGET})

add_test(NAME GasPhysicsTestingSuite COMMAND gas_tests)

# State unit testing suite
add_executable(state_tests
   unit_test_main.cpp
   state_tests.cpp
)

target_include_directories(state_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_PREFIX_PATH}/include/mfem
)

target_link_libraries(state_tests PRIVATE ${THESEUS_MFEM_TARGET})

add_test(NAME StateTestingSuite COMMAND state_tests)

# Riemann solvers tests
add_executable(rflux_tests
   unit_test_main.cpp
   test_riemann_fluxes.cpp
)

target_include_directories(rflux_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_PREFIX_PATH}/include/mfem
)

target_link_libraries(rflux_tests PRIVATE ${THESEUS_MFEM_TARGET})

add_test(NAME RiemannTestingSuite COMMAND rflux_tests)

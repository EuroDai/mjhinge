cmake_minimum_required(VERSION 3.20)
project(mjhinge VERSION 0.2.0 LANGUAGES CXX)

# Prefer the CMake package from an official MuJoCo SDK installation.
find_package(mujoco CONFIG QUIET)
if(NOT TARGET mujoco::mujoco)
  set(MUJOCO_ROOT "" CACHE PATH "MuJoCo SDK directory containing include/ and lib/")
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
    HINTS "${MUJOCO_ROOT}" ENV MUJOCO_ROOT
    PATH_SUFFIXES include REQUIRED)
  find_library(MUJOCO_LIBRARY NAMES mujoco
    HINTS "${MUJOCO_ROOT}" ENV MUJOCO_ROOT
    PATH_SUFFIXES lib bin REQUIRED)
  add_library(mujoco::mujoco UNKNOWN IMPORTED)
  set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME hinge_friction
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hinge_friction.py"
        --plugin "$<TARGET_FILE:mjhinge>")
  endif()
endif()

add_library(mjhinge SHARED src/hinge_friction.cc)
target_compile_features(mjhinge PRIVATE cxx_std_17)
target_link_libraries(mjhinge PRIVATE mujoco::mujoco)
set_target_properties(mjhinge PROPERTIES
  CXX_EXTENSIONS OFF
  PREFIX "")

if(MSVC)
  target_compile_options(mjhinge PRIVATE /W4)
else()
  target_compile_options(mjhinge PRIVATE -Wall -Wextra -Wpedantic)
endif()


cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(DEFINED ENV{VITASDK})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VITASDK}/share/vita.toolchain.cmake" CACHE PATH "toolchain file")
  else()
    message(FATAL_ERROR "Please define VITASDK to point to your SDK path!")
  endif()
endif()

project(simulation)
include("${VITASDK}/share/vita.cmake" REQUIRED)

set(VITA_APP_NAME "simulation")
set(VITA_TITLEID  "PSVSPHSIM")
set(VITA_VERSION  "01.00")

# compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O2 -ftree-vectorize")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -ftree-vectorize")

# code to compile
add_executable(${PROJECT_NAME} src/main.cpp)

# third party headers
target_include_directories(${PROJECT_NAME} PRIVATE include)

# dynamic libs to link against
target_link_libraries(${PROJECT_NAME}
  vitaGL
  c
  SceCommonDialog_stub
  SceLibKernel_stub
  SceDisplay_stub
  SceGxm_stub
  SceSysmodule_stub
  SceCtrl_stub
  SceTouch_stub
  m
  SceAppMgr_stub
  SceAppUtil_stub
  mathneon
  SceKernelDmacMgr_stub
  vitashark
  SceShaccCgExt
  taihen_stub
  SceShaccCg_stub
  SceVshBridge_stub
  stdc++
  "-Wl,--whole-archive" pthread "-Wl,--no-whole-archive"
  zip
  z
)

vita_create_self(eboot.bin ${PROJECT_NAME})
vita_create_vpk(${PROJECT_NAME}.vpk ${VITA_TITLEID} eboot.bin VERSION ${VITA_VERSION} NAME ${VITA_APP_NAME})
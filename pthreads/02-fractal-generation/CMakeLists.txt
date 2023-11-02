cmake_minimum_required(VERSION 3.5)
project(mandelbrot CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-m64 -O3 -Wall -fPIC")

set(APP_NAME mandelbrot)
set(OBJDIR src)

add_executable(${APP_NAME} "")

target_include_directories(${APP_NAME} PRIVATE ${OBJDIR})

target_sources(${APP_NAME}
    PRIVATE
        ${OBJDIR}/main.cpp
        ${OBJDIR}/mandelbrotSerial.cpp
        ${OBJDIR}/mandelbrotThread.cpp
        ${OBJDIR}/ppm.cpp
)

target_link_libraries(${APP_NAME} PRIVATE pthread m)
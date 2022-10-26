add_executable(splinegui 
splinegui.cpp app.h app.cpp)

set_target_properties(splinegui     PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(splinegui    PUBLIC ${CMAKE_SOURCE_DIR}/libs)

target_link_libraries(splinegui yocto splinesurf yocto_gui realtime)

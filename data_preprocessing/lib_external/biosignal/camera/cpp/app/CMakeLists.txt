
# opencv
find_package( OpenCV REQUIRED ) 

# cam
add_library(cam STATIC ${CMAKE_CURRENT_SOURCE_DIR}/cam.cc)
target_compile_options(cam PRIVATE -Wall -Werror -O2)
target_compile_features(cam PRIVATE cxx_std_11)
target_link_libraries(cam ${OpenCV_LIBS})

# signalgo
add_library(signalgo STATIC ${CMAKE_CURRENT_SOURCE_DIR}/signalgo.cc)
target_compile_options(signalgo PRIVATE -Wall -Werror -O2)
target_compile_features(signalgo PRIVATE cxx_std_11)

# resp                             
add_executable(resp ${CMAKE_CURRENT_SOURCE_DIR}/resp.cc)                      
target_compile_options(resp PUBLIC -Wall -Werror -O2)
target_compile_features(resp PRIVATE cxx_std_11)
target_link_libraries(resp ${OpenCV_LIBS})
target_link_libraries(resp cam)
target_link_libraries(resp signalgo)

# heart
add_executable(heart ${CMAKE_CURRENT_SOURCE_DIR}/heart.cc)
target_compile_options(heart PUBLIC -Wall -Werror -O2)
target_compile_features(heart PRIVATE cxx_std_11)
target_link_libraries(heart ${OpenCV_LIBS})
target_link_libraries(heart cam)
target_link_libraries(heart signalgo)

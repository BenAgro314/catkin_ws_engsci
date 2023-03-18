# CMake generated Testfile for 
# Source directory: /home/agrobenj/catkin_ws/src/avoidance/avoidance
# Build directory: /home/agrobenj/catkin_ws/build/avoidance
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_avoidance_gtest_avoidance-test "/home/agrobenj/catkin_ws/build/avoidance/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/agrobenj/catkin_ws/build/avoidance/test_results/avoidance/gtest-avoidance-test.xml" "--return-code" "/home/agrobenj/catkin_ws/devel/.private/avoidance/lib/avoidance/avoidance-test --gtest_output=xml:/home/agrobenj/catkin_ws/build/avoidance/test_results/avoidance/gtest-avoidance-test.xml")
set_tests_properties(_ctest_avoidance_gtest_avoidance-test PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;98;catkin_run_tests_target;/opt/ros/noetic/share/catkin/cmake/test/gtest.cmake;37;_catkin_add_google_test;/home/agrobenj/catkin_ws/src/avoidance/avoidance/CMakeLists.txt;89;catkin_add_gtest;/home/agrobenj/catkin_ws/src/avoidance/avoidance/CMakeLists.txt;0;")
subdirs("gtest")

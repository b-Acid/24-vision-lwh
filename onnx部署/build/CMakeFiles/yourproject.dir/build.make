# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/bacid/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/bacid/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bacid/文档/code/10.5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bacid/文档/code/10.5/build

# Include any dependencies generated for this target.
include CMakeFiles/yourproject.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/yourproject.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/yourproject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yourproject.dir/flags.make

CMakeFiles/yourproject.dir/test.cpp.o: CMakeFiles/yourproject.dir/flags.make
CMakeFiles/yourproject.dir/test.cpp.o: /home/bacid/文档/code/10.5/test.cpp
CMakeFiles/yourproject.dir/test.cpp.o: CMakeFiles/yourproject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bacid/文档/code/10.5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yourproject.dir/test.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yourproject.dir/test.cpp.o -MF CMakeFiles/yourproject.dir/test.cpp.o.d -o CMakeFiles/yourproject.dir/test.cpp.o -c /home/bacid/文档/code/10.5/test.cpp

CMakeFiles/yourproject.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yourproject.dir/test.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bacid/文档/code/10.5/test.cpp > CMakeFiles/yourproject.dir/test.cpp.i

CMakeFiles/yourproject.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yourproject.dir/test.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bacid/文档/code/10.5/test.cpp -o CMakeFiles/yourproject.dir/test.cpp.s

# Object files for target yourproject
yourproject_OBJECTS = \
"CMakeFiles/yourproject.dir/test.cpp.o"

# External object files for target yourproject
yourproject_EXTERNAL_OBJECTS =

yourproject: CMakeFiles/yourproject.dir/test.cpp.o
yourproject: CMakeFiles/yourproject.dir/build.make
yourproject: /home/bacid/文档/onnxruntime/lib/libonnxruntime.so
yourproject: /usr/local/lib/libopencv_highgui.so.4.7.0
yourproject: /usr/local/lib/libopencv_ml.so.4.7.0
yourproject: /usr/local/lib/libopencv_objdetect.so.4.7.0
yourproject: /usr/local/lib/libopencv_photo.so.4.7.0
yourproject: /usr/local/lib/libopencv_stitching.so.4.7.0
yourproject: /usr/local/lib/libopencv_video.so.4.7.0
yourproject: /usr/local/lib/libopencv_videoio.so.4.7.0
yourproject: /usr/local/lib/libopencv_imgcodecs.so.4.7.0
yourproject: /usr/local/lib/libopencv_calib3d.so.4.7.0
yourproject: /usr/local/lib/libopencv_dnn.so.4.7.0
yourproject: /usr/local/lib/libopencv_features2d.so.4.7.0
yourproject: /usr/local/lib/libopencv_flann.so.4.7.0
yourproject: /usr/local/lib/libopencv_imgproc.so.4.7.0
yourproject: /usr/local/lib/libopencv_core.so.4.7.0
yourproject: CMakeFiles/yourproject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bacid/文档/code/10.5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable yourproject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yourproject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yourproject.dir/build: yourproject
.PHONY : CMakeFiles/yourproject.dir/build

CMakeFiles/yourproject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yourproject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yourproject.dir/clean

CMakeFiles/yourproject.dir/depend:
	cd /home/bacid/文档/code/10.5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bacid/文档/code/10.5 /home/bacid/文档/code/10.5 /home/bacid/文档/code/10.5/build /home/bacid/文档/code/10.5/build /home/bacid/文档/code/10.5/build/CMakeFiles/yourproject.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yourproject.dir/depend


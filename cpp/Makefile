#!/bin/make
CXX = g++
CXXFLAGS = -O3 -shared -Wall -march=native -mtune=native -mavx2 -mfma -g -finline-functions -ffast-math
# -mavx2 -mfma
CPPFLAGS = -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64
INCLUDES = /home/rh/rh_dir/rh_env/include/python3.6m  `python3.6m-config --cflags --libs --extension-suffix`
LIBS = -lm -lopenblas -llapack -fopenmp -lpthread -lgsl -lgslcblas
OUTPUT_FILE = ghlda.so
OBJECT_FILES = ghlda.cpp

$(OUTPUT_FILE) : $(OBJECT_FILES)
	$(CXX) $(CXXFLAGS) -std=c++14 -fPIC $(CPPFLAGS) $(OBJECT_FILES) -o $(OUTPUT_FILE) $(LIBS)

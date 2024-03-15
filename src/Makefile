PROGRAM   = main
CXX       = g++
CXXFLAGS  = -g -std=c++17 -Ofast -Wall

$(PROGRAM): main.o vns.o greedy.o utils.o
	$(CXX) $(CXXFLAGS) -o $(PROGRAM) $^

pybind:
	c++ -O3 -Wall -shared -std=c++17 -fPIC $$(python3 -m pybind11 --includes) python_binding.cpp vns.cpp greedy.cpp utils.cpp -o ksrdp$$(python3-config --extension-suffix)

.PHONY: clean

clean:
	-rm -f *.o $(PROGRAM)
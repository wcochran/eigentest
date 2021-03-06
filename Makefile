CXX=g++
#CXXOPTS=-Wall -std=c++14
CXXOPTS=-O3 -std=c++14 -DNDEBUG
ALL=eigentest

all: $(ALL)

JUNK=*~ *.o *.dSYM

clean:
	-rm -rf $(JUNK)

clobber:
	-rm -rf $(JUNK) $(ALL)

eigentest: eigentest.cpp
	$(CXX) $(CXXOPTS) $^ -o $@

eigentest2: eigentest2.cpp
	$(CXX) $(CXXOPTS) $^ -o $@

test: eigentest
	./eigentest

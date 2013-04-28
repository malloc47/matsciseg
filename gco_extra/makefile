CXX = g++
#CCFLAGS = -pthread -fno-strict-aliasing -march=x86-64 -mtune=generic -O2 -pipe -DNDEBUG -march=x86-64 -mtune=generic -O2 -pipe -fPIC
CCFLAGS = -fPIC -Wall -fpermissive
#LDFLAGS = -pthread -shared -Wl,--hash-style=gnu -Wl,--as-needed -Wl,-Bsymbolic
LDFLAGS = 
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

all: gco 

gco: $(OBJECTS)
	${CXX} ${CCFLAGS} GCoptimization.o maxflow.o graph.o LinkedBlockList.o example.o -o example

%.o: %.cpp
	${CXX} -c $(CCFLAGS) $< -o $@

library:
	ar rcs libgco.a *.o

clean:
	-rm *.o
	-rm example
	-rm *.a

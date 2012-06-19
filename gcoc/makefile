CPP = g++
CCFLAGS = -pthread -fno-strict-aliasing -march=x86-64 -mtune=generic -O2 -pipe -DNDEBUG -march=x86-64 -mtune=generic -O2 -pipe -fPIC -Wl,-no-undefined
INCLUDE = -I/usr/lib/python2.7/site-packages/numpy/core/include/ -I/usr/include/python2.7/ -I/home/malloc47/src/programs/gco
LDFLAGS = -pthread -shared -Wl,--hash-style=gnu -Wl,--as-needed -Wl,-Bsymbolic
LIBINCLUDE = -L/home/malloc47/src/programs/gco -lgco -L/usr/lib -lpython2.7
# LIBINCLUDE = -L/usr/lib -lpython2.7
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
# GCO_OBJECTS = $(wildcard /home/malloc47/src/programs/gco/*.o)
EXEC = $(SOURCES:.cpp=.so)

all: $(EXEC)

%.so: %.o
	$(CPP) $(LDFLAGS) $< $(LIBINCLUDE) -o $@

%.o: %.cpp
	$(CPP) -c $(CCFLAGS) $< -o $@ $(INCLUDE)

clean:
	-rm -f $(EXEC) $(OBJECTS)
	-rm *~

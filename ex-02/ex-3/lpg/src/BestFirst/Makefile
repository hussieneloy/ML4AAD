###########################
# Makefile for BestFirst  #
###########################


####### FLAGS

CFLAGS = -O3 -DNDEBUG

CC	= gcc 

####### INCLUDE CONFIGURATION FILE

include ../configure.make


####### COMPILE

PATH_HEADERS = -I$(SRC_HEADER)

SRC_HEADER =../include

SOURCES = relax.c \
	  output.c \
	  search.c

OBJECTS = $(SOURCES:.c=.o)

find_header = $(wildcard $(dir)/*.h)

HEADERS := $(foreach dir,../include,$(find_header))


####### Implicit rules

.SUFFIXES:

.SUFFIXES: .c .o

.c.o:; $(CC) $(PATH_HEADERS) -c $(CFLAGS) $<


####### Build rules

BestFirst.a: 	$(OBJECTS)
		$(AR) rc $@ $^ 

$(OBJECTS): $(HEADERS)

clean:
	rm -f *.o *.a *~

#######

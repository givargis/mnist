#
# Tony Givargis
# Copyright (C), 2023
# University of California, Irvine
#
# Makefile
#

CC     = gcc
CFLAGS = -ansi -pedantic -Wall -Wextra -Werror -Wfatal-errors -O3
LDLIBS = -lm
DEST   = mnist
SRCS  := $(wildcard *.c)
OBJS  := $(SRCS:.c=.o)

all: $(OBJS)
	@echo "[LN]" $(DEST)
	@$(CC) -o $(DEST) $(OBJS) $(LDLIBS)

%.o: %.c
	@echo "[CC]" $<
	@$(CC) $(CFLAGS) -c $<
	@$(CC) $(CFLAGS) -MM $< > $*.d

clean:
	@rm -f $(DEST) *.o *.d *~ *#

-include $(OBJS:.o=.d)

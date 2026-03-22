CC ?= cc
CFLAGS ?= -O2
CFLAGS += -std=c11 -Wall -Wextra -pedantic
LDFLAGS ?=
LDLIBS ?= -lm

TARGET ?= app
SRC := $(wildcard src/*.c) main.c

.PHONY: all build clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $(SRC) $(LDLIBS)

clean:
	$(RM) $(TARGET)


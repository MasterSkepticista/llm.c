# somehow `cc` faster than `clang` in math mode
CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDLIBS = -lm
CFLAGS_COND = -march=native

$(info ---------------------------------------------)
# Check for USE_OMP environment variable
ifeq ($(NO_OMP), 1)
  $(info OpenMP is manually disabled)
else
  # Add OpenMP flags if supported
  ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
    CFLAGS += -fopenmp -DOMP
    LDLIBS += -lgomp
    $(info OpenMP support detected. Enabling OpenMP flags.)
  else
    $(info OpenMP not found. Compiling without OpenMP.)
  endif
endif
$(info ---------------------------------------------)

TARGETS = train_gpt2 matmul

all: $(TARGETS)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(CFLAGS_COND) $^ $(LDLIBS) -o $@

matmul: matmul.c
	$(CC) $(CFLAGS) $(CFLAGS_COND) $^ $(LDLIBS) -o $@

clean:
	rm -f $(TARGETS)

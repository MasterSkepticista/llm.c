CC = clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDLIBS = -lm
CFLAGS_COND = -march=native

$(info ---------------------------------------------)

# Add OpenMP flags if supported
ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
  CFLAGS += -fopenmp -DOMP
  LDLIBS += -lgomp
  $(info OpenMP support detected. Enabling OpenMP flags.)
else
  $(info OpenMP support not detected. Compiling without OpenMP.)
endif
$(info ---------------------------------------------)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(LDLIBS) $(CFLAGS_COND) $^ -o $@

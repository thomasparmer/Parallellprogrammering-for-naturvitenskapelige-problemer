CC = mpicc
CFLAGS = -fopenmp -O3
PROJ = parallel_smart_memory_main
OBJS = parallel_smart_memory_main.o

all : $(PROJ)

parallel_smart_memory_main : $(OBJS)
	$(CC) $(CFLAGS) $< -o $@

obj-clean:
	$(RM) *.o

exec-clean:
	$(RM) $(PROJ)

autosave-clean:
	$(RM) *~

clean:
	$(MAKE) obj-clean
	$(MAKE) exec-clean
	$(MAKE) autosave-clean

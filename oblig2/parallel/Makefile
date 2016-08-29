CC = mpicc
CFLAGS = -g -Wall
PROJ = parallel_main
OBJS = parallel_main.o

all : $(PROJ)

parallel_main : $(OBJS)
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

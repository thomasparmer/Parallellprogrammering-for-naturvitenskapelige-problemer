CC = gcc
CFLAGS = -o2 	#-g -Wall
PROJ = serial_main
OBJS = serial_main.o

all : $(PROJ)

serial_main : $(OBJS)
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

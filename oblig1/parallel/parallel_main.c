#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "../simple-jpeg/import_export_jpeg.h"

#define MASTER 0


struct image
{
    float **image_data;  /* a 2D array of floats */
    int m;               /* # pixels in y-direction */
    int n;               /* # pixels in x-direction */
};


float **alloc_2d_float(int rows, int cols) 
{
    float *data = (float *)malloc(rows*cols*sizeof(float));
    float **array= (float **)malloc(rows*sizeof(float*));
    int i;
    for (i=0; i<rows; i++)
        array[i] = &(data[cols*i]);
    
    return array;
}


void allocate_image(struct image *u, int m, int n) 
{
    u->image_data = alloc_2d_float(m, n);
    u->m = m;
    u->n = n;
}


void deallocate_image(struct image *u) 
{
    free(u->image_data[0]);
    free(u->image_data);
}


void convert_jpeg_to_image(const unsigned char* image_chars, struct image *u) 
{
    int i, j, k = 0;
    for (i = 0; i < u->m; i++) {
	for (j = 0; j < u->n; j++) {
	    u->image_data[i][j] = (float)image_chars[k++];
	}
    }
}


void convert_image_to_jpeg(const struct image *u, unsigned char* image_chars) 
{
    int i, j, k = 0;
    for (i = 0; i < u->m; i++) {
	for (j = 0; j < u->n; j++) {
	    image_chars[k++] = u->image_data[i][j];
	}
    }
}


void iso_diffusion_denoising(struct image *u, struct image *u_bar, float kappa) 
{
    int i, j;
    struct image tmp;
    for (i = 1; i < u->m-1; i++) {
	for (j = 1; j < u->n-1; j++) {
	    u_bar->image_data[i][j] = u->image_data[i][j] + kappa * (u->image_data[i-1][j] + u->image_data[i][j-1] -4*u->image_data[i][j]+u->image_data[i][j+1] + u->image_data[i+1][j]);
	}
    }
    
    tmp = *u; 
    *u = *u_bar;
    *u_bar = tmp;
}


/* each process defines its region size */ 
void partition_region(int num_procs, int my_rank, int m, int *my_m, int *from_m) 
{
    *my_m = m/num_procs + 1;
    *from_m = (m/num_procs) * my_rank;
    if (my_rank == num_procs-1) {
	*from_m -= 1;
	*my_m += m%num_procs;
    } else if (my_rank > 0) {
	*from_m -= 1;
	*my_m += 1;
    }
}


/* each process asks process 0 to distribute a region */
void distribute_region(int num_procs, int my_rank, int my_m, int from_m, struct image *u, struct image *u_bar, struct image *whole_image) 
{
    int size[2];
    if (my_rank == MASTER) {
	int i;
	for (i = 1; i < num_procs; i++) {
	    MPI_Recv(&size, 2, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Send(&(whole_image->image_data[size[1]][0]), size[0]*u->n, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
	}

	memcpy(&(u->image_data[0][0]), &(whole_image->image_data[0][0]), u->m*u->n*sizeof(float));
	memcpy(&(u_bar->image_data[0][0]), &(whole_image->image_data[0][0]), u_bar->m*u_bar->n*sizeof(float));
    } else {
	size[0] = my_m;
	size[1] = from_m;
	MPI_Send(&size, 2, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
	MPI_Recv(&(u->image_data[0][0]), u->m*u->n, MPI_FLOAT, MASTER, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}


void swap_borders(int num_procs, int my_rank, int my_m, struct image *u) 
{
    int n = u->n;
    if (my_rank > 0 && my_rank < num_procs-1) {	   
	if (my_rank % 2 == 0) {
	    MPI_Send(&(u->image_data[my_m-2][0]), n, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD);
	    MPI_Send(&(u->image_data[1][0]), n, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD);
	    MPI_Recv(&(u->image_data[0][0]), n, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Recv(&(u->image_data[my_m-1][0]), n, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	} else {
	    MPI_Recv(&(u->image_data[0][0]), n, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Recv(&(u->image_data[my_m-1][0]), n, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Send(&(u->image_data[my_m-2][0]), n, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD);
	    MPI_Send(&(u->image_data[1][0]), n, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD);
	}
    } else if (my_rank == num_procs -1) {
	if (my_rank % 2 != 0) {
	    MPI_Recv(&(u->image_data[0][0]), n, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Send(&(u->image_data[1][0]), n, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD);
	} else {
	    MPI_Send(&(u->image_data[1][0]), n, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD);
	    MPI_Recv(&(u->image_data[0][0]), n, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
    } else {
	MPI_Send(&(u->image_data[my_m-2][0]), n, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD);
	MPI_Recv(&(u->image_data[my_m-1][0]), n, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}


void combine_regions(int num_procs, int my_rank, int my_m, int from_m, struct image *u, struct image *whole_image)
{
    int n = u->n;
    int size[2];
    if (my_rank != MASTER) {
	size[0] = my_m;
	size[1] = from_m;	  
	MPI_Send(&size, 2, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
	MPI_Send(&(u->image_data[0][0]), my_m*n, MPI_FLOAT, MASTER, 1, MPI_COMM_WORLD);
    } else {
	int i;
	memcpy(&(whole_image->image_data[0][0]), &(u->image_data[0][0]), u->m*u->n*sizeof(float));
	for (i = 1; i < num_procs; i++) {
	    MPI_Recv(&size, 2, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    MPI_Recv(&(whole_image->image_data[size[1]][0]), size[0]*n, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
    }
}


int main(int argc, char *argv[])
{
    if (argc != 5) {
	printf("argc = %d\n", argc);
	printf("Use: mpirun -np [cores] %s [number_of_iterations][kappa_value][infile][outfile]\n", argv[0]);
	return -1;
    }

    clock_t begin, end;
    double time_spent;	
    int m, n, c, iters;
    int my_m, from_m, my_rank, num_procs, size[2], w;
    float kappa;
    struct image u; 
    struct image u_bar;
    struct image whole_image;	
    unsigned char *image_chars;
    char *input_jpeg_filename, *output_jpeg_filename;	
    begin = clock();
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_filename */
    iters = strtol(argv[1], (char **)NULL, 10);
    kappa = strtof(argv[2], (char **)NULL);
    input_jpeg_filename = argv[3];
    output_jpeg_filename = argv[4];
    if (my_rank == MASTER) {
	import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
	allocate_image(&whole_image, m, n);
	convert_jpeg_to_image(image_chars, &whole_image);
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* divide the m x n pixels evenly among the MPI processes */
    partition_region(num_procs, my_rank, m, &my_m, &from_m);
    size[0] = my_m;
    size[1] = from_m;
    allocate_image(&u, my_m, n);
    allocate_image(&u_bar, my_m, n);

    /* each process asks process 0 for a partitioned region */
    distribute_region(num_procs, my_rank, my_m, from_m, &u, &u_bar, &whole_image);	
    for (w = 0; w < iters; w++) {
	iso_diffusion_denoising(&u, &u_bar, kappa);
	swap_borders(num_procs, my_rank, my_m, &u); 
    }	

    combine_regions(num_procs, my_rank, my_m, from_m, &u, &whole_image);	
    if (my_rank==0) {
	convert_image_to_jpeg(&whole_image, image_chars);
	export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
	deallocate_image(&whole_image);
	free(image_chars);
    }

    deallocate_image(&u);
    deallocate_image(&u_bar);	
    MPI_Finalize();
    if (my_rank == MASTER) {
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("%f seconds used on %d iterations with %d threads\n", time_spent, iters, num_procs);
    }

    return 0;
}

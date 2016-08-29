#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../simple-jpeg/import_export_jpeg.h"


struct image
{
    float **image_data;  /* a 2D array of floats */
    int m;             /* # pixels in x-direction */
    int n;             /* # pixels in y-direction */
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


void deallocate_image(struct image *u) {
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


void iso_diffusion_denoising(struct image *u, struct image *u_bar, float kappa, int iters) 
{
    int i, j, k;
    struct image tmp;
    for (k = 0; k < iters; k++) {
	for (i = 1; i < u->m-1; i++) {
	    for (j = 1; j < u->n-1; j++) {
		u_bar->image_data[i][j] = u->image_data[i][j] + kappa * (u->image_data[i-1][j] + u->image_data[i][j-1] -4*u->image_data[i][j]+u->image_data[i][j+1] + u->image_data[i+1][j]);
	    }
	}

	tmp = *u; 
	*u = *u_bar;
	*u_bar = tmp;
    }
}


int main(int argc, char *argv[])
{
    if (argc != 5) {
	printf("argc = %d\n", argc);
	printf("Use: %s [number_of_iterations][kappa_value][infile][outfile]\n", argv[0]);
	return -1;
    }

    clock_t begin, end;
    double time_spent;	
    struct image u;
    struct image u_bar;
    int m, n, c;
    begin = clock();
    int iters = strtol(argv[1], (char **)NULL, 10);
    float kappa = strtof(argv[2], (char **)NULL);
    unsigned char *image_chars;
    import_JPEG_file(argv[3], &image_chars, &m, &n, &c);
    allocate_image (&u, m, n);
    allocate_image (&u_bar, m, n);
    convert_jpeg_to_image(image_chars, &u);
    convert_jpeg_to_image(image_chars, &u_bar);
    iso_diffusion_denoising(&u, &u_bar, kappa, iters);
    convert_image_to_jpeg(&u, image_chars);
    deallocate_image(&u);
    deallocate_image(&u_bar);
    export_JPEG_file(argv[4], image_chars, m, n, c, 75);
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%f seconds used on %d iterations serial\n", time_spent, iters);
    free(image_chars);
    return 0;
}

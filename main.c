/**
 * Tony Givargis
 * Copyright (C), 2023
 * University of California, Irvine
 *
 * main.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "ann.h"

#define BATCH  8
#define EPOCHS 4

static int32_t
swap(int32_t x)
{
	union { int32_t i; char b[4]; } in, out;

	in.i = x;
	out.b[0] = in.b[3];
	out.b[1] = in.b[2];
	out.b[2] = in.b[1];
	out.b[3] = in.b[0];
	return out.i;
}

static int
argmax(const double *a, int n)
{
	double max;
	int i, j;

	max = a[0];
	for (i=j=0; i<n; ++i) {
		if (max < a[i]) {
			max = a[i];
			j = i;
		}
	}
	return j;
}

static int
train_and_test(struct ann *ann,
	       const uint8_t *train_y,
	       const uint8_t *train_x,
	       const uint8_t *test_y,
	       const uint8_t *test_x,
	       int train_n,
	       int test_n)
{
	const uint8_t *labels, *images;
	int i, j, k, m, error;
	const double *z;
	double *x, *y;

	x = (double *)malloc(BATCH * 28 * 28 * sizeof (x[0]));
	y = (double *)malloc(BATCH * 10 * sizeof (y[0]));
	assert( x && y );

	/* train */

	m = train_n / BATCH;
	labels = train_y;
	images = train_x;
	for (i=0; i<m; ++i) {
		for (j=0; j<BATCH; ++j) {
			for (k=0; k<(28*28); ++k) {
				x[j * (28*28) + k] = (*images++) / 255.0;
			}
			for (k=0; k<10; ++k) {
				y[j * 10 + k] = 0.0;
			}
			y[j * 10 + (*labels++)] = 1.0;
		}
		ann_train(ann, x, y, 0.1, BATCH);
		printf("\r%06d/%06d", i, m);
		fflush(stdout);
	}

	/* test */

	error = 0;
	labels = test_y;
	images = test_x;
	for (i=0; i<test_n; ++i) {
		for (k=0; k<(28*28); ++k) {
			x[k] = (*images++) / 255.0;
		}
		z = ann_activate(ann, x);
		if (argmax(z, 10) != (int)(*labels++)) {
			error++;
		}
		printf("\r%06d/%06d", i, test_n);
		fflush(stdout);
	}

	printf("\rAccuracy  : %.4f\n", 1.0 - ((double)error / test_n));

	/* done */

	free(x);
	free(y);
	return 0;
}

static uint8_t *
load_labels(const char *pathname, int *n)
{
	int32_t meta[2];
	uint8_t *data;
	FILE *file;

	file = fopen(pathname, "r");
	if (!file) {
		fprintf(stderr, "unable to open file\n");
		return 0;
	}
	if (sizeof (meta) != fread(meta, 1, sizeof (meta), file)) {
		fclose(file);
		fprintf(stderr, "unable to read file\n");
		return 0;
	}
	if ((0x1080000 != meta[0]) || (0 >= swap(meta[1]))) {
		fclose(file);
		fprintf(stderr, "invalid file\n");
		return 0;
	}
	(*n) = swap(meta[1]);
	meta[1] = (*n);
	data = (uint8_t *)malloc(meta[1]);
	assert( data );
	if ((size_t)meta[1] != fread(data, 1, meta[1], file)) {
		free(data);
		fclose(file);
		fprintf(stderr, "unable to read file\n");
		return 0;
	}
	fclose(file);
	return data;
}

static uint8_t *
load_images(const char *pathname, int *n)
{
	int32_t meta[4];
	uint8_t *data;
	FILE *file;

	file = fopen(pathname, "r");
	if (!file) {
		fprintf(stderr, "unable to open file\n");
		return 0;
	}
	if (sizeof (meta) != fread(meta, 1, sizeof (meta), file)) {
		fclose(file);
		fprintf(stderr, "unable to read file\n");
		return 0;
	}
	if ((0x3080000 != meta[0]) ||
	    (0  >= swap(meta[1])) ||
	    (28 != swap(meta[2])) ||
	    (28 != swap(meta[3]))) {
		fclose(file);
		fprintf(stderr, "invalid file\n");
		return 0;
	}
	(*n) = swap(meta[1]);
	meta[1] = (*n) * 28 * 28;
	data = (uint8_t *)malloc(meta[1]);
	assert( data );
	if ((size_t)meta[1] != fread(data, 1, meta[1], file)) {
		free(data);
		fclose(file);
		fprintf(stderr, "unable to read file\n");
		return 0;
	}
	fclose(file);
	return data;
}

int
main()
{
	int train_y_n, train_x_n, test_y_n, test_x_n;
	uint8_t *train_y, *train_x, *test_y, *test_x;
	struct ann *ann;
	int i;

	/* open ANN */

	ann = ann_open(28 * 28, 10, 100, 4);
	assert( ann );

	/* load train/test data */

	train_y = load_labels("data/train-labels", &train_y_n);
	train_x = load_images("data/train-images", &train_x_n);
	test_y = load_labels("data/test-labels", &test_y_n);
	test_x = load_images("data/test-images", &test_x_n);
	assert( train_y && train_x && test_y && test_x );

	/* train and test */

	for (i=0; i<EPOCHS; ++i) {
		printf("--- EPOCH %d ---\n", i);
		if (train_and_test(ann,
				   train_y,
				   train_x,
				   test_y,
				   test_x,
				   train_y_n,
				   test_y_n)) {
			fprintf(stderr, "failed to train/test");
			break;
		}
	}

	/* close */

	ann_close(ann);
	free(train_y);
	free(train_x);
	free(test_y);
	free(test_x);
	return 0;
}

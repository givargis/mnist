/**
 * Tony Givargis
 * Copyright (C), 2023
 * University of California, Irvine
 *
 * ann.c
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "ann.h"

#define FREE(p)					\
	do {					\
		if ((p)) {			\
			free((void *)(p));	\
			(p) = NULL;		\
		}				\
	} while (0)

struct ann {
	int input;
	int output;
	int hidden;
	int layers;
	struct {
		double *w;
		double *b;
		double *a_;
		double *d_;
		double *w_;
		double *b_;
	} *net; /* [layers] */
};

static void
mac1(double *z, const double *a, const double *b, int n, int m)
{
	int i, j;

	for (i=0; i<n; ++i) {
		z[i] = 0.0;
		for (j=0; j<m; ++j) {
			z[i] += a[i * m + j] * b[j];
		}
	}
}

static void
mac2(double *z, const double *a, const double *b, int n, int m)
{
	int i, j;

	for (i=0; i<m; ++i) {
		z[i] = 0.0;
		for (j=0; j<n; ++j) {
			z[i] += a[j * m + i] * b[j];
		}
	}
}

static void
mac3(double *za, const double *b, const double *c, int n, int m)
{
	int i, j;

	for (i=0; i<n; ++i) {
		for (j=0; j<m; ++j) {
			za[i * m + j] += b[i] * c[j];
		}
	}
}

static void
mac4(double *za, const double *b, double s, int n)
{
	int i;

	for (i=0; i<n; ++i) {
		za[i] += b[i] * s;
	}
}

static void
add(double *za, const double *b, int n)
{
	int i;

	for (i=0; i<n; ++i) {
		za[i] += b[i];
	}
}

static void
sub(double *z, const double *a, const double *b, int n)
{
	int i;

	for (i=0; i<n; ++i) {
		z[i] = a[i] - b[i];
	}
}

static void
relu(double *za, int n)
{
	int i;

	for (i=0; i<n; ++i) {
		if (0.0 >= za[i]) {
			za[i] = 0.0;
		}
	}
}

static void
relud(double *za, const double *b, int n)
{
	int i;

	for (i=0; i<n; ++i) {
		if (0.0 >= b[i]) {
			za[i] = 0.0;
		}
	}
}

static int
size(const struct ann *ann, int l)
{
	if (0 == l) {
		return ann->input;
	}
	if (ann->layers == (l + 1)) {
		return ann->output;
	}
	return ann->hidden;
}

static void
randomize(struct ann *ann)
{
	int i, l, n, m;
	double a, b;

	for (l=1; l<ann->layers; ++l) {
		n = size(ann, l);
		m = size(ann, l - 1);
		a = -sqrt(6.0 / (n * m)) * 1.0;
		b = +sqrt(6.0 / (n * m)) * 2.0;
		for (i=0; i<(n*m); ++i) {
			ann->net[l].w[i] = a + (rand() / (double)RAND_MAX) * b;
		}
	}
}

static void
activate_(struct ann *ann, const double *x)
{
	int l, n, m;

	/*
	 * a_[0] := x
	 * a_[l] := activation( w[l] * a_[l - 1] + b[l] )
	 *
	 * activation:
	 *    RELU - internal
	 *    LINEAR - output
	 */

	memcpy(ann->net[0].a_, x, size(ann, 0) * sizeof (ann->net[0].a_[0]));
	for (l=1; l<ann->layers; ++l) {
		n = size(ann, l);
		m = size(ann, l - 1);
		mac1(ann->net[l].a_, ann->net[l].w, ann->net[l - 1].a_, n, m);
		add(ann->net[l].a_, ann->net[l].b, n);
		if ((l + 1) < ann->layers) {
			relu(ann->net[l].a_, n);
		}
	}
}

static void
backprop_(struct ann *ann, const double *y)
{
	int l, n, m;

	/*
	 * start with last layer
	 */

	l = ann->layers - 1;

	/*
	 * using: Quadratic Cost Function
	 *
	 * d_[L] := a_[L] − y
	 */

	sub(ann->net[l].d_, ann->net[l].a_, y, size(ann, l));

	/*
	 * d_[l] := (w[l+1]' * d_[l+1]) ⊙ σ′(a_[l])
	 */

	while (1 < l) {
		n = size(ann, l);
		m = size(ann, l - 1);
		mac2(ann->net[l - 1].d_, ann->net[l].w, ann->net[l].d_, n, m);
		relud(ann->net[l - 1].d_, ann->net[l - 1].a_, m);
		--l;
	}

	/*
	 * b_[l] := b_[l] + d_[l]
	 * w_[l] := w_[l] + d_[l] * a_[l - 1]
	 */

	for (l=1; l<ann->layers; ++l) {
		n = size(ann, l);
		m = size(ann, l - 1);
		add(ann->net[l].b_, ann->net[l].d_, n);
		mac3(ann->net[l].w_, ann->net[l].d_, ann->net[l - 1].a_, n, m);
	}
}

struct ann *
ann_open(int input, int output, int hidden, int layers)
{
	struct ann *ann;
	int l, n, m;

	assert( (1 <= input) && (1000000 >= input) );
	assert( (1 <= output) && (1000000 >= output) );
	assert( (1 <= hidden) && (1000000 >= hidden) );
	assert( (3 <= layers) && (20 >= layers) );

	/* initialize */

	if (!(ann = malloc(sizeof (struct ann)))) {
		fprintf(stderr, "out of memory");
		return NULL;
	}
	memset(ann, 0, sizeof (struct ann));

	/* initialize */

	ann->input = input;
	ann->output = output;
	ann->hidden = hidden;
	ann->layers = layers;

	/* initialize */

	if (!(ann->net = malloc(ann->layers * sizeof (ann->net[0])))) {
		ann_close(ann);
		fprintf(stderr, "out of memory");
		return NULL;
	}

	/* initialize */

	for (l=0; l<ann->layers; ++l) {
		n = size(ann, l);
		ann->net[l].a_ = malloc(n * sizeof (ann->net[0].a_));
		ann->net[l].d_ = malloc(n * sizeof (ann->net[0].d_));
		if (!ann->net[l].a_ || !ann->net[l].d_) {
			ann_close(ann);
			fprintf(stderr, "out of memory");
			return NULL;
		}
		memset(ann->net[l].a_, 0, n * sizeof (ann->net[0].a_));
		memset(ann->net[l].d_, 0, n * sizeof (ann->net[0].d_));
	}

	/* initialize */

	for (l=1; l<ann->layers; ++l) {
		n = size(ann, l);
		m = size(ann, l - 1);
		ann->net[l].w = malloc(n * m * sizeof (ann->net[0].w));
		ann->net[l].b = malloc(n * 1 * sizeof (ann->net[0].b));
		ann->net[l].w_ = malloc(n * m * sizeof (ann->net[0].w_));
		ann->net[l].b_ = malloc(n * 1 * sizeof (ann->net[0].b_));
		if (!ann->net[l].w ||
		    !ann->net[l].b ||
		    !ann->net[l].w_ ||
		    !ann->net[l].b_) {
			ann_close(ann);
			fprintf(stderr, "out of memory");
			return NULL;
		}
		memset(ann->net[l].w, 0, n * m * sizeof (ann->net[0].w));
		memset(ann->net[l].b, 0, n * 1 * sizeof (ann->net[0].b));
		memset(ann->net[l].w_, 0, n * m * sizeof (ann->net[0].w_));
		memset(ann->net[l].b_, 0, n * 1 * sizeof (ann->net[0].b_));
	}
	randomize(ann);
	return ann;
}

void
ann_close(struct ann *ann)
{
	int l;

	if (ann) {
		if (ann->net) {
			for (l=0; l<ann->layers; ++l) {
				FREE(ann->net[l].a_);
				FREE(ann->net[l].d_);
			}
			for (l=1; l<ann->layers; ++l) {
				FREE(ann->net[l].w);
				FREE(ann->net[l].b);
				FREE(ann->net[l].w_);
				FREE(ann->net[l].b_);
			}
		}
		FREE(ann->net);
		memset(ann, 0, sizeof (struct ann));
	}
	FREE(ann);
}

const double *
ann_activate(struct ann *ann, const double *x)
{
	assert( ann );
	assert( x );

	activate_(ann, x);
	return ann->net[ann->layers - 1].a_;
}

void
ann_train(struct ann *ann,
	  const double *x,
	  const double *y,
	  double eta,
	  int k)
{
	int i, l, n, m;

	assert( ann );
	assert( x && y );
	assert( (0.0 < eta) && (1.0 >= eta) );
	assert( (1 <= k) && (128 >= k) );

	/*
	 * w_[*] := 0.0
	 * b_[*] := 0.0
	 */

	for (l=1; l<ann->layers; ++l) {
		n = size(ann, l);
		m = size(ann, l - 1);
		memset(ann->net[l].w_, 0, n * m * sizeof (ann->net[0].w[0]));
		memset(ann->net[l].b_, 0, n * 1 * sizeof (ann->net[0].b[0]));
	}

	/*
	 * for all (x -> y):
	 *   activate()
	 *   backprop()
	 */

	for (i=0; i<k; ++i) {
		activate_(ann, x + i * ann->input);
		backprop_(ann, y + i * ann->output);
	}

	/*
	 * w[l] := w[l] - ( (η / k) * w_[l] )
	 * b[l] := b[l] - ( (η / k) * b_[l] )
	 */

	for (l=1; l<ann->layers; ++l) {
		n = size(ann, l);
		m = size(ann, l - 1);
		mac4(ann->net[l].w, ann->net[l].w_, -eta / k, n * m);
		mac4(ann->net[l].b, ann->net[l].b_, -eta / k, n * 1);
	}
}

/**
 * Tony Givargis
 * Copyright (C), 2023
 * University of California, Irvine
 *
 * ann.h
 */

#ifndef _ANN_H_
#define _ANN_H_

struct ann;

/**
 * Opens (i.e., creates and initializes) a new ann handle.
 *
 * input : number of input neurons
 * output: number of output neurons
 * hidden: number of hidden neurons per layer
 * layers: number of layers including input/output layers
 *
 * return: an ann handle or NULL on error
 */

struct ann *ann_open(int input, int output, int hidden, int layers);

/**
 * Closes a previously opened ann handle.
 *
 * ann: an ann handle previously obtained by calling ann_open() or NULL
 */

void ann_close(struct ann *ann);

/**
 * Activates the ann.
 *
 * ann: an ann handle previously obtained by calling ann_open()
 * x  : an array of values with size equal to input argument of ann_open()
 *
 * return: an array of values with size equal to output argument of ann_open()
 *
 * note: The return vector memory is only valid while the ann handle is
 *       open. The content of the return vector will be stable until a
 *       subsequent call to ann_activate() or ann_train(). This function
 *       is not re-entrant.
 */

const double *ann_activate(struct ann *ann, const double *x);

/**
 * Trains the ann.
 *
 * ann: an ann handle previously obtained by calling ann_open()
 * x  : an array of input vectors equal to input argument of ann_open() x k
 * y  : an array of output vectors equal to output argument of ann_open() x k
 * eta: learning rate
 * k  : batch size equal to 1 or more
 *
 * note: The function will use k to determine the number of input/output pairs
 *       used in this round of training. The ann, when opened will be in a
 *       random state of training. Each subsequent call to ann_train() will
 *       learn at the specified learning rate. The x/y vectors are assumed to
 *       be 2d C arrays (i.e., row-wise ordering of the input/output pairs).
 */

void ann_train(struct ann *ann,
	       const double *x,
	       const double *y,
	       double eta,
	       int k);

#endif /* _ANN_H_ */

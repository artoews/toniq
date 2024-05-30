""" Functions extending the sigpy Linop module.

"""
import numpy as np
import sigpy as sp

def get_matrix(op: sp.linop.Linop, verify=False, x=None):
    """ Returns the matrix equivalent of a sigpy Linop. """
    output_size = np.prod(op.oshape)
    input_size = np.prod(op.ishape)
    if input_size != np.max(op.ishape):
        op = op * sp.linop.Reshape(op.ishape, (input_size,))
    if output_size != np.max(op.oshape):
        op = sp.linop.Reshape((output_size,), op.oshape) * op
    unit_vectors = np.eye(input_size)
    matrix_columns = []
    for i in range(input_size):
        col_i = op(unit_vectors[i])
        matrix_columns.append(col_i)
    mtx = np.stack(matrix_columns, axis=-1)
    if verify:
        if x is None:
            x = np.random.rand(input_size)
        y1 = op(x)
        y2 = mtx @ x
        if not np.allclose(y1, y2, rtol=1e-4, atol=1e-7):
            print('Warning: computed matrix is not equivalent to the provided linop. Input {} yields outputs {} and {} for linop and matrix respectively, with max error {}'.format(x, y1, y2, np.max(np.abs(y2 - y1))))
    return mtx

if __name__ == '__main__':

    shape = (128, 128)
    op = sp.linop.FFT(shape)
    x = np.random.rand(*shape)

    mtx = get_matrix(op, verify=True, x=x.ravel())
    y1 = mtx @ x.ravel()
    y2 = sp.fft(x).ravel()
    if not np.allclose(y1, y2, rtol=1e-4, atol=1e-7):
        print('Warning: computed matrix does not give the same result as built-in fft operation.')

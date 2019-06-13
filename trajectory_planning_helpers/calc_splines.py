import numpy as np


def calc_splines(path: np.ndarray,
                 el_lengths: np.ndarray = None,
                 psi_s: float = None,
                 psi_e: float = None,
                 use_dist_scaling: bool = True) -> tuple:
    """
    Created by:
    Tim Stahl & Alexander Heilmeier

    Documentation:
    Solve for a curvature continous cubic spline between given poses

                    P_{x,y}   = a3 *t³ + a2 *t² + a1*t + a0
                    P_{x,y}'  = 3a3*t² + 2a2*t  + a1
                    P_{x,y}'' = 6a3*t² + 2a2

                    a * {x; y} = {b_x; b_y}

    Input:
    - path:             x and y coordinates as the basis for the spline construction
    - el_lengths:       distances between path points
    - psi_{s,e}:        orientation of the {start, end} point
    - use_dist_scaling: bool flag to indicate if heading and curvature scaling should be performed. This is required
                        if the distances between the points in the path are not equal.

    path and el_lengths inputs can either be closed or unclosed, but must be consistent!

    Output:
    - x_coeff:              spline coefficients of the x-component
    - y_coeff:              spline coefficients of the y-component
    - M:                    LES coefficients
    - normvec_normalized:   normalized normal vectors

    Coefficient matrices have the form a_i, b_i * t, c_i * t^2, d_i * t^3.
    """

    # check if path is closed
    if np.all(np.isclose(path[0], path[-1])):
        closed = True
    else:
        closed = False

    # check inputs
    if not closed and (psi_s is None or psi_e is None):
        raise IOError("Headings must be provided for unclosed spline calculation!")

    if el_lengths is not None and path.shape[0] != el_lengths.size + 1:
        raise IOError("el_lengths input must be one element smaller than path input!")

    # if distances between path coordinates are not provided but required, calculate euclidean distances as el_lengths
    if use_dist_scaling and el_lengths is None:
        el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))

    # get number of splines
    no_splines = path.shape[0] - 1

    # calculate scaling factors between every pair of splines
    if use_dist_scaling:
        scaling = el_lengths[:-1] / el_lengths[1:]
    else:
        scaling = np.ones(no_splines - 1)

    # ------------------------------------------------------------------------------------------------------------------
    # DEFINE LINEAR EQUATION SYSTEM ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # M_{x,y} * a_{x,y} = b_{x,y}) with a_{x,y} being the desired spline param
    # *4 because of 4 parameters in cubic spline
    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))

    # create template for M array entries
    template_M = np.array(                          # current time step           | next time step          | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0

    for i in range(no_splines):
        j = i * 4

        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M

            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= np.power(scaling[i], 2)

        else:
            M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],  # no curvature and heading bounds on last element
                                     [1,  1,  1,  1]]

        b_x[j: j + 2] = [[path[i,     0]],      # NOTE: the bounds of the two last equations remain zero
                         [path[i + 1, 0]]]
        b_y[j: j + 2] = [[path[i,     1]],
                         [path[i + 1, 1]]]

    # ------------------------------------------------------------------------------------------------------------------
    # SET BOUNDARY CONDITIONS FOR FIRST AND LAST POINT -----------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not closed:
        # We want to fix heading at the start and end point (heading and curvature at start gets unbound at spline end)
        # heading and curvature boundary condition
        M[-2, 1] = 1                  # heading start
        M[-1, -4:] = [0,  1,  2,  3]  # heading end

        # heading start
        if el_lengths is None:
            el_length_s = 1.0
        else:
            el_length_s = el_lengths[0]

        b_x[-2] = np.cos(psi_s + np.pi / 2) * el_length_s
        b_y[-2] = np.sin(psi_s + np.pi / 2) * el_length_s

        # heading end
        if el_lengths is None:
            el_length_e = 1.0
        else:
            el_length_e = el_lengths[-1]

        b_x[-1] = np.cos(psi_e + np.pi / 2) * el_length_e
        b_y[-1] = np.sin(psi_e + np.pi / 2) * el_length_e

    else:
        # gradient boundary conditions (for a closed spline)
        M[-2, 1] = scaling[-1]
        M[-2, -3:] = [-1, -2, -3]
        # b_x[-2] = 0
        # b_y[-2] = 0

        # curvature boundary conditions (for a closed spline)
        M[-1, 2] = 2 * np.power(scaling[-1], 2)
        M[-1, -2:] = [-2, -6]
        # b_x[-1] = 0
        # b_y[-1] = 0

    # ------------------------------------------------------------------------------------------------------------------
    # SOLVE ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    x_les = np.squeeze(np.linalg.solve(M, b_x))  # squeeze removes single-dimensional entries
    y_les = np.squeeze(np.linalg.solve(M, b_y))

    # get coefficients of every piece into one row -> reshape
    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))

    # get normal vector (second coefficient of cubic splines is relevant for the gradient)
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

    # normalize normal vectors
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    norm_factors = np.expand_dims(norm_factors, axis=1)  # second dimension must be inserted for next step
    normvec_normalized = norm_factors * normvec

    return coeffs_x, coeffs_y, M, normvec_normalized


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass

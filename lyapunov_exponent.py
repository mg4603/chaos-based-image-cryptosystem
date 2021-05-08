from numpy import *
from numpy.linalg import *


def lyapunov_exponent(f_df, initial_conditions=None, tol=0.01, max_it=1000,
                         min_it_percentage=0.1):
    """Numerical approximation of all Lyapunov Exponents of a map.
    Parameters
    ----------
    f_df : callable
        The map to be considered for computation of Lyapunov Exponents
        f_df (x, w) -> (f(x), df(x, w)), where
        f(x) is the map evaluated at the point x and
        df(x, w) is the differential of this map evaluated at x in the direction of w.
    single_initial_conditions : ndarray of shape (n)
        Initial conditions to compute the Lyapunov Exponent.
    tol : float
        Tolerance to stop the approximation.
    max_it : int
        Max numbers of iterations.
    min_it_percentage : float, optional
        Min number of iterations as a percentage of the max_it.
    Returns
    -------
    : ndarray of shape (n)
        The Lyapunov exponents computed associated to a single initial condition or
        the average value considering several initial conditions
    """

    if initial_conditions is None:
        raise Exception('initial conditions must be provided.')

    n = len(initial_conditions)
    x = initial_conditions
    w = eye(n)
    h = zeros(n)
    trans_it = int(max_it * min_it_percentage)
    l = -1

    for i in range(0, max_it):
        x_next, w_next = f_df(x, w)
        w_next = orthogonalize_columns(w_next)

        h_next = h + log_of_the_norm_of_the_columns(w_next)
        l_next = h_next / (i + 1)

        if i > trans_it and norm(l_next - l) < tol:
            return sort(l_next)

        h = h_next
        x = x_next
        w = normalize_columns(w_next)
        l = l_next

    raise Exception('Lyapunov Exponents computation did no convergence' +
                    ' at ' + str(initial_conditions) +
                    ' with tol=' + str(tol) +
                    ' max_it=' + str(max_it) +
                    ' min_it_percentage=' + str(min_it_percentage))


  

    
def orthogonalize_columns(a):
    q, r = qr(a)
    return q @ diag(r.diagonal())


def normalize_columns(a):
    return apply_along_axis(lambda v: v / norm(v), 0, a)


def log_of_the_norm_of_the_columns(a):
    return apply_along_axis(lambda v: log(norm(v)), 0, a)
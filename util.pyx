from libc.math cimport exp
from libc.math cimport log
from libc.math cimport floor
from libc.math cimport fabs


cdef double log_sum_exp_pair(double d1, double d2):
    """
    Do the log-sum-exp trick for a pair of numbers
    :param d1: first number
    :param d2: second number
    :return: log(exp(d1) + exp(d2))
    """
    cdef double result
    if d1 > d2:
        result = d1 + log(1 + exp(d2 - d1))
    else:
        result = d2 + log(1 + exp(d1 - d2))
    return result


cdef double log_sum_exp_vec(double[:] vec, int length):
    """
    Do the log-sum-exp trick for a vector of numbers of a given length
    :param vec: a one-dimensional typed memory view
    :param length: the number of elements in vec
    :return log(\sum_i exp(vec_i))
    """
    cdef double running_sum = vec[0]
    cdef int k = 1
    # do the log-sum-exp one element at a time
    while k < length:
        running_sum = log_sum_exp_pair(running_sum, vec[k])
        k += 1
    return running_sum


cdef double digamma(double x):
    """
    Compute the digamma function of a single number.
    This implementation was adapted from the original lda-c code.
    It seems pretty solid; almost no differences more than 1e-7 from scipy, and much faster.
    :param x: the input value
    :return: digamma(x), i.e. d/dx ln (Gamma(x))
    """
    x = x + 6
    cdef double p = 1 / (x * x)
    p = ((( 0.004166666666667 * p - 0.003968253986254 ) * p + 0.008333333333333) * p - 0.083333333333333) * p
    p = p + log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3) - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6)
    return p


cdef double Gamma(double x):
    """
    Compute gamma for x > 0 and x < 171.624:
    Ported from https://www.johndcook.com/Gamma.cpp
    :param x: the input value
    :return: Gamma(x)
    """
    assert x > 0.0
    assert x < 171.624

    # Split the function domain into three intervals:
    # (0, 0.001), [0.001, 12), and (12, 171.624)

    # First interval: (0, 0.001)
    # For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
    # So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
    # The relative error over this interval is less than 6e-7.

    cdef:
        double gamma_const = 0.577215664901532860606512090  # Euler's gamma constant

        double y = 0
        int n = 0
        int arg_was_less_than_one = 0
        double num = 0.0
        double den = 1.0
        int i = 0
        double z = 0
        double result = 0


    # numerator coefficients for approximation over the interval (1,2)
    cdef double p[8]
    p[:] = [
        -1.71618513886549492533811E+0,
         2.47656508055759199108314E+1,
        -3.79804256470945635097577E+2,
         6.29331155312818442661052E+2,
         8.66966202790413211295064E+2,
        -3.14512729688483675254357E+4,
        -3.61444134186911729807069E+4,
         6.64561438202405440627855E+4
    ]

    # denominator coefficients for approximation over the interval (1,2)
    cdef double q[8]
    q[:] = [
        -3.08402300119738975254353E+1,
         3.15350626979604161529144E+2,
        -1.01515636749021914166146E+3,
        -3.10777167157231109440444E+3,
         2.25381184209801510330112E+4,
         4.75584627752788110767815E+3,
        -1.34659959864969306392456E+5,
        -1.15132259675553483497211E+5
    ]


    if x < 0.001:
        return 1.0 / (x * (1.0 + gamma_const * x))

    # Second interval: [0.001, 12)

    elif x < 12.0:
        # The algorithm directly approximates gamma over (1,2) and uses
        # reduction identities to reduce other arguments to this interval.

        y = x
        n = 0
        if x < 1.0:
            arg_was_less_than_one = 1

        # Add or subtract integers as necessary to bring y into (1,2)
        # Will correct for this below
        if arg_was_less_than_one > 0:
            y += 1.0
        else:
            n = int(floor(y)) - 1  # will use n later
            y -= n

        z = y -1

        i = 0
        while i < 8:
            num = (num + p[i]) * z
            den = den * z + q[i]
            i += 1

        result = num/den + 1.0

        # Apply correction if argument was not initially in (1,2)
        if arg_was_less_than_one > 0:
            # Use identity gamma(z) = gamma(z+1)/z
            # The variable "result" now holds gamma of the original y + 1
            # Thus we use y-1 to get back the orginal y.
            result = result / (y-1.0)
        else:
            # Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
            i = 0
            while i < n:
                result *= y
                y += 1
                i += 1

        return result

    # Third interval: [12, 171.624)
    else:
        return exp(log_Gamma(x))


cdef double log_Gamma(double x):
    """
    Compute ln(Gamma(x)) for x > 0
    Ported from https://www.johndcook.com/Gamma.cpp
    :param x: the input value
    :return: ln(Gamma(x))
    """

    assert x > 0.0

    if x < 12.0:
        return log(fabs(Gamma(x)))

    # Abramowitz and Stegun 6.1.41
    # Asymptotic series should be good to at least 11 or 12 figures
    # For error analysis, see Whittiker and Watson
    # A Course in Modern Analysis (1927), page 252

    cdef double c[8]
    c[:] = [
         1.0/12.0,
        -1.0/360.0,
        1.0/1260.0,
        -1.0/1680.0,
        1.0/1188.0,
        -691.0/360360.0,
        1.0/156.0,
        -3617.0/122400.0
    ]

    cdef double z = 1.0 / (x * x)
    cdef double w_sum = c[7]
    cdef int i = 6
    while i > -1:
        w_sum *= z
        w_sum += c[i]
        i -= 1

    cdef double series = w_sum/x

    cdef double halfLogTwoPi = 0.91893853320467274178032973640562
    cdef double logGamma = (x - 0.5) * log(x) - x + halfLogTwoPi + series
    return logGamma


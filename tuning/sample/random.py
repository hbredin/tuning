#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013  Herve BREDIN (http://herve.niderb.fr/)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np


class RandomComposite(object):
    """
    >>> generator = RandomComposite(alphas=RandomWeights(3),
                                    threshold=RandomFloat(low=0, high=1))
    >>> next(generator)
    {'alphas': (0.52164076924710034, 0.081846657781224491, 0.3965125729716753),
     'threshold': 0.14020107784554892}
    """

    def __init__(self, **kwargs):
        super(RandomComposite, self).__init__()
        self.params = dict(**kwargs)

    def __iter__(self):
        return self

    def next(self):
        return dict([(key, next(v))
                    for key, v in self.params.iteritems()])


class RandomChoice(object):

    def __init__(self, *args):
        super(RandomChoice, self).__init__()
        self.choices = tuple(args)
        self.n = len(self.choices)

    def __iter__(self):
        return self

    def next(self):
        i = np.random.randint(0, self.n)
        return self.choices[i]


class RandomInt(object):
    """
    Generate random integers from `low` (inclusive) to `high` (exclusive).

    Generate random integers from the "discrete uniform" distribution in the
    "half-open" interval [`low`, `high`). If `high` is None (the default),
    then results are from [0, `low`).

    >>> generator = RandomInt(10)
    >>> i = next(generator)
    >>> assert isinstance(i, int) and i >= 0 and i < 10
    """

    def __init__(self, low, high=None):
        super(RandomInt, self).__init__()
        self.low = low
        self.high = high

    def __iter__(self):
        return self

    def next(self):
        return np.random.randint(self.low, high=self.high)


class RandomFloat(object):
    """
    Generate random floats from `low` (inclusive) to `high` (exclusive).

    Generate random floats from the uniform distribution in the
    "half-open" interval [`low`, `high`). If `high` is None (the default),
    then results are from [0, `low`).

    >>> generator = RandomFloat(10)
    >>> v = next(generator)
    >>> assert isinstance(v, float) and v >= 0 and v < 10
    """

    def __init__(self, low, high=None):
        super(RandomFloat, self).__init__()
        if high:
            self.low = low
            self.high = high
        else:
            self.low = 0.
            self.high = low

    def __iter__(self):
        return self

    def next(self):
        value = np.random.random()
        return self.low + value * (self.high-self.low)


class RandomWeights(object):
    """
    Generate random tuples of `number` positive floats summing to 1
    with Dirichlet distribution

    >>> generator = RandomWeights(4)
    >>> w = next(generator)
    >>> assert isinstance(w, tuple) and len(w) == 4 and sum(w) == 1.
    """

    def __init__(self, number):
        super(RandomWeights, self).__init__()
        self.alpha = (1,) * number

    def __iter__(self):
        return self

    def next(self):
        return tuple(np.random.dirichlet(self.alpha, 1)[0])

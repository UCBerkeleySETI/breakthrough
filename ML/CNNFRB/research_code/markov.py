import numpy as np

def markov(L, p, q=None):
    if q is None:
        q = p
    rands = np.random.random(L)
    change = rands > p; qchange = rands > q
    func = np.zeros_like(rands)
    for i in xrange(L-1):
        if (func[i] == 1 and change[i]) or (func[i] == 0 and qchange[i]):
            func[i+1] = 1-func[i]
        else:
            func[i+1] = func[i]
    #print func
    return func

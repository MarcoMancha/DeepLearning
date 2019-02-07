
def main():
    xs = [59,52,44,51,42]
    ys = [56,63,55,50,66]
    n = len(xs)
    m = 0.5
    b = 1
    a = 0.0003

    for i in range(100000):
        acum0 = 0
        acum1 = 0
        for i,x in enumerate(xs):
            acum0 = acum0 + ((b + (xs[i] * m)) - ys[i])
            acum1 = acum1 + (((b + (xs[i] * m)) - ys[i]) * xs[i])
        t0 = b - (a * (acum0 / n))
        t1 = m - (a * (acum1 / n))
        b = t0
        m = t1
    print(str(b) + " + "+str(m)+"x")
    new = b + 43 * m
    print("VALOR 43: "+str(new))
if __name__ == '__main__':
    main()

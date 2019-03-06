def main():
    xs = [59,52,44,51,42]
    ys = [56,63,55,50,66]
    n = len(xs)
    m = -0.317668947095 # 0.5
    b = 73.7436118606 #1
    a = 0.0003
    old = 0
    b0 = 0
    b1 = 0
    epochs = 0
    x = input("Valor a predecir: ")
    while(True and epochs < 1000000):
        acum0 = 0
        acum1 = 0
        for i,x in enumerate(xs):
            acum0 = acum0 + ((b + (xs[i] * m)) - ys[i])
            acum1 = acum1 + (((b + (xs[i] * m)) - ys[i]) * xs[i])
        t0 = b - (a * (acum0 / n))
        t1 = m - (a * (acum1 / n))
        b = t0
        m = t1
        print(str(b) + " "+str(m)+"x")
        new = b + x * m
        if old == new or b0 == b or b1 == m:
            break
        else:
            old = new
            b0 = b
            b1 = m
        epochs = epochs + 1
    print("VALOR PREDECIDO: "+str(new))
if __name__ == '__main__':
    main()

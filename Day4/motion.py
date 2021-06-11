def read(filename, rotate=True, alpha=1.0/12.0):
    """Read and preprocess the motion file.

    A simple moving average filter with window size N can be
    approximated by an EMA with alpha=2/(N+1).
    """
    Acc = list()
    Grav = list()
    Mag = list()
    Rot = list()

    # Read data
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        for row in csvreader:
            if row[1] == '2':
                a = np.array([float(row[0]), float(row[2]),
                             float(row[3]), float(row[4])])
                Acc.append(a)
            elif row[1] == '1':  # Gravity
                g = np.array([float(row[0]), float(row[2]),
                             float(row[3]), float(row[4])])
                Grav.append(g)
                if len(Mag) > 0:
                    Rot.append(
                        [float(row[0]), getRotationMatrix(Grav[-1], Mag[-1])])
            elif row[1] == '4':  # Magnetic field
                m = np.array([float(row[0]), float(row[2]),
                             float(row[3]), float(row[4])])
                Mag.append(m)
                if len(Grav) > 0:
                    Rot.append(
                        [float(row[0]), getRotationMatrix(Grav[-1], Mag[-1])])

    # Rotate acceleration vectors
    if rotate:
        k = 0
        for i in range(len(Acc)):
            while k < len(Rot) and Acc[k][0] >= Rot[k][0]:
                k = k + 1
            av = np.array(Acc[i][1:])
            # a = av.dot(Rot[k][1])
            a = (Rot[k][1]).dot(av)
            for j in range(1, 4):
                Acc[i][j] = a[0, j-1]

    if alpha:  # Exponentially fading filter
        for i in range(1, len(Acc)):
            ts = Acc[i][0]  # remember timestamp, we do not want to filter it
            Acc[i] = Acc[i] * alpha + Acc[i-1] * (1.0-alpha)
            Acc[i][0] = ts
    return Acc

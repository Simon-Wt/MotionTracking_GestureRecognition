import csv
import statistics


def calcAvg(values):
    if (len(values) == 0):
        return 0
    return sum(values) / len(values)


def calcAndPrintSingle(v, sensor):
    print("######################################")
    print("Sensor:", sensor)
    if v == []:
        print("no values found")
        return
    print("Average of v", calcAvg(v))
    print("Standart Deviation of v", statistics.pstdev(v))


def calcAndPrint(x, y, z, sensor):
    if x == [] or y == [] or z == []:
        print("no values found for the sensor", sensor)
        return
    print("######################################")
    print("Sensor:", sensor)
    print("Average of x", calcAvg(x))
    print("Average of y", calcAvg(y))
    print("Average of z", calcAvg(z))
    print("Standart Deviation of x", statistics.pstdev(x))
    print("Standart Deviation of y", statistics.pstdev(y))
    print("Standart Deviation of z", statistics.pstdev(z))


counter = 0
a1, a2, a3 = [], [], []
g1, g2, g3 = [], [], []
m1, m2, m3 = [], [], []
p = []
with open("Tag2/5-5-5-5.csv") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=";")
    # offset = 13000 * 0.05
    # end = 90000 * 0.95
    for row in csvreader:
        # counter += 1
        # if counter < offset or counter > end:
        #     continue
        if row[1] == '0':  # ACCELEROMETER
            a1.append(float(row[2]))
            a2.append(float(row[3]))
            a3.append(float(row[4]))

        elif row[1] == '3':  # GYROSCOPE
            g1.append(float(row[2]))
            g2.append(float(row[3]))
            g3.append(float(row[4]))

        elif row[1] == '4':  # MAGNETIC FIELD
            m1.append(float(row[2]))
            m2.append(float(row[3]))
            m3.append(float(row[4]))

        elif row[1] == '5':
            p.append(float(row[2]))

    calcAndPrint(a1, a2, a3, "Accelerometer")
    calcAndPrint(g1, g2, g3, "Gyroscope")
    calcAndPrint(m1, m2, m3, "Mgnetic Field")
    calcAndPrintSingle(p, "Pressure")

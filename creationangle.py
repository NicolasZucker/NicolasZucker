import numpy
from matplotlib import pyplot
import math
N=10000;
def creaangles(N):
    OR=numpy.ndarray(shape=(N,4), dtype=float);
    OR[:,:]=2;
    for i in range(N):
        normmm=numpy.linalg.norm(OR[i,:]);
        while normmm> 1:
            OR[i,:]= [  ((numpy.random.rand(1)[0] - 0.5) * 2.0),
            ((numpy.random.rand(1)[0] - 0.5) * 2.0),
            ((numpy.random.rand(1)[0] - 0.5) * 2.0),
            ((numpy.random.rand(1)[0] - 0.5) * 2.0)]
            normmm=numpy.linalg.norm(OR[i,:]);
    Rad=numpy.ndarray(shape=(N,3), dtype=float);
    return(OR)

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

print(euler_from_quaternion( 8.12348062e-01,-1.64967985e-02,-1.24628608e-01 ,4.36515467e-01))

creaangles(N)
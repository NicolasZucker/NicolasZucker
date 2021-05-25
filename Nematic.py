import hoomd
import hoomd.md
import numpy
import math
from numpy.core.numeric import True_
from numpy.lib.shape_base import expand_dims


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
    return(OR)

hoomd.context.initialize(""); 

N=60; #nombre de partciules
L=100;

## Création d'un snapshot pour randomiser les positions initiales
snap = hoomd.data.make_snapshot(N, box=hoomd.data.boxdim(L=100));
numpy.random.seed(12);
my_position= (numpy.random.random((N,3)) * 2 - 1)*(L/2); #10 = taille de la box/2 ici
snap.particles.position[:] = my_position[:];

snap.particle.moment_inertia[:]=my_position[:];
## Création d'orientation random avec quaternions
for i in range(N):  
    snap.particles.orientation[i] = creaangles(N)[i];


hoomd.init.read_snapshot(snap);
#print(snap.particles.position) #ça marche pas ??

## Définition du potentiel d'intéraction
nl = hoomd.md.nlist.cell()

mie = hoomd.md.pair.mie(r_cut=1.0, nlist=nl)
mie.pair_coeff.set('A', 'A', epsilon=1.0, sigma=(2.0)**(1/6), n=12, m=6)
dip = hoomd.md.pair.dipole(r_cut=25, nlist=nl, name=None)
dip.pair_coeff.set('A', 'A', mu=1.0, A=1.0, kappa=0)


##Création de l'activité
#import numpy
#activity = [ ( ((numpy.random.rand(1)[0] - 0.5) * 2.0),
#               ((numpy.random.rand(1)[0] - 0.5) * 2.0),
 #              ((numpy.random.rand(1)[0] - 0.5) * 2.0)) 
  #           for i in range(N)];
#all=hoomd.group.all();

#hoomd.md.force.active(group=all,seed=123,f_lst=activity,rotation_diff=0.005, orientation_link=False);



hoomd.md.integrate.mode_standard(dt=0.001,aniso=None);


hoomd.md.integrate.brownian(group=all, kT=0.02, seed=123);

hoomd.analyze.log(filename="log-output.log",quantities=['potential_energy', 'temperature'],period=100,overwrite=True);

hoomd.dump.gsd("trajectory.gsd", period=2e2, group=all, overwrite=True);

hoomd.run(1e6);



# Analysis 
import numpy
from matplotlib import pyplot

data = numpy.genfromtxt(fname='log-output.log', skip_header=True);

pyplot.figure(figsize=(4,2.2), dpi=140);
pyplot.plot(data[:,0], data[:,1]);
pyplot.xlabel('time step');
pyplot.ylabel('potential_energy');



pyplot.figure(figsize=(4,2.2), dpi=140);
pyplot.plot(data[:,0], data[:,2]);
pyplot.xlabel('time step');
pyplot.ylabel('temperature');


pyplot.show()


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

# cd Documents/Pythonhommd
# cd Documents/Python_ext/ovito-basic-3.5.0-x86_64/bin